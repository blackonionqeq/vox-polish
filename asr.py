import whisperx
import gc
import json
import torch
import os
import sys
import argparse
import inspect
from typing import Any, Dict, List, Optional

# Optional: load HF_TOKEN from .env (like JS dotenv)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

# Satisfy PyTorch 2.6+ security requirements for loading models
# Some dependencies (pyannote, whisperx) still use torch.load with complex types
import torch.serialization
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# Ensure cuDNN and other CUDA DLLs are found on Windows
if sys.platform == "win32":
    venv_site_packages = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages")
    cuda_paths = [
        os.path.join(venv_site_packages, "nvidia", "cudnn", "bin"),
        os.path.join(venv_site_packages, "nvidia", "cublas", "bin"),
        os.path.join(venv_site_packages, "nvidia", "cuda_runtime", "bin"),
    ]
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"Adding DLL directory: {path}")
            os.add_dll_directory(path)

def transcribe_video(
    video_path: str,
    language: str = "ja",
    *,
    vad_filter: bool = True,
    vad_threshold: float = 0.5,
    vad_min_silence_ms: int = 200,
    vad_speech_pad_ms: int = 200,
    condition_on_previous_text: bool = True,
    chunk_length: Optional[int] = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4 # reduce if low on GPU mem
    compute_type = "float16" if device == "cuda" else "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

    print(f"Loading WhisperX model (large-v3) on {device}...")
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)

    print(f"Loading audio from {video_path}...")
    audio = whisperx.load_audio(video_path)
    
    print("Transcribing...")
    vad_parameters = None
    if vad_filter:
        # faster-whisper VAD options (dict is accepted)
        vad_parameters = {
            "threshold": float(vad_threshold),
            "min_silence_duration_ms": int(vad_min_silence_ms),
            "speech_pad_ms": int(vad_speech_pad_ms),
        }

    # WhisperX returns different pipeline classes across versions; their transcribe()
    # signatures are not stable. Filter kwargs by supported parameter names to stay compatible.
    transcribe_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "language": language,
        "condition_on_previous_text": condition_on_previous_text,
        "chunk_length": chunk_length,
        "vad_filter": vad_filter,
        "vad_parameters": vad_parameters,
    }
    try:
        sig = inspect.signature(model.transcribe)
        supported = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in transcribe_kwargs.items() if k in supported and v is not None}

        if vad_filter and ("vad_filter" not in supported and "vad_parameters" not in supported):
            print("提示：当前 WhisperX pipeline 的 transcribe() 不支持 VAD 参数（vad_filter/vad_parameters），将跳过 VAD。")
        result = model.transcribe(audio, **filtered_kwargs)
    except TypeError as e:
        # Fallback: try without any optional kwargs
        msg = str(e)
        if "vad_filter" in msg or "vad_parameters" in msg:
            print("提示：当前 WhisperX pipeline 不支持 VAD 参数，已自动回退为不使用 VAD 的转写。")
            try:
                # remove vad keys and retry with remaining supported ones
                sig = inspect.signature(model.transcribe)
                supported = set(sig.parameters.keys())
                filtered_kwargs = {
                    k: v
                    for k, v in transcribe_kwargs.items()
                    if k in supported and v is not None and k not in {"vad_filter", "vad_parameters"}
                }
                result = model.transcribe(audio, **filtered_kwargs)
            except Exception:
                result = model.transcribe(audio)
        else:
            raise
    
    # 2. Align whisper output (adds word-level timestamps; may be huge for JA/ZH)
    print(f"Loading alignment model for {language}...")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # Free up memory
    del model
    del model_a
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def diarize_and_assign_speakers(
    audio: Any,
    aligned_result: Dict[str, Any],
    device: str,
    hf_token: Optional[str] = None,
    diarize_model_name: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    fill_nearest: bool = True,
) -> Dict[str, Any]:
    """
    Run speaker diarization (pyannote) and assign speaker labels to words/segments.

    Notes:
    - Requires Hugging Face token + acceptance of the model agreement:
      `pyannote/speaker-diarization-community-1`
    - If token/deps are missing, returns input unchanged.
    """
    if not hf_token:
        print("No HF token provided; skip diarization.")
        return aligned_result

    try:
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers
    except Exception as e:
        print(f"Failed to import diarization components; skip diarization. ({e})")
        return aligned_result

    try:
        # WhisperX versions differ on token kwarg name.
        try:
            diarize_model = DiarizationPipeline(model_name=diarize_model_name, token=hf_token, device=device)  # type: ignore[arg-type]
        except TypeError:
            diarize_model = DiarizationPipeline(model_name=diarize_model_name, use_auth_token=hf_token, device=device)  # type: ignore[arg-type]
        try:
            diarize_df = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        except TypeError:
            # older versions may not accept these kwargs
            diarize_df = diarize_model(audio)
        return assign_word_speakers(diarize_df, aligned_result, fill_nearest=fill_nearest)
    except Exception as e:
        msg = str(e)
        if "unexpected keyword argument 'plda'" in msg or "unexpected keyword argument \"plda\"" in msg:
            print("Diarization pipeline 参数不兼容（plda）。")
            print("- 你当前是 pyannote.audio 3.x，这通常意味着 `speaker-diarization-community-1` 不兼容。")
            print("- 推荐改用：`--diarize-model pyannote/speaker-diarization-3.1`")
            print("- 并在 HF 上同时接受：`pyannote/speaker-diarization-3.1` 与 `pyannote/segmentation-3.0` 的条件。")
        if ("accept the user conditions" in msg) or ("gated" in msg) or ("authenticate" in msg) or ("Could not download" in msg):
            print("Diarization model download/auth failed (likely gated on Hugging Face).")
            print("- Make sure your HF token has read access.")
            print("- Visit the model page and click to accept the user conditions, then retry.")
            if diarize_model_name:
                print(f"- Model: {diarize_model_name}")
            else:
                print("- Model: (whisperx default; see --diarize-model)")
        print(f"Diarization failed; keep transcript without speakers. ({e})")
        return aligned_result


def build_utterances(
    aligned_result: Dict[str, Any],
    include_words: bool = False,
    *,
    resegment: bool = True,
    split_on_speaker_change: bool = True,
    split_on_gap_seconds: float = 0.6,
    split_on_punct: bool = True,
    punctuations: str = "。！？?!",
    max_utterance_seconds: float = 9.0,
    max_utterance_chars: int = 48,
    min_utterance_chars: int = 4,
    merge_short_utterances: bool = True,
    merge_max_gap_seconds: float = 0.25,
    synthetic_word_times_on_bad_alignment: bool = True,
    bad_alignment_token_max_seconds: float = 2.0,
    bad_alignment_short_token_chars: int = 8,
) -> List[Dict[str, Any]]:
    """
    Convert WhisperX aligned segments into speaker-aware utterances.

    - If `words[].speaker` exists, splits on speaker changes inside a segment.
    - Otherwise falls back to segment-level utterances.

    If `resegment=True`, applies additional splitting rules to avoid overly long utterances:
    - split on long pauses (word gap)
    - split on punctuation boundaries
    - split when exceeding max duration or max char length
    """
    utterances: List[Dict[str, Any]] = []
    segments = aligned_result.get("segments", []) or []
    for seg in segments:
        if not isinstance(seg, dict):
            continue

        words = seg.get("words")
        if not (isinstance(words, list) and words):
            utt = {
                "start": float(seg.get("start")) if seg.get("start") is not None else None,
                "end": float(seg.get("end")) if seg.get("end") is not None else None,
                "speaker": seg.get("speaker"),
                "text": (seg.get("text") or "").strip(),
            }
            utterances.append(utt)
            continue

        seg_start = float(seg.get("start")) if seg.get("start") is not None else None
        seg_end = float(seg.get("end")) if seg.get("end") is not None else None
        word_dicts: List[Dict[str, Any]] = [w for w in words if isinstance(w, dict)]

        def _normalize_word_times(
            words_in: List[Dict[str, Any]],
            s0: Optional[float],
            e0: Optional[float],
            default_word_dur: float = 0.12,
        ) -> None:
            """
            Fill/repair per-token timestamps to avoid pathological splits.
            Mutates `words_in` in-place.
            """
            starts: List[Optional[float]] = []
            ends: List[Optional[float]] = []
            for w in words_in:
                ws = w.get("start")
                we = w.get("end")
                starts.append(float(ws) if ws is not None else None)
                ends.append(float(we) if we is not None else None)

            # Backward pass: fill missing ends from next known start.
            next_start: Optional[float] = None
            for i in range(len(words_in) - 1, -1, -1):
                if starts[i] is not None:
                    next_start = starts[i]
                if ends[i] is None and next_start is not None:
                    ends[i] = next_start

            # Forward pass: fill missing starts from previous known end.
            prev_end: Optional[float] = None
            for i in range(len(words_in)):
                if ends[i] is not None:
                    prev_end = ends[i]
                if starts[i] is None and prev_end is not None:
                    starts[i] = prev_end

            for i, w in enumerate(words_in):
                ws = starts[i]
                we = ends[i]

                if ws is None and we is not None:
                    ws = we - float(default_word_dur)
                if we is None and ws is not None:
                    we = ws + float(default_word_dur)

                if ws is None:
                    ws = s0
                if we is None:
                    we = e0

                if ws is not None and we is not None and ws > we:
                    ws = we

                if s0 is not None and ws is not None:
                    ws = max(s0, ws)
                if e0 is not None and we is not None:
                    we = min(e0, we)
                if ws is not None and we is not None and ws > we:
                    ws = we

                if ws is not None:
                    w["start"] = float(ws)
                if we is not None:
                    w["end"] = float(we)

        if word_dicts:
            _normalize_word_times(word_dicts, seg_start, seg_end)

        def _looks_like_bad_alignment(words_in: List[Dict[str, Any]]) -> bool:
            if not (synthetic_word_times_on_bad_alignment and seg_start is not None and seg_end is not None):
                return False
            for w in words_in:
                ws = w.get("start")
                we = w.get("end")
                if ws is None or we is None:
                    continue
                try:
                    dur = float(we) - float(ws)
                except Exception:
                    continue
                token = (w.get("word") or "").strip()
                if token and len(token) <= int(bad_alignment_short_token_chars) and dur >= float(bad_alignment_token_max_seconds):
                    return True
            return False

        if word_dicts and _looks_like_bad_alignment(word_dicts):
            # Fallback: ignore alignment word times and assign synthetic times linearly within the segment.
            # This avoids absurd boundaries like a short token spanning tens of seconds.
            total = len(word_dicts)
            dur = float(seg_end - seg_start) if (seg_end is not None and seg_start is not None) else 0.0
            if total > 0 and dur > 0 and seg_start is not None:
                for i, w in enumerate(word_dicts):
                    w["start"] = float(seg_start + dur * (i / total))
                    w["end"] = float(seg_start + dur * ((i + 1) / total))

        cur_speaker = None
        cur_words: List[Dict[str, Any]] = []
        cur_text_parts: List[str] = []
        prev_word_end: Optional[float] = None

        def cur_start() -> Optional[float]:
            for w in cur_words:
                s = w.get("start")
                if s is not None:
                    return float(s)
            s0 = seg.get("start")
            return float(s0) if s0 is not None else None

        def cur_end() -> Optional[float]:
            for w in reversed(cur_words):
                e = w.get("end")
                if e is not None:
                    return float(e)
            e0 = seg.get("end")
            return float(e0) if e0 is not None else None

        def flush():
            nonlocal cur_words, cur_text_parts
            if not cur_words:
                return
            text = "".join(cur_text_parts).strip()
            utt: Dict[str, Any] = {
                "start": cur_start(),
                "end": cur_end(),
                "speaker": cur_speaker or seg.get("speaker"),
                "text": text,
            }
            if include_words:
                utt["words"] = cur_words
            utterances.append(utt)
            cur_words = []
            cur_text_parts = []

        def should_split_before(next_word: Dict[str, Any], next_speaker: Any) -> bool:
            if not resegment:
                return False
            # 1) speaker turn
            if split_on_speaker_change and cur_speaker is not None and next_speaker != cur_speaker and cur_words:
                return True
            # 2) long pause
            if split_on_gap_seconds is not None and split_on_gap_seconds > 0 and prev_word_end is not None:
                ns = next_word.get("start")
                if ns is not None:
                    gap = float(ns) - float(prev_word_end)
                    if gap >= float(split_on_gap_seconds) and cur_words:
                        return True
            return False

        def should_split_after() -> bool:
            if not resegment or not cur_words:
                return False
            cur_len = len("".join(cur_text_parts).strip())
            # duration cap
            s = cur_start()
            e = cur_end()
            if s is not None and e is not None and max_utterance_seconds is not None and max_utterance_seconds > 0:
                if cur_len >= int(min_utterance_chars) and (e - s) >= float(max_utterance_seconds):
                    return True
            # length cap
            if max_utterance_chars is not None and max_utterance_chars > 0:
                if cur_len >= int(min_utterance_chars) and cur_len >= int(max_utterance_chars):
                    return True
            # punctuation boundary
            if split_on_punct and punctuations:
                tail = ("".join(cur_text_parts)).rstrip()
                if cur_len >= int(min_utterance_chars) and tail and tail[-1] in punctuations:
                    return True
            return False

        for w in word_dicts:
            spk = w.get("speaker") or seg.get("speaker")
            if cur_speaker is None:
                cur_speaker = spk

            if should_split_before(w, spk):
                flush()
                cur_speaker = spk
                prev_word_end = None

            token = (w.get("word") or "")
            cur_words.append(w)
            cur_text_parts.append(token)

            we = w.get("end")
            if we is not None:
                prev_word_end = float(we)

            if should_split_after():
                flush()
                cur_speaker = spk
                prev_word_end = None

        flush()

    if merge_short_utterances and utterances:
        merged: List[Dict[str, Any]] = []

        def _len_text(u: Dict[str, Any]) -> int:
            return len((u.get("text") or "").strip())

        def _gap(prev: Dict[str, Any], cur: Dict[str, Any]) -> Optional[float]:
            pe = prev.get("end")
            cs = cur.get("start")
            if pe is None or cs is None:
                return None
            try:
                return float(cs) - float(pe)
            except Exception:
                return None

        for u in utterances:
            if not merged:
                merged.append(u)
                continue

            prev = merged[-1]
            same_speaker = (prev.get("speaker") == u.get("speaker"))
            g = _gap(prev, u)
            prev_short = _len_text(prev) > 0 and _len_text(prev) < int(min_utterance_chars)
            cur_short = _len_text(u) > 0 and _len_text(u) < int(min_utterance_chars)

            # Merge small fragments back to neighbors to avoid splitting inside a word
            if same_speaker and (prev_short or cur_short) and (g is None or g <= float(merge_max_gap_seconds)):
                prev["text"] = ((prev.get("text") or "") + (u.get("text") or "")).strip()
                if prev.get("start") is None:
                    prev["start"] = u.get("start")
                prev["end"] = u.get("end") if u.get("end") is not None else prev.get("end")
                if include_words and "words" in prev and "words" in u and isinstance(prev["words"], list) and isinstance(u["words"], list):
                    prev["words"].extend(u["words"])
            else:
                merged.append(u)
        utterances = merged

    return utterances


def drop_words_in_segments(aligned_result: Dict[str, Any]) -> Dict[str, Any]:
    """Remove `segments[].words` to shrink JSON size."""
    segs = aligned_result.get("segments")
    if not isinstance(segs, list):
        return aligned_result
    for seg in segs:
        if isinstance(seg, dict) and "words" in seg:
            seg.pop("words", None)
    return aligned_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhisperX ASR POC (with optional diarization + compact output).")
    parser.add_argument("input", nargs="?", default="test.mp4", help="Input video/audio path (default: test.mp4)")
    parser.add_argument("--language", default="ja", help="Language code (default: ja)")
    parser.add_argument("--output", default="asr_result.json", help="Output json path (default: asr_result.json)")

    # Transcription segmentation knobs (VAD + decoding behavior)
    parser.add_argument("--vad-filter", action=argparse.BooleanOptionalAction, default=True, help="Enable VAD-based audio segmentation (default: enabled)")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="VAD threshold (higher=more strict)")
    parser.add_argument("--vad-min-silence-ms", type=int, default=200, help="VAD min silence duration to split (ms)")
    parser.add_argument("--vad-speech-pad-ms", type=int, default=200, help="VAD speech padding (ms)")
    parser.add_argument("--condition-on-previous-text", action=argparse.BooleanOptionalAction, default=True, help="Decode conditioning on previous text (default: enabled)")
    parser.add_argument("--chunk-length", type=int, default=None, help="Max chunk length in seconds (optional)")

    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization (requires HF token)")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"), help="Hugging Face token (or env HF_TOKEN)")
    parser.add_argument("--diarize-model", default=None, help="pyannote diarization model id (optional)")
    parser.add_argument("--min-speakers", type=int, default=None, help="Min speakers (optional)")
    parser.add_argument("--max-speakers", type=int, default=None, help="Max speakers (optional)")

    # Post re-segmentation (utterances) knobs
    parser.add_argument("--resegment", action=argparse.BooleanOptionalAction, default=True, help="Re-segment utterances by pauses/punct/length (default: enabled)")
    parser.add_argument("--split-on-speaker-change", action=argparse.BooleanOptionalAction, default=True, help="Split utterances on speaker change (default: enabled)")
    parser.add_argument("--split-on-gap-seconds", type=float, default=0.6, help="Split utterances if word gap >= this seconds")
    parser.add_argument("--split-on-punct", action=argparse.BooleanOptionalAction, default=True, help="Split utterances on punctuation boundary (default: enabled)")
    parser.add_argument("--punctuations", default="。！？?!", help="Punctuations used for splitting")
    parser.add_argument("--max-utterance-seconds", type=float, default=9.0, help="Max utterance duration before forced split")
    parser.add_argument("--max-utterance-chars", type=int, default=48, help="Max utterance char length before forced split")
    parser.add_argument("--min-utterance-chars", type=int, default=4, help="Don't finalize splits for utterances shorter than this")
    parser.add_argument("--merge-short-utterances", action=argparse.BooleanOptionalAction, default=True, help="Merge very short utterances back to neighbors (default: enabled)")
    parser.add_argument("--merge-max-gap-seconds", type=float, default=0.25, help="Max gap allowed when merging short utterances")

    parser.add_argument("--keep-words", action="store_true", help="Keep words array in output (huge for ja/zh)")
    args = parser.parse_args()

    input_video = args.input
    if not os.path.exists(input_video):
        print(f"Error: {input_video} not found.")
        raise SystemExit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = transcribe_video(
        input_video,
        language=args.language,
        vad_filter=args.vad_filter,
        vad_threshold=args.vad_threshold,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
        condition_on_previous_text=args.condition_on_previous_text,
        chunk_length=args.chunk_length,
    )

    if args.diarize:
        results = diarize_and_assign_speakers(
            audio=input_video,
            aligned_result=results,
            device=device,
            hf_token=args.hf_token,
            diarize_model_name=args.diarize_model,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )

    utterances = build_utterances(
        results,
        include_words=args.keep_words,
        resegment=args.resegment,
        split_on_speaker_change=args.split_on_speaker_change,
        split_on_gap_seconds=args.split_on_gap_seconds,
        split_on_punct=args.split_on_punct,
        punctuations=args.punctuations,
        max_utterance_seconds=args.max_utterance_seconds,
        max_utterance_chars=args.max_utterance_chars,
        min_utterance_chars=args.min_utterance_chars,
        merge_short_utterances=args.merge_short_utterances,
        merge_max_gap_seconds=args.merge_max_gap_seconds,
    )

    output: Dict[str, Any] = {
        "language": args.language,
        "input": input_video,
        "utterances": utterances,
    }
    if args.keep_words:
        output["segments"] = results.get("segments", [])
    else:
        output["segments"] = drop_words_in_segments(results).get("segments", [])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"ASR result saved to {args.output}")
    for u in utterances[:8]:
        spk = u.get("speaker") or "SPEAKER?"
        s = u.get("start")
        e = u.get("end")
        t = u.get("text") or ""
        if s is not None and e is not None:
            print(f"[{spk}] [{s:.2f} - {e:.2f}] {t}")
        else:
            print(f"[{spk}] {t}")
