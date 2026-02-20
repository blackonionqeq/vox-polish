import whisperx
import gc
import json
import torch
import os
import sys
import argparse
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

def transcribe_video(video_path, language="ja"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4 # reduce if low on GPU mem
    compute_type = "float16" if device == "cuda" else "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

    print(f"Loading WhisperX model (large-v3) on {device}...")
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)

    print(f"Loading audio from {video_path}...")
    audio = whisperx.load_audio(video_path)
    
    print("Transcribing...")
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    
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
) -> List[Dict[str, Any]]:
    """
    Convert WhisperX aligned segments into speaker-aware utterances.

    - If `words[].speaker` exists, splits on speaker changes inside a segment.
    - Otherwise falls back to segment-level utterances.
    """
    utterances: List[Dict[str, Any]] = []
    for seg in aligned_result.get("segments", []) or []:
        words = seg.get("words")
        if isinstance(words, list) and words and any(isinstance(w, dict) and "speaker" in w for w in words):
            cur_speaker = None
            cur_words: List[Dict[str, Any]] = []

            def flush():
                nonlocal cur_speaker, cur_words
                if not cur_words:
                    return
                text = "".join((w.get("word") or "") for w in cur_words).strip()
                start = next((w.get("start") for w in cur_words if w.get("start") is not None), seg.get("start"))
                end = next((w.get("end") for w in reversed(cur_words) if w.get("end") is not None), seg.get("end"))
                utt: Dict[str, Any] = {
                    "start": float(start) if start is not None else None,
                    "end": float(end) if end is not None else None,
                    "speaker": cur_speaker or seg.get("speaker"),
                    "text": text,
                }
                if include_words:
                    utt["words"] = cur_words
                utterances.append(utt)
                cur_words = []

            for w in words:
                if not isinstance(w, dict):
                    continue
                spk = w.get("speaker") or seg.get("speaker")
                if cur_speaker is None:
                    cur_speaker = spk
                if spk != cur_speaker and cur_words:
                    flush()
                    cur_speaker = spk
                # Avoid copying per-word dicts (JA/ZH char-level can be huge).
                cur_words.append(w)
            flush()
        else:
            utt = {
                "start": float(seg.get("start")) if seg.get("start") is not None else None,
                "end": float(seg.get("end")) if seg.get("end") is not None else None,
                "speaker": seg.get("speaker"),
                "text": (seg.get("text") or "").strip(),
            }
            if include_words and isinstance(words, list):
                utt["words"] = words
            utterances.append(utt)
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
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization (requires HF token)")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"), help="Hugging Face token (or env HF_TOKEN)")
    parser.add_argument("--diarize-model", default=None, help="pyannote diarization model id (optional)")
    parser.add_argument("--min-speakers", type=int, default=None, help="Min speakers (optional)")
    parser.add_argument("--max-speakers", type=int, default=None, help="Max speakers (optional)")
    parser.add_argument("--keep-words", action="store_true", help="Keep words array in output (huge for ja/zh)")
    args = parser.parse_args()

    input_video = args.input
    if not os.path.exists(input_video):
        print(f"Error: {input_video} not found.")
        raise SystemExit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = transcribe_video(input_video, language=args.language)

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

    utterances = build_utterances(results, include_words=args.keep_words)

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
