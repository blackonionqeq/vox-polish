# Vox Polish - ASR POC

这是一个使用 WhisperX 进行日语视频识别 (ASR) 的 POC 实验。

## 环境要求

1. **Python**: >= 3.10
2. **uv**: 用于包管理
3. **FFmpeg**: 系统需要安装 FFmpeg。WhisperX 需要 FFmpeg 来处理视频/音频文件。
   - Windows 用户可以通过 `scoop install ffmpeg` 或从官网下载并添加到 PATH。

## 安装步骤

1. 安装依赖：
   ```bash
   uv sync
   ```

2. 准备测试视频：
   将 `test.mp4` 放在项目根目录下。

## 运行

执行识别脚本：
```bash
uv run asr.py
```

你也可以指定输入/输出与语言：

```bash
uv run asr.py test.mp4 --language ja --output asr_result.json
```

### 说话人识别（Diarization，可选）

WhisperX 的说话人识别依赖 pyannote 的 diarization pipeline。

在你当前依赖（`pyannote.audio 3.x`）下，**推荐使用** `pyannote/speaker-diarization-3.1`（它要求 `pyannote.audio >= 3.1`）。

> 说明：`pyannote/speaker-diarization-community-1` 在部分 `pyannote.audio 3.x` 环境会出现 `plda` 参数不兼容（你遇到的就是这个）。

- 在 Hugging Face 上**接受该模型的协议**
- 准备一个 **HF Token（read 权限）**

推荐把 token 放到环境变量里：

```bash
export HF_TOKEN=xxxxxxxx
```

然后运行：

```bash
uv run asr.py test.mp4 --diarize --min-speakers 2 --max-speakers 2
```

如果你需要显式指定模型（用于匹配你当前环境的 whisperx 默认行为），可以加 `--diarize-model`：

```bash
uv run asr.py test.mp4 --diarize --diarize-model pyannote/speaker-diarization-3.1
```

注意：`pyannote/speaker-diarization-3.1` 还要求你额外接受 `pyannote/segmentation-3.0` 的条件（否则也会下载失败）。

- `pyannote/speaker-diarization-3.1`: `https://hf.co/pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`: `https://hf.co/pyannote/segmentation-3.0`

## 输出

识别结果将保存为 `asr_result.json`，其中包含：

- `utterances`: **推荐用于后续翻译/字幕**（包含 `start/end/speaker/text`，在 diarization 启用后会按说话人变化切分）
- `segments`: WhisperX 的原始 segment（默认已移除 `words`，避免文件过大）

如果你确实需要保留 `segments[].words`（日语/中文会非常大），可以加 `--keep-words`：

```bash
uv run asr.py test.mp4 --keep-words
```

## 后续计划

- [ ] 增加翻译逻辑
- [ ] 优化识别效果
- [ ] 增加字幕生成功能 (.srt)
