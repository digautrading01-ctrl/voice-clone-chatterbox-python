# Flask Voice Clone (chatterbox-tts)

A free, locally deployed voice cloning + subtitle-to-audio web app powered by Resemble AI’s open-source **[`chatterbox-tts`](https://pypi.org/project/chatterbox-tts/)**.

The server **auto-detects** which local Chatterbox checkpoint is present under `./model/` and loads it accordingly:

| Model variant | Class | Languages | Notes |
|---|---|---|---|
| **Turbo** (`ChatterboxTurboTTS`) | `chatterbox.tts_turbo` | English only | Faster; loaded from `ResembleAI/chatterbox-turbo` |
| **Multilingual V3** (`ChatterboxMultilingualTTS`) | `chatterbox.mtl_tts` | English + Chinese | Loaded from `ResembleAI/chatterbox` multilingual checkpoints |
| **Chatterbox** (`ChatterboxTTS`) | `chatterbox.tts` | English only | Original English checkpoint from `ResembleAI/chatterbox` |

The app runs fully offline once the model is pre-downloaded.

## Features

| Feature | Description |
|---|---|
| Voice Cloning | Upload a 3–30 s reference clip; the first clip is used for voice conditioning |
| Chinese & English | Supported by the Multilingual model; Turbo is English-only |
| In-browser Preview | Play generated audio directly in the browser |
| WAV Download | One-click download of generated WAV files |
| Subtitles (.srt) | Upload an `.srt` file and generate a single WAV/MP3 track aligned to subtitle start times |
| History | Browse, preview, and delete all previously generated files |
| Long-text reliability | Long input is chunked by sentence and stitched to reduce occasional truncation on long sentences |

> Note on reference clips: `chatterbox-tts` voice conditioning takes a single `audio_prompt_path`.
> The server accepts multiple uploads for API compatibility, but only the **first** clip is used.

## Requirements

- Python 3.10+ (recommended: **3.10–3.12** on Windows)
- CUDA GPU recommended (CPU works but is slower)
- Python deps installed via `requirements.txt` (includes `chatterbox-tts`, `torch`, `torchaudio`, `soundfile`, `numpy`)
- `pydub` is used to stitch chunked WAV segments for long text (installed via `requirements.txt`)
- Optional (for **MP3** output): **ffmpeg** installed and available as the `ffmpeg` command

## GPU / CUDA Acceleration

This app will **automatically use CUDA** for speech synthesis **if**:

- You have an NVIDIA GPU
- Your installed PyTorch build supports CUDA
- `torch.cuda.is_available()` returns `True`

You can **force CPU** (even if CUDA is available) by setting:

```bash
# Windows (PowerShell)
$env:VOICECLONE_DEVICE="cpu"
python app.py
```

Or you can try to **force CUDA** explicitly:

```bash
# Windows (PowerShell)
$env:VOICECLONE_DEVICE="cuda"
python app.py
```

If CUDA is forced but not available, the server will log a warning and fall back to CPU.

The web UI also shows a small **CPU / CUDA** badge at runtime (based on what the backend is using).

## Long text chunking

Some very long sentences can get truncated when synthesized in a single pass. To reduce this, the backend will split long input into sentence-like chunks and stitch the resulting audio into a single WAV.

You can control the chunk size in two ways:

1. **Environment variable default**
```bash
# Windows (PowerShell)
$env:VOICECLONE_MAX_CHARS="260"   # default: 260 (clamped to 80..1200)
python app.py
```

2. **UI override**
In the web UI, set **Max chars / chunk** (in the Voice Cloning or Default Voice tab). Leaving it blank uses the environment-variable default.

## Download the Model (offline)

The backend loads Chatterbox via Hugging Face and stores caches under the project-root **`./model/`** folder.
To run offline, pre-download the model **once** while you have internet access.

Choose **one** of the model variants below:

### Turbo (English-only, faster)

Recommended model ID: **`ResembleAI/chatterbox-turbo`**

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('ResembleAI/chatterbox-turbo', cache_dir='model')"
```

### Multilingual V3 (English + Chinese)

Recommended model ID: **`ResembleAI/chatterbox`**

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('ResembleAI/chatterbox', cache_dir='model')"
```

### Original Chatterbox (English-only)

Recommended model ID: **`ResembleAI/chatterbox`**

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('ResembleAI/chatterbox', cache_dir='model')"
```

### Using the Hugging Face CLI (either variant)

```bash
pip install -U 'huggingface_hub[cli]'
huggingface-cli download ResembleAI/chatterbox-turbo --cache-dir model
# or
huggingface-cli download ResembleAI/chatterbox --cache-dir model
```

Once downloaded, you can disconnect from the internet and the app will keep working.

> **Which model gets loaded?** The server inspects the files under `./model/` and matches them to the
> expected checkpoint layout for Turbo, Multilingual V3, or the original Chatterbox model. In auto mode,
> it prefers **Turbo**, then **Multilingual V3**, then **Chatterbox** if more than one checkpoint is present.
>
> To force a specific local model when multiple checkpoints exist, set `VOICECLONE_MODEL` to one of:
> `turbo`, `multilingual`, or `chatterbox`.

### Supported local layouts

The loader supports either of these offline layouts:

1. A Hugging Face cache/snapshot tree under `./model/` created by `snapshot_download(..., cache_dir='model')`
2. A direct checkpoint folder under `./model/` containing the model files themselves

Examples of files the app looks for:

- Turbo: `ve.safetensors`, `t3_turbo_v1.safetensors`, `s3gen_meanflow.safetensors`
- Multilingual V3: `ve.pt`, `t3_mtl23ls_v2.safetensors`, `s3gen.pt`
- Original Chatterbox: `ve.safetensors`, `t3_cfg.safetensors`, `s3gen.safetensors`, `tokenizer.json`

## Setup

```bash
# 1. Enter the project folder
cd voice-clone-chatterbox-python

# 2. Pre-download the model into ./model (see "Download the Model" above)

# 3. Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Start the server
python app.py
```

Open your browser at **http://localhost:5001**

> The banner at the top of the page will change to "Model ready" once the model has finished loading.
> If the `./model/` folder is missing/empty, the server will print an error and exit immediately.

If you keep more than one model under `./model/`, you can force the selection:

```bash
# Windows (PowerShell)
$env:VOICECLONE_MODEL="turbo"
python app.py
```

## Usage

### Voice Cloning (recommended)
1. Click **Add Clip(s)** and select a clean voice recording (WAV / MP3 / FLAC, 3–30 s). Only the first uploaded clip is used for voice conditioning.
2. Select the output language (English / Chinese) and enter your text
3. Click **Generate Speech**
4. Preview the result in the browser, then click **Download WAV**

> Note: for longer inputs, the backend automatically splits text into sentence-like chunks and stitches the generated audio into a single WAV. This helps avoid occasional truncation that can happen when synthesizing very long sentences in a single pass.

### Default Voice (no reference)
The “Built-in Speakers” tab is kept for compatibility. Since chatterbox-tts is a **zero-shot** model (no discrete built-in speaker list),
this mode uses a single option: **Default** (no reference audio).

### Subtitles (.srt)
1. Switch to the **Subtitles (.srt)** tab
2. Choose **Use uploaded reference audio** (voice cloning) or **Use built-in speaker** (Default voice)
3. Upload an `.srt` subtitles file
4. Select language and output format (WAV / MP3)
5. Click **Generate from SRT**
6. If needed, click **Pause** during background generation to stop starting new cues until you click **Resume**

> Pause/resume is cooperative: the current subtitle cue is allowed to finish, then the worker waits before synthesizing the next cue. This lowers ongoing resource use for long jobs without interrupting model inference mid-cue.

**How timing is enforced:**

The app aligns segments by inserting silence to match each subtitle **start time**.
When synthesized speech is longer than the cue duration (`end − start`), it applies these steps in order:

1. **Punctuation normalization** — Before synthesis, each cue’s text is checked for terminal punctuation. If it is missing, a period (sentence end) or comma (continuation — when the next cue begins with a lowercase letter) is appended. This guides the TTS model to produce the correct prosody and intonation, reducing flat or "confused-sounding" output on short fragments.

2. **ffmpeg `atempo` time-stretch (up to 4.0×)** — If the speech is longer than the cue window, the app now prefers pitch-preserving ffmpeg `atempo` compression much more aggressively before giving up. This keeps substantially more words in dense subtitle lines while preserving pitch; for extreme overruns it applies up to `4.0×` time-stretch first.

3. **Tail trim** — If the speech still exceeds the cue window even after `atempo` has been pushed up to `4.0×`, only the remaining tail is trimmed. The previous fallback that used `np.interp` linear resampling has been removed; it compressed the waveform in the time domain and raised pitch in direct proportion to the compression ratio, which was the primary cause of the high-pitch artifact.

If speech is shorter than the cue duration, silence is padded to fill the gap.

> Installing **ffmpeg** is strongly recommended for step 2. Without it, all overruns go straight to tail trimming.

## Project Structure

```
voice-clone-chatterbox-python/
├── app.py                  # Flask backend & API routes (chatterbox-tts)
├── requirements.txt
├── README.md
├── model/                  # Hugging Face cache for chatterbox-tts (pre-download; not committed)
│   └── ...                 # (contents depend on your download method)
├── uploads/                # Reference audio files (auto-created)
├── outputs/                # Generated files (auto-created)
├── templates/
│   └── index.html          # Single-page UI
└── static/
    ├── css/style.css
    └── js/app.js
```

## API Reference

### Upload reference audio
```
POST /upload-ref
Content-Type: multipart/form-data
file: <audio file>
```

### Clone synthesis
```
POST /synthesize
Content-Type: application/json

{
  "text":    "Text to synthesize",
  "ref_ids": ["uploaded_ref1.wav", "uploaded_ref2.wav"],  // accepted, but only the first is used
  "ref_id":  "uploaded_reference.wav",                    // backward compat
  "lang":    "en"                                         // en | zh
}
```

### Default voice synthesis (no reference)
```
POST /synthesize-builtin
Content-Type: application/json

{
  "text":    "Hello, world!",
  "speaker": "Default",
  "lang":    "en"
}
```

### SRT (subtitle) synthesis — cloned voice
```
POST /synthesize-srt
Content-Type: multipart/form-data

file:      <subtitles.srt>
ref_ids[]: <uploaded_reference_audio_id>   # repeat allowed; first is used
ref_id:    <uploaded_reference_audio_id>   # backward compat
lang:      en | zh
format:    wav | mp3
async:     0 | 1   # 1 enables background job mode with progress (used by the web UI)
```

### SRT (subtitle) synthesis — default voice
```
POST /synthesize-srt-builtin
Content-Type: multipart/form-data

file:    <subtitles.srt>
speaker: Default
lang:    en | zh
format:  wav | mp3
async:   0 | 1
```

### SRT job status (progress polling)
```
GET /srt-job/<job_id>
```

Response fields:
`status` = `running | paused | canceled | done | error`, plus `paused`, `canceled`, `progress` (0–1), `done`, `total`, and `output_id` when complete.

### Pause / resume an SRT job
```
POST /srt-job/<job_id>/pause
POST /srt-job/<job_id>/resume
```

Pause is cooperative: if a cue is already being synthesized, that cue finishes first and the job pauses before the next cue begins.

### Cancel an SRT job
```
POST /srt-job/<job_id>/cancel
```

Cancel is also cooperative: the current cue (if any) finishes first, then the job stops before the next cue begins and no output is produced.

### Other endpoints
```
GET    /model-status          # Check if model is loaded
GET    /speakers              # Returns ["Default"] for UI compatibility
GET    /list-outputs          # List all generated files
GET    /play/outputs/<id>     # Stream audio for playback
GET    /download/outputs/<id> # Download audio file
DELETE /delete-output/<id>    # Delete a generated file
```

## FAQ

**Q: The server exits immediately on startup.**  
A: The `./model/` folder is missing or empty. Pre-download the Chatterbox model into `./model/` (see “Download the Model”), then restart.

**Q: I get `expected scalar type Float but found Double` (usually with Turbo).**  
A: Some Turbo checkpoints can contain float64 tensors. This project forces model weights and cached conditionals to float32 at load time to avoid this error. Make sure you’re running the updated `app.py`, then restart the server.

**Q: MP3 export fails with an error about `ffmpeg`.**  
A: MP3 output requires **ffmpeg**. Install ffmpeg and ensure the `ffmpeg` command works in your terminal, then retry.

**Q: How fast is synthesis on CPU?**  
A: It depends heavily on your hardware; GPU inference is strongly recommended for interactive use.
