import os
import uuid
import threading
import re
import shutil
import subprocess
import time
import inspect
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, abort

app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("outputs")
TMP_FOLDER = OUTPUT_FOLDER / "_tmp"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)
TMP_FOLDER.mkdir(exist_ok=True)

ALLOWED_AUDIO = {"wav", "mp3", "flac", "ogg", "m4a"}
ALLOWED_SUBTITLES = {"srt"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# TTS model (loaded once at startup)
tts_model = None
tts_lock = threading.Lock()
tts_device = "cpu"
tts_supports_speed = False  # chatterbox-tts currently does not expose a stable speaking-rate parameter
tts_speed_param = None  # kept for backward compat in the subtitle pipeline
tts_model_type = None  # "turbo" | "multilingual"

SRT_JOBS = {}
SRT_JOB_CONTROLS = {}
SRT_JOBS_LOCK = threading.Lock()
SRT_JOB_TTL_SEC = 60 * 60  # keep job status for 1 hour


class _SrtJobCancelled(Exception):
    pass


MODEL_DIR = Path("model")

MODEL_SPECS = {
    "turbo": {
        "label": "Turbo",
        "module": "chatterbox.tts_turbo",
        "class_name": "ChatterboxTurboTTS",
        "required_files": (
            "ve.safetensors",
            "t3_turbo_v1.safetensors",
            "s3gen_meanflow.safetensors",
        ),
        "supports_language_id": False,
        "default_repo": "ResembleAI/chatterbox-turbo",
    },
    "multilingual": {
        "label": "Multilingual V3",
        "module": "chatterbox.mtl_tts",
        "class_name": "ChatterboxMultilingualTTS",
        "required_files": (
            "ve.pt",
            "t3_mtl23ls_v2.safetensors",
            "s3gen.pt",
            "grapheme_mtl_merged_expanded_v1.json",
        ),
        "supports_language_id": True,
        "default_repo": "ResembleAI/chatterbox",
    },
    "chatterbox": {
        "label": "Chatterbox",
        "module": "chatterbox.tts",
        "class_name": "ChatterboxTTS",
        "required_files": (
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
        ),
        "supports_language_id": False,
        "default_repo": "ResembleAI/chatterbox",
    },
}

MODEL_ENV_ALIASES = {
    "auto": "auto",
    "turbo": "turbo",
    "chatterbox-turbo": "turbo",
    "resembleai/chatterbox-turbo": "turbo",
    "multilingual": "multilingual",
    "multilingual-v3": "multilingual",
    "chatterbox-multilingual": "multilingual",
    "resembleai/chatterbox-multilingual": "multilingual",
    "chatterbox": "chatterbox",
    "original": "chatterbox",
    "resembleai/chatterbox": "chatterbox",
}

def _check_model_dir():
    """
    Verify that the local ./model folder exists and is not empty.

    This project uses Resemble AI's open-source `chatterbox-tts` and expects any required
    Hugging Face model files to be pre-downloaded into the project-root `./model/` folder.
    The exact file layout depends on how you downloaded (HF cache vs snapshot), so we only
    validate that the folder exists and contains at least one file.
    """
    if not MODEL_DIR.is_dir():
        msg = (
            "[Voice Clone] ERROR: Local model folder './model/' was not found.\n"
            "  This app runs fully offline once models are present.\n"
            "  Download a Chatterbox model into './model/' (see README), then restart."
        )
        print(msg)
        raise SystemExit(1)

    has_any = any(p.is_file() for p in MODEL_DIR.rglob("*"))
    if not has_any:
        msg = (
            "[Voice Clone] ERROR: Local model folder './model/' is empty.\n"
            "  Download a Chatterbox model into './model/' (see README), then restart."
        )
        print(msg)
        raise SystemExit(1)


def _configure_hf_offline_cache():
    """
    Ensure Hugging Face caches live under ./model for offline use.

    `chatterbox-tts` uses Hugging Face under the hood. By pointing HF_HOME/HUGGINGFACE_HUB_CACHE
    to `./model`, users can pre-download the model once and then run the app without internet.
    """
    model_abs = str(MODEL_DIR.resolve())
    os.environ.setdefault("HF_HOME", model_abs)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((MODEL_DIR / "hub").resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", str((MODEL_DIR / "hub").resolve()))


def _resolve_model_preference() -> str:
    raw = (os.getenv("VOICECLONE_MODEL") or "auto").strip().lower()
    return MODEL_ENV_ALIASES.get(raw, raw or "auto")


def _is_model_checkpoint_dir(path: Path, model_type: str) -> bool:
    spec = MODEL_SPECS[model_type]
    return path.is_dir() and all((path / name).is_file() for name in spec["required_files"])


def _candidate_checkpoint_dirs(root: Path):
    """
    Yield plausible checkpoint directories under ./model.

    Supported local layouts:
    - direct checkpoint files in ./model/
    - Hugging Face snapshot/cache folders created below ./model/
    """
    seen = set()
    candidates = [root]

    for parent_name in ("snapshots",):
        for p in root.rglob(parent_name):
            if p.is_dir():
                candidates.extend(x for x in p.iterdir() if x.is_dir())

    for p in root.rglob("*"):
        if p.is_dir():
            candidates.append(p)

    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        yield candidate


def _discover_local_models():
    matches = []
    for candidate in _candidate_checkpoint_dirs(MODEL_DIR):
        for model_type in ("turbo", "multilingual", "chatterbox"):
            if _is_model_checkpoint_dir(candidate, model_type):
                matches.append(
                    {
                        "model_type": model_type,
                        "label": MODEL_SPECS[model_type]["label"],
                        "ckpt_dir": candidate,
                    }
                )
    return matches


def _select_local_model():
    matches = _discover_local_models()
    if not matches:
        supported = ", ".join(
            f"{spec['label']} ({', '.join(spec['required_files'])})"
            for spec in MODEL_SPECS.values()
        )
        raise RuntimeError(
            "No supported Chatterbox checkpoint layout was found under './model/'.\n"
            "Place either a direct checkpoint folder or a Hugging Face snapshot cache there.\n"
            f"Supported layouts: {supported}"
        )

    preference = _resolve_model_preference()
    if preference != "auto":
        for match in matches:
            if match["model_type"] == preference:
                return match
        available = ", ".join(f"{m['label']} @ {m['ckpt_dir']}" for m in matches)
        raise RuntimeError(
            f"VOICECLONE_MODEL={preference!r} was requested, but that model was not found under './model/'.\n"
            f"Available local models: {available}"
        )

    # Auto mode preference: Turbo first, then multilingual, then original Chatterbox.
    order = {"turbo": 0, "multilingual": 1, "chatterbox": 2}
    matches.sort(key=lambda m: (order.get(m["model_type"], 99), len(m["ckpt_dir"].parts)))
    return matches[0]


def _import_model_class(model_type: str):
    spec = MODEL_SPECS[model_type]
    try:
        module = __import__(spec["module"], fromlist=[spec["class_name"]])
        return getattr(module, spec["class_name"])
    except Exception as e:
        print(
            "[Voice Clone] ERROR: Python package 'chatterbox-tts' is missing a required model class.\n"
            "  Install or upgrade dependencies first:\n"
            "    pip install -r requirements.txt\n"
            f"  Missing class: {spec['module']}.{spec['class_name']}\n"
            f"  Details: {e}"
        )
        raise SystemExit(1)


def _force_model_float32(model):
    """
    Force the underlying torch modules/tensors to float32.

    Some Chatterbox checkpoints (especially Turbo) may contain float64 tensors,
    which can trigger runtime errors like:
      "expected scalar type Float but found Double"
    """
    try:
        import torch  # type: ignore
    except Exception:
        return model

    # Cast known sub-modules if present (the model object itself is not an nn.Module).
    for attr in ("t3", "s3gen", "ve"):
        m = getattr(model, attr, None)
        if m is None:
            continue
        try:
            # Prefer .to(dtype=...) to avoid changing device
            m.to(dtype=torch.float32)
        except Exception:
            try:
                m.float()
            except Exception:
                pass

    # Cast cached conditionals if present (voice prompt cache / built-in voice).
    conds = getattr(model, "conds", None)
    if conds is not None:
        # conds.t3 is a dataclass-like object containing tensors
        t3_cond = getattr(conds, "t3", None)
        if t3_cond is not None:
            try:
                for k, v in vars(t3_cond).items():
                    if torch.is_tensor(v):
                        setattr(t3_cond, k, v.to(dtype=torch.float32))
            except Exception:
                pass

        gen = getattr(conds, "gen", None)
        if isinstance(gen, dict):
            try:
                for k, v in list(gen.items()):
                    if torch.is_tensor(v):
                        gen[k] = v.to(dtype=torch.float32)
            except Exception:
                pass

    return model


def load_model():
    global tts_model, tts_device, tts_supports_speed, tts_speed_param, tts_model_type
    _check_model_dir()

    # -----------------------------------------------------------------------
    # Device selection (CUDA GPU acceleration when available)
    # -----------------------------------------------------------------------
    # Default: auto-detect CUDA via torch and use it if available.
    # Override: set environment variable VOICECLONE_DEVICE to "cpu" or "cuda".
    forced_device = (os.getenv("VOICECLONE_DEVICE") or "").strip().lower()
    use_gpu = False
    device = "cpu"
    try:
        import torch  # type: ignore
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False

    if forced_device in ("cpu", "cuda", "cuda:0"):
        device = "cuda" if forced_device.startswith("cuda") else "cpu"
        use_gpu = device == "cuda" and cuda_ok
    else:
        device = "cuda" if cuda_ok else "cpu"
        use_gpu = device == "cuda"

    if device == "cuda" and not cuda_ok:
        # User forced CUDA but torch can't see a CUDA device.
        print("[Voice Clone] CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
        device = "cpu"
        use_gpu = False

    tts_device = device

    _configure_hf_offline_cache()

    try:
        selected = _select_local_model()
    except Exception as e:
        print(
            "[Voice Clone] ERROR: Failed to locate a supported local Chatterbox model under './model/'.\n"
            "  Make sure the model is fully pre-downloaded into './model' (see README).\n"
            f"  Details: {e}"
        )
        raise SystemExit(1)

    detected_type = selected["model_type"]
    ckpt_dir = selected["ckpt_dir"]
    spec = MODEL_SPECS[detected_type]
    ChatterboxClass = _import_model_class(detected_type)

    print(
        f"[Voice Clone] Loading chatterbox-tts model "
        f"({spec['label']}) from '{ckpt_dir}' on device: {tts_device} …"
    )

    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        tts_model = ChatterboxClass.from_local(
            device=tts_device,
            ckpt_dir=str(ckpt_dir.resolve())
        )
    except Exception as e:
        print(
            "[Voice Clone] ERROR: Failed to load chatterbox-tts model.\n"
            f"  Model type: {spec['label']}\n"
            f"  Expected repo: {spec['default_repo']}\n"
            f"  Checkpoint dir: {ckpt_dir}\n"
            "  Make sure the model is fully pre-downloaded into './model' (see README).\n"
            f"  Details: {e}"
        )
        raise SystemExit(1)

    # Force float32 to avoid Float vs Double runtime errors during inference.
    #tts_model = _force_model_float32(tts_model)
    
    tts_model_type = detected_type

    # chatterbox-tts currently doesn't provide a consistent speed/speaking-rate knob exposed via API.
    tts_supports_speed = False
    tts_speed_param = None

    print(f"[Voice Clone] {spec['label']} model ready — server is accepting requests.")


def _save_wav_tensor_to_file(wav, file_path: Path, sample_rate: int):
    """
    Save a Chatterbox-generated waveform tensor to a WAV file.

    chatterbox-tts returns a torch Tensor suitable for torchaudio.save.
    """
    import torch  # type: ignore
    import torchaudio as ta  # type: ignore

    if isinstance(wav, torch.Tensor):
        t = wav.detach()
        # Ensure CPU float32
        if t.is_cuda:
            t = t.cpu()
        t = t.to(dtype=torch.float32)
        # Ensure (channels, time)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        elif t.ndim == 2:
            pass
        else:
            # Unexpected shape; best-effort flatten
            t = t.reshape(1, -1)
        ta.save(str(file_path), t, sample_rate)
        return

    # Best-effort fallback for numpy arrays / lists
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore

    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim == 1:
        sf.write(str(file_path), arr, sample_rate)
    elif arr.ndim == 2:
        # (channels, time) -> (time, channels)
        if arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
            arr = arr.T
        sf.write(str(file_path), arr, sample_rate)
    else:
        sf.write(str(file_path), arr.reshape(-1), sample_rate)


def allowed_audio(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO


def allowed_subtitles(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_SUBTITLES


def unique_path(folder: Path, suffix: str) -> Path:
    return folder / f"{uuid.uuid4().hex}{suffix}"


# ---------------------------------------------------------------------------
# Subtitle (.srt) helpers
# ---------------------------------------------------------------------------

_SRT_TIME_RE = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})"
)


def _srt_time_to_ms(s: str) -> int:
    m = _SRT_TIME_RE.fullmatch(s.strip())
    if not m:
        raise ValueError(f"Invalid SRT time: {s!r}")
    h = int(m.group("h"))
    mi = int(m.group("m"))
    se = int(m.group("s"))
    ms = int(m.group("ms"))
    return ((h * 60 + mi) * 60 + se) * 1000 + ms


def _strip_srt_tags(text: str) -> str:
    # Remove simple HTML-like tags often found in SRT files (<i>, <b>, <font>, ...)
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def parse_srt(content: str):
    """
    Parse .srt into a list of cues:
      [{"start_ms": int, "end_ms": int, "text": str}, ...]
    """
    # Normalize line endings and split into blocks
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content:
        return []

    blocks = re.split(r"\n\s*\n", content)
    cues = []
    for b in blocks:
        lines = [ln.strip() for ln in b.split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            continue

        # Optional numeric index in the first line
        if re.fullmatch(r"\d+", lines[0]):
            lines = lines[1:]
            if len(lines) < 2:
                continue

        # Timing line
        if "-->" not in lines[0]:
            continue
        left, right = [x.strip() for x in lines[0].split("-->", 1)]
        try:
            start_ms = _srt_time_to_ms(left)
            end_ms = _srt_time_to_ms(right.split()[0])  # ignore trailing alignment attrs
        except Exception:
            continue

        text = _strip_srt_tags(" ".join(lines[1:]))
        if not text:
            continue

        cues.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})

    cues.sort(key=lambda x: x["start_ms"])
    return cues


def _load_audio_deps():
    """
    Lazy-import optional deps used for SRT concatenation.
    This keeps the basic (single-text) endpoints working even if these deps
    aren't installed.
    """
    try:
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for subtitle synthesis. Install requirements first:\n"
            "  pip install -r requirements.txt\n"
            f"Details: {e}"
        )
    return np, sf


def _wav_to_mp3(wav_path: Path, mp3_path: Path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg was not found on PATH. MP3 export requires ffmpeg.\n"
            "Install ffmpeg and make sure the 'ffmpeg' command is available."
        )
    # -q:a 2 gives good quality/size tradeoff
    proc = subprocess.run(
        [ffmpeg, "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-q:a", "2", str(mp3_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg conversion failed")


def _ffmpeg_atempo_chain(tempo: float) -> str:
    """
    ffmpeg atempo only supports 0.5..2.0 per filter.
    Chain multiple atempo filters to achieve a wider range.
    """
    tempo = float(tempo)
    if tempo <= 0:
        tempo = 1.0
    filters = []
    # Speed-up: tempo > 2
    while tempo > 2.0:
        filters.append("atempo=2.0")
        tempo /= 2.0
    # Slow-down: tempo < 0.5
    while tempo < 0.5:
        filters.append("atempo=0.5")
        tempo /= 0.5
    filters.append(f"atempo={tempo:.6f}")
    return ",".join(filters)


def _time_stretch_wav_ffmpeg(in_wav: Path, out_wav: Path, tempo: float):
    """
    Pitch-preserving time-stretch using ffmpeg's atempo filter.
    - tempo > 1.0 speeds up (shorter)
    - tempo < 1.0 slows down (longer)
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    chain = _ffmpeg_atempo_chain(tempo)
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(in_wav),
            "-filter:a",
            chain,
            "-acodec",
            "pcm_s16le",
            str(out_wav),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg atempo failed")


def _tts_to_file_speed(*, text: str, language: str, file_path: Path, speed: float | None, tts_kwargs: dict):
    """
    Render text to a WAV file using chatterbox-tts.

    Notes:
    - `speed` is currently ignored (kept for backward compatibility with the subtitle pipeline).
    - chatterbox-tts voice conditioning uses a single `audio_prompt_path`. If the caller passes
      multiple refs, we use the first one.
    """
    kwargs = dict(tts_kwargs)
    lang_id = (language or "en").strip().lower()
    if lang_id == "zh-cn":
        lang_id = "zh"
    if lang_id not in ("en", "zh"):
        lang_id = "en"

    # speaker conditioning (optional)
    audio_prompt_path = None
    if "speaker_wav" in kwargs:
        sw = kwargs.pop("speaker_wav")
        if isinstance(sw, (list, tuple)):
            audio_prompt_path = str(sw[0]) if sw else None
        else:
            audio_prompt_path = str(sw)

    with tts_lock:
        # Only the multilingual model accepts language_id.
        gen_kwargs = {}
        if MODEL_SPECS.get(tts_model_type, {}).get("supports_language_id"):
            gen_kwargs["language_id"] = lang_id
        if audio_prompt_path:
            gen_kwargs["audio_prompt_path"] = audio_prompt_path
        wav = tts_model.generate(text, **gen_kwargs)  # type: ignore
        sr = int(getattr(tts_model, "sr", 24000))
        _save_wav_tensor_to_file(wav, file_path, sr)
        del wav  # release the output tensor before the lock drops
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _cleanup_old_jobs():
    now = time.time()
    with SRT_JOBS_LOCK:
        dead = [k for k, v in SRT_JOBS.items() if now - v.get("updated_at", now) > SRT_JOB_TTL_SEC]
        for k in dead:
            SRT_JOBS.pop(k, None)
            SRT_JOB_CONTROLS.pop(k, None)


def _job_update(job_id: str, **fields):
    with SRT_JOBS_LOCK:
        job = SRT_JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = time.time()


def _job_wait_if_paused(job_id: str):
    """
    Block a background SRT job while it is paused.

    Pause/resume is cooperative: an in-flight cue is allowed to finish, then the
    worker sleeps before starting the next cue. This avoids interrupting model
    inference mid-generation while still reducing resource usage during long jobs.
    """
    while True:
        with SRT_JOBS_LOCK:
            job = SRT_JOBS.get(job_id)
            controls = SRT_JOB_CONTROLS.get(job_id)
            if not job or not controls:
                raise RuntimeError("SRT job no longer exists.")
            if job.get("canceled"):
                raise _SrtJobCancelled()
            paused = bool(job.get("paused"))
            resume_event = controls["resume_event"]
        if not paused:
            return
        resume_event.wait(timeout=0.25)


def _job_check_cancelled(job_id: str):
    with SRT_JOBS_LOCK:
        job = SRT_JOBS.get(job_id)
        if not job:
            raise RuntimeError("SRT job no longer exists.")
        if job.get("canceled"):
            raise _SrtJobCancelled()


# Maximum pitch-preserving ffmpeg atempo speed-up ratio applied to an overlong
# subtitle cue before falling back to tail trimming. We keep this noticeably
# higher than before so fewer words are lost on dense subtitle lines.
_MAX_ATEMPO_RATIO = 4.0

# Characters that count as terminal punctuation in a cue.
_SENTENCE_END = frozenset(".?!…")


def _normalize_cue_text(text: str, next_text: str | None) -> str:
    """
    Append terminal punctuation to a subtitle cue that lacks it.

    TTS models use punctuation to shape prosody (pitch contour, pause length,
    breath placement). SRT lines frequently omit it, which causes the model to
    produce flat, uncertain, or cut-off intonation.

    Rules applied in order:
    - Already ends with recognized punctuation → return unchanged.
    - Next cue begins with a lowercase letter (the sentence continues across the
      cue boundary) → append a comma (continuation / slight pause).
    - Otherwise → append a period (falling intonation / sentence end).
    """
    if not text:
        return text
    if text[-1] in _SENTENCE_END or text[-1] in ",;:":
        return text
    if next_text and next_text[0].islower():
        return text + ","
    return text + "."


def _srt_to_audio_file(*, cues, lang: str, fmt: str, tts_kwargs: dict, progress_cb=None, wait_if_paused=None, check_cancelled=None):
    """
    Render cues into a single output file.
    - Normalizes cue text with terminal punctuation before synthesis to improve
      TTS prosody (Approach 7).
    - Aligns segments by inserting silence to match each cue's start time.
    - If synthesized audio exceeds the cue duration:
        a) Re-synthesize at a faster speaking rate (skipped: chatterbox has no speed knob).
        b) Time-stretch via ffmpeg atempo, up to _MAX_ATEMPO_RATIO (Approach 2).
           For extreme overruns we still apply the maximum allowed atempo first,
           then trim only the small remainder if needed.
        c) Trim the tail (Approach 1). The np.interp resampling fallback has been
           removed because it compresses the waveform in time and raises pitch.
    - If synthesized audio is shorter than the cue duration, pad with silence.
    """
    np, sf = _load_audio_deps()

    # Always render to WAV first, then convert to MP3 if requested.
    wav_out = unique_path(OUTPUT_FOLDER, ".wav")

    sr = None
    parts = []
    current_ms = 0

    total = len(cues)
    for i, cue in enumerate(cues, start=1):
        if wait_if_paused:
            wait_if_paused()
        if check_cancelled:
            check_cancelled()
        start_ms = int(cue["start_ms"])
        end_ms = int(cue["end_ms"])
        if end_ms <= start_ms:
            continue
        # Approach 7: normalize punctuation to guide TTS prosody.
        next_text = cues[i]["text"] if i < len(cues) else None
        text = _normalize_cue_text(cue["text"], next_text)
        target_ms = end_ms - start_ms

        # Overlapping cues can't be represented in a single mono track without mixing.
        # We clamp to the current timeline position to keep output monotonic.
        if start_ms < current_ms:
            start_ms = current_ms
            end_ms = max(end_ms, start_ms)
            target_ms = max(1, end_ms - start_ms)

        # If we know sample rate already, insert silence to match the cue start time.
        if sr is not None and start_ms > current_ms:
            silence_len = int((start_ms - current_ms) * sr / 1000)
            if silence_len > 0:
                parts.append(np.zeros(silence_len, dtype=np.float32))
                current_ms = start_ms

        seg_path = unique_path(TMP_FOLDER, ".wav")
        try:
            # -----------------------------------------------------------------
            # 1) Synthesize at normal speed first
            # -----------------------------------------------------------------
            _tts_to_file_speed(text=text, language=lang, file_path=seg_path, speed=None, tts_kwargs=tts_kwargs)

            data, seg_sr = sf.read(seg_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr is None:
                sr = int(seg_sr)
                # For the first cue, we can now also honor any leading silence.
                if start_ms > 0:
                    silence_len = int(start_ms * sr / 1000)
                    if silence_len > 0:
                        parts.append(np.zeros(silence_len, dtype=np.float32))
                        current_ms = start_ms
            elif int(seg_sr) != sr:
                raise RuntimeError(
                    f"Sample rate mismatch across segments: expected {sr}, got {seg_sr}."
                )

            target_len = int(target_ms * sr / 1000)
            target_len = max(1, target_len)
            if len(data) > target_len:
                # a) Try faster speaking rate (only one retry to avoid large overhead).
                if tts_supports_speed and tts_speed_param:
                    try:
                        # tempo_needed > 1 means we need to speed up
                        tempo_needed = len(data) / target_len
                        # clamp to keep it natural-ish
                        speed = max(1.05, min(1.75, float(tempo_needed)))
                        _tts_to_file_speed(text=text, language=lang, file_path=seg_path, speed=speed, tts_kwargs=tts_kwargs)
                        data, seg_sr = sf.read(seg_path, dtype="float32")
                        if data.ndim > 1:
                            data = data.mean(axis=1)
                        if int(seg_sr) != sr:
                            raise RuntimeError(
                                f"Sample rate mismatch across segments: expected {sr}, got {seg_sr}."
                            )
                    except Exception:
                        pass

            if len(data) > target_len:
                # b) Pitch-preserving time-stretch via ffmpeg atempo (Approach 2).
                # Apply as much atempo speed-up as we safely allow before any
                # trimming. This preserves pitch and keeps more words than
                # skipping atempo entirely for large overruns.
                tempo = float(len(data) / target_len)
                stretch_tempo = min(tempo, _MAX_ATEMPO_RATIO)
                if stretch_tempo > 1.0:
                    try:
                        stretched = unique_path(TMP_FOLDER, ".wav")
                        _time_stretch_wav_ffmpeg(seg_path, stretched, stretch_tempo)
                        data, seg_sr = sf.read(stretched, dtype="float32")
                        if data.ndim > 1:
                            data = data.mean(axis=1)
                        if int(seg_sr) != sr:
                            raise RuntimeError(
                                f"Sample rate mismatch across segments: expected {sr}, got {seg_sr}."
                            )
                        try:
                            stretched.unlink(missing_ok=True)
                        except Exception:
                            pass
                    except Exception:
                        pass

            if len(data) > target_len:
                # c) Trim the tail (Approach 1).
                # The previous np.interp resampling fallback has been removed: it
                # compressed the waveform in the time domain and raised pitch in
                # direct proportion to the compression ratio, which was the primary
                # cause of the high-pitch artifact in SRT synthesis.
                # Any overrun falls at trailing silence or the very end of speech
                # — far less noticeable than pitch distortion across the whole cue.
                data = data[:target_len]
            elif len(data) < target_len:
                pad = np.zeros(target_len - len(data), dtype=np.float32)
                data = np.concatenate([data, pad])

            parts.append(data)
            current_ms = start_ms + target_ms

            if progress_cb:
                try:
                    progress_cb(i, total)
                except Exception:
                    pass
        finally:
            try:
                seg_path.unlink(missing_ok=True)
            except Exception:
                pass

    if sr is None or not parts:
        raise RuntimeError("No valid subtitle cues found to synthesize.")

    audio = np.concatenate(parts) if len(parts) > 1 else parts[0]
    del parts  # free per-cue buffers; only the concatenated array is needed now
    sf.write(wav_out, audio, sr)

    if fmt == "wav":
        return wav_out

    # MP3 requested
    mp3_out = wav_out.with_suffix(".mp3")
    _wav_to_mp3(wav_out, mp3_out)
    try:
        wav_out.unlink(missing_ok=True)
    except Exception:
        pass
    return mp3_out


def _start_srt_job(*, cues, lang: str, fmt: str, tts_kwargs: dict) -> str:
    _cleanup_old_jobs()
    job_id = uuid.uuid4().hex
    now = time.time()
    with SRT_JOBS_LOCK:
        SRT_JOBS[job_id] = {
            "status": "running",
            "paused": False,
            "canceled": False,
            "progress": 0.0,
            "done": 0,
            "total": len(cues),
            "output_id": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        resume_event = threading.Event()
        resume_event.set()
        cancel_event = threading.Event()
        SRT_JOB_CONTROLS[job_id] = {
            "resume_event": resume_event,
            "cancel_event": cancel_event,
        }

    def _worker():
        try:
            def cb(done, total):
                with SRT_JOBS_LOCK:
                    paused = bool(SRT_JOBS.get(job_id, {}).get("paused"))
                _job_update(
                    job_id,
                    progress=(done / total) if total else 1.0,
                    done=done,
                    total=total,
                    status="paused" if paused else "running",
                )

            out_path = _srt_to_audio_file(
                cues=cues,
                lang=lang,
                fmt=fmt,
                tts_kwargs=tts_kwargs,
                progress_cb=cb,
                wait_if_paused=lambda: _job_wait_if_paused(job_id),
                check_cancelled=lambda: _job_check_cancelled(job_id),
            )
            _job_update(job_id, status="done", progress=1.0, output_id=out_path.name)
        except _SrtJobCancelled:
            # Cooperative cancel: stop before starting the next cue.
            _job_update(job_id, status="canceled", error=None, output_id=None)
        except Exception as e:
            _job_update(job_id, status="error", error=str(e))

    threading.Thread(target=_worker, daemon=True).start()
    return job_id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model-status")
def model_status():
    return jsonify({"ready": tts_model is not None, "device": tts_device, "model_type": tts_model_type})


@app.route("/upload-ref", methods=["POST"])
def upload_ref():
    """Upload a reference audio file for voice cloning."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "" or not allowed_audio(f.filename):
        return jsonify({"error": "Invalid file"}), 400

    ext = f.filename.rsplit(".", 1)[1].lower()
    save_path = unique_path(UPLOAD_FOLDER, f".{ext}")
    f.save(save_path)
    return jsonify({"ref_id": save_path.name})


@app.route("/synthesize", methods=["POST"])
def synthesize():
    """
    Generate speech from text using one or more cloned voice references.

    Body (JSON):
    {
        "text":    "Text to synthesize",
        "ref_ids": ["ref1.wav", "ref2.wav"],  // preferred: list of reference audio ids
        "ref_id":  "reference_audio.wav",      // backward compat: single reference
        "lang":    "zh"   // zh | en
    }
    chatterbox-tts uses a single reference clip (`audio_prompt_path`) for voice conditioning.
    For compatibility with the previous API, this server accepts multiple `ref_ids` but will
    currently use the first clip.
    """
    if tts_model is None:
        return jsonify({"error": "Model not loaded yet, please wait."}), 503

    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    lang = data.get("lang", "en").strip().lower()

    # Accept ref_ids (list) or ref_id (single string, backward compat)
    ref_ids = data.get("ref_ids") or []
    if not ref_ids:
        ref_id = (data.get("ref_id") or "").strip()
        if ref_id:
            ref_ids = [ref_id]

    if not text:
        return jsonify({"error": "text is required"}), 400
    if not ref_ids:
        return jsonify({"error": "ref_ids is required — upload at least one reference audio clip first"}), 400
    if lang not in ("zh", "en", "zh-cn"):
        lang = "en"
    if lang == "zh-cn":
        lang = "zh"

    ref_paths = []
    for rid in ref_ids:
        p = UPLOAD_FOLDER / rid.strip()
        if not p.exists():
            return jsonify({"error": f"Reference audio not found: {rid}"}), 404
        ref_paths.append(str(p))

    # chatterbox-tts uses a single audio prompt path; we use the first uploaded clip.
    speaker_wav = ref_paths[0]
    out_path = unique_path(OUTPUT_FOLDER, ".wav")

    try:
        _tts_to_file_speed(
            text=text,
            language=lang,
            file_path=out_path,
            speed=None,
            tts_kwargs={"speaker_wav": speaker_wav},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"output_id": out_path.name})


@app.route("/synthesize-builtin", methods=["POST"])
def synthesize_builtin():
    """
    Generate speech using the default chatterbox-tts voice (no reference audio needed).

    Body (JSON):
    {
        "text":    "Text content",
        "speaker": "Default",
        "lang":    "en"
    }
    """
    if tts_model is None:
        return jsonify({"error": "Model not loaded yet, please wait."}), 503

    data    = request.get_json(force=True)
    text    = (data.get("text") or "").strip()
    # chatterbox-tts is a zero-shot model and does not expose a discrete speaker list;
    # we keep the `speaker` field for UI compatibility.
    speaker = data.get("speaker", "Default")
    lang    = data.get("lang", "en").strip().lower()

    if not text:
        return jsonify({"error": "text is required"}), 400
    if lang == "zh-cn":
        lang = "zh"

    out_path = unique_path(OUTPUT_FOLDER, ".wav")
    try:
        # Default voice (no reference audio)
        _tts_to_file_speed(
            text=text,
            language=lang,
            file_path=out_path,
            speed=None,
            tts_kwargs={},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"output_id": out_path.name})


@app.route("/speakers")
def speakers():
    """
    Return available "speakers".

    chatterbox-tts is a zero-shot model and does not provide a discrete built-in speaker list.
    For UI compatibility, we expose a single option: "Default" (no reference audio).
    """
    return jsonify({"speakers": ["Default"] if tts_model is not None else []})


@app.route("/download/outputs/<file_id>")
def download_output(file_id: str):
    return send_from_directory(OUTPUT_FOLDER.resolve(), file_id, as_attachment=True)


@app.route("/play/outputs/<file_id>")
def play_output(file_id: str):
    """Stream audio for in-browser playback."""
    return send_from_directory(OUTPUT_FOLDER.resolve(), file_id)


@app.route("/list-outputs")
def list_outputs():
    files = sorted(
        [{"name": f.name, "size": f.stat().st_size} for f in OUTPUT_FOLDER.iterdir() if f.is_file()],
        key=lambda x: x["name"], reverse=True,
    )
    return jsonify(files)


@app.route("/delete-output/<file_id>", methods=["DELETE"])
def delete_output(file_id: str):
    p = OUTPUT_FOLDER / file_id
    if p.exists():
        p.unlink()
    return jsonify({"ok": True})


@app.route("/synthesize-srt", methods=["POST"])
def synthesize_srt():
    """
    Generate a single audio file from a .srt subtitle file using a cloned voice.

    multipart/form-data:
      file:      <subtitles.srt>
      ref_ids[]: <uploaded reference audio id>  (repeat for multiple clips)
      ref_id:    <single reference audio id>     (backward compat)
      lang:      zh|en
      format:    wav|mp3   (default: wav)
    """
    if tts_model is None:
        return jsonify({"error": "Model not loaded yet, please wait."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]
    if f.filename == "" or not allowed_subtitles(f.filename):
        return jsonify({"error": "Invalid subtitles file (expected .srt)"}), 400

    # Accept ref_ids[] (list) or ref_id (single, backward compat)
    ref_ids = request.form.getlist("ref_ids[]")
    if not ref_ids:
        ref_id = (request.form.get("ref_id") or "").strip()
        if ref_id:
            ref_ids = [ref_id]

    lang = (request.form.get("lang") or "en").strip().lower()
    fmt = (request.form.get("format") or "wav").strip().lower()
    async_mode = (request.form.get("async") or "0").strip().lower() in ("1", "true", "yes")

    if not ref_ids:
        return jsonify({"error": "ref_ids is required — upload at least one reference audio clip first"}), 400
    if lang not in ("zh", "en", "zh-cn"):
        lang = "zh"
    if lang == "zh-cn":
        lang = "zh"
    if fmt not in ("wav", "mp3"):
        fmt = "wav"

    ref_paths = []
    for rid in ref_ids:
        p = UPLOAD_FOLDER / rid.strip()
        if not p.exists():
            return jsonify({"error": f"Reference audio not found: {rid}"}), 404
        ref_paths.append(str(p))

    speaker_wav = ref_paths if len(ref_paths) > 1 else ref_paths[0]

    # Read + parse SRT
    raw = f.read()
    try:
        content = raw.decode("utf-8-sig")
    except Exception:
        content = raw.decode("latin-1", errors="ignore")
    cues = parse_srt(content)
    if not cues:
        return jsonify({"error": "No valid subtitle cues found in the .srt file."}), 400

    if async_mode:
        job_id = _start_srt_job(cues=cues, lang=lang, fmt=fmt, tts_kwargs={"speaker_wav": speaker_wav})
        return jsonify({"job_id": job_id})

    try:
        out_path = _srt_to_audio_file(
            cues=cues,
            lang=lang,
            fmt=fmt,
            tts_kwargs={"speaker_wav": speaker_wav},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"output_id": out_path.name})


@app.route("/synthesize-srt-builtin", methods=["POST"])
def synthesize_srt_builtin():
    """
    Generate a single audio file from a .srt subtitle file using a built-in speaker.

    multipart/form-data:
      file:   <subtitles.srt>
      speaker:<built-in speaker name>
      lang:   zh|en
      format: wav|mp3   (default: wav)
    """
    if tts_model is None:
        return jsonify({"error": "Model not loaded yet, please wait."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]
    if f.filename == "" or not allowed_subtitles(f.filename):
        return jsonify({"error": "Invalid subtitles file (expected .srt)"}), 400

    speaker = (request.form.get("speaker") or "").strip()
    lang = (request.form.get("lang") or "en").strip().lower()
    fmt = (request.form.get("format") or "wav").strip().lower()
    async_mode = (request.form.get("async") or "0").strip().lower() in ("1", "true", "yes")

    if not speaker:
        return jsonify({"error": "speaker is required"}), 400
    if lang == "zh-cn":
        lang = "zh"
    if fmt not in ("wav", "mp3"):
        fmt = "wav"

    # Read + parse SRT
    raw = f.read()
    try:
        content = raw.decode("utf-8-sig")
    except Exception:
        content = raw.decode("latin-1", errors="ignore")
    cues = parse_srt(content)
    if not cues:
        return jsonify({"error": "No valid subtitle cues found in the .srt file."}), 400

    if async_mode:
        # Built-in mode = default voice (no reference)
        job_id = _start_srt_job(cues=cues, lang=lang, fmt=fmt, tts_kwargs={})
        return jsonify({"job_id": job_id})

    try:
        out_path = _srt_to_audio_file(
            cues=cues,
            lang=lang,
            fmt=fmt,
            # Built-in mode = default voice (no reference)
            tts_kwargs={},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"output_id": out_path.name})


@app.route("/srt-job/<job_id>")
def srt_job(job_id: str):
    _cleanup_old_jobs()
    with SRT_JOBS_LOCK:
        job = SRT_JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        # return a shallow copy to avoid accidental mutation
        return jsonify(dict(job))


@app.route("/srt-job/<job_id>/pause", methods=["POST"])
def pause_srt_job(job_id: str):
    _cleanup_old_jobs()
    with SRT_JOBS_LOCK:
        job = SRT_JOBS.get(job_id)
        controls = SRT_JOB_CONTROLS.get(job_id)
        if not job or not controls:
            return jsonify({"error": "Job not found"}), 404
        if job.get("status") in ("done", "error"):
            return jsonify(dict(job))
        controls["resume_event"].clear()
        job["paused"] = True
        job["status"] = "paused"
        job["updated_at"] = time.time()
        return jsonify(dict(job))


@app.route("/srt-job/<job_id>/resume", methods=["POST"])
def resume_srt_job(job_id: str):
    _cleanup_old_jobs()
    with SRT_JOBS_LOCK:
        job = SRT_JOBS.get(job_id)
        controls = SRT_JOB_CONTROLS.get(job_id)
        if not job or not controls:
            return jsonify({"error": "Job not found"}), 404
        if job.get("status") in ("done", "error"):
            return jsonify(dict(job))
        controls["resume_event"].set()
        job["paused"] = False
        job["status"] = "running"
        job["updated_at"] = time.time()
        return jsonify(dict(job))


@app.route("/srt-job/<job_id>/cancel", methods=["POST"])
def cancel_srt_job(job_id: str):
    _cleanup_old_jobs()
    with SRT_JOBS_LOCK:
        job = SRT_JOBS.get(job_id)
        controls = SRT_JOB_CONTROLS.get(job_id)
        if not job or not controls:
            return jsonify({"error": "Job not found"}), 404
        if job.get("status") in ("done", "error", "canceled"):
            return jsonify(dict(job))
        job["canceled"] = True
        job["paused"] = False
        job["status"] = "canceled"
        job["error"] = None
        job["output_id"] = None
        job["updated_at"] = time.time()
        # Wake any paused worker so it can exit quickly.
        try:
            controls["resume_event"].set()
        except Exception:
            pass
        return jsonify(dict(job))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _start_job_cleanup_scheduler(interval_sec: int = 1800):
    """
    Periodically prune stale SRT job entries from SRT_JOBS and SRT_JOB_CONTROLS.

    Without this, entries only get cleaned when an SRT API endpoint is hit.
    On idle servers the dicts would grow indefinitely. The scheduler runs as a
    daemon thread so it never blocks a clean process exit.
    """
    def _loop():
        while True:
            time.sleep(interval_sec)
            _cleanup_old_jobs()
    t = threading.Thread(target=_loop, daemon=True, name="job-cleanup-scheduler")
    t.start()


if __name__ == "__main__":
    # Load model in background thread so the server starts immediately
    t = threading.Thread(target=load_model, daemon=True)
    t.start()
    _start_job_cleanup_scheduler()
    app.run(debug=False, host="0.0.0.0", port=5001)
