import json
import logging
import subprocess
from pathlib import Path
from typing import Tuple, List

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return proc.stdout.decode("utf-8")


def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = ""
        if exc.stderr:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        msg = f"ffmpeg failed (exit {exc.returncode})"
        if stderr:
            msg = f"{msg}: {stderr}"
        raise RuntimeError(msg) from exc


def probe_video(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    output = _run(cmd)
    return json.loads(output)


def get_video_metadata(path: Path) -> Tuple[float | None, int | None, int | None]:
    data = probe_video(path)
    width = None
    height = None
    duration = None

    streams = data.get("streams", [])
    if streams:
        width = streams[0].get("width")
        height = streams[0].get("height")
        duration = streams[0].get("duration")

    if duration is None:
        fmt = data.get("format", {})
        duration = fmt.get("duration")

    if duration is not None:
        duration = float(duration)

    return duration, width, height


def extract_catalog_image(
    video_path: Path,
    output_path: Path,
    duration: float | None,
    width: int | None,
    height: int | None,
) -> Tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if duration and duration > 0:
        ts = duration / 2.0
    else:
        ts = 0

    max_w = settings.max_catalog_width
    max_h = settings.max_catalog_height

    if width and height and (width > max_w or height > max_h):
        scale_filter = f"scale='min(iw,{max_w})':'min(ih,{max_h})':force_original_aspect_ratio=decrease"
    else:
        scale_filter = "scale=iw:ih"

    attempts = [
        output_path.with_suffix(".png"),
    ]
    last_error: Exception | None = None
    for candidate in attempts:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(ts),
            "-i",
            str(video_path),
            "-vf",
            scale_filter,
            "-vframes",
            "1",
            "-q:v",
            "80",
            str(candidate),
        ]
        try:
            _run_ffmpeg(cmd)
            output_path = candidate
            last_error = None
            break
        except RuntimeError as exc:
            last_error = exc
            logger.warning("ffmpeg failed for %s: %s", candidate.suffix, exc)
    if last_error is not None:
        raise last_error

    meta = probe_video(output_path)
    streams = meta.get("streams", [])
    if streams:
        out_w = streams[0].get("width")
        out_h = streams[0].get("height")
    else:
        out_w, out_h = width or 0, height or 0

    return int(out_w), int(out_h)


def extract_catalog_images(
    video_path: Path,
    output_dir: Path,
    duration: float | None,
    width: int | None,
    height: int | None,
) -> List[Tuple[Path, int, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    max_w = settings.max_catalog_width
    max_h = settings.max_catalog_height

    if width and height and (width > max_w or height > max_h):
        scale_filter = f"scale='min(iw,{max_w})':'min(ih,{max_h})':force_original_aspect_ratio=decrease"
    else:
        scale_filter = "scale=iw:ih"

    output_patterns = [
        output_dir / "catalog_%05d.png",
    ]
    last_error: Exception | None = None
    used_pattern: Path | None = None
    for pattern in output_patterns:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1,{scale_filter}",
            "-q:v",
            "80",
            str(pattern),
        ]
        try:
            _run_ffmpeg(cmd)
            used_pattern = pattern
            last_error = None
            break
        except RuntimeError as exc:
            last_error = exc
            logger.warning("ffmpeg failed for %s: %s", pattern.suffix, exc)
    if used_pattern is None:
        raise last_error if last_error is not None else RuntimeError("ffmpeg failed")

    from PIL import Image

    results: List[Tuple[Path, int, int]] = []
    for img_path in sorted(output_dir.glob(f"catalog_*.{used_pattern.suffix.lstrip('.')}")):
        with Image.open(img_path) as img:
            out_w, out_h = img.size
        results.append((img_path, int(out_w), int(out_h)))

    return results
