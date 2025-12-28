import subprocess
from pathlib import Path
import imageio_ffmpeg

def extract_audio(video_path: str, out_wav: str):
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        out_wav
    ]

    subprocess.run(cmd, check=True)
