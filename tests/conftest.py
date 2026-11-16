import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        check=True,
    )


@pytest.fixture(scope="session")
def sample_clip(tmp_path_factory):
    """10 s, 640x360, 30 fps, 440 Hz tone."""
    path = tmp_path_factory.mktemp("media") / "sample.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=10",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=10",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    )
    return path


@pytest.fixture(scope="session")
def silent_clip(tmp_path_factory):
    """3 s, 640x360, no audio stream."""
    path = tmp_path_factory.mktemp("media") / "silent.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(path),
    )
    return path


@pytest.fixture(scope="session")
def music_file(tmp_path_factory):
    """4 s tone, shorter than the sample clip so looping is exercised."""
    path = tmp_path_factory.mktemp("media") / "music.wav"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=4", str(path))
    return path
