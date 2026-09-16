"""Stream game frames into a standard H.264 MP4 without retaining a whole run."""

import platform
from fractions import Fraction
from pathlib import Path

import av


class VideoRecorder:
    def __init__(
        self, path: Path, *, width: int, height: int, fps: int, clock: str = "game"
    ):
        if clock not in ("game", "wall"):
            raise ValueError("Video clock must be game or wall")
        self.clock = clock
        self.fps = fps
        self.time_base = Fraction(1, 1_000_000) if clock == "wall" else Fraction(1, fps)
        self.last_pts = -1
        self.frames = 0
        self.last_screen = None
        self.encoder = "libx264"
        if platform.system() == "Darwin" and "h264_videotoolbox" in av.codecs_available:
            self.encoder = "h264_videotoolbox"

        def initialize():
            self.container = av.open(
                str(path), mode="w", options={"movflags": "+faststart"}
            )
            self.stream = self.container.add_stream(self.encoder, rate=fps)
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.codec_context.time_base = self.time_base
            if self.encoder == "h264_videotoolbox":
                self.stream.bit_rate = max(8_000_000, width * height * 12)
                self.stream.options = {"allow_sw": "0", "realtime": "1"}
            else:
                self.stream.options = {"crf": "20", "preset": "veryfast"}
            self.stream.codec_context.open()

        try:
            initialize()
        except av.FFmpegError:
            self.container.close()
            if self.encoder != "h264_videotoolbox":
                raise
            self.encoder = "libx264"
            initialize()

    def write(self, screen, *, elapsed_seconds: float | None = None) -> None:
        # A terminal episode has no screen; hold its last frame until the reset.
        if screen is not None:
            self.last_screen = screen.copy()
        if self.last_screen is None:
            raise ValueError("Video recording needs an initial game screen")
        frame = av.VideoFrame.from_ndarray(self.last_screen, format="rgb24")
        if self.clock == "wall":
            if elapsed_seconds is None or elapsed_seconds < 0:
                raise ValueError(
                    "Wall-clock recording requires a nonnegative elapsed time"
                )
            frame.pts = max(self.last_pts + 1, round(elapsed_seconds / self.time_base))
        else:
            frame.pts = self.frames
        self.last_pts = frame.pts
        frame.time_base = self.time_base
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def ready(self, elapsed_seconds: float) -> bool:
        """Bound display/encoding work without ever delaying a policy decision."""
        return (
            self.clock == "game"
            or self.frames == 0
            or elapsed_seconds - float(self.last_pts * self.time_base) >= 1 / self.fps
        )

    @property
    def duration_seconds(self) -> float:
        return (
            float(self.last_pts * self.time_base) + 1 / self.fps if self.frames else 0.0
        )

    def close(self) -> None:
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
