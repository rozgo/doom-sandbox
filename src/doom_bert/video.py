"""Stream game frames into a standard H.264 MP4 without retaining a whole run."""

import platform
from fractions import Fraction
from pathlib import Path

import av


class VideoRecorder:
    def __init__(self, path: Path, *, width: int, height: int, fps: int):
        self.time_base = Fraction(1, fps)
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
                self.stream.bit_rate = 8_000_000
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

    def write(self, screen) -> None:
        # A terminal episode has no screen; hold its last frame until the reset.
        if screen is not None:
            self.last_screen = screen.copy()
        if self.last_screen is None:
            raise ValueError("Video recording needs an initial game screen")
        frame = av.VideoFrame.from_ndarray(self.last_screen, format="rgb24")
        frame.pts = self.frames
        frame.time_base = self.time_base
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
