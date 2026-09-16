"""A live instrumentation panel shared by the desktop preview and recorded video."""

import math
import statistics
import textwrap
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from doom_bert.policy import BUTTONS

WIDTH, HEIGHT = 1440, 900
BG = "#111316"
PANEL = "#1b1e23"
EDGE = "#343a43"
ACCENT = "#73caff"
TEXT = "#edf0f5"
MUTED = "#98a2b2"
AMBER = "#f3bd72"


@lru_cache(maxsize=32)
def font(size: int, bold: bool = False, mono: bool = False):
    names = (
        [
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
        if mono
        else [
            f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",
            f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
        ]
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


class LiveMetrics:
    """Use completed decisions, including all intervening overhead, for throughput."""

    def __init__(self):
        self.latencies = deque(maxlen=60)
        self.completions = deque(maxlen=60)

    def update(self, latency_ms: float, completed_at: float) -> dict:
        self.latencies.append(latency_ms)
        self.completions.append(completed_at)
        elapsed = self.completions[-1] - self.completions[0]
        rate = (len(self.completions) - 1) / elapsed if elapsed > 0 else None
        ordered = sorted(self.latencies)
        return {
            "decision_p50_ms": statistics.median(ordered),
            "decision_p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1],
            "rolling_decisions_per_second": rate,
            "metric_samples": len(ordered),
        }


class StatsOverlay:
    width, height = WIDTH, HEIGHT

    def __init__(
        self,
        *,
        policy_name: str,
        device: str | None,
        dtype: str | None,
        instruction: str,
        untrained: bool,
        video_clock: str,
    ):
        self.policy_name = policy_name
        self.device = "METAL / MPS" if device == "mps" else (device or "CPU").upper()
        self.dtype = (dtype or "PYTHON").upper()
        self.instruction = instruction
        self.untrained = untrained
        self.video_clock = video_clock
        self.small = font(12, mono=True)
        self.body = font(15)
        self.mono = font(14, mono=True)
        self.title = font(28, bold=True)
        self.number = font(43, bold=True)
        self.last_screen = None
        self.cached_panel = None
        self.refreshed_at = -1.0

    def _panel(self, stats: dict) -> Image.Image:
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 20), f"DOOM / {self.policy_name}", font=self.title, fill=TEXT)
        draw.text(
            (26, 63),
            "STATE + INSTRUCTION  >  BUTTON SCORES  >  GAME ACTION",
            font=self.small,
            fill=MUTED,
        )
        draw.text(
            (1008, 24), f"{self.device}  /  {self.dtype}", font=self.mono, fill=ACCENT
        )
        label = (
            "UNTRAINED HEAD / SPEED DEMO" if self.untrained else "LIVE POLICY INFERENCE"
        )
        if self.policy_name == "RANDOM BASELINE":
            label = "RANDOM POLICY / NO MODEL"
        draw.text(
            (1008, 53), label, font=self.small, fill=AMBER if self.untrained else MUTED
        )

        def card(box):
            draw.rounded_rectangle(box, radius=8, fill=PANEL, outline=EDGE, width=1)

        card((1008, 100, 1204, 226))
        card((1216, 100, 1416, 226))
        draw.text((1024, 113), "DECISION p50 / MS", font=self.small, fill=MUTED)
        latency = stats.get("decision_p50_ms")
        draw.text(
            (1024, 133),
            f"{latency:.1f}" if latency is not None else "--",
            font=self.number,
            fill=ACCENT,
        )
        draw.text(
            (1024, 187),
            f"p95 {stats.get('decision_p95_ms', 0):.1f} ms",
            font=self.mono,
            fill=TEXT,
        )
        draw.text((1024, 208), "text + model + decode", font=font(10), fill=MUTED)
        draw.text((1232, 113), "LOOP / DECISIONS/S", font=self.small, fill=MUTED)
        rate = stats.get("rolling_decisions_per_second")
        draw.text(
            (1232, 133),
            f"{rate:.1f}" if rate is not None else "--",
            font=self.number,
            fill=ACCENT,
        )
        draw.text(
            (1232, 187),
            f"{35 / stats.get('action_tics', 1):g}/s = real time",
            font=self.mono,
            fill=TEXT,
        )
        draw.text((1232, 208), "includes capture + HUD", font=font(10), fill=MUTED)

        card((1008, 240, 1416, 326))
        tokens = stats.get("input_tokens")
        draw.text(
            (1024, 252),
            "INSTRUCTION" + (f" / {tokens} INPUT TOKENS" if tokens else ""),
            font=self.small,
            fill=MUTED,
        )
        for i, line in enumerate(
            textwrap.wrap(self.instruction or "Random baseline", width=48)[:3]
        ):
            draw.text((1024, 274 + 16 * i), line, font=self.body, fill=TEXT)

        card((1008, 340, 1416, 648))
        draw.text((1024, 353), "BUTTON CLASSES", font=self.small, fill=ACCENT)
        draw.text((1260, 353), "SCORE   ON", font=self.small, fill=MUTED)
        scores = stats.get("action_scores") or {}
        selected = set(stats.get("pressed", []))
        for index, name in enumerate(BUTTONS):
            y = 388 + index * 34
            probability = scores.get(name)
            active = name in selected
            color = (
                ACCENT
                if active
                else (AMBER if probability is not None and probability > 0.5 else MUTED)
            )
            draw.text((1024, y), name, font=self.mono, fill=color)
            draw.rounded_rectangle(
                (1150, y + 3, 1270, y + 14), radius=3, fill="#303641"
            )
            if probability is not None:
                fill = round(120 * probability)
                if fill > 0:
                    draw.rounded_rectangle(
                        (1150, y + 3, 1150 + fill, y + 14), radius=3, fill=color
                    )
            draw.line((1210, y + 1, 1210, y + 16), fill=AMBER, width=1)
            draw.text(
                (1282, y),
                f"{probability:.3f}" if probability is not None else "  --",
                font=self.mono,
                fill=color,
            )
            draw.rectangle(
                (1380, y + 1, 1396, y + 17), fill=ACCENT if active else BG, outline=EDGE
            )
            if active:
                draw.line((1383, y + 9, 1387, y + 13, 1393, y + 5), fill=BG, width=2)
        draw.text(
            (1024, 628),
            "marker: 0.5 / cyan: executed / amber: suppressed",
            font=font(10),
            fill=MUTED,
        )

        card((1008, 662, 1416, 820))
        draw.text(
            (1024, 675), "SELECTION / VALID COMBINATIONS", font=self.small, fill=ACCENT
        )
        total = 2 ** len(stats.get("buttons", BUTTONS))
        combo = stats.get("combo_score")
        draw.text(
            (1024, 698),
            f"{stats.get('legal_action_count', 0)}/{total} legal"
            + (f" / chosen {combo:.4f}" if combo is not None else ""),
            font=self.mono,
            fill=TEXT,
        )
        draw.text(
            (1024, 722),
            "Max product: ON = p, OFF = 1-p."
            if scores
            else "Uniform draw from legal combinations.",
            font=self.body,
            fill=MUTED,
        )
        notes = stats.get("selection_notes", [])
        for index, note in enumerate(notes[:3]):
            draw.text((1024, 747 + index * 19), note, font=font(12), fill=AMBER)
        if not notes:
            draw.text(
                (1024, 752),
                "No conflicting buttons above 0.5.",
                font=self.body,
                fill=MUTED,
            )
        draw.text(
            (1024, 795),
            "Opposite directions and empty-ammo fire masked.",
            font=font(11),
            fill=MUTED,
        )

        card((1008, 834, 1416, 886))
        draw.text(
            (1024, 844),
            f"WALL {stats.get('display_wall_seconds', 0):6.2f}s  GAME {stats.get('second', 0):6.2f}s",
            font=self.mono,
            fill=TEXT,
        )
        draw.text(
            (1024, 867),
            f"{stats.get('decision_count', 0):,} decisions / last {stats.get('metric_samples', 0)} measured",
            font=self.small,
            fill=MUTED,
        )

        draw.rounded_rectangle((23, 99, 985, 821), radius=6, outline=EDGE, width=2)
        playback = (
            "ACTUAL WALL-CLOCK PLAYBACK"
            if self.video_clock == "wall"
            else "GAME-TIME PLAYBACK / 35 FPS"
        )
        draw.text((26, 827), playback, font=self.small, fill=ACCENT)
        variables = stats.get("variables", {})
        draw.text(
            (525, 827),
            f"HEALTH {variables.get('HEALTH', 0):.0f}  AMMO {variables.get('SELECTED_WEAPON_AMMO', 0):.0f}  KILLS {stats.get('total_kills', 0)}  EP {stats.get('episode', 1)}",
            font=self.small,
            fill=MUTED,
        )
        nodes = [
            (24, 224, "STRUCTURED STATE"),
            (254, 484, "ModernBERT / ENCODER"),
            (514, 694, "7 SIGMOID SCORES"),
            (724, 984, "VALID BUTTON VECTOR"),
        ]
        for index, (left, right, label) in enumerate(nodes):
            draw.rounded_rectangle(
                (left, 853, right, 884), radius=6, fill=PANEL, outline=EDGE
            )
            draw.text((left + 12, 862), label, font=self.small, fill=ACCENT)
            if index < len(nodes) - 1:
                draw.text((right + 9, 861), ">", font=self.mono, fill=AMBER)
        return canvas

    def render(self, screen, stats: dict):
        if screen is not None:
            self.last_screen = screen
        if self.last_screen is None:
            raise ValueError("Overlay requires an initial game screen")
        now = time.monotonic()
        # Keep labels readable and avoid repainting every glyph on every inference.
        if self.cached_panel is None or now - self.refreshed_at >= 0.1:
            self.cached_panel = self._panel(stats)
            self.refreshed_at = now
        canvas = self.cached_panel.copy()
        game = Image.fromarray(self.last_screen).resize(
            (960, 720), Image.Resampling.NEAREST
        )
        canvas.paste(game, (24, 100))
        return np.asarray(canvas)


class LivePreview:
    def __init__(self):
        import pygame

        self.pygame = pygame
        pygame.display.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Doom Bert | ModernBERT live decisions")
        self.last_presented = 0.0

    def ready(self) -> bool:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                raise KeyboardInterrupt
        return time.monotonic() - self.last_presented >= 1 / 30

    def present(self, screen):
        now = time.monotonic()
        surface = self.pygame.image.frombuffer(screen.tobytes(), (WIDTH, HEIGHT), "RGB")
        self.window.blit(surface, (0, 0))
        self.pygame.display.flip()
        self.last_presented = now

    def close(self):
        self.pygame.display.quit()


class RenderWorker:
    """Draw and encode away from inference; retain only the latest pending frame."""

    def __init__(self, recorder, overlay, fps: int = 30):
        self.recorder = recorder
        self.overlay = overlay
        self.fps = fps
        self.pending = Queue(maxsize=1)
        self.last_submitted = -1.0
        self.latest_frame = None
        self.dropped_frames = 0
        self.error = None
        self.thread = Thread(target=self._run, name="doom-renderer", daemon=True)
        self.thread.start()

    def ready(self, elapsed: float) -> bool:
        self._check_error()
        return elapsed - self.last_submitted >= 1 / self.fps

    def submit(self, screen, stats: dict, elapsed: float):
        self._check_error()
        payload = (screen.copy() if screen is not None else None, dict(stats), elapsed)
        try:
            self.pending.put_nowait(payload)
        except Full:
            try:
                self.pending.get_nowait()
                self.pending.task_done()
                self.dropped_frames += 1
            except Empty:
                pass
            self.pending.put_nowait(payload)
        self.last_submitted = elapsed

    def _check_error(self):
        if self.error is not None:
            raise RuntimeError("Dashboard rendering failed") from self.error

    def _run(self):
        while True:
            payload = self.pending.get()
            try:
                if payload is None:
                    return
                if self.error is not None:
                    continue
                screen, stats, elapsed = payload
                if self.overlay is not None:
                    screen = self.overlay.render(
                        screen, {**stats, "display_wall_seconds": elapsed}
                    )
                if self.recorder is not None:
                    self.recorder.write(screen, elapsed_seconds=elapsed)
                self.latest_frame = screen
            except Exception as error:
                self.error = error
            finally:
                self.pending.task_done()

    def close(self):
        self.pending.put(None, timeout=10)
        self.thread.join(timeout=20)
        if self.thread.is_alive():
            raise RuntimeError("Dashboard renderer did not finish")
        self._check_error()
