"""Showcase dashboard: what EmbeddingGemma 2 sees and how cosine matches become buttons."""

from __future__ import annotations

import textwrap

import numpy as np
from gemma_doom import AIM_WINDOW, AMMO_BOX, BAND_BOTTOM, BAND_TOP, HEALTH_BOX
from PIL import Image, ImageDraw

from doom_bert.overlay import ACCENT, AMBER, BG, EDGE, MUTED, PANEL, TEXT, font
from doom_bert.policy import BUTTONS

WIDTH, HEIGHT = 1440, 900
SCALE = 1.5  # 640x480 game -> 960x720
GAME_X, GAME_Y = 24, 104
RIGHT = 1008
GREEN = "#7fdc9a"
RED = "#ff7a7a"
DIM = "#4a515c"


def game_xy(x: float, y: float) -> tuple[float, float]:
    return GAME_X + x * SCALE, GAME_Y + y * SCALE


def draw_eye(draw, x, y, colour):
    """Modality icon for image inputs."""
    draw.ellipse((x, y + 2, x + 18, y + 12), outline=colour, width=2)
    draw.ellipse((x + 6, y + 4, x + 12, y + 10), fill=colour)


def draw_hash(draw, x, y, colour):
    """Modality icon for text/data inputs."""
    for dx in (6, 12):
        draw.line((x + dx, y, x + dx - 2, y + 14), fill=colour, width=2)
    for dy in (4, 10):
        draw.line((x + 2, y + dy, x + 17, y + dy), fill=colour, width=2)


class GemmaOverlay:
    width, height = WIDTH, HEIGHT

    def __init__(self, *, instruction, device, dtype, window_bearings, window_width, video_clock):
        self.instruction = instruction
        self.hardware = {"cuda": "NVIDIA RTX 4090 / CUDA", "mps": "APPLE METAL / MPS"}.get(device, device.upper())
        self.dtype = dtype.upper()
        self.window_centres = 320 + np.asarray(window_bearings) * 320
        self.window_width = window_width
        self.video_clock = video_clock
        self.title = font(30, bold=True)
        self.h2 = font(13, bold=True)
        self.body = font(15)
        self.mono = font(13, mono=True)
        self.small = font(11, mono=True)
        self.big = font(22, bold=True)
        self.last_screen = None
        self.panel = None
        self.panel_decision = None

    _eye = staticmethod(lambda draw, x, y, colour: draw_eye(draw, x, y, colour))
    _hash = staticmethod(lambda draw, x, y, colour: draw_hash(draw, x, y, colour))

    def _match(self, draw, right, y, left_kind, right_kind):
        """Right-aligned tag such as [eye] image <-> [#] text."""
        items = [(left_kind, AMBER), ("vs", None), (right_kind, ACCENT)]
        widths = [draw.textlength(k, font=self.small) + (24 if c else 0) for k, c in items]
        x = right - sum(widths) - 10 * (len(items) - 1)
        for (kind, colour), width in zip(items, widths, strict=True):
            if colour is None:
                draw.text((x, y), kind, font=self.small, fill=MUTED)
            else:
                (self._eye if kind == "image" else self._hash)(draw, x, y - 1, colour)
                draw.text((x + 24, y), kind, font=self.small, fill=colour)
            x += width + 10

    # Static-per-decision panel ---------------------------------------------------------
    def _card(self, draw, box, title, match=None):
        draw.rounded_rectangle(box, radius=8, fill=PANEL, outline=EDGE)
        draw.text((box[0] + 14, box[1] + 10), title, font=self.h2, fill=ACCENT)
        if match:
            self._match(draw, box[2] - 14, box[1] + 11, *match)

    def _bar(self, draw, x, y, w, h, value, lo, hi, colour):
        draw.rectangle((x, y, x + w, y + h), fill="#262a31")
        fill = max(0.0, min(1.0, (value - lo) / (hi - lo)))
        draw.rectangle((x, y, x + w * fill, y + h), fill=colour)

    def _panel(self, stats: dict) -> Image.Image:
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        d = stats.get("policy_details") or {}
        draw.text((24, 18), "DOOM  ×  EmbeddingGemma 2", font=self.title, fill=TEXT)
        draw.text(
            (26, 62),
            "ZERO-SHOT · NO TRAINING · EACH DECISION: "
            f"{d.get('crops', 19)} CROPS → 768-D EMBEDDINGS → COSINE vs TEXT PROMPTS",
            font=self.mono,
            fill=MUTED,
        )
        draw.text((RIGHT, 20), f"{self.hardware}  /  {self.dtype}", font=self.mono, fill=ACCENT)
        draw.text((RIGHT, 42), "google/embeddinggemma-2 · frozen · 740M params", font=self.small, fill=MUTED)
        draw.text((RIGHT, 60), "no game state is read by the policy", font=self.small, fill=MUTED)

        # Instruction -> behaviour by cosine.
        box = (RIGHT, 84, 1416, 214)
        self._card(draw, box, "INSTRUCTION  →  MODE", ("text", "text"))
        for i, line in enumerate(textwrap.wrap(self.instruction, width=46)[:2]):
            draw.text((RIGHT + 14, 32 + box[1] + 18 * i), line, font=self.body, fill=TEXT)
        cos = d.get("behaviour_cos") or {}
        for i, (name, value) in enumerate(cos.items()):
            y = box[1] + 78 + 16 * i
            chosen = name == d.get("behaviour")
            draw.text((RIGHT + 14, y - 2), f"{name:<10s}", font=self.small, fill=TEXT if chosen else MUTED)
            self._bar(draw, RIGHT + 104, y, 200, 9, value, 0.6, 0.95, AMBER if chosen else DIM)
            draw.text(
                (RIGHT + 314, y - 2),
                f"{value:.3f}{'  ◀' if chosen else ''}",
                font=self.small,
                fill=TEXT if chosen else MUTED,
            )

        # Monster detector: per-window margins.
        box = (RIGHT, 226, 1416, 486)
        self._card(draw, box, "MONSTER DETECTOR", ("image", "text"))
        draw.text(
            (RIGHT + 14, box[1] + 30),
            'per window: cos("a monster"…) − cos(wall·floor·sky)',
            font=self.small,
            fill=MUTED,
        )
        where = np.asarray(d.get("where") or [0.0])
        chart = (RIGHT + 14, box[1] + 50, 1402, box[1] + 150)
        zero_y = (chart[1] + chart[3]) / 2
        span = max(0.08, float(np.abs(where).max()))
        draw.line((chart[0], zero_y, chart[2], zero_y), fill=EDGE, width=1)
        draw.text((chart[0], chart[1] - 2), f"+{span:.2f}", font=self.small, fill=MUTED)
        draw.text((chart[0], chart[3] - 12), f"−{span:.2f}", font=self.small, fill=MUTED)
        n = len(where)
        best = int(where.argmax())
        step = (chart[2] - chart[0] - 44) / n
        for i, value in enumerate(where):
            x0 = chart[0] + 44 + i * step
            h = (value / span) * (chart[3] - chart[1]) / 2
            colour = (AMBER if i == best else ACCENT) if value > 0 else DIM
            draw.rectangle((x0 + 2, min(zero_y, zero_y - h), x0 + step - 2, max(zero_y, zero_y - h)), fill=colour)
        draw.text((chart[0] + 44, chart[3] + 4), "left", font=self.small, fill=MUTED)
        draw.text((chart[2], chart[3] + 4), "right", font=self.small, fill=MUTED, anchor="ra")
        monster_cos, scene_cos = d.get("monster_cos") or [0.0], d.get("scene_cos") or [0.0]
        if d.get("bearing") is not None:
            verdict = f"VISIBLE  bearing {d['bearing']:+.2f}" + (
                "  ·  ON TARGET" if abs(d["bearing"]) <= AIM_WINDOW else ""
            )
            colour = GREEN
        else:
            verdict, colour = "NO MONSTER  (every margin < 0)", MUTED
        draw.text((RIGHT + 14, box[1] + 170), verdict, font=self.h2, fill=colour)
        draw.text(
            (RIGHT + 14, box[1] + 190),
            f"best window: monster {monster_cos[best]:.3f}  vs  scene {scene_cos[best]:.3f}  →  {where[best]:+.3f}",
            font=self.small,
            fill=TEXT,
        )
        if d.get("behaviour") == "evasive":
            # Evade's own contrasts on the same crops.
            danger = d.get("danger")
            if danger is None:
                line = "danger: no threat in view"
            else:
                line = f"danger: up close {d['close_cos']:.3f} vs far {d['far_cos']:.3f}  →  {danger:+.3f} " + (
                    "CLOSE" if danger > 0 else "FAR"
                )
            draw.text((RIGHT + 14, box[1] + 210), line, font=self.small, fill=RED if danger and danger > 0 else TEXT)
            draw.text(
                (RIGHT + 14, box[1] + 228),
                f"escape: cos(open floor) − cos(monster), best {d.get('escape_bearing', 0):+.2f}",
                font=self.small,
                fill=GREEN,
            )

        # HUD readings: crop thumbnails and their best prompts.
        box = (RIGHT, 498, 1416, 708)
        self._card(draw, box, "STATUS BAR READING", ("image", "text"))
        variables = stats.get("variables", {})
        rows = (
            ("health", HEALTH_BOX, d.get("health_top") or [], variables.get("HEALTH")),
            ("ammo", AMMO_BOX, d.get("ammo_top") or [], variables.get("SELECTED_WEAPON_AMMO")),
        )
        for r, (_name, crop_box, top, truth) in enumerate(rows):
            y = box[1] + 34 + r * 86
            if self.last_screen is not None:
                thumb = Image.fromarray(self.last_screen).crop(crop_box)
                thumb = thumb.resize((int(thumb.width * 0.95), int(thumb.height * 0.95)), Image.Resampling.NEAREST)
                canvas.paste(thumb, (RIGHT + 14, y))
            for k, (prompt, value) in enumerate(top[:3]):
                draw.text(
                    (RIGHT + 140, y + 4 + 22 * k),
                    f"{prompt:<12s} {value:.3f}",
                    font=self.mono,
                    fill=TEXT if k == 0 else MUTED,
                )
            if truth is not None:
                draw.text((1402, y + 4), f"game: {truth:.0f}", font=self.small, fill=MUTED, anchor="ra")

        # Decision.
        box = (RIGHT, 720, 1416, 820)
        self._card(draw, box, "BUTTONS  (fixed rules on the readings)")
        pressed = set(stats.get("pressed") or [])
        x = RIGHT + 14
        for name in BUTTONS:
            label = {"MOVE_FORWARD": "FWD", "MOVE_BACKWARD": "BACK", "TURN_LEFT": "TURN L", "TURN_RIGHT": "TURN R"}.get(
                name, name.replace("MOVE_", "")
            )
            w = draw.textlength(label, font=self.small) + 14
            on = name in pressed
            draw.rounded_rectangle((x, box[1] + 34, x + w, box[1] + 54), radius=5, fill=ACCENT if on else "#262a31")
            draw.text((x + 7, box[1] + 38), label, font=self.small, fill=BG if on else MUTED)
            x += w + 6
        reasons = d.get("reasons") or []
        if reasons:
            draw.text((RIGHT + 14, box[1] + 66), textwrap.shorten(reasons[-1], 58), font=self.small, fill=TEXT)

        # Pipeline footer.
        nodes = ("19 CROPS OF THE FRAME", "EmbeddingGemma 2 · 768-D", "COSINE vs TEXT PROMPTS", "RULES → BUTTONS")
        x = 24
        for i, label in enumerate(nodes):
            w = draw.textlength(label, font=self.small) + 24
            draw.rounded_rectangle((x, 846, x + w, 876), radius=6, fill=PANEL, outline=EDGE)
            draw.text((x + 12, 855), label, font=self.small, fill=ACCENT)
            x += w
            if i < len(nodes) - 1:
                draw.text((x + 6, 853), ">", font=self.mono, fill=AMBER)
                x += 22
        return canvas

    # Per-frame layers ----------------------------------------------------------------
    def _game(self, canvas: Image.Image, stats: dict) -> None:
        game = Image.fromarray(self.last_screen).resize((960, 720), Image.Resampling.NEAREST)
        canvas.paste(game, (GAME_X, GAME_Y))
        draw = ImageDraw.Draw(canvas, "RGBA")
        d = stats.get("policy_details") or {}
        where = np.asarray(d.get("where") or [])
        top, bottom = game_xy(0, BAND_TOP)[1], game_xy(0, BAND_BOTTOM)[1]
        draw.rectangle((GAME_X, top, GAME_X + 960, bottom), outline=(115, 202, 255, 140), width=1)
        self._eye(draw, GAME_X + 8, top + 4, (243, 189, 114, 255))
        draw.text((GAME_X + 32, top + 5), "17 WINDOW CROPS", font=self.small, fill=(243, 189, 114, 255))
        if len(where):
            # Heat strip: one cell per window centre, coloured by its cosine margin.
            span = max(0.08, float(np.abs(where).max()))
            cell = (self.window_centres[1] - self.window_centres[0]) * SCALE
            for centre, value in zip(self.window_centres, where, strict=True):
                x = game_xy(centre, 0)[0]
                alpha = int(60 + 160 * min(1.0, abs(value) / span))
                colour = (115, 202, 255, alpha) if value > 0 else (70, 76, 86, 120)
                draw.rectangle((x - cell / 2 + 1, bottom + 2, x + cell / 2 - 1, bottom + 14), fill=colour)
            best = int(where.argmax())
            if where[best] > 0:
                left = game_xy(self.window_centres[best] - self.window_width / 2, 0)[0]
                draw.rectangle(
                    (left, top, left + self.window_width * SCALE, bottom), outline=(243, 189, 114, 230), width=3
                )
        # Aim window around the crosshair.
        for side in (-1, 1):
            x = game_xy(320 + side * AIM_WINDOW * 320, 0)[0]
            draw.line((x, top - 8, x, top + 8), fill=(237, 240, 245, 200), width=2)
        if d.get("bearing") is not None:
            x = game_xy(320 + d["bearing"] * 320, 0)[0]
            draw.polygon(((x - 9, top - 16), (x + 9, top - 16), (x, top - 3)), fill=(243, 189, 114, 255))
            draw.line((x, top, x, bottom), fill=(243, 189, 114, 200), width=2)
        if d.get("behaviour") == "evasive" and d.get("escape_bearing") is not None:
            x = game_xy(320 + d["escape_bearing"] * 320, 0)[0]
            draw.polygon(((x - 10, bottom + 30), (x + 10, bottom + 30), (x, bottom + 16)), fill=(127, 220, 154, 255))
            draw.text((x, bottom + 34), "ESCAPE", font=self.small, fill=(127, 220, 154, 255), anchor="ma")
        for crop_box, label in ((HEALTH_BOX, "HEALTH crop"), (AMMO_BOX, "AMMO crop")):
            x0, y0 = game_xy(crop_box[0], crop_box[1])
            x1, y1 = game_xy(crop_box[2], crop_box[3])
            draw.rectangle((x0, y0, x1, y1), outline=(243, 189, 114, 220), width=2)
            self._eye(draw, x0 + 2, y0 - 17, (243, 189, 114, 255))
            draw.text((x0 + 24, y0 - 15), label, font=self.small, fill=(243, 189, 114, 255))

        # Clocks and score under the game.
        clock = "GAME-TIME PLAYBACK / 35 FPS" if self.video_clock == "game" else "WALL-CLOCK PLAYBACK"
        draw.text((24, 828), clock, font=self.small, fill=ACCENT)
        ms = stats.get("decision_ms")
        info = (
            f"GAME {stats.get('second', 0):5.1f}s   KILLS {stats.get('total_kills', 0)}   EPISODE {stats.get('episode', 1)}"
            + (f"   DECISION {ms:.0f} ms" if ms else "")
        )
        draw.text((984, 828), info, font=self.small, fill=MUTED, anchor="ra")

    def render(self, screen, stats: dict):
        if screen is not None:
            self.last_screen = screen
        if self.last_screen is None:
            raise ValueError("Overlay requires an initial game screen")
        decision = stats.get("decision_count")
        if self.panel is None or decision != self.panel_decision:
            self.panel = self._panel(stats)
            self.panel_decision = decision
        canvas = self.panel.copy()
        self._game(canvas, stats)
        return np.asarray(canvas)
