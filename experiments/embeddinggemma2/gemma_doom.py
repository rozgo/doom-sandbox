"""Zero-shot Doom control with EmbeddingGemma 2: cosine similarity to text prompts.

Nothing is trained. Every decision cuts the 640x480 frame into crops, embeds
them all in one batch, and compares each with fixed text prompts. Each side of a
contrast is the normalized mean of its prompt embeddings, and every decision is
either a sign (margin > 0) or an argmax across crops; no threshold is fitted.

- Monsters: overlapping windows slide across the band where monsters stand.
  Each window's margin is cos(monster prompts) - cos(scene prompts: wall, floor,
  sky). An enemy is visible when some margin is positive. The best window gives
  its horizontal position; a parabola through that window and its two
  neighbours refines the peak between window centres, fine enough to aim.
- Status: the HEALTH and AMMO boxes of the status bar are matched against
  "HEALTH 40%" and "AMMO 12" prompts.
- Instruction: matched against three behaviour descriptions.

A fixed rule controller with the same shape as the scripted teacher turns those
readings into buttons. It never reads game state.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from doom_bert.policy import BUTTONS

MODEL = "google/embeddinggemma-2"
STATUS_BAR_Y = 403  # 640x480 render: Doom's 32-row status bar scaled by 2.4
# Band where monsters appear (logged boxes: tops ~191, bottoms ~233 px) and above the
# pistol, which bobs up to y~249 while firing or moving. It also drops the sky.
BAND_TOP, BAND_BOTTOM = 120, 245
HEALTH_BOX = (92, STATUS_BAR_Y, 210, 480)
AMMO_BOX = (0, STATUS_BAR_Y, 92, 480)
WINDOW_WIDTH, WINDOW_STRIDE = 128, 32
AIM_WINDOW = 0.12  # |bearing| the scripted teacher treats as lined up (~38 px)

MONSTER_PROMPTS = ("a monster", "a pink demon", "a zombie soldier")
SCENE_PROMPTS = ("an empty brown stone wall", "a grey tiled floor", "a dark night sky")
# Evade is its own policy with its own contrasts on the same window crops.
CLOSE_PROMPTS = (
    "a monster right in front of you, up close",
    "a big monster attacking you",
    "a monster filling the view",
)
FAR_PROMPTS = ("a small monster far away", "a distant monster across the room", "a tiny figure in the distance")
OPEN_PROMPTS = ("an empty open floor", "a clear path with nothing in the way", "open empty space")
HEALTH_LEVELS = tuple(range(0, 101, 5))
AMMO_LEVELS = tuple(range(0, 51))
BEHAVIOURS = {
    "aggressive": "attack: shoot every enemy",
    "cautious": "fight, but retreat and stop shooting when health is low",
    "evasive": "pacifist: never shoot, run away from enemies",
}
RETREAT_HEALTH = {"aggressive": 20, "cautious": 40}


def window_starts(width: int = WINDOW_WIDTH, stride: int = WINDOW_STRIDE) -> list[int]:
    starts = list(range(0, 640 - width + 1, stride))
    return starts if starts[-1] == 640 - width else [*starts, 640 - width]


def select_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def default_dtype(device: str) -> str:
    # The model card forbids float16: activations overflow it.
    return "bfloat16" if device == "cuda" else "float32"


def load_encoder(device: str, dtype: str, vision_tokens: int = 140):
    from sentence_transformers import SentenceTransformer

    # Audio is never used; skipping its 300M-parameter encoder saves memory.
    model = SentenceTransformer(
        MODEL,
        device=device,
        model_kwargs={"dtype": getattr(torch, dtype)},
        config_kwargs={"audio_config": None},
    )
    model[0].processor.image_processor.max_soft_tokens = vision_tokens
    return model


def crops(screen: np.ndarray | Image.Image, width: int = WINDOW_WIDTH, stride: int = WINDOW_STRIDE) -> dict:
    frame = screen if isinstance(screen, Image.Image) else Image.fromarray(screen)
    frame = frame.convert("RGB")
    if frame.size != (640, 480):
        raise ValueError(f"Expected a 640x480 frame, got {frame.size}")
    return {
        "windows": [frame.crop((s, BAND_TOP, s + width, BAND_BOTTOM)) for s in window_starts(width, stride)],
        "health": frame.crop(HEALTH_BOX),
        "ammo": frame.crop(AMMO_BOX),
    }


class Perception:
    """Embeds every crop on each call; prompt embeddings are fixed class labels."""

    def __init__(self, model, window_width: int = WINDOW_WIDTH, window_stride: int = WINDOW_STRIDE):
        self.model = model
        self.window_width, self.window_stride = window_width, window_stride
        starts = np.array(window_starts(window_width, window_stride))
        self.window_bearings = (starts + window_width / 2 - 320) / 320
        self.monster_prototype = self.prototype(MONSTER_PROMPTS)
        self.scene_prototype = self.prototype(SCENE_PROMPTS)
        self.close_prototype = self.prototype(CLOSE_PROMPTS)
        self.far_prototype = self.prototype(FAR_PROMPTS)
        self.open_prototype = self.prototype(OPEN_PROMPTS)
        self.health = self.text(f"HEALTH {n}%" for n in HEALTH_LEVELS)
        self.ammo = self.text(f"AMMO {n}" for n in AMMO_LEVELS)
        self.behaviour_names = list(BEHAVIOURS)
        self.behaviours = self.text(BEHAVIOURS.values())

    def text(self, prompts) -> torch.Tensor:
        return self.model.encode(list(prompts), prompt_name="SearchQuery", convert_to_tensor=True).float()

    def prototype(self, prompts) -> torch.Tensor:
        mean = self.text(prompts).mean(0)
        return mean / mean.norm()

    @torch.inference_mode()
    def __call__(self, screen) -> dict:
        parts = crops(screen, self.window_width, self.window_stride)
        images = [*parts["windows"], parts["health"], parts["ammo"]]
        emb = self.model.encode(images, convert_to_tensor=True, batch_size=len(images)).float()
        windows, health, ammo = emb[:-2], emb[-2], emb[-1]
        # Embeddings are unit length, so every dot product is a cosine similarity.
        monster = (windows @ self.monster_prototype).cpu().numpy()
        scene = (windows @ self.scene_prototype).cpu().numpy()
        where = monster - scene
        visible = bool(where.max() > 0)
        best = int(where.argmax())
        # Evade contrasts: is the nearest-looking threat close, and which way is open?
        close = float(windows[best] @ self.close_prototype)
        far = float(windows[best] @ self.far_prototype)
        escape = (windows @ self.open_prototype).cpu().numpy() - monster
        health_cos = (health @ self.health.T).cpu().numpy()
        ammo_cos = (ammo @ self.ammo.T).cpu().numpy()
        return {
            "where": where,
            "monster_cos": monster,
            "scene_cos": scene,
            "visible": visible,
            "bearing": self.peak(where) if visible else None,
            "close_cos": close,
            "far_cos": far,
            "danger": close - far if visible else None,
            "escape": escape,
            "escape_bearing": self.peak(escape),
            "health": HEALTH_LEVELS[int(health_cos.argmax())],
            "ammo": AMMO_LEVELS[int(ammo_cos.argmax())],
            "health_top": top_matches(health_cos, [f"HEALTH {n}%" for n in HEALTH_LEVELS]),
            "ammo_top": top_matches(ammo_cos, [f"AMMO {n}" for n in AMMO_LEVELS]),
            "crops": len(images),
        }

    def peak(self, where: np.ndarray) -> float:
        k = int(where.argmax())
        bearing = float(self.window_bearings[k])
        if 0 < k < len(where) - 1:
            left, centre, right = where[k - 1], where[k], where[k + 1]
            curvature = left - 2 * centre + right
            if curvature < 0:
                step = self.window_bearings[1] - self.window_bearings[0]
                bearing += float(0.5 * (left - right) / curvature * step)
        return bearing

    @torch.inference_mode()
    def behaviour(self, instruction: str) -> tuple[str, np.ndarray]:
        emb = self.model.encode(instruction, prompt_name="SearchQuery", convert_to_tensor=True).float()
        sims = (self.behaviours @ emb).cpu().numpy()
        return self.behaviour_names[int(sims.argmax())], sims


def top_matches(cosines: np.ndarray, prompts: list[str], k: int = 3) -> list[tuple[str, float]]:
    return [(prompts[i], round(float(cosines[i]), 4)) for i in np.argsort(-cosines)[:k]]


def control(seen: dict, behaviour: str) -> tuple[set[str], list[str]]:
    """Fixed rules over perceived values only; no game state is read."""
    return evade(seen) if behaviour == "evasive" else fight(seen, behaviour)


def evade(seen: dict) -> tuple[set[str], list[str]]:
    """Never fire. Back away from close threats, head for open space, keep moving."""
    if seen["bearing"] is None:
        return {"TURN_RIGHT", "MOVE_BACKWARD", "MOVE_LEFT"}, ["Nothing in view: circle backwards and scan."]
    bearing, escape = seen["bearing"], seen["escape_bearing"]
    away = "MOVE_LEFT" if bearing >= 0 else "MOVE_RIGHT"
    if seen["danger"] > 0:
        pressed = {"MOVE_BACKWARD", away}
        if abs(bearing) > AIM_WINDOW:
            pressed.add("TURN_RIGHT" if bearing > 0 else "TURN_LEFT")
        return pressed, [f"Threat up close at {bearing:+.2f}: face it, back off, strafe away."]
    pressed = {away, "MOVE_FORWARD" if abs(escape) <= AIM_WINDOW else ("TURN_RIGHT" if escape > 0 else "TURN_LEFT")}
    return pressed, [f"Threat far at {bearing:+.2f}: head for open space at {escape:+.2f}."]


def fight(seen: dict, behaviour: str) -> tuple[set[str], list[str]]:
    """Teacher-shaped attack rules; cautious also retreats below 40 health."""
    if seen["bearing"] is None:
        return {"TURN_RIGHT"}, ["No window looks more like a monster than the scene: turn to search."]
    pressed, reasons = set(), []
    bearing = seen["bearing"]
    aligned = abs(bearing) <= AIM_WINDOW
    if not aligned:
        pressed.add("TURN_RIGHT" if bearing > 0 else "TURN_LEFT")
        reasons.append(f"Monster at bearing {bearing:+.2f}: turn toward it.")
    has_ammo = seen["ammo"] > 0
    hurt = seen["health"] < RETREAT_HEALTH[behaviour]
    if hurt or not has_ammo:
        pressed.add("MOVE_BACKWARD")
        pressed.add("MOVE_LEFT" if bearing >= 0 else "MOVE_RIGHT")
        reasons.append("Back off and strafe: hurt or out of ammo.")
    if has_ammo and aligned and not hurt:
        pressed.add("ATTACK")
        reasons.append("Monster under the crosshair: fire.")
    return pressed, reasons


class EmbeddingGemmaPolicy:
    """Pixels + instruction -> buttons; every call embeds all crops in one batch."""

    uses_screen = True
    display_name = "EmbeddingGemma 2"
    mode_label = "ZERO-SHOT / COSINE TO TEXT PROMPTS"
    pipeline_labels = ("VIEW + HUD CROPS", "EmbeddingGemma 2", "COSINE VS PROMPTS", "VALID BUTTON VECTOR")
    untrained_head = False
    training_scope = None

    def __init__(
        self,
        *,
        device: str = "auto",
        dtype: str = "auto",
        vision_tokens: int = 140,
        window_width: int = WINDOW_WIDTH,
        window_stride: int = WINDOW_STRIDE,
    ):
        self.device = select_device(device)
        self.dtype = default_dtype(self.device) if dtype == "auto" else dtype
        self.vision_tokens = vision_tokens
        self.perception = Perception(load_encoder(self.device, self.dtype, vision_tokens), window_width, window_stride)
        self.behaviours: dict[str, tuple[str, dict[str, float]]] = {}
        self.last_state_text = None
        self.last_token_count = None
        self.last_details = None

    def __call__(self, instruction: str, record: dict, *, screen: np.ndarray) -> dict[str, float]:
        # The instruction is constant for a run; it is embedded once, not per decision.
        if instruction not in self.behaviours:
            name, sims = self.perception.behaviour(instruction)
            self.behaviours[instruction] = (
                name,
                dict(zip(self.perception.behaviour_names, sims.round(4).tolist(), strict=True)),
            )
        behaviour, behaviour_cos = self.behaviours[instruction]
        seen = self.perception(screen)
        pressed, reasons = control(seen, behaviour)
        target = "none" if seen["bearing"] is None else f"{seen['bearing']:+.2f}"
        self.last_state_text = f"mode={behaviour} monster_bearing={target} health~{seen['health']} ammo~{seen['ammo']}"
        self.last_details = {
            "behaviour": behaviour,
            "bearing": seen["bearing"],
            "health": seen["health"],
            "ammo": seen["ammo"],
            "behaviour_cos": behaviour_cos,
            "where": np.round(seen["where"], 4).tolist(),
            "monster_cos": np.round(seen["monster_cos"], 4).tolist(),
            "scene_cos": np.round(seen["scene_cos"], 4).tolist(),
            "danger": None if seen["danger"] is None else round(seen["danger"], 4),
            "close_cos": round(seen["close_cos"], 4),
            "far_cos": round(seen["far_cos"], 4),
            "escape_bearing": round(seen["escape_bearing"], 4),
            "health_top": seen["health_top"],
            "ammo_top": seen["ammo_top"],
            "crops": seen["crops"],
            "reasons": reasons,
        }
        return {button: 0.98 if button in pressed else 0.02 for button in BUTTONS}

    def make_overlay(self, *, instruction: str, video_clock: str):
        from gemma_overlay import GemmaOverlay

        return GemmaOverlay(
            instruction=instruction,
            device=self.device,
            dtype=self.dtype,
            window_bearings=self.perception.window_bearings,
            window_width=self.perception.window_width,
            video_clock=video_clock,
        )

    def synchronize(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
