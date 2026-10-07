"""Zero-shot Doom control with EmbeddingGemma 2: cosine similarity to text prompts.

Nothing is trained. Every decision cuts the 640x480 frame into crops, embeds
them all in one batch, and compares each with fixed text prompts. Each side of a
contrast is the normalized mean of its prompt embeddings, and every decision is
either a sign (margin > 0) or an argmax across crops; no threshold is fitted.

- Presence: the band where monsters stand, above the player's pistol, is cut
  into left / front / right thirds. An enemy is visible when some third is closer to the monster prompts
  than to their negations ("a monster" vs "not a monster").
- Bearing: overlapping windows slide across the same band. The window that most
  prefers the monster prompts over the scene prompts (walls, floor, sky) gives
  the enemy's horizontal position; a parabola through that window and its two
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
THIRDS = {"left": (0, 213), "front": (213, 427), "right": (427, 640)}
WINDOW_WIDTH, WINDOW_STRIDE = 128, 32
AIM_WINDOW = 0.12  # |bearing| the scripted teacher treats as lined up (~38 px)

MONSTER_PROMPTS = ("a monster", "a pink demon", "a zombie soldier")
NEGATION_PROMPTS = ("not a monster", "no monster", "no enemies")
SCENE_PROMPTS = ("an empty brown stone wall", "a grey tiled floor", "a dark night sky")
HEALTH_LEVELS = tuple(range(0, 101, 5))
AMMO_LEVELS = tuple(range(0, 51))
BEHAVIOURS = {
    "aggressive": "attack: shoot every enemy",
    "cautious": "fight, but retreat and stop shooting when health is low",
    "evasive": "pacifist: never shoot, run away from enemies",
}
RETREAT_HEALTH = {"aggressive": 20, "cautious": 40, "evasive": 101}


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


def crops(screen: np.ndarray | Image.Image, width: int = WINDOW_WIDTH) -> dict:
    frame = screen if isinstance(screen, Image.Image) else Image.fromarray(screen)
    frame = frame.convert("RGB")
    if frame.size != (640, 480):
        raise ValueError(f"Expected a 640x480 frame, got {frame.size}")
    return {
        "thirds": [frame.crop((a, BAND_TOP, z, BAND_BOTTOM)) for a, z in THIRDS.values()],
        "windows": [frame.crop((s, BAND_TOP, s + width, BAND_BOTTOM)) for s in window_starts(width)],
        "health": frame.crop(HEALTH_BOX),
        "ammo": frame.crop(AMMO_BOX),
    }


class Perception:
    """Embeds every crop on each call; prompt embeddings are fixed class labels."""

    def __init__(self, model, window_width: int = WINDOW_WIDTH):
        self.model = model
        self.window_width = window_width
        starts = np.array(window_starts(window_width))
        self.window_bearings = (starts + window_width / 2 - 320) / 320
        monster = self.prototype(MONSTER_PROMPTS)
        self.present_axis = monster - self.prototype(NEGATION_PROMPTS)
        self.where_axis = monster - self.prototype(SCENE_PROMPTS)
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
        parts = crops(screen, self.window_width)
        images = [*parts["thirds"], *parts["windows"], parts["health"], parts["ammo"]]
        emb = self.model.encode(images, convert_to_tensor=True, batch_size=len(images)).float()
        n_windows = len(parts["windows"])
        thirds, windows = emb[:3], emb[3 : 3 + n_windows]
        health, ammo = emb[3 + n_windows], emb[4 + n_windows]
        # Embeddings are unit length, so each dot product is a cosine difference.
        present = (thirds @ self.present_axis).cpu().numpy()
        where = (windows @ self.where_axis).cpu().numpy()
        visible = bool(present.max() > 0)
        return {
            "present": present,
            "where": where,
            "visible": visible,
            "bearing": self.peak(where) if visible else None,
            "health": HEALTH_LEVELS[int((health @ self.health.T).argmax())],
            "ammo": AMMO_LEVELS[int((ammo @ self.ammo.T).argmax())],
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


def control(seen: dict, behaviour: str) -> tuple[set[str], list[str]]:
    """Teacher-shaped rules over perceived values only; no game state is read."""
    if seen["bearing"] is None:
        return {"TURN_RIGHT"}, ["No monster in any third: turn to search."]
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
        reasons.append("Back off and strafe: " + ("evasive mode." if behaviour == "evasive" else "hurt or no ammo."))
    if behaviour != "evasive" and has_ammo and aligned and not hurt:
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
    ):
        self.device = select_device(device)
        self.dtype = default_dtype(self.device) if dtype == "auto" else dtype
        self.vision_tokens = vision_tokens
        self.perception = Perception(load_encoder(self.device, self.dtype, vision_tokens), window_width)
        self.behaviours: dict[str, str] = {}
        self.last_state_text = None
        self.last_token_count = None
        self.last_details = None

    def __call__(self, instruction: str, record: dict, *, screen: np.ndarray) -> dict[str, float]:
        # The instruction is constant for a run; it is embedded once, not per decision.
        if instruction not in self.behaviours:
            self.behaviours[instruction] = self.perception.behaviour(instruction)[0]
        behaviour = self.behaviours[instruction]
        seen = self.perception(screen)
        pressed, reasons = control(seen, behaviour)
        target = "none" if seen["bearing"] is None else f"{seen['bearing']:+.2f}"
        self.last_state_text = f"mode={behaviour} monster_bearing={target} health~{seen['health']} ammo~{seen['ammo']}"
        self.last_details = {
            "behaviour": behaviour,
            "bearing": seen["bearing"],
            "health": seen["health"],
            "ammo": seen["ammo"],
            "present": np.round(seen["present"], 4).tolist(),
            "where": np.round(seen["where"], 4).tolist(),
            "crops": seen["crops"],
            "reasons": reasons,
        }
        return {button: 0.98 if button in pressed else 0.02 for button in BUTTONS}

    def synchronize(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
