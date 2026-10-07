"""GLiNER2.5 button scores with optional adapted head or fine-tuned weights; no gameplay rules."""

import json
from pathlib import Path

import torch
from gliner2.classification import (
    ClassificationConfig,
    ClassificationSchema,
    Classifier,
)
from huggingface_hub import snapshot_download

from doom_bert.policy import BUTTONS, serialize_categorical_observation

ACTION_DESCRIPTIONS = {
    "ATTACK": "Fire the equipped weapon at a target in front of the player.",
    "MOVE_LEFT": "Strafe sideways to the player's left without changing aim.",
    "MOVE_RIGHT": "Strafe sideways to the player's right without changing aim.",
    "MOVE_FORWARD": "Walk forward in the direction the player is facing.",
    "MOVE_BACKWARD": "Walk backward, away from what the player is facing.",
    "TURN_LEFT": "Rotate the player's view and aim toward the left.",
    "TURN_RIGHT": "Rotate the player's view and aim toward the right.",
}
TASK_INSTRUCTION = (
    "Choose the game-control buttons to press now, given the player's instruction "
    "and current game state. Select actions that carry out the instruction. "
    "Several buttons can be pressed together, or none."
)
MODEL_REVISIONS = {
    "fastino/gliner2.5-small-v1": "f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e",
    "fastino/gliner2.5-base-v1": "78cea040597df251eedefa9d7ee2a756af39fe64",
}


def describe_state(observation: dict) -> str:
    fields = dict(
        item.split("=", 1)
        for item in serialize_categorical_observation(observation).split()
    )
    variables = observation["variables"]
    text = (
        f"Health: {variables['HEALTH']:.0f} out of 100, {fields['health']}. "
        f"Ammunition: {variables['SELECTED_WEAPON_AMMO']:.0f}. "
    )
    if fields["nearest_enemy"] == "none":
        return text + "No enemy is visible."
    side = {
        "left": "to the left of the player's aim",
        "center": "directly ahead, lined up with the player's aim",
        "right": "to the right of the player's aim",
    }[fields["nearest_enemy"]]
    return text + f"The nearest visible enemy is {side}, at {fields['range']} range."


def input_text(instruction: str, state_text: str) -> str:
    """The exact text scored at play time; training uses the same function."""
    return f"Player instruction: {instruction}\nGame state: {state_text}"


class GLiNERPolicy:
    name = "GLiNER2.5"
    state_format = "natural-language-v1"
    training_scope = "pretrained GLiNER2.5; zero Doom-specific training"
    untrained_head = False
    pipeline_labels = (
        "STATE TEXT + INSTRUCTION",
        "GLiNER2.5 ENCODER",
        "7 LABEL SCORES",
        "VALID BUTTON VECTOR",
    )

    def __init__(
        self, checkpoint: str, *, device="mps", dtype="float16", head=None, weights=None
    ):
        self.checkpoint, self.device, self.dtype = checkpoint, device, dtype
        self.revision = MODEL_REVISIONS[checkpoint]
        snapshot = snapshot_download(
            checkpoint,
            revision=self.revision,
            allow_patterns=["*.json", "*.safetensors", "*.model", "encoder_config/*"],
        )
        self.classifier = (
            Classifier.from_pretrained(snapshot)
            .to(device=device, dtype=getattr(torch, dtype))
            .eval()
        )
        self.schema = self.classifier.compile_schema(
            ClassificationSchema().multi(
                "buttons",
                ACTION_DESCRIPTIONS,
                instruction=TASK_INSTRUCTION,
                activation="sigmoid",
            )
        )
        self.config = ClassificationConfig(max_len=512, batch_size=1)
        self.doom_training_examples = 0
        if head is not None:
            from safetensors.torch import load_file

            head = Path(head)
            metadata = json.loads((head / "metadata.json").read_text())
            if (
                metadata["base_model"] != checkpoint
                or metadata["revision"] != self.revision
            ):
                raise ValueError(
                    "Adapted head does not match the pinned base checkpoint"
                )
            if metadata["schema"] != self.schema.build():
                raise ValueError("Adapted head does not match the action schema")
            self.classifier.model.classifier.load_state_dict(
                load_file(head / "head.safetensors"), strict=True
            )
            self.doom_training_examples = metadata["training_examples"]
            self.training_scope = (
                f"frozen GLiNER2.5 encoder; classifier head adapted on "
                f"{self.doom_training_examples} Doom examples"
            )
        if weights is not None:
            from safetensors.torch import load_file

            weights = Path(weights)
            metadata = json.loads((weights / "metadata.json").read_text())
            if (
                metadata["base_model"] != checkpoint
                or metadata["revision"] != self.revision
            ):
                raise ValueError(
                    "Fine-tuned weights do not match the pinned base checkpoint"
                )
            if metadata["schema"] != self.schema.build():
                raise ValueError("Fine-tuned weights do not match the action schema")
            state = load_file(weights / "model.safetensors")
            target = (
                self.classifier.model
                if metadata["scope"] == "full"
                else self.classifier.model.classifier
            )
            target.load_state_dict(
                {k: v.to(torch.float32) for k, v in state.items()}, strict=True
            )
            self.classifier.to(device=device, dtype=getattr(torch, dtype))
            self.doom_training_examples = metadata["training_examples"]
            self.training_scope = (
                f"GLiNER2.5 {'fully fine-tuned' if metadata['scope'] == 'full' else 'head adapted, encoder frozen'} "
                f"on {self.doom_training_examples:,} teacher-labelled Doom examples"
            )
        size = checkpoint.split("gliner2.5-")[1].split("-")[0].capitalize()
        self.display_name = f"GLiNER2.5 {size}"
        self.mode_label = (
            "ZERO-SHOT / NO DOOM TRAINING"
            if not self.doom_training_examples
            else f"TRAINED ON {self.doom_training_examples:,} TEACHER EXAMPLES"
        )
        self.encoder_forward_calls = 0
        self._hook = self.classifier.model.encoder.register_forward_pre_hook(
            self._count_forward, with_kwargs=True
        )

    def _count_forward(self, module, args, kwargs):
        self.encoder_forward_calls += 1
        self.last_token_count = int(kwargs["input_ids"].shape[-1])

    @torch.inference_mode()
    def __call__(self, instruction: str, observation: dict) -> dict[str, float]:
        self.last_state_text = describe_state(observation)
        self.last_input_text = input_text(instruction, self.last_state_text)
        scores = self.classifier.score(
            self.last_input_text, self.schema, config=self.config
        )
        return {button: scores.probability("buttons", button) for button in BUTTONS}

    def synchronize(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()
