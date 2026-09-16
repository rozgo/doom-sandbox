"""ModernBERT inference on Apple Metal, CUDA, or CPU."""

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from doom_bert.policy import BUTTONS, serialize_observation

BASE_MODEL = "answerdotai/ModernBERT-base"


def select_device(requested: str = "auto") -> str:
    available = {
        "cpu": True,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }
    if requested == "auto":
        return next(device for device in ("mps", "cuda", "cpu") if available[device])
    if requested not in available or not available[requested]:
        raise ValueError(f"Requested device {requested!r} is unavailable")
    return requested


class ModernBertPolicy:
    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        allow_untrained_head: bool = False,
    ):
        self.device = select_device(device)
        self.dtype = (
            ("float32" if self.device == "cpu" else "float16")
            if dtype == "auto"
            else dtype
        )
        config = AutoConfig.from_pretrained(checkpoint)
        if config.model_type != "modernbert":
            raise ValueError("Expected a ModernBERT checkpoint")
        self.untrained_head = allow_untrained_head
        if allow_untrained_head:
            config.num_labels = len(BUTTONS)
            config.id2label = dict(enumerate(BUTTONS))
            config.label2id = {name: index for index, name in enumerate(BUTTONS)}
        elif tuple(
            config.id2label.get(i) for i in range(len(BUTTONS))
        ) != BUTTONS or config.num_labels != len(BUTTONS):
            raise ValueError(
                "Checkpoint must contain a fine-tuned seven-button action head with "
                f"id2label in this order: {BUTTONS}. The base model is not a trained policy."
            )
        config.problem_type = "multi_label_classification"
        # Older Transformers releases auto-compiled CUDA-specific paths.
        if hasattr(config, "reference_compile"):
            config.reference_compile = False
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model, loading = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            config=config,
            dtype=getattr(torch, self.dtype),
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        if not allow_untrained_head and (
            loading.get("missing_keys") or loading.get("mismatched_keys")
        ):
            raise ValueError(
                "Checkpoint has missing or mismatched weights; refusing an incomplete policy"
            )
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, instruction: str, observation: dict) -> dict[str, float]:
        encoded = self.tokenizer(
            instruction,
            serialize_observation(observation),
            return_tensors="pt",
            truncation=True,
            max_length=256,
            return_token_type_ids=False,
        ).to(self.device)
        self.last_token_count = int(encoded["input_ids"].shape[-1])
        # Copying results back synchronizes GPU work, so loop timing is real latency.
        probabilities = self.model(**encoded).logits[0].float().sigmoid().cpu().tolist()
        return dict(zip(BUTTONS, probabilities, strict=True))

    def synchronize(self) -> None:
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()
