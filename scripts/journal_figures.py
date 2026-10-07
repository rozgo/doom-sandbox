"""Draw journal figures from committed reports (no numbers typed in by hand).

    uv run python scripts/journal_figures.py

Writes to media/journals/:
- gliner2-results.png: the first GLiNER2.5 attempt (reports/gliner2-experiment/summary.json),
  with a 16:9 padded copy, gliner2-card.png;
- gliner2-v2-results.png: the second attempt on the RTX 4090 (reports/gliner2-v2/summary.json).
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from doom_bert.overlay import font

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "media/journals"

# GLiNER journal palette (light): surface, ink, muted ink, gridline, bar (validated: chroma, contrast).
SURFACE, INK, MUTED, GRID, BAR = "#ffffff", "#1c1a16", "#67625a", "#e8e3d9", "#0b8f7a"
FIRST = {
    "modernbert": ("ModernBERT", "372 Doom examples"),
    "small-zero-shot": ("GLiNER2.5 Small", "zero-shot"),
    "base-zero-shot": ("GLiNER2.5 Base", "zero-shot"),
    "small-128": ("GLiNER2.5 Small", "128 examples, head only"),
    "base-32": ("GLiNER2.5 Base", "32 examples, head only"),
}
SECOND = {
    "modernbert": ("ModernBERT", "372 examples, head only"),
    "small-zero-shot": ("GLiNER2.5 Small", "zero-shot"),
    "base-zero-shot": ("GLiNER2.5 Base", "zero-shot"),
    "small-head": ("GLiNER2.5 Small", "12,000 examples, head only"),
    "base-head": ("GLiNER2.5 Base", "12,000 examples, head only"),
    "small-full": ("GLiNER2.5 Small", "12,000 examples, fine-tuned"),
    "base-full": ("GLiNER2.5 Base", "12,000 examples, fine-tuned"),
}


def chart(
    summary: Path,
    names: dict,
    panels: list,
    title: str,
    subtitle: str,
    footer: str,
    out: str,
) -> Path:
    policies = {p["id"]: p for p in json.loads(summary.read_text())["policies"]}
    width, label_w, top, row_h = 1800, 400, 230, 92 if len(names) <= 5 else 80
    height = top + len(names) * row_h + 90
    image = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(image)
    draw.text((60, 44), title, font=font(40, bold=True), fill=INK)
    draw.text((60, 100), subtitle, font=font(22), fill=MUTED)
    panel_w = (width - 60 - label_w - 2 * 50) / len(panels)
    for row, pid in enumerate(names):
        y = top + row * row_h
        name, detail = names[pid]
        draw.text((60, y + 6), name, font=font(25, bold=True), fill=INK)
        draw.text((60, y + 40), detail, font=font(20), fill=MUTED)
    for col, (heading, sub, value_of, limit, fmt) in enumerate(panels):
        x0 = 60 + label_w + col * (panel_w + 50)
        draw.text((x0, 168), heading, font=font(24, bold=True), fill=INK)
        draw.text((x0, 198), sub, font=font(19), fill=MUTED)
        draw.line((x0, top - 6, x0, top + len(names) * row_h - 18), fill=GRID, width=2)
        space = panel_w - 110
        for row, pid in enumerate(names):
            value = value_of(policies[pid])
            y = top + row * row_h + 14
            length = max(2.0, space * value / limit)
            # 24 px bars, square at the baseline, rounded at the data end.
            draw.rounded_rectangle((x0, y, x0 + length, y + 24), radius=4, fill=BAR)
            draw.rectangle((x0, y, x0 + min(length, 4), y + 24), fill=BAR)
            draw.text(
                (x0 + length + 12, y - 1),
                fmt(value),
                font=font(23, bold=True),
                fill=INK,
            )
    draw.text((60, height - 64), footer, font=font(18), fill=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / out
    image.save(path)
    return path


def pad_16_9(path: Path, out: str) -> Path:
    image = Image.open(path)
    card = Image.new("RGB", (image.width, round(image.width * 9 / 16)), SURFACE)
    card.paste(image, (0, (card.height - image.height) // 2))
    card.save(OUT / out)
    return OUT / out


def percent(v: float) -> str:
    return f"{v:.0%}"


def main() -> None:
    first = chart(
        ROOT / "reports/gliner2-experiment/summary.json",
        FIRST,
        [
            (
                "Headless game loop",
                "decisions per second",
                lambda p: p["headless_decisions_per_second"],
                60,
                "{:.1f}".format,
            ),
            (
                "Matches the teacher",
                "held-out states, 48 pairs",
                lambda p: p["test_decoded_teacher_agreement"],
                1,
                percent,
            ),
            (
                "Matches the teacher",
                "new phrasings, 48 pairs",
                lambda p: p["new_instruction_decoded_teacher_agreement"],
                1,
                percent,
            ),
        ],
        "Five pipelines on the same Doom observations",
        "Apple M3 Max, MPS float16, batch one. Agreement is the exact 7-button vector against the scripted teacher.",
        "Source: reports/gliner2-experiment/summary.json. Small exploratory comparison; "
        "the held-out sets have eight distinct states.",
        "gliner2-results.png",
    )
    pad_16_9(first, "gliner2-card.png")
    second = ROOT / "reports/gliner2-v2/summary.json"
    if second.exists():
        chart(
            second,
            SECOND,
            [
                (
                    "Matches the teacher",
                    "held-out states, 48 pairs",
                    lambda p: p["test_decoded_teacher_agreement"],
                    1,
                    percent,
                ),
                (
                    "Matches the teacher",
                    "new phrasings, 48 pairs",
                    lambda p: p["new_instruction_decoded_teacher_agreement"],
                    1,
                    percent,
                ),
                (
                    "Conditional probes",
                    "fire only at 40+ health, 8 cases",
                    lambda p: p["conditional_checks_passed"],
                    8,
                    lambda v: f"{v}/8",
                ),
            ],
            "Second attempt: what fine-tuning changed",
            "RTX 4090, float16, batch one. Agreement is the exact 7-button vector against the scripted teacher.",
            "Source: reports/gliner2-v2/summary.json. Training data: data/gliner2-teacher-v2.jsonl; "
            "test states and phrasings never trained on.",
            "gliner2-v2-results.png",
        )


if __name__ == "__main__":
    main()
