"""Draw journal figures from committed reports (no numbers typed in by hand).

    uv run python scripts/journal_figures.py

Writes media/journals/gliner2-results.png (three small panels comparing the GLiNER2.5
pipelines with the ModernBERT policy, read from reports/gliner2-experiment/summary.json)
and a 16:9 padded copy, gliner2-card.png, for the home page card.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from doom_bert.overlay import font

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "media/journals"

# GLiNER journal palette (light): surface, ink, muted ink, gridline, bar (validated: chroma, contrast).
SURFACE, INK, MUTED, GRID, BAR = "#ffffff", "#1c1a16", "#67625a", "#e8e3d9", "#0b8f7a"
NAMES = {
    "modernbert": ("ModernBERT", "372 Doom examples"),
    "small-zero-shot": ("GLiNER2.5 Small", "zero-shot"),
    "base-zero-shot": ("GLiNER2.5 Base", "zero-shot"),
    "small-128": ("GLiNER2.5 Small", "128 examples, head only"),
    "base-32": ("GLiNER2.5 Base", "32 examples, head only"),
}


def gliner_results() -> Path:
    summary = json.loads((ROOT / "reports/gliner2-experiment/summary.json").read_text())
    policies = {p["id"]: p for p in summary["policies"]}
    panels = [
        (
            "Headless game loop",
            "decisions per second",
            "headless_decisions_per_second",
            60,
            "{:.1f}",
        ),
        (
            "Matches the teacher",
            "held-out states, 48 pairs",
            "test_decoded_teacher_agreement",
            1,
            "{:.0%}",
        ),
        (
            "Matches the teacher",
            "new phrasings, 48 pairs",
            "new_instruction_decoded_teacher_agreement",
            1,
            "{:.0%}",
        ),
    ]
    width, height = 1800, 820
    image = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(image)
    draw.text(
        (60, 44),
        "Five pipelines on the same Doom observations",
        font=font(40, bold=True),
        fill=INK,
    )
    draw.text(
        (60, 100),
        "Apple M3 Max, MPS float16, batch one. Agreement is the exact 7-button vector against the scripted teacher.",
        font=font(22),
        fill=MUTED,
    )
    label_w, top, row_h = 380, 230, 96
    panel_w = (width - 60 - label_w - 2 * 50) / 3
    for row, pid in enumerate(NAMES):
        y = top + row * row_h
        name, detail = NAMES[pid]
        draw.text((60, y + 8), name, font=font(25, bold=True), fill=INK)
        draw.text((60, y + 42), detail, font=font(20), fill=MUTED)
    for col, (title, subtitle, key, limit, fmt) in enumerate(panels):
        x0 = 60 + label_w + col * (panel_w + 50)
        draw.text((x0, 168), title, font=font(24, bold=True), fill=INK)
        draw.text((x0, 198), subtitle, font=font(19), fill=MUTED)
        draw.line((x0, top - 6, x0, top + len(NAMES) * row_h - 18), fill=GRID, width=2)
        bar_space = panel_w - 110
        for row, pid in enumerate(NAMES):
            value = policies[pid][key]
            y = top + row * row_h + 16
            length = max(2.0, bar_space * value / limit)
            # 24 px bars, square at the baseline, rounded at the data end.
            draw.rounded_rectangle((x0, y, x0 + length, y + 24), radius=4, fill=BAR)
            draw.rectangle((x0, y, x0 + min(length, 4), y + 24), fill=BAR)
            draw.text(
                (x0 + length + 12, y - 1),
                fmt.format(value),
                font=font(23, bold=True),
                fill=INK,
            )
    draw.text(
        (60, height - 64),
        "Source: reports/gliner2-experiment/summary.json. Small exploratory comparison; "
        "the held-out sets have eight distinct states.",
        font=font(18),
        fill=MUTED,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gliner2-results.png"
    image.save(path)
    # The home page crops card posters to 16:9; pad a copy instead of cutting the chart.
    card = Image.new("RGB", (width, round(width * 9 / 16)), SURFACE)
    card.paste(image, (0, (card.height - height) // 2))
    card.save(OUT / "gliner2-card.png")
    return path


if __name__ == "__main__":
    print(gliner_results())
