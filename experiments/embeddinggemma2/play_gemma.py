"""Play Doom zero-shot with EmbeddingGemma 2 and record the dashboard video.

uv run python play_gemma.py --instruction "Attack enemies on sight." --seconds 60 --headless
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from gemma_doom import WINDOW_STRIDE, WINDOW_WIDTH, EmbeddingGemmaPolicy

from doom_bert.play import SCENARIOS, play

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instruction", default="Attack enemies on sight.")
    parser.add_argument("--seconds", type=int, default=60, help="Game seconds, not wall time")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenario", choices=SCENARIOS, default="defend_the_center")
    parser.add_argument("--action-tics", type=int, default=2)
    parser.add_argument("--headless", action="store_true", help="No game window (use over SSH)")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-stats", action="store_true", help="Record only the game, without the dashboard")
    parser.add_argument(
        "--video-clock",
        choices=("game", "wall"),
        default="game",
        help="game: 35 fps replay at normal game speed; wall: actual inference speed",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"))
    parser.add_argument("--vision-tokens", type=int, default=140, choices=[70, 140, 280, 560, 1120])
    parser.add_argument("--window-width", type=int, default=WINDOW_WIDTH)
    parser.add_argument("--window-stride", type=int, default=WINDOW_STRIDE)
    parser.add_argument(
        "--show-weapon", action="store_true", help="Render the pistol (its recoil and flash cross the view)"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    policy = EmbeddingGemmaPolicy(
        device=args.device,
        dtype=args.dtype,
        vision_tokens=args.vision_tokens,
        window_width=args.window_width,
        window_stride=args.window_stride,
    )
    behaviour, sims = policy.perception.behaviour(args.instruction)
    print(
        f"EmbeddingGemma 2: {policy.device} / {policy.dtype}; instruction -> {behaviour} "
        f"(cosine {dict(zip(policy.perception.behaviour_names, sims.round(3).tolist(), strict=True))})",
        flush=True,
    )
    output = args.output or REPO / "runs" / f"eg2-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    play(
        scenario=args.scenario,
        seconds=args.seconds,
        seed=args.seed,
        output=output,
        headless=args.headless,
        video=not args.no_video,
        action_tics=args.action_tics,
        instruction=args.instruction,
        policy=policy,
        stats=not args.no_stats,
        video_clock=args.video_clock,
        render_weapon=args.show_weapon,
    )


if __name__ == "__main__":
    main()
