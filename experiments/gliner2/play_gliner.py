"""Record GLiNER2.5 playing Doom with the decision dashboard.

    uv run python play_gliner.py --model base --weights ../../runs/gliner2-v2/base-full \
        --instruction "Attack enemies on sight." --output ../../runs/gliner2-show-attack
"""

import argparse
from pathlib import Path

from gliner_policy import GLiNERPolicy

from doom_bert.play import play


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("small", "base"), required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = GLiNERPolicy(
        f"fastino/gliner2.5-{args.model}-v1", device=args.device, dtype=args.dtype, weights=args.weights
    )
    play(
        scenario="defend_the_center",
        seconds=args.seconds,
        seed=args.seed,
        output=args.output,
        headless=True,
        video=True,
        stats=True,
        action_tics=2,
        instruction=args.instruction,
        policy=policy,
        video_clock="game",
    )


if __name__ == "__main__":
    main()
