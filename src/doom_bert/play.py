"""Observe, decide, and advance one Doom tic as fast as the policy can run."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import vizdoom as vzd
from PIL import Image

from doom_bert.overlay import LiveMetrics, LivePreview, RenderWorker, StatsOverlay
from doom_bert.policy import BUTTONS, decode_scores, selection_notes
from doom_bert.video import VideoRecorder

TICS_PER_SECOND = 35
SCENARIOS = ("defend_the_center", "basic", "deadly_corridor")
VARIABLES = (
    vzd.GameVariable.HEALTH,
    vzd.GameVariable.ARMOR,
    vzd.GameVariable.SELECTED_WEAPON,
    vzd.GameVariable.SELECTED_WEAPON_AMMO,
    vzd.GameVariable.KILLCOUNT,
    vzd.GameVariable.POSITION_X,
    vzd.GameVariable.POSITION_Y,
    vzd.GameVariable.POSITION_Z,
    vzd.GameVariable.ANGLE,
    vzd.GameVariable.DEAD,
    vzd.GameVariable.DAMAGE_TAKEN,
)
OPPOSITES = (
    ("TURN_LEFT", "TURN_RIGHT"),
    ("MOVE_LEFT", "MOVE_RIGHT"),
    ("MOVE_FORWARD", "MOVE_BACKWARD"),
)


def legal_actions(buttons: list[str], variables: dict[str, float]) -> list[list[int]]:
    """Mask contradictory directions and shooting without ammo; retain a no-op."""
    if variables["DEAD"] or variables["HEALTH"] <= 0:
        return [[0] * len(buttons)]
    actions = []
    for values in product((0, 1), repeat=len(buttons)):
        pressed = {name for name, value in zip(buttons, values, strict=True) if value}
        if any(left in pressed and right in pressed for left, right in OPPOSITES):
            continue
        # Fist and chainsaw do not consume ammunition.
        if (
            "ATTACK" in pressed
            and variables["SELECTED_WEAPON_AMMO"] <= 0
            and variables["SELECTED_WEAPON"] not in (0, 1)
        ):
            continue
        actions.append(list(values))
    return actions


def create_game(scenario: str, *, headless: bool, seed: int) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(str(Path(vzd.scenarios_path) / f"{scenario}.cfg"))
    game.set_mode(vzd.Mode.PLAYER)
    game.set_seed(seed)
    game.set_window_visible(not headless)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(True)
    game.set_render_crosshair(True)
    game.set_labels_buffer_enabled(True)
    game.set_available_game_variables(list(VARIABLES))
    game.set_sound_enabled(False)
    return game


def advance_tics(
    game: vzd.DoomGame,
    deadline: float,
    *,
    tics: int,
    realtime: bool,
    recorder: VideoRecorder | None = None,
    overlay: StatsOverlay | None = None,
    preview: LivePreview | None = None,
    stats: dict | None = None,
    started: float = 0,
    render_worker: RenderWorker | None = None,
) -> None:
    """Advance the chosen action; only the optional realtime mode sleeps."""
    if not realtime and recorder is None and preview is None:
        game.advance_action(tics)
        return
    for tic in range(tics):
        if render_worker is not None:
            elapsed = time.monotonic() - started
            if render_worker.ready(elapsed):
                frame_state = game.get_state()
                render_worker.submit(
                    frame_state.screen_buffer if frame_state else None,
                    stats or {},
                    elapsed,
                )
            if preview is not None and preview.ready():
                if render_worker.latest_frame is not None:
                    preview.present(render_worker.latest_frame)
        elif recorder is not None or preview is not None:
            elapsed = time.monotonic() - started
            record_frame = recorder is not None and recorder.ready(elapsed)
            preview_frame = preview is not None and preview.ready()
            if record_frame or preview_frame:
                frame_state = game.get_state()
                screen = frame_state.screen_buffer if frame_state else None
                if overlay is not None:
                    screen = overlay.render(
                        screen, {**(stats or {}), "display_wall_seconds": elapsed}
                    )
                if record_frame:
                    recorder.write(screen, elapsed_seconds=elapsed)
                if preview_frame:
                    preview.present(screen)
        if not game.is_episode_finished():
            game.advance_action(1)
        if realtime:
            target = deadline - (tics - tic - 1) / TICS_PER_SECOND
            time.sleep(max(0, target - time.monotonic()))


def play(
    *,
    scenario: str,
    seconds: int,
    seed: int,
    output: Path,
    headless: bool = False,
    realtime: bool = False,
    video: bool = True,
    action_tics: int = 1,
    instruction: str = "",
    policy: Callable[[str, dict], dict[str, float]] | None = None,
    stats: bool = False,
    video_clock: str = "game",
) -> dict:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if action_tics < 1:
        raise ValueError("action_tics must be positive")
    if video_clock not in ("game", "wall"):
        raise ValueError("video_clock must be game or wall")
    # Refuse to mix observations from different runs or overwrite recordings.
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    rng = random.Random(seed)
    game = create_game(scenario, headless=headless or stats, seed=seed)
    if policy is not None:
        game.set_available_buttons([getattr(vzd.Button, button) for button in BUTTONS])
    summary = {
        "scenario": scenario,
        "seed": seed,
        "requested_seconds": seconds,
        "observation_interval_game_seconds": action_tics / TICS_PER_SECOND,
        "screenshot_interval_game_seconds": 1,
        "tics_per_action": action_tics,
        "policy": "random" if policy is None else type(policy).__name__,
        "policy_device": getattr(policy, "device", None),
        "policy_dtype": getattr(policy, "dtype", None),
        "untrained_action_head": getattr(policy, "untrained_head", None),
        "instruction": instruction,
        "realtime": realtime,
        "vizdoom_version": vzd.__version__,
        "episodes": 1,
        "completed_episodes": 0,
        "deaths": 0,
        "kills": 0,
        "observations": 0,
        "decisions": 0,
        "interrupted": False,
        "video": "replay.mp4" if video else None,
        "video_fps": TICS_PER_SECOND if video and video_clock == "game" else None,
        "video_clock": video_clock if video else None,
        "video_capture_limit_fps": 30 if video_clock == "wall" else TICS_PER_SECOND,
        "stats_overlay": stats,
        "headless": headless,
        "preview_enabled": stats and not headless,
    }
    (output / "config.json").write_text(json.dumps(summary, indent=2) + "\n")
    completed_kills = 0
    recorder = None
    overlay = None
    preview = None
    render_worker = None
    metrics = LiveMetrics()
    metric_values = {}
    decision_seconds = 0.0
    previous_damage = None
    next_capture_tic = 0
    elapsed_tics = 0
    total_tics = seconds * TICS_PER_SECOND
    started = time.monotonic()
    try:
        game.init()
        if stats:
            overlay = StatsOverlay(
                policy_name="ModernBERT" if policy is not None else "RANDOM BASELINE",
                device=getattr(policy, "device", None),
                dtype=getattr(policy, "dtype", None),
                instruction=instruction,
                untrained=bool(getattr(policy, "untrained_head", False)),
                video_clock=video_clock,
            )
            if not headless:
                preview = LivePreview()
        if video:
            recorder = VideoRecorder(
                output / "replay.mp4",
                width=overlay.width if overlay else game.get_screen_width(),
                height=overlay.height if overlay else game.get_screen_height(),
                fps=30 if video_clock == "wall" else TICS_PER_SECOND,
                clock=video_clock,
            )
        if video_clock == "wall" and (recorder is not None or preview is not None):
            render_worker = RenderWorker(recorder, overlay)
        buttons = [button.name for button in game.get_available_buttons()]
        started = time.monotonic()
        print(
            f"{scenario} | seed={seed} | {action_tics} tic(s)/decision | "
            f"{'realtime' if realtime else 'uncapped'}",
            flush=True,
        )
        print(f"Captures: {output.resolve()}", flush=True)
        with (output / "states.jsonl").open("w") as log:
            while elapsed_tics <= total_tics:
                second = elapsed_tics / TICS_PER_SECOND
                if game.is_episode_finished():
                    dead = game.is_player_dead()
                    completed_kills += int(
                        game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                    )
                    summary["completed_episodes"] += 1
                    summary["deaths"] += int(dead)
                    summary["kills"] = completed_kills
                    log.write(
                        json.dumps(
                            {
                                "type": "episode_end",
                                "second": second,
                                "episode": summary["episodes"],
                                "episode_tic": game.get_episode_time(),
                                "dead": dead,
                                "total_reward": game.get_total_reward(),
                            }
                        )
                        + "\n"
                    )
                    print(
                        f"  Episode {summary['episodes']} ended; dead={dead}",
                        flush=True,
                    )
                    if second == seconds:
                        break
                    game.new_episode()
                    summary["episodes"] += 1
                    previous_damage = None

                state = game.get_state()
                if state is None:
                    raise RuntimeError(
                        "ViZDoom returned no state during an active episode"
                    )
                observed_at = time.monotonic() - started
                variables = {
                    variable.name: float(value)
                    for variable, value in zip(
                        VARIABLES, state.game_variables, strict=True
                    )
                }
                record = {
                    "type": "observation",
                    "second": second,
                    "elapsed_tics": elapsed_tics,
                    "wall_seconds": round(observed_at, 6),
                    "episode": summary["episodes"],
                    "episode_tic": state.tic,
                    "state_number": state.number,
                    "variables": variables,
                    "damage_since_observation": (
                        0
                        if previous_damage is None
                        else max(0, variables["DAMAGE_TAKEN"] - previous_damage)
                    ),
                    "screen_width": game.get_screen_width(),
                    "visible_objects": [
                        {
                            "id": label.object_id,
                            "name": label.object_name,
                            "category": label.object_category,
                            "box": [label.x, label.y, label.width, label.height],
                            "position": [
                                label.object_position_x,
                                label.object_position_y,
                                label.object_position_z,
                            ],
                        }
                        for label in state.labels
                    ],
                    "buttons": buttons,
                    "last_reward": game.get_last_reward(),
                    "total_reward": game.get_total_reward(),
                }
                previous_damage = variables["DAMAGE_TAKEN"]
                candidates = legal_actions(buttons, variables)
                action = None
                scores = None
                inference_ms = None
                if elapsed_tics < total_tics:
                    decision_started = time.monotonic()
                    if policy is None:
                        action = rng.choice(candidates)
                    else:
                        scores = policy(instruction, record)
                        action = decode_scores(scores, buttons, candidates)
                    inference_ms = (time.monotonic() - decision_started) * 1000
                    decision_seconds += inference_ms / 1000
                    summary["decisions"] += 1
                    metric_values = metrics.update(
                        inference_ms, time.monotonic() - started
                    )
                pressed = [
                    name
                    for name, value in zip(
                        buttons, action or [0] * len(buttons), strict=True
                    )
                    if value
                ]
                # PNG and terminal output do not run on every model decision.
                capture = elapsed_tics >= next_capture_tic or action is None
                frame_name = None
                if capture:
                    frame_name = f"frames/{elapsed_tics:06d}.png"
                    Image.fromarray(state.screen_buffer).save(output / frame_name)
                    next_capture_tic = (
                        elapsed_tics // TICS_PER_SECOND + 1
                    ) * TICS_PER_SECOND
                record.update(
                    {
                        "screen": frame_name,
                        "action": action,
                        "pressed": pressed,
                        "action_scores": scores,
                        "decision_ms": inference_ms,
                        "legal_action_count": len(candidates),
                        "selection_notes": selection_notes(
                            scores, buttons, action, candidates
                        )
                        if action is not None
                        else [],
                        "decision_count": summary["decisions"],
                        "action_tics": action_tics,
                        "combo_score": math.prod(
                            scores[button] if value else 1 - scores[button]
                            for button, value in zip(buttons, action, strict=True)
                        )
                        if scores is not None and action is not None
                        else None,
                        "input_tokens": getattr(policy, "last_token_count", None),
                        **metric_values,
                    }
                )
                summary["observations"] += 1
                summary["kills"] = completed_kills + int(variables["KILLCOUNT"])
                record["total_kills"] = summary["kills"]
                log.write(json.dumps(record) + "\n")
                action_text = "+".join(pressed) if action is not None else "STOP"
                if capture:
                    log.flush()
                    print(
                        f"[{second:6.2f}s] ep={summary['episodes']} tic={state.tic:4d} "
                        f"health={variables['HEALTH']:3.0f} "
                        f"ammo={variables['SELECTED_WEAPON_AMMO']:3.0f} "
                        f"kills={summary['kills']} → {action_text or 'WAIT'}",
                        flush=True,
                    )
                if action is None:
                    break
                game.set_action(action)
                tics = min(action_tics, total_tics - elapsed_tics)
                advance_tics(
                    game,
                    started + (elapsed_tics + tics) / TICS_PER_SECOND,
                    tics=tics,
                    realtime=realtime,
                    recorder=recorder,
                    overlay=overlay,
                    preview=preview,
                    stats=record,
                    started=started,
                    render_worker=render_worker,
                )
                elapsed_tics += tics
    except KeyboardInterrupt:
        summary["interrupted"] = True
        print("\nStopped; captured states are saved.", flush=True)
    finally:
        wall_seconds = time.monotonic() - started
        summary["wall_seconds"] = round(wall_seconds, 3)
        summary["game_seconds"] = elapsed_tics / TICS_PER_SECOND
        summary["decisions_per_wall_second"] = summary["decisions"] / max(
            wall_seconds, 1e-9
        )
        summary["mean_decision_ms"] = (
            decision_seconds * 1000 / max(summary["decisions"], 1)
        )
        game.close()
        if render_worker is not None:
            render_worker.close()
            summary["dropped_video_frames"] = render_worker.dropped_frames
            summary["background_rendering"] = True
        if preview is not None:
            preview.close()
        if recorder is not None:
            recorder.close()
            summary["video_frames"] = recorder.frames
            summary["video_seconds"] = recorder.duration_seconds
            summary["video_encoder"] = recorder.encoder
            summary["video_width"] = recorder.stream.width
            summary["video_height"] = recorder.stream.height
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="defend_the_center")
    parser.add_argument(
        "--seconds", type=int, default=60, help="Duration in game seconds"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true", help="Hide the game window")
    pacing = parser.add_mutually_exclusive_group()
    pacing.add_argument(
        "--realtime", action="store_true", help="Optionally pace game tics to wall time"
    )
    pacing.add_argument(
        "--fast",
        dest="realtime",
        action="store_false",
        help="Uncapped execution (default)",
    )
    parser.set_defaults(realtime=False)
    parser.add_argument(
        "--action-tics", type=int, default=1, help="Game tics per decision (default: 1)"
    )
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument(
        "--checkpoint", help="Fine-tuned ModernBERT policy directory or model ID"
    )
    model_source.add_argument(
        "--benchmark-demo",
        action="store_true",
        help="Show pretrained ModernBERT throughput using an explicitly untrained action head",
    )
    parser.add_argument("--instruction", default="attack every enemy you see")
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record an H.264 MP4 (default: enabled)",
    )
    parser.add_argument(
        "--stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show live performance and button-score panel (default: enabled)",
    )
    parser.add_argument(
        "--video-clock",
        choices=("wall", "game"),
        default="wall",
        help="Record actual wall-clock timing or fixed game-time playback (default: wall)",
    )
    parser.add_argument(
        "--output", type=Path, help="New output directory (must not already exist)"
    )
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.action_tics <= 0:
        parser.error("--action-tics must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    output = args.output or Path("runs") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S-%fZ"
    )
    if output.exists():
        parser.error(f"output directory already exists: {output}")
    policy = None
    if args.checkpoint or args.benchmark_demo:
        try:
            import torch

            from doom_bert.model import BASE_MODEL, ModernBertPolicy
        except ImportError:
            parser.error("Model dependencies are missing; run uv sync --extra model")

        try:
            torch.manual_seed(args.seed)
            policy = ModernBertPolicy(
                args.checkpoint or BASE_MODEL,
                device=args.device,
                dtype=args.dtype,
                allow_untrained_head=args.benchmark_demo,
            )
        except ValueError as error:
            parser.error(str(error))
        print(f"ModernBERT: {policy.device} / {policy.dtype}", flush=True)
        from doom_bert.benchmark import sample_observation

        for index in range(5):
            policy(args.instruction, sample_observation(index))
        policy.synchronize()
    play(
        scenario=args.scenario,
        seconds=args.seconds,
        seed=args.seed,
        output=output,
        headless=args.headless,
        realtime=args.realtime,
        video=args.video,
        action_tics=args.action_tics,
        instruction=args.instruction,
        policy=policy,
        stats=args.stats,
        video_clock=args.video_clock,
    )


if __name__ == "__main__":
    main()
