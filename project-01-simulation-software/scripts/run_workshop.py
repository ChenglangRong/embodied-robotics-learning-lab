"""Generate three controlled experiments and one arm-motion GIF."""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_lab.planar_arm import Config, run_episode, save_run


def make_gif(rows: list[dict], summary: dict, cfg: Config, destination: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    l1, l2 = cfg.link_lengths
    frames = list(range(0, len(rows), max(1, len(rows) // 60)))
    if frames[-1] != len(rows) - 1:
        frames.append(len(rows) - 1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xlabel="x [m]", ylabel="y [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.plot(*summary["goal"], marker="x", markersize=10, label="goal")
    arm, = ax.plot([], [], "o-", linewidth=4, label="arm")
    trail, = ax.plot([], [], linewidth=1.5, label="tool path")
    ax.legend(loc="lower left")

    def update(index: int):
        row = rows[frames[index]]
        q1, q2 = row["q1"], row["q2"]
        elbow = (l1 * np.cos(q1), l1 * np.sin(q1))
        tip = (elbow[0] + l2 * np.cos(q1 + q2), elbow[1] + l2 * np.sin(q1 + q2))
        arm.set_data([0, elbow[0], tip[0]], [0, elbow[1], tip[1]])
        trail.set_data([r["x"] for r in rows[: frames[index] + 1]],
                       [r["y"] for r in rows[: frames[index] + 1]])
        ax.set_title(f"t={row['time_s']:.2f} s  error={row['error']:.3f} m")
        return arm, trail

    animation = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
    animation.save(destination, writer=PillowWriter(fps=12))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "workshop")
    args = parser.parse_args()
    scenarios = {
        "reachable": ((0.55, 0.55), Config()),
        "unreachable": ((1.05, 0.0), Config(max_steps=300)),
        "speed_limited": ((0.55, 0.55), Config(max_steps=180, velocity_limit=0.05)),
    }
    for name, (goal, cfg) in scenarios.items():
        rows, summary = run_episode(seed=7, goal=goal, cfg=cfg)
        folder = args.output_root / name
        save_run(rows, summary, cfg, folder)
        (folder / "run_metadata.json").write_text(json.dumps({
            "command": "python scripts/run_workshop.py --output-root <path>",
            "scenario": name,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "source_sha256": hashlib.sha256((ROOT / "src" / "robot_lab" / "planar_arm.py").read_bytes()).hexdigest(),
        }, indent=2), encoding="utf-8")
        if name == "reachable":
            make_gif(rows, summary, cfg, folder / "animation.gif")
        print(f"{name}: {summary['termination']}, steps={summary['steps']}, "
              f"final_error={summary['final_error']:.4f} m, "
              f"clipped={summary['clipped_fraction']:.1%}, "
              f"geometric_reachable={summary['geometric_reachable']}")


if __name__ == "__main__":
    main()
