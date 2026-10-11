import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_lab.planar_arm import Config, run_episode, save_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "demo")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--goal-x", type=float, default=0.55)
    parser.add_argument("--goal-y", type=float, default=0.55)
    parser.add_argument("--velocity-limit", type=float, default=2.0)
    args = parser.parse_args()
    cfg = Config(velocity_limit=args.velocity_limit)
    rows, summary = run_episode(args.seed, (args.goal_x, args.goal_y), cfg)
    save_run(rows, summary, cfg, args.output)
    print(summary)


if __name__ == "__main__":
    main()
