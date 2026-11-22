"""Slow-play the six measured states from the Day 1 MuJoCo control step."""

import argparse
import hashlib
import json
from math import cos, sin
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

from day1_control_step import ACTIONS, RESET_SEED, physical_env

PROJECT = Path(__file__).resolve().parent
EVIDENCE = PROJECT / "outputs" / "day1-control-step-20261122.json"
ANGLE_SCALE = 180
FRAME_MS = [1600, 630, 630, 630, 630, 3500]
SIZE = (1320, 776)
INK, MUTED, BORDER = "#11283b", "#426072", "#d5e1e6"
BRANCHES = (
    ("positive", "正向指令", "#076d73", "#e5f5f3"),
    ("negative", "反向指令", "#af461d", "#fdf0e7"),
    ("zero", "零指令", "#506477", "#edf1f5"),
)


def sample_substeps():
    """Observe the ordinary physics step, with no extra forward/integration calls."""
    frozen = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    env = physical_env.ArmPhysics(split=frozen["environment_split"])
    mujoco = physical_env.mujoco
    mj_step = mujoco.mj_step

    def sample():
        q = env.d.qpos.copy()
        return {
            "time_ms": round(float(env.d.time) * 1000),
            "q_rad": q.tolist(),
            "dq_rad_s": env.d.qvel.tolist(),
            "tip_m": env.fk(q).tolist(),
        }

    traces = {}
    try:
        for name, action in ACTIONS.items():
            env.reset(RESET_SEED)
            samples = [sample()]

            def traced_step(model, data):
                mj_step(model, data)
                samples.append(sample())

            with patch.object(mujoco, "mj_step", traced_step):
                _, _, _, _, info = env.step(action)

            if [state["time_ms"] for state in samples] != [0, 4, 8, 12, 16, 20]:
                raise RuntimeError(f"Unexpected substeps for {name}")
            for state, saved in ((samples[0], frozen["initial"]), (samples[-1], frozen["cases"][name]["final"])):
                for field in ("q_rad", "dq_rad_s"):
                    np.testing.assert_allclose(state[field], saved[field], rtol=0, atol=1e-12)
            np.testing.assert_allclose(samples[-1]["tip_m"], env.d.site_xpos[env.tip_id, :2], rtol=0, atol=1e-12)
            traces[name] = {
                "action": action,
                "applied_velocity_ref_rad_s": info["applied_velocity"],
                "samples": samples,
            }
    finally:
        env.close()
    return traces


def pose_pixels(angle, base):
    q1, q2 = angle
    x, y = base
    elbow = (x + 250 * 0.6 * cos(q1), y - 250 * 0.6 * sin(q1))
    tip = (elbow[0] + 250 * 0.4 * cos(q1 + q2), elbow[1] - 250 * 0.4 * sin(q1 + q2))
    return base, elbow, tip


def circle(draw, point, radius, fill, outline=None, width=1):
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=width)


def dashed_rod(draw, start, end):
    a, b = np.asarray(start), np.asarray(end)
    delta = b - a
    length = float(np.linalg.norm(delta))
    for pos in range(0, int(length), 14):
        left = tuple(a + delta * (pos / length))
        right = tuple(a + delta * (min(pos + 8, length) / length))
        draw.line((left, right), fill="#8b9fac", width=4)


def speed_text(value):
    if abs(value) < 1e-8:
        return "0.00000"
    return f"{value:+.5f}".replace("-", "−")


def signed_pair(values):
    formatted = [f"{value:+g}".replace("-", "−") if value else "0" for value in values]
    return "[" + ", ".join(formatted) + "]"


def draw_case(draw, font, x, branch, trace, index):
    _, label, accent, tint = branch
    action = signed_pair(trace["action"])
    reference = signed_pair(trace["applied_velocity_ref_rad_s"])
    states = trace["samples"]
    initial, current = states[0], states[index]
    draw.rounded_rectangle((x, 203, x + 406, 652), radius=22, fill="#ffffff", outline=BORDER, width=2)
    draw.rounded_rectangle((x + 19, 221, x + 148, 262), radius=12, fill=tint)
    draw.text((x + 33, 227), label, fill=accent, font=font(24, True))
    draw.text((x + 22, 275), f"动作  {action}", fill=INK, font=font(22))
    draw.text((x + 22, 307), f"速度参考  {reference} rad/s", fill=MUTED, font=font(20))

    base = (x + 80, 433)
    draw.rounded_rectangle((x + 21, 345, x + 384, 531), radius=12, fill="#f5f8fa")
    for y in (365, 402, 439, 476, 513):
        draw.line((x + 40, y, x + 364, y), fill="#e4edf0", width=1)
    for offset in (45, 95, 145, 195, 245, 295, 345):
        draw.line((x + offset, 354, x + offset, 522), fill="#e4edf0", width=1)
    q0 = np.asarray(initial["q_rad"])
    q_view = q0 + ANGLE_SCALE * (np.asarray(current["q_rad"]) - q0)
    _, rest_elbow, rest_tip = pose_pixels(q0, base)
    if index:
        dashed_rod(draw, base, rest_elbow)
        dashed_rod(draw, rest_elbow, rest_tip)
        circle(draw, rest_tip, 5, "#ffffff", "#8b9fac", 2)

    _, elbow, tip = pose_pixels(q_view, base)
    draw.line((base, elbow), fill=accent, width=13)
    draw.line((elbow, tip), fill=accent, width=9)
    circle(draw, base, 13, INK, "#ffffff", 2)
    circle(draw, elbow, 10, "#ffffff", accent, 4)
    circle(draw, tip, 7, accent, "#ffffff", 2)
    draw.text((base[0] - 38, base[1] - 36), "关节 1", fill=INK, font=font(16, True))
    label_y = elbow[1] - 23 if elbow[1] > 505 else elbow[1] + 1
    draw.text((elbow[0] + 12, label_y), "关节 2", fill=INK, font=font(16, True))

    draw.text((x + 22, 538), "实测角速度（rad/s）", fill=MUTED, font=font(20))
    for j, dq in enumerate(current["dq_rad_s"], start=1):
        column = x + 22 + (j - 1) * 198
        draw.text((column, 569), f"关节 {j}", fill=INK, font=font(17))
        draw.text((column + 61, 563), speed_text(dq), fill=accent, font=font(23, True))
    tip_mm = float(np.linalg.norm(np.asarray(current["tip_m"]) - initial["tip_m"]) * 1000)
    draw.text((x + 22, 610), f"末端实际移动  {tip_mm:.3f} mm", fill=MUTED, font=font(20))


def draw_frame(font, traces, index):
    image = Image.new("RGB", SIZE, "#f0f5f7")
    draw = ImageDraw.Draw(image)
    draw.text((36, 23), "同一个起点，三种动作", fill=INK, font=font(46, True))
    draw.text((38, 91), "MuJoCo 二连杆  /  同一静止初态  /  五个 4 ms 仿真子步", fill=MUTED, font=font(24))

    draw.text((38, 148), "仿真时间", fill=INK, font=font(21, True))
    draw.line((172, 158, 597, 158), fill="#b4c6cf", width=3)
    for step in range(6):
        x = 172 + step * 85
        circle(draw, (x, 158), 8 if step == index else 5, INK if step == index else "#a5bac4")
        draw.text((x, 169), str(step * 4), fill=INK if step == index else MUTED, font=font(18), anchor="mt")
    draw.text((615, 170), "ms", fill=MUTED, font=font(16))
    draw.text((1279, 137), f"关节角变化放大 {ANGLE_SCALE}×", fill=INK, font=font(25, True), anchor="rt")
    draw.text((1279, 171), "灰虚线表示起始姿势 · 速度和位移读数为实际值", fill=MUTED, font=font(18), anchor="rt")

    for column, branch in enumerate(BRANCHES):
        draw_case(draw, font, 36 + column * 421, branch, traces[branch[0]], index)

    draw.rounded_rectangle((36, 672, 1284, 745), radius=15, fill="#e0ebef")
    if index == 0:
        title = "0 ms  /  三组都在同一位置静止"
        detail = "接下来只切换给关节 1 的速度指令，观察两关节各自如何响应。"
    elif index == 5:
        title = "20 ms  /  关节 1 随指令换向，关节 2 出现较小的反向转动"
        detail = "关节 2 的读数表示第二根杆相对第一根杆的角速度。两杆相连，转动会通过惯性相互影响。"
    else:
        title = f"{index * 4} ms  /  电机根据速度误差施力，机械臂逐步加速"
        detail = "对照每组的速度参考与实测角速度。三组始终显示同一个仿真时刻。"
    draw.text((56, 679), title, fill=INK, font=font(24, True))
    draw.text((56, 716), detail, fill=MUTED, font=font(19))
    return image


def main():
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, default=fonts / "msyh.ttc")
    parser.add_argument("--bold-font", type=Path, default=fonts / "msyhbd.ttc")
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE.parent)
    args = parser.parse_args()
    traces = sample_substeps()
    cache = {}

    def font(size, bold=False):
        key = size, bold
        if key not in cache:
            cache[key] = ImageFont.truetype(args.bold_font if bold else args.font, size)
        return cache[key]

    frames = [draw_frame(font, traces, index) for index in range(len(FRAME_MS))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output_dir / "day1-motion.gif", save_all=True, append_images=frames[1:],
        duration=FRAME_MS, loop=0, optimize=True, disposal=1,
    )
    report = {
        "frozen_control_result_sha256": hashlib.sha256(EVIDENCE.read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pillow_version": pillow_version,
        "fonts": [args.font.name, args.bold_font.name],
        "angle_magnification": ANGLE_SCALE,
        "gif_frame_hold_ms": FRAME_MS,
        "reset_seed": RESET_SEED,
        "cases": traces,
    }
    (args.output_dir / "day1-motion-samples.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
