"""Single-factor physics experiments and read-only diagnostics; no new success claim."""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import html
import json
from pathlib import Path
import platform
import subprocess
import sys
import webbrowser
import mujoco
import numpy as np
import lab

ROOT = Path(__file__).resolve().parent
SUITES = {
    "panda": [("baseline", {}), ("partly_open", {"close_ctrl": 160}), ("open", {"close_ctrl": 255})],
    "go2": [("baseline", {}), ("soft_pd", {"kp": 55}), ("large_stride", {"amp": .15})],
    "h1": [("baseline", {}), ("low_feedback", {"com_gain": 3.5}), ("deep_squat", {"bend": .20})],
    "allegro": [("baseline", {}), ("open_fingers", {"closed": [0, .15, .15, .15] * 3 + [.6, .2, .2, .2]}),
                ("tilted_wrist", {"wrist_tilt": .6})],
}
LABELS = {"panda": "Panda：夹爪命令与搬运", "go2": "Go2：关节跟踪与支撑相滑移",
          "h1": "H1：反馈强度与失稳", "allegro": "Allegro：接触计数与竖直受力"}


def contact_on_geom(contact, local_force, geom_id):
    """World force acting on geom_id. Frame rows are contact basis vectors."""
    if geom_id not in (contact.geom1, contact.geom2):
        raise ValueError("Requested geom is outside this contact")
    sign = 1. if contact.geom2 == geom_id else -1.
    return sign * (contact.frame.reshape(3, 3).T @ np.asarray(local_force)[:3])


def tangent_speed(velocity, center, point, normal):
    """Rigid surface point speed tangent to a stationary ground contact."""
    point_velocity = np.asarray(velocity)[3:] + np.cross(np.asarray(velocity)[:3], np.asarray(point) - center)
    return float(np.linalg.norm(point_velocity - normal * np.dot(normal, point_velocity)))


def observe(kind, m, d, c):
    """Read cached post-mj_step data. Never call mj_forward or change simulator state."""
    # mj_step has integrated qpos; cached geometry/contact is from the previous
    # dynamics evaluation (~one physics step earlier). Preserve that distinction.
    row = {"time_s": float(d.time), "cached_geometry_time_s": float(d.time - m.opt.timestep)}
    n = 7 if kind == "panda" else len(c.qa)
    row["tracking_rmse_rad"] = float(np.sqrt(np.mean((c.target[:n] - d.qpos[c.qa[:n]]) ** 2)))
    if kind in ("go2", "h1"):
        limits = m.actuator_ctrlrange
        saturated = (d.ctrl <= limits[:, 0] + 1e-6) | (d.ctrl >= limits[:, 1] - 1e-6)
        row["saturated_actuator_fraction"] = float(saturated.mean())
        row["base_x_m"], row["base_height_m"] = map(float, d.qpos[[0, 2]])
        body = m.body("base" if kind == "go2" else "pelvis").id
        row["tilt_deg"] = float(np.degrees(np.arccos(np.clip(d.xmat[body].reshape(3, 3)[2, 2], -1, 1))))
    if kind == "go2":
        floor = m.geom("floor").id
        for foot in ("FL", "FR", "RL", "RR"):
            gid = m.geom(foot).id
            contacts = [k for k in d.contact if gid in (k.geom1, k.geom2) and floor in (k.geom1, k.geom2)]
            row[f"{foot}_contact"] = int(bool(contacts))
            row[f"{foot}_x_m"], row[f"{foot}_y_m"] = map(float, d.geom_xpos[gid, :2])
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, gid, velocity, 0)
            speeds = [tangent_speed(velocity, d.geom_xpos[gid], k.pos, k.frame[:3]) for k in contacts]
            row[f"{foot}_contact_tangent_speed_m_s"] = float(np.mean(speeds)) if speeds else 0.
    if kind == "h1":
        pelvis = m.body("pelvis").id
        feet_x = .5 * (d.xpos[m.body("left_ankle_link").id, 0] + d.xpos[m.body("right_ankle_link").id, 0])
        row["com_reference_error_m"] = float(d.subtree_com[pelvis, 0] - feet_x - c.options.get("foot_offset", .03))
        row["ankle_saturation_fraction"] = float(saturated[[4, 9]].mean())
    if kind in ("panda", "allegro"):
        oid, gid = m.body("object").id, m.geom("object_geom").id
        obj = d.xpos[oid]
        row["object_height_m"] = float(obj[2])
        row["object_weight_N"] = float(m.body_mass[oid] * np.linalg.norm(m.opt.gravity))
        if kind == "panda":
            ref, rot = d.site_xpos[m.site("tcp").id], d.site_xmat[m.site("tcp").id].reshape(3, 3)
        else:
            ref, rot = d.xpos[m.body("palm").id], d.xmat[m.body("palm").id].reshape(3, 3)
        rel = rot.T @ (obj - ref)
        for axis, value in zip("xyz", rel):
            row[f"object_relative_{axis}_m"] = float(value)
        for group in ("finger", "palm", "external"):
            row[f"{group}_normal_sum_N"] = 0.
            row[f"{group}_world_fz_N"] = 0.
            row[f"{group}_contacts"] = 0
        groups = set()
        for i, contact in enumerate(d.contact):
            if gid not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom1 if contact.geom2 == gid else contact.geom2
            name = m.body(int(m.geom_bodyid[other])).name or ""
            finger = name.startswith(("ff_", "mf_", "rf_", "th_", "left_finger", "right_finger"))
            group = "finger" if finger else "palm" if name in ("palm", "hand") else "external"
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            world_force = contact_on_geom(contact, force, gid)
            row[f"{group}_normal_sum_N"] += float(abs(force[0]))
            row[f"{group}_world_fz_N"] += float(world_force[2])
            row[f"{group}_contacts"] += 1
            if finger:
                groups.add(name[:2] if kind == "allegro" else name)
        row["distinct_contacting_fingers"] = len(groups)
    return row


def stance_segments(rows, foot, start=2., end=9.):
    """50-Hz contact streaks, >=4 samples; planar net travel and path length."""
    chunks, current = [], []
    for r in rows:
        if start < r["time_s"] < end and r[f"{foot}_contact"]:
            current.append(r)
        else:
            if len(current) >= 4:
                chunks.append(current)
            current = []
    if len(current) >= 4:
        chunks.append(current)
    result = []
    for chunk in chunks:
        xy = np.array([[r[f"{foot}_x_m"], r[f"{foot}_y_m"]] for r in chunk])
        result.append({"start_s": chunk[0]["time_s"], "end_s": chunk[-1]["time_s"],
                       "net_travel_m": float(np.linalg.norm(xy[-1] - xy[0])),
                       "path_length_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())})
    return result


def diagnostic_metrics(kind, rows):
    out = {"mean_tracking_rmse_rad": float(np.mean([r["tracking_rmse_rad"] for r in rows]))}
    if kind == "go2":
        out["stance"] = {}
        for foot in ("FL", "FR", "RL", "RR"):
            segments = stance_segments(rows, foot)
            out["stance"][foot] = {"segments": len(segments),
                "mean_net_travel_m": float(np.mean([s["net_travel_m"] for s in segments])) if segments else None,
                "mean_path_length_m": float(np.mean([s["path_length_m"] for s in segments])) if segments else None,
                "records": segments}
            active_speeds = [r[f"{foot}_contact_tangent_speed_m_s"] for r in rows if 2 < r["time_s"] < 9 and r[f"{foot}_contact"]]
            out["stance"][foot]["mean_contact_tangent_speed_m_s"] = float(np.mean(active_speeds)) if active_speeds else None
    if kind in ("go2", "h1"):
        out["mean_saturated_actuator_fraction"] = float(np.mean([r["saturated_actuator_fraction"] for r in rows]))
    if kind == "h1":
        crossed = next((r for r in rows if r["tilt_deg"] >= np.degrees(.25)), None)
        out["first_tilt_threshold_crossing_s"] = crossed["time_s"] if crossed else None
        out["mean_ankle_saturation_fraction"] = float(np.mean([r["ankle_saturation_fraction"] for r in rows]))
    if kind in ("panda", "allegro"):
        window = (7., 9.) if kind == "panda" else (8., float("inf"))
        late = [r for r in rows if window[0] < r["time_s"] < window[1]]
        xyz = np.array([[r[f"object_relative_{a}_m"] for a in "xyz"] for r in late])
        out["relative_position_window_s"] = [window[0], window[1] if np.isfinite(window[1]) else rows[-1]["time_s"]]
        out["max_relative_displacement_from_window_start_m"] = float(np.linalg.norm(xyz - xyz[0], axis=1).max())
        for group in ("finger", "palm", "external"):
            for field in ("world_fz_N", "normal_sum_N"):
                key = f"{group}_{field}"
                out[f"late_mean_{key}"] = float(np.mean([r[key] for r in late]))
        out["object_weight_N"] = late[0]["object_weight_N"]
        out["late_finger_contact_fraction"] = float(np.mean([r["distinct_contacting_fingers"] >= 2 for r in late]))
    return out


def series_svg(cases, key, title, unit):
    """Static scientific plot, all points retained in source CSV; report downsamples."""
    palette = ("#2563eb", "#c2410c", "#15803d")
    series = [(name, [(r["time_s"], r[key]) for r in rows]) for name, rows in cases]
    ys = [y for _, pts in series for _, y in pts]
    if not ys:
        return ""
    ymin, ymax = min(ys), max(ys)
    pad = max((ymax - ymin) * .08, .01)
    ymin, ymax = ymin - pad, ymax + pad
    xmax = max(pts[-1][0] for _, pts in series)
    def point(x, y):
        return 70 + x / xmax * 590, 250 - (y - ymin) / (ymax - ymin) * 200
    parts = [f'<figure><figcaption>{html.escape(title)}</figcaption><svg viewBox="0 0 720 310" role="img" aria-label="{html.escape(title)}"><rect x="70" y="50" width="590" height="200" fill="none" stroke="#94a3b8"/>']
    for i in range(5):
        x = xmax * i / 4
        px, _ = point(x, ymin)
        y = ymin + (ymax - ymin) * i / 4
        _, py = point(0, y)
        parts += [f'<text x="{px}" y="273" text-anchor="middle">{x:.1f}</text>',
                  f'<text x="62" y="{py+4}" text-anchor="end">{y:.3g}</text>']
    for (name, pts), color in zip(series, palette):
        sampled = pts[::max(1, len(pts) // 240)]
        if sampled[-1] != pts[-1]:
            sampled.append(pts[-1])
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(x, y) for x, y in sampled))
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.7"/>')
    parts += [f'<text x="365" y="300" text-anchor="middle">时间 (s)</text><text x="12" y="32">{html.escape(unit)}</text></svg><div>']
    for (name, _), color in zip(series, palette):
        parts.append(f'<span style="color:{color};margin-right:18px">━ {html.escape(name)}</span>')
    parts.append('</div></figure>')
    return "".join(parts)


def report_html(output, results, traces):
    charts = {
        "panda": [("object_height_m", "物体高度", "m"), ("tracking_rmse_rad", "七个臂关节跟踪误差", "rad"), ("object_relative_z_m", "物体相对 TCP 的局部 z", "m")],
        "go2": [("base_x_m", "机身前进", "m"), ("FL_x_m", "前左足世界系 x", "m"), ("FL_contact_tangent_speed_m_s", "前左足接触表面切向速度（离地时置零）", "m/s"), ("tracking_rmse_rad", "关节目标跟踪误差", "rad")],
        "h1": [("tilt_deg", "机身倾斜", "deg"), ("base_height_m", "骨盆高度", "m"), ("com_reference_error_m", "质心与踝部参考 x 偏差", "m")],
        "allegro": [("object_height_m", "球心高度", "m"), ("finger_world_fz_N", "手指对球的世界系竖直合力（带符号）", "N"), ("palm_world_fz_N", "掌面对球的世界系竖直合力（带符号）", "N")],
    }
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>项目03单因素实验报告</title><style>body{font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 18px;color:#182233;background:#fff}svg{width:100%;max-width:720px}svg text{font-size:13px;fill:#182233}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cbd5e1;padding:8px;text-align:left}pre{white-space:pre-wrap;background:#f1f5f9;padding:14px}figure{margin:24px 0}a{color:#2563eb}</style><h1>项目03：单因素实验与诊断</h1><p>旧判据用于历史对照；新列用于诊断。每组只改变配置中的一个字段。支撑相为50 Hz记录中连续接地至少4个样本，取2–9秒。阈值标定、抗扰测试与新控制器另设实验。</p>']
    parts.append(f'<p>运行目录：{html.escape(str(output))}</p>')
    for kind in traces:
        parts.append(f'<h2>{html.escape(LABELS[kind])}</h2><table><tr><th>案例</th><th>单项改动</th><th>旧判据</th><th>基线CSV一致</th><th>原始证据</th></tr>')
        for name, _ in traces[kind]:
            item = results[kind][name]
            link = f"{kind}/{name}/diagnostics.csv"
            parts.append(f'<tr><td>{name}</td><td>{html.escape(json.dumps(item["changed_options"], ensure_ascii=False))}</td><td>{item["legacy_result"]["passed"]}</td><td>{item["saved_baseline_csv_identical"]}</td><td><a href="{link}">诊断CSV</a> · <a href="{kind}/{name}/diagnostic_summary.json">指标</a></td></tr>')
        parts.append('</table>')
        for key, title, unit in charts[kind]:
            parts.append(series_svg(traces[kind], key, title, unit))
        for name, _ in traces[kind]:
            metrics = dict(results[kind][name]["diagnostics"])
            if "stance" in metrics:
                metrics["stance"] = {f: {k: v for k, v in s.items() if k != "records"} for f, s in metrics["stance"].items()}
            parts.append(f'<details><summary>{name}：定量指标</summary><pre>{html.escape(json.dumps(metrics, ensure_ascii=False, indent=2))}</pre></details>')
    parts.append('<h2>个人验收</h2><p>每组选一幅图，标明首次差异出现的时间、修改字段、预期与实际，并写一个可被下一次实验推翻的解释。当前报告记录程序结果；学习闸门由你的解释和独立修改实验核验。</p></html>')
    (output / "report.html").write_text("".join(parts), encoding="utf-8")


def run_suite(robot="all", output=None, render=False):
    output = Path(output) if output else ROOT / "outputs/deep-practice" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    # Fresh directory protects both old baselines and personal runs.
    output.mkdir(parents=True, exist_ok=False)
    actual_commit = subprocess.check_output(["git", "-C", str(lab.CACHE), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != lab.PIN:
        raise RuntimeError(f"Model commit mismatch: {actual_commit}")
    results, traces = {}, {}
    for kind in lab.SOURCES if robot == "all" else [robot]:
        results[kind], traces[kind] = {}, []
        for name, options in SUITES[kind]:
            print(f"RUN {kind}/{name}: {options}", flush=True)
            rows = []
            path = output / kind / name
            result = lab.run(kind, output=path, render=render, options=options,
                             observer=lambda m, d, c: rows.append(observe(kind, m, d, c)))
            with (path / "diagnostics.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            saved = ROOT / "outputs/baseline" / kind / "trajectory.csv"
            identical = (path / "trajectory.csv").read_bytes() == saved.read_bytes() if name == "baseline" and saved.exists() else None
            item = {"changed_options": options, "legacy_result": result, "diagnostics": diagnostic_metrics(kind, rows),
                    "saved_baseline_csv_identical": identical,
                    "note": "diagnostics only; legacy pass does not certify new skill claims"}
            (path / "diagnostic_summary.json").write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
            results[kind][name] = item
            traces[kind].append((name, rows))
            print(f"DONE {kind}/{name}: legacy={result['passed']}, baseline_csv_identical={identical}", flush=True)
    payload = {"date": datetime.datetime.now().astimezone().isoformat(), "platform": platform.platform(),
               "python": sys.version, "mujoco": mujoco.__version__, "numpy": np.__version__,
               "measured_model_commit": actual_commit, "expected_model_commit": lab.PIN,
               "lab_source_sha256": hashlib.sha256((ROOT / "lab.py").read_bytes()).hexdigest(),
               "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "geometry_timestamp_note": "cached geometry/forces precede integrated qpos by one physics step",
               "personal_learning_gate": "pending student explanation and independent modification", "results": results}
    (output / "study.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_html(output, results, traces)
    print(f"REPORT {output / 'report.html'}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["all", *lab.SOURCES], default="all")
    parser.add_argument("--output")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--open-report", action="store_true")
    args = parser.parse_args()
    out = run_suite(args.robot, args.output, args.render)
    if args.open_report:
        webbrowser.open((out / "report.html").resolve().as_uri())
