"""Render a static four-panel scientific overview from saved diagnostic CSVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def overview(root):
    root = Path(root)
    study = json.loads((root / "study.json").read_text(encoding="utf-8"))
    font_file = Path("C:/Windows/Fonts/segoeui.ttf")
    def font(size):
        return ImageFont.truetype(str(font_file), size) if font_file.exists() else ImageFont.load_default(size=size)
    canvas = Image.new("RGB", (1440, 1020), "white")
    draw = ImageDraw.Draw(canvas)
    ink, muted, grid = "#152536", "#506174", "#dae1e8"
    colors = ("#2563eb", "#c2410c", "#15803d")
    draw.text((38, 22), "Project 03 | Single-factor physics experiments", font=font(30), fill=ink)
    draw.text((38, 65), "Measured diagnostic traces; historical success criteria retained separately", font=font(19), fill=muted)
    panels = [("panda", "object_height_m", "Panda: object height", "m"),
              ("go2", "base_x_m", "Go2: base forward position", "m"),
              ("h1", "tilt_deg", "H1: base tilt", "deg"),
              ("allegro", "finger_world_fz_N", "Allegro: signed finger force on ball", "N")]
    for index, (kind, key, title, unit) in enumerate(panels):
        x0, y0 = 38 + (index % 2) * 710, 125 + (index // 2) * 430
        left, right, top, bottom = x0 + 75, x0 + 620, y0 + 65, y0 + 305
        draw.text((x0, y0), title, font=font(23), fill=ink)
        draw.text((x0, y0 + 34), f"y: {unit}  |  x: simulation time (s)", font=font(17), fill=muted)
        series = []
        for name in study["results"][kind]:
            with (root / kind / name / "diagnostics.csv").open(encoding="utf-8", newline="") as stream:
                series.append((name, [(float(r["time_s"]), float(r[key])) for r in csv.DictReader(stream)]))
        ys = [value for _, points in series for _, value in points]
        lo, hi = min(ys), max(ys)
        padding = max((hi - lo) * .08, .005)
        lo, hi = lo - padding, hi + padding
        xmax = max(points[-1][0] for _, points in series)
        def scale(x, y):
            return left + x / xmax * (right - left), bottom - (y - lo) / (hi - lo) * (bottom - top)
        for i in range(5):
            x, y = xmax * i / 4, lo + (hi - lo) * i / 4
            px, _ = scale(x, lo)
            _, py = scale(0, y)
            draw.line((left, py, right, py), fill=grid, width=1)
            draw.text((left - 10, py), f"{y:.3g}", anchor="rm", font=font(17), fill=muted)
            draw.text((px, bottom + 12), f"{x:.1f}", anchor="mt", font=font(17), fill=muted)
        draw.rectangle((left, top, right, bottom), outline=muted, width=1)
        for (name, points), color in zip(series, colors):
            draw.line([scale(x, y) for x, y in points], fill=color, width=2)
        for i, ((name, _), color) in enumerate(zip(series, colors)):
            lx, ly = x0 + 10, y0 + 350 + i * 22
            draw.line((lx, ly + 8, lx + 24, ly + 8), fill=color, width=3)
            draw.text((lx + 34, ly), name, font=font(17), fill=ink)
    target = root / "overview.png"
    canvas.save(target)
    metadata = {"source_study_sha256": hashlib.sha256((root / "study.json").read_bytes()).hexdigest(),
                "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "units": {kind: {"field": key, "unit": unit} for kind, key, _, unit in panels}}
    (root / "figure_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    print(overview(parser.parse_args().directory))
