from pathlib import Path
from PIL import Image, ImageDraw
from lab import SOURCES, run
from panel import FAILURE

root = Path(__file__).parent
for kind in SOURCES:
    print(kind, run(kind, root / "outputs/baseline" / kind, render=True), flush=True)
    print(kind, "failure", run(kind, root / "outputs/failures" / kind, render=True, options=FAILURE[kind]), flush=True)

# Scientific simulation output combined into a 2x2 overview, synchronized in wall-clock playback.
gifs = [Image.open(root / "outputs/baseline" / kind / "demo.gif") for kind in SOURCES]
frames = []
for i in range(max(g.n_frames for g in gifs)):
    board = Image.new("RGB", (960, 720), (14, 22, 33))
    for j, gif in enumerate(gifs):
        gif.seek(min(i, gif.n_frames - 1))
        board.paste(gif.convert("RGB").resize((480, 360)), ((j % 2) * 480, (j // 2) * 360))
    frames.append(board)
frames[0].save(root / "outputs/four-robots.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
frames[50].save(root / "outputs/four-robots.png")
for gif in gifs:
    gif.close()
