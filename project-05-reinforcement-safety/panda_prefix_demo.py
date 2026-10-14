"""Show the fixed physical grasp/transport prefix used to reset placement trials."""
import argparse
import hashlib
from pathlib import Path
import time
import mujoco
from PIL import Image, ImageDraw
from panda_residual_env import lab
from tabular_lab import save_json


def run(out=None, live=False):
    m,d,qa,va,_=lab.build("panda")
    c=lab.Controller("panda",m,d,qa,va)
    frames=[];handle=None
    if live:
        from mujoco import viewer
        handle=viewer.launch_passive(m,d)
        handle.cam.lookat[:],handle.cam.distance=[.35,0,.4],1.7
        handle.cam.azimuth,handle.cam.elevation=125,-22
    try:
        for step in range(4500):
            started=time.monotonic()
            if step%5==0:c.step()
            mujoco.mj_step(m,d)
            if out and step%80==0:
                mujoco.mj_forward(m,d)
                if step==0:renderer=mujoco.Renderer(m,width=640,height=480)
                renderer.update_scene(d,camera=lab.camera("panda"))
                im=Image.fromarray(renderer.render().copy())
                ImageDraw.Draw(im).text((12,12),f"Frozen prefix | {c.stage} | t={d.time:.2f}s",fill="white")
                frames.append(im)
            if handle and step%5==0:
                handle.sync()
                if not handle.is_running():break
                time.sleep(max(0,.01-(time.monotonic()-started)))
        if out:
            out.mkdir(parents=True,exist_ok=False)
            frames[0].save(out/"demo.gif",save_all=True,append_images=frames[1:],duration=160,loop=0)
            frames[-1].save(out/"final.png")
            save_json(out/"config.json",{"model_commit":lab.PIN,"physics_steps":4500,"physics_dt_s":.002,"controller_period_s":.01,"scope":"Fixed nominal grasp/transport; actual physical integration; placement begins at9s"})
            save_json(out/"source.json",{"panda_prefix_demo.py":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            renderer.close()
    finally:
        if handle:handle.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path);p.add_argument("--live",action="store_true");a=p.parse_args()
    if not a.output and not a.live:p.error("--output or --live required")
    run(a.output,a.live)
