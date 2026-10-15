"""Command-conditioned Go2 gait: planar velocity/yaw feedback and matched ablations."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
import argparse
from dataclasses import dataclass,asdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import mujoco
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
LAB_FILE=ROOT/"lab.py" if (ROOT/"lab.py").exists() else ROOT.parent/"project-03-robot-platforms/lab.py"
SPEC=importlib.util.spec_from_file_location("go2_nominal_lab",LAB_FILE)
lab=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=lab;SPEC.loader.exec_module(lab)
from tabular_lab import save_csv,save_json

SCENARIOS=("slow","fast","left","right","step_stop")
CONTROLLERS=("fixed","feedforward","feedback","wrong_sign")


@dataclass(frozen=True)
class Config:
    dt:float=.02
    horizon:int=500
    freq_hz:float=1.8
    stride_scale_s:float=.45
    velocity_gain:float=2.
    yaw_gain:float=2.
    filter_alpha:float=.12
    stride_limit_m:float=.09
    stride_slew_m:float=.002
    lift_m:float=.07
    kp:float=100.
    kd:float=3.5


def command(scenario,t):
    if scenario not in SCENARIOS:raise ValueError("unknown scenario")
    if scenario=="slow":return np.array([.08,0.])
    if scenario=="fast":return np.array([.16,0.])
    if scenario=="left":return np.array([.12,.25])
    if scenario=="right":return np.array([.12,-.25])
    return np.array([.08 if t<3 else .16 if t<6 else 0.,0.])


def world_to_body(rotation,velocity):return np.asarray(rotation).reshape(3,3).T@np.asarray(velocity)


def stride_reference(mode,cmd,measured,cfg):
    if mode not in CONTROLLERS:raise ValueError("unknown controller")
    if mode=="fixed":return np.full(4,.055)
    if np.max(np.abs(cmd))<1e-12:return np.zeros(4)
    correction=np.array([cfg.velocity_gain,cfg.yaw_gain])*(cmd-measured)
    desired=cmd+(correction if mode=="feedback" else -correction if mode=="wrong_sign" else 0.)
    # FL,FR,RL,RR: v_x(foot)=v_x(body)-omega_z*y(foot).
    return cfg.stride_scale_s*(desired[0]-desired[1]*np.array([.095,-.095,.095,-.095]))


class Go2:
    def __init__(self,cfg=Config()):
        if not np.isfinite(list(asdict(cfg).values())).all() or cfg.velocity_gain<0 or cfg.yaw_gain<0 or cfg.dt!=.02 or cfg.horizon!=500 or not 0<cfg.filter_alpha<=1 or cfg.stride_limit_m<=0 or cfg.stride_slew_m<=0 or cfg.kp<=0 or cfg.kd<0:raise ValueError("invalid config or changed time protocol")
        self.cfg=cfg;self.m,self.d,self.qa,self.va,self.model_hash=lab.build("go2")
        self.base=self.m.body("base").id;self.home=self.d.qpos[self.qa].copy()
        self.feet=[self.m.geom(k).id for k in ("FL","FR","RL","RR")]
        self.floor=self.m.geom("floor").id
        self.reset(0)
        self.renderer=None

    def reset(self,seed):
        rng=np.random.default_rng(seed);mujoco.mj_resetData(self.m,self.d)
        self.d.qpos[:7]=[rng.uniform(-.02,.02),rng.uniform(-.02,.02),.30,1,0,0,0]
        self.d.qpos[self.qa]=self.home+rng.normal(0,.005,12)
        self.d.qvel[:2]=rng.normal(0,.01,2)
        for _ in range(1000):
            self.d.ctrl[:]=np.clip(self.cfg.kp*(self.home-self.d.qpos[self.qa])-self.cfg.kd*self.d.qvel[self.va],self.m.actuator_ctrlrange[:,0],self.m.actuator_ctrlrange[:,1])
            mujoco.mj_step(self.m,self.d)
        mujoco.mj_forward(self.m,self.d)
        self.start_xy=self.d.qpos[:2].copy();self.steps=0;self.rows=[];self.filtered=np.zeros(2);self.stride=np.zeros(4);self.ended=False
        return self.measure()

    def measure(self):
        rotation=self.d.xmat[self.base].reshape(3,3)
        spatial=np.zeros(6);mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_BODY,self.base,spatial,0)
        linear=world_to_body(rotation,spatial[3:])
        return np.array([linear[0],spatial[2]])

    def foot_contacts(self):
        contacts=np.zeros(4,dtype=bool);slips=[]
        for i,c in enumerate(self.d.contact):
            for j,gid in enumerate(self.feet):
                if gid not in (c.geom1,c.geom2) or self.floor not in (c.geom1,c.geom2):continue
                force=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,i,force)
                if abs(force[0])<=1.:continue
                contacts[j]=True
                velocity=np.zeros(6);mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_GEOM,gid,velocity,0)
                at_contact=velocity[3:]+np.cross(velocity[:3],c.pos-self.d.geom_xpos[gid])
                slips.append(float(np.linalg.norm(at_contact[:2])))
        return contacts,slips

    def step(self,mode,scenario):
        if self.ended:raise RuntimeError("reset required")
        t=self.steps*self.cfg.dt;cmd=command(scenario,t);measurement=self.measure()
        self.filtered+=(measurement-self.filtered)*self.cfg.filter_alpha
        requested=stride_reference(mode,cmd,self.filtered,self.cfg)
        clipped=np.clip(requested,-self.cfg.stride_limit_m,self.cfg.stride_limit_m)
        self.stride+=np.clip(clipped-self.stride,-self.cfg.stride_slew_m,self.cfg.stride_slew_m)
        q=self.home.copy();ramp=lab.smooth(t/1.5)
        moving=1. if mode=="fixed" else min(1.,abs(cmd[0])/.04+abs(cmd[1])/.2)
        for i,offset in enumerate([0,np.pi,np.pi,0]):
            phase=2*np.pi*self.cfg.freq_hz*t+offset
            x=ramp*self.stride[i]*np.sin(phase)
            z=-.265+ramp*moving*self.cfg.lift_m*max(0,np.cos(phase))
            knee=-np.arccos(np.clip((x*x+z*z-2*.213**2)/(2*.213**2),-1,1))
            q[3*i+1]=np.arctan2(-x,-z)-knee/2;q[3*i+2]=knee
        torque_peak=0.;saturated=0
        for _ in range(10):
            raw=self.cfg.kp*(q-self.d.qpos[self.qa])-self.cfg.kd*self.d.qvel[self.va]
            torque=np.clip(raw,self.m.actuator_ctrlrange[:,0],self.m.actuator_ctrlrange[:,1]);self.d.ctrl[:]=torque
            saturated+=int(np.any(np.abs(raw-torque)>1e-9));torque_peak=max(torque_peak,float(np.max(np.abs(torque))))
            mujoco.mj_step(self.m,self.d)
        mujoco.mj_forward(self.m,self.d);self.steps+=1
        measured=self.measure();rotation=self.d.xmat[self.base].reshape(3,3)
        tilt=float(np.arccos(np.clip(rotation[2,2],-1,1)));height=float(self.d.qpos[2])
        fallen=bool(height<.18 or tilt>.75 or not np.isfinite(self.d.qpos).all())
        contacts,slips=self.foot_contacts()
        row={"step":self.steps,"time_s":self.steps*self.cfg.dt,"command_v_m_s":float(cmd[0]),"command_yaw_rad_s":float(cmd[1]),
             "v_body_m_s":float(measured[0]),"yaw_rate_rad_s":float(measured[1]),"filtered_v_m_s":float(self.filtered[0]),"filtered_yaw_rad_s":float(self.filtered[1]),
             "height_m":height,"tilt_rad":tilt,"yaw_rad":float(np.arctan2(rotation[1,0],rotation[0,0])),"base_xy_m":self.d.qpos[:2].tolist(),
             "requested_stride_m":requested.tolist(),"executed_stride_m":self.stride.tolist(),"stride_intervention":bool(np.any(abs(requested-self.stride)>1e-9)),
             "foot_contacts":contacts.tolist(),"contact_slip_m_s":slips,"torque_peak_nm":torque_peak,"saturated_substeps":saturated,"fallen":fallen,
             "joint_reference_rad":q.tolist(),"joint_position_rad":self.d.qpos[self.qa].tolist(),"joint_velocity_rad_s":self.d.qvel[self.va].tolist(),"applied_torque_nm":self.d.ctrl.tolist()}
        window=self.rows[-24:]+[row]
        row["cycle_v_m_s"]=float(np.mean([r["v_body_m_s"] for r in window]))
        row["cycle_yaw_rad_s"]=float(np.mean([r["yaw_rate_rad_s"] for r in window]))
        self.rows.append(row);self.ended=fallen or self.steps==self.cfg.horizon
        return row

    def assess(self,scenario):
        # Fixed, predeclared windows exclude startup and command-step transients.
        rows=[r for r in self.rows if r["time_s"]>=2 and (scenario!="step_stop" or (r["time_s"]<3 or 4<=r["time_s"]<6 or r["time_s"]>=7))]
        ev=np.array([r["cycle_v_m_s"]-r["command_v_m_s"] for r in rows]);ew=np.array([r["cycle_yaw_rad_s"]-r["command_yaw_rad_s"] for r in rows])
        raw_v=float(np.sqrt(np.mean([(r["v_body_m_s"]-r["command_v_m_s"])**2 for r in rows]))) if rows else None
        raw_w=float(np.sqrt(np.mean([(r["yaw_rate_rad_s"]-r["command_yaw_rad_s"])**2 for r in rows]))) if rows else None
        vrmse=float(np.sqrt(np.mean(ev**2))) if rows else None;wrmse=float(np.sqrt(np.mean(ew**2))) if rows else None
        slips=[s for r in rows for s in r["contact_slip_m_s"]]
        complete=self.steps==self.cfg.horizon
        criteria={"complete":complete,"upright":not any(r["fallen"] for r in self.rows),"speed_tracking":vrmse is not None and vrmse<=.07,"yaw_tracking":wrmse is not None and wrmse<=.18,
                  "raw_oscillation":raw_v is not None and raw_v<=.2 and raw_w<=.5,"slip":bool(slips) and float(np.mean(slips))<=.20,"torque_saturation":sum(r["saturated_substeps"] for r in self.rows)/(10*len(self.rows))<=.1}
        return {"success":bool(all(criteria.values())),"criteria":criteria,"steps":self.steps,"speed_rmse_m_s":vrmse,"yaw_rmse_rad_s":wrmse,
                "mean_speed_m_s":float(np.mean([r["v_body_m_s"] for r in rows])) if rows else None,"mean_yaw_rate_rad_s":float(np.mean([r["yaw_rate_rad_s"] for r in rows])) if rows else None,
                "displacement_m":float(np.linalg.norm(self.d.qpos[:2]-self.start_xy)),"min_height_m":min(r["height_m"] for r in self.rows),"max_tilt_rad":max(r["tilt_rad"] for r in self.rows),
                "raw_speed_rmse_m_s":raw_v,"raw_yaw_rmse_rad_s":raw_w,
                "mean_contact_slip_m_s":float(np.mean(slips)) if slips else None,"stride_interventions":sum(r["stride_intervention"] for r in self.rows)}

    def render(self):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.m,width=640,height=480)
        camera=lab.camera("go2");camera.lookat[:2]=self.d.qpos[:2]
        self.renderer.update_scene(self.d,camera=camera);return self.renderer.render().copy()

    def close(self):
        if self.renderer:self.renderer.close()


def episode(env,mode,scenario,seed,render=False,viewer=False):
    env.reset(seed);frames=[];handle=None
    if viewer:
        from mujoco import viewer as mv
        handle=mv.launch_passive(env.m,env.d);handle.cam.distance=1.8;handle.cam.azimuth=125;handle.cam.elevation=-25
    try:
        for i in range(env.cfg.horizon):
            tick=time.monotonic();row=env.step(mode,scenario)
            if render and i%5==0:
                im=Image.fromarray(env.render());d=ImageDraw.Draw(im)
                d.text((10,10),f"{mode} {scenario} t={row['time_s']:.2f}s",fill="white")
                d.text((10,28),f"v {row['v_body_m_s']:.2f}/{row['command_v_m_s']:.2f} m/s yaw {row['yaw_rate_rad_s']:.2f}/{row['command_yaw_rad_s']:.2f} rad/s",fill="white");frames.append(im)
            if handle:
                handle.cam.lookat[:]=env.d.qpos[:3];handle.sync()
                if not handle.is_running():break
                time.sleep(max(0,.02-(time.monotonic()-tick)))
            if env.ended:break
        return {"controller":mode,"scenario":scenario,"seed":seed,**env.assess(scenario)},list(env.rows),frames
    finally:
        if handle:handle.close()


def run(out,cfg,seeds,render=True):
    if out.exists():raise FileExistsError("new output folder required")
    out.mkdir(parents=True);source=out/"source";source.mkdir()
    for name in ("go2_feedback.py","tabular_lab.py"):(source/name).write_bytes((ROOT/name).read_bytes())
    (source/"lab.py").write_bytes(LAB_FILE.read_bytes())
    assets={p.relative_to(lab.CACHE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (lab.CACHE/"unitree_go2").rglob("*") if p.is_file()}
    assert subprocess.check_output(["git","-C",str(lab.CACHE),"rev-parse","HEAD"],text=True).strip()==lab.PIN
    save_json(out/"protocol.json",{"date":"2026-10-15","config":asdict(cfg),"seeds":seeds,"scenarios":list(SCENARIOS),"controllers":list(CONTROLLERS),"model_commit":lab.PIN,"model_assets_sha256":assets,
         "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},"dependencies":{"numpy":np.__version__,"mujoco":mujoco.__version__},
         "acceptance":{"cycle_average_samples":25,"cycle_average_period_s":.5,"speed_rmse_m_s":.07,"yaw_rmse_rad_s":.18,"raw_speed_rmse_m_s":.2,"raw_yaw_rmse_rad_s":.5,"mean_contact_slip_m_s":.20,"saturation_fraction":.1,"fall_height_m":.18,"fall_tilt_rad":.75},
         "development_seeds":[0],"reset_scope":"Initial XY uniform+-20mm, joint noise std.005rad and horizontal velocity std.01m/s, followed by2s joint-PD settle; flat floor",
         "scope":"Analytic feedback engineering baseline, no RL training; matched reset seeds across controllers; fixed flat floor, twelve-joint PD and scripted gait; thresholds frozen before evaluation"})
    cases=[];env=Go2(cfg)
    with gzip.open(out/"trajectories.jsonl.gz","wt",encoding="utf-8") as stream:
        try:
            for mode in CONTROLLERS:
                for scenario in SCENARIOS:
                    for seed in seeds:
                        record,rows,frames=episode(env,mode,scenario,seed,render and seed==seeds[0] and scenario in ("fast","left","step_stop") and mode in ("fixed","feedback"))
                        cases.append(record);stream.write(json.dumps({"case":record,"trajectory":rows})+"\n")
                        if frames:
                            demos=out/"demos";demos.mkdir(exist_ok=True);name=f"{mode}-{scenario}"
                            frames[0].save(demos/f"{name}.gif",save_all=True,append_images=frames[1:],duration=100,loop=0);frames[-1].save(demos/f"{name}.png")
                            save_csv(demos/f"{name}.csv",rows)
                        print(f"{mode} {scenario} seed{seed} success={record['success']} v={record['speed_rmse_m_s']} yaw={record['yaw_rmse_rad_s']}",flush=True)
        finally:env.close()
    save_json(out/"cases.json",cases)
    aggregate={mode:{"successes":sum(c["success"] for c in cases if c["controller"]==mode),"cases":sum(c["controller"]==mode for c in cases),"falls":sum(not c["criteria"]["upright"] for c in cases if c["controller"]==mode)} for mode in CONTROLLERS}
    save_json(out/"summary.json",{"aggregate":aggregate,"evaluation_cases":len(cases),"retained_failure_cases":sum(not c["success"] for c in cases),"task_status":"Per-scenario physical outcomes retained; learner evidence and RL residual training assessed separately"})
    save_json(out/"manifest.json",[{"path":p.relative_to(out).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.rglob("*")) if p.is_file() and "__pycache__" not in p.parts])


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path);p.add_argument("--demo",action="store_true");p.add_argument("--controller",choices=CONTROLLERS,default="feedback");p.add_argument("--scenario",choices=SCENARIOS,default="left");p.add_argument("--seed",type=int,default=0);p.add_argument("--seeds",type=int,nargs="+",default=[1000000,1000001,1000002,1000003,1000004]);p.add_argument("--velocity-gain",type=float,default=2.);p.add_argument("--yaw-gain",type=float,default=2.);p.add_argument("--no-render",action="store_true")
    a=p.parse_args();cfg=Config(velocity_gain=a.velocity_gain,yaw_gain=a.yaw_gain)
    if a.demo:
        e=Go2(cfg)
        try:print(json.dumps(episode(e,a.controller,a.scenario,a.seed,viewer=True)[0],indent=2))
        finally:e.close()
    elif a.output:
        if len(a.seeds)!=len(set(a.seeds)) or min(a.seeds)<0:p.error("unique nonnegative seeds")
        run(a.output,cfg,a.seeds,not a.no_render)
    else:p.error("--output or --demo required")
