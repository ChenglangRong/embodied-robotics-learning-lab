"""Phase-conditioned reference BC on four existing MuJoCo robot platforms.

The network supplies joint references; existing servos execute them. Student
rollouts never query the reference expert. Wrist/tray motion is an explicit
external fixture for Allegro. Every result states these scope boundaries.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import copy
import datetime
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import mujoco
from PIL import Image, ImageDraw
from learning import ROOT, MLP, fit, mse, save_json, write_csv

PLATFORM_ROOT=ROOT.parent/"project-03-robot-platforms"
sys.path.insert(0,str(PLATFORM_ROOT))
import lab
import deep_practice

KINDS=("panda","go2","h1","allegro")
DURATION={"panda":14.,"go2":12.,"h1":10.,"allegro":11.}
TITLES={"panda":"Panda：抓取—搬运—放置", "go2":"Go2：步态参考模仿", "h1":"H1：平衡—下蹲—抬臂", "allegro":"Allegro：多指接触保持"}
SCOPE={"panda":"Joint-reference BC with a separate gripper command; position servos retained; simulator object pose and task clock observed.",
       "go2":"Joint-reference BC; torque PD retained; flat terrain; phase and commanded gait parameters observed.",
       "h1":"Joint-reference BC including expert balance corrections; torque PD and bias compensation retained; double support, squat and arm motion.",
       "allegro":"Finger-reference BC; palm-assisted caging; prescribed mocap wrist and withdrawing tray are external fixtures."}


def specifications(kind,n,seed,split):
    rng=np.random.default_rng(seed)
    result=[]
    for i in range(n):
        options=copy.deepcopy(lab.DEFAULT_OPTIONS[kind])
        spec={"id":f"{kind}-{split}-{i:03d}","split":split,"seed":seed*1000+i,
              "q_noise":.001 if kind=="h1" else .002,"options":options,"push_N":0.,"mass_scale":1.}
        if kind=="panda":
            delta=rng.uniform(-.008,.008,4)
            spec.update(object_xy=(np.array([.5,.12])+delta[:2]).tolist(),
                        goal_xy=(np.array([.5,-.19])+delta[2:]).tolist())
        elif kind=="go2":
            options.update(amp=float(rng.uniform(.050,.060)),freq=float(rng.uniform(1.7,1.9)),lift=float(rng.uniform(.065,.075)))
        elif kind=="h1":options["bend"]=float(rng.uniform(.14,.16))
        else:
            options["radius"]=float(rng.uniform(.039,.041))
            options["object_pos"]=[float(.03+rng.uniform(-.001,.001)),float(rng.uniform(-.001,.001)),.215]
        if split=="ood":
            if kind in ("panda","allegro"):spec["mass_scale"]=1.5
            else:spec["push_N"]=12. if kind=="h1" else 8.
        result.append(spec)
    return result


def protocol(kind,mode):
    sizes=(6,2,2,2) if mode=="quick" else (16,4,6,6)
    specs={s:specifications(kind,n,seed,s) for s,n,seed in zip(("train","validation","test","ood"),sizes,(111,222,333,444))}
    return {"kind":kind,"mode":mode,"specs":specs,"control_dt_s":.01,"physics_dt_s":.002,
            "sample_dt_s":.02,"duration_s":DURATION[kind],"scope":SCOPE[kind],
            "ood":"Object mass x1.5 (Panda/Allegro) or base x-direction force during 4.5–4.8 s (Go2/H1). One factor per platform.",
            "checkpoint":"Lowest validation normalized reference-action MSE; no test or OOD selection.",
            "seed_scope":"Quick: one training seed. Full: five training seeds sharing a frozen dataset/test set.",
            "criteria":"Fixed project03 thresholds plus declared task-process checks; failed experts retained; no threshold tuning after results."}


class ReferenceEnv:
    def __init__(self,kind,spec):
        self.kind,self.spec=kind,copy.deepcopy(spec)
        self.options=copy.deepcopy(spec["options"])
        self.m,self.d,self.qa,self.va,self.model_hash=lab.build(kind,self.options)
        rng=np.random.default_rng(spec["seed"])
        self.d.qpos[self.qa]+=rng.normal(0,spec["q_noise"],len(self.qa))
        if kind in ("panda","allegro"):
            oid=self.m.body("object").id
            self.m.body_mass[oid]*=spec["mass_scale"]
            self.m.body_inertia[oid]*=spec["mass_scale"]
            if kind=="panda":
                adr=int(self.m.joint("object_free").qposadr[0])
                self.d.qpos[adr:adr+2]=spec["object_xy"]
                self.m.site("goal").pos[:2]=spec["goal_xy"]
        # mj_setConst uses the model reference configuration internally. Restore
        # the explicitly randomized reset state before observing/constructing the expert.
        reset_qpos,reset_qvel,reset_ctrl=self.d.qpos.copy(),self.d.qvel.copy(),self.d.ctrl.copy()
        reset_mocap_pos,reset_mocap_quat=self.d.mocap_pos.copy(),self.d.mocap_quat.copy()
        mujoco.mj_setConst(self.m,self.d)
        self.d.qpos[:],self.d.qvel[:],self.d.ctrl[:]=reset_qpos,reset_qvel,reset_ctrl
        self.d.mocap_pos[:],self.d.mocap_quat[:]=reset_mocap_pos,reset_mocap_quat
        mujoco.mj_forward(self.m,self.d)
        self.c=lab.Controller(kind,self.m,self.d,self.qa,self.va,self.options)
        self.home=self.d.qpos[self.qa].copy()
        self.initial_height=float(self.d.qpos[2]) if kind in ("go2","h1") else float(self.d.xpos[self.m.body("object").id,2])
        if kind=="panda":
            points=[]
            for t,xyz,g in self.c.waypoints:
                xy=spec["object_xy"] if t<7 else spec["goal_xy"]
                points.append((t,np.r_[xy,xyz[2]],g))
            self.c.waypoints=points
            self.action_low=np.r_[self.m.actuator_ctrlrange[:7,0],0.]
            self.action_high=np.r_[self.m.actuator_ctrlrange[:7,1],1.]
            self.hold=np.r_[self.home[:7],1.]
        else:
            ids=self.m.actuator_trnid[:,0]
            self.action_low=self.m.jnt_range[ids,0].copy()
            self.action_high=self.m.jnt_range[ids,1].copy()
            self.hold=self.home.copy()
        self.expert_calls=0
        self.last_action=self.hold.copy()
        self.fixture()

    def fixture(self):
        if self.kind!="allegro":return
        t=self.d.time
        wrist=int(self.m.body("palm").mocapid[0])
        tray=int(self.m.body("tray").mocapid[0])
        self.d.mocap_pos[wrist,2]=.15+.16*lab.smooth((t-4)/3)
        support=self.options["object_pos"][2]-self.options["radius"]-.006
        self.d.mocap_pos[tray,2]=support-.3*lab.smooth((t-3.5)/1)
        angle=self.options.get("wrist_tilt",0.)*lab.smooth((t-7)/1)
        mujoco.mju_mulQuat(self.d.mocap_quat[wrist],np.array([np.cos(angle/2),0,np.sin(angle/2),0]),self.c.wrist_home[wrist])

    def observation(self):
        phase=float(self.d.time/DURATION[self.kind])
        # A disclosed clock basis makes this first lesson phase-conditioned.
        features=[phase,*np.exp(-((phase-np.linspace(0,1,24))/.06)**2),
                  *self.d.qpos[self.qa],*self.d.qvel[self.va],*self.home]
        if self.kind in ("panda","allegro"):
            oid=self.m.body("object").id
            obj=self.d.xpos[oid]
            if self.kind=="panda":
                ref=self.d.site_xpos[self.m.site("tcp").id]
                goal=np.r_[self.spec["goal_xy"],.275]
                features.extend([*(obj-ref),*(goal-obj),*self.spec["object_xy"],*self.spec["goal_xy"]])
            else:
                palm=self.m.body("palm").id
                rel=self.d.xmat[palm].reshape(3,3).T@(obj-self.d.xpos[palm])
                features.extend([*rel,self.options["radius"],*self.d.mocap_pos[int(self.m.body("palm").mocapid[0])]])
            velocity=np.zeros(6)
            mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_BODY,oid,velocity,0)
            features.extend(velocity.tolist())
        else:
            body=self.m.body("base" if self.kind=="go2" else "pelvis").id
            features.extend([self.d.qpos[2],*self.d.qvel[:6],*(self.d.xmat[body].reshape(3,3).T@np.array([0.,0.,-1.]))])
            if self.kind=="go2":features.extend([self.options[k] for k in ("amp","freq","lift")])
            else:
                center=.5*(self.d.xpos[self.m.body("left_ankle_link").id]+self.d.xpos[self.m.body("right_ankle_link").id])
                features.extend([*(self.d.subtree_com[body]-center),self.options["bend"]])
        return np.array(features,dtype=float)

    def expert(self):
        self.expert_calls+=1
        self.c.step()
        action=np.r_[self.c.target[:7],self.d.ctrl[7]/255.] if self.kind=="panda" else self.c.target.copy()
        return np.clip(action,self.action_low,self.action_high)

    def apply(self,action):
        requested=np.asarray(action,dtype=float)
        if requested.shape!=self.action_low.shape or not np.isfinite(requested).all():raise ValueError("Invalid reference action")
        action=np.clip(requested,self.action_low,self.action_high)
        self.last_action=action.copy()
        if self.kind=="panda":
            self.d.ctrl[:7]=action[:7]
            self.d.ctrl[7]=255*action[7]
            self.c.target=np.r_[action[:7],.04*action[7]]
        elif self.kind=="allegro":
            self.d.ctrl[:]=action
            self.c.target=action.copy()
        else:
            if self.kind=="go2":kp=np.full(12,self.options["kp"]); kd=np.full(12,self.options["kd"]); bias=0.
            else:
                kp=np.array([160,240,300,300,100]*2+[180]+[60]*8)
                kd=2*np.sqrt(kp)*.6
                bias=self.d.qfrc_bias[self.va]
            torque=kp*(action-self.d.qpos[self.qa])-kd*self.d.qvel[self.va]+bias
            self.d.ctrl[:]=np.clip(torque,self.m.actuator_ctrlrange[:,0],self.m.actuator_ctrlrange[:,1])
            self.c.target=action.copy()
        return bool(np.any(requested!=action))

    def advance(self,action):
        clipped=self.apply(action)
        for _ in range(5):
            self.fixture()
            self.d.xfrc_applied[:]=0
            if self.kind in ("go2","h1") and 4.5<=self.d.time<4.8:
                body=self.m.body("base" if self.kind=="go2" else "pelvis").id
                self.d.xfrc_applied[body,0]=self.spec["push_N"]
            mujoco.mj_step(self.m,self.d)
        self.fixture()
        mujoco.mj_forward(self.m,self.d)
        return clipped


class ReferenceMLP(MLP):
    def __init__(self,seed,mean,std,action_mean,action_std,width=64):
        self.seed,self.width=seed,width
        self.mean,self.std=np.asarray(mean).copy(),np.asarray(std).copy()
        self.action_mean,self.action_std=np.asarray(action_mean).copy(),np.asarray(action_std).copy()
        rng=np.random.default_rng(seed)
        dims=[len(mean),width,width,len(action_mean)]
        self.params={}
        for i,(n,m) in enumerate(zip(dims,dims[1:])):
            self.params[f"W{i}"]=rng.normal(0,np.sqrt(2/(n+m)),(n,m))
            self.params[f"b{i}"]=np.zeros(m)
        self.m={k:np.zeros_like(v) for k,v in self.params.items()}
        self.v=copy.deepcopy(self.m)
        self.t=0

    def action(self,obs):return self.forward(obs)[0]*self.action_std+self.action_mean

    def save(self,path):
        np.savez_compressed(path,**self.params,mean=self.mean,std=self.std,action_mean=self.action_mean,
                            action_std=self.action_std,seed=self.seed,width=self.width)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            model=cls(int(d["seed"]),d["mean"],d["std"],d["action_mean"],d["action_std"],int(d["width"]))
            model.params={k:d[k].copy() for k in model.params}
        return model


class PhaseBC:
    """A disclosed clock-only, piecewise-linear BC reference baseline.

    At each sampled phase, minimize squared reference error by averaging only
    training-episode labels. Interpolation supplies intermediate control steps.
    This baseline has no state-feedback/command-parameter adaptation.
    """
    def __init__(self,phase,actions):self.phase,self.actions=np.asarray(phase),np.asarray(actions)

    @classmethod
    def fit(cls,data,duration):
        times=np.unique(data["time_s"])
        actions=np.stack([data["action"][data["time_s"]==t].mean(0) for t in times])
        return cls(times/duration,actions)

    def action(self,obs):return np.array([np.interp(obs[0],self.phase,self.actions[:,i]) for i in range(self.actions.shape[1])])

    def save(self,path):np.savez_compressed(path,phase=self.phase,actions=self.actions)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:return cls(data["phase"].copy(),data["actions"].copy())


def assess(env,rows,diagnostics,completed):
    kind=env.kind
    old=lab.assess(kind,rows) if completed else {"passed":False,"metrics":{}}
    criteria={"original_project03":old["passed"]}
    metrics=old["metrics"]
    if completed and kind=="panda":
        late=np.array([r[:8] for r in rows if r[0]>13],dtype=float)
        error=float(np.linalg.norm(late[:,1:3].mean(0)-env.spec["goal_xy"]))
        lifted=[r for r in diagnostics if 7<r["time_s"]<9]
        released=[r for r in diagnostics if r["time_s"]>12]
        metrics["placement_error_m"]=error
        # Override only the original fixed goal; thresholds and process checks are frozen.
        criteria["original_project03"]=metrics["max_height_m"]>.38 and error<.035 and abs(metrics["final_height_m"]-.275)<.015
        criteria["two_finger_transport"]=bool(lifted) and np.mean([r["distinct_contacting_fingers"]>=2 for r in lifted])>.6
        criteria["release"]=bool(released) and np.mean([r["finger_contacts"]==0 for r in released])>.6
    if completed and kind in ("panda","allegro"):
        late=[r for r in diagnostics if (7<r["time_s"]<9 if kind=="panda" else r["time_s"]>8)]
        rel=np.array([[r[f"object_relative_{axis}_m"] for axis in "xyz"] for r in late])
        drift=float(np.linalg.norm(rel-rel[0],axis=1).max())
        metrics["relative_object_drift_m"]=drift
        criteria["relative_object_retention"]=drift<(.04 if kind=="panda" else .03)
        metrics["late_finger_world_fz_N"]=float(np.mean([r["finger_world_fz_N"] for r in late]))
        metrics["late_palm_world_fz_N"]=float(np.mean([r["palm_world_fz_N"] for r in late]))
    if completed and kind=="h1":
        before=[r[4] for r in rows if 1<r[0]<2]
        after=[r[4] for r in rows if r[0]>9]
        metrics["recovery_height_error_m"]=float(abs(np.mean(before)-np.mean(after)))
        criteria["recover_height"]=metrics["recovery_height_error_m"]<.05
    if kind=="go2":
        speeds=[r[f"{foot}_contact_tangent_speed_m_s"] for r in diagnostics for foot in ("FL","FR","RL","RR") if 2<r["time_s"]<9 and r[f"{foot}_contact"]]
        metrics["mean_contact_tangent_speed_m_s"]=float(np.mean(speeds)) if speeds else None
    if kind in ("go2","h1"):
        metrics["mean_saturated_actuator_fraction"]=float(np.mean([r["saturated_actuator_fraction"] for r in diagnostics]))
    return {"passed":bool(completed and all(criteria.values())),"criteria":{k:bool(v) for k,v in criteria.items()},"metrics":metrics}


def episode(kind,spec,policy="expert",model=None,output=None,render=False,viewer=False,record_labels=False):
    env=ReferenceEnv(kind,spec)
    observations,actions,next_observations,pre_qpos,next_qpos,times=[],[],[],[],[],[]
    rows,diagnostics,frames=[],[],[]
    renderer=mujoco.Renderer(env.m,height=300,width=400) if render else None
    cam=lab.camera(kind)
    view=None
    if viewer:
        from mujoco import viewer as mjviewer
        view=mjviewer.launch_passive(env.m,env.d)
        view.cam.lookat[:],view.cam.distance=cam.lookat,cam.distance
        view.cam.azimuth,view.cam.elevation=cam.azimuth,cam.elevation
    started=time.perf_counter()
    initial_obs=env.observation()
    clipped_count=0
    expected_steps=int(round(DURATION[kind]/.01))
    completed=False
    for step in range(expected_steps):
        if view and not view.is_running():break
        obs=env.observation()
        pre=env.d.qpos.copy()
        if policy=="expert":action=env.expert()
        elif policy=="hold":action=env.hold.copy()
        elif policy in ("bc","phase_bc"):action=model.action(obs)
        elif policy=="frozen_state":
            ablated=initial_obs.copy()
            ablated[:25]=obs[:25]
            action=model.action(ablated)
        else:raise ValueError(policy)
        clipped_count+=env.advance(action)
        if not np.isfinite(env.d.qpos).all() or abs(env.d.time-(step+1)*.01)>1e-6:break
        # Sparse training rows remain aligned: obs and expert reference precede integration.
        if step%2==0:
            if record_labels:
                if policy!="expert":raise ValueError("This collector labels expert rollouts only")
                observations.append(obs.copy()); actions.append(env.last_action.copy()); next_observations.append(env.observation())
                pre_qpos.append(pre); next_qpos.append(env.d.qpos.copy()); times.append(step*.01)
            env.c.stage=policy
            rows.append(lab.sample(kind,env.m,env.d,env.c))
            diagnostic=deep_practice.observe(kind,env.m,env.d,env.c)
            diagnostic["cached_geometry_time_s"]=float(env.d.time)
            diagnostics.append(diagnostic)
        if renderer and step%10==0:
            if kind=="go2":cam.lookat[:2]=env.d.qpos[:2]
            renderer.update_scene(env.d,camera=cam)
            frame=Image.fromarray(renderer.render())
            draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,400,29),fill="#142231")
            draw.text((8,8),f"{kind} | {policy} | {env.d.time:.2f}s",fill="white")
            frames.append(frame.quantize(colors=128))
        if view:
            if step%2==0:view.sync()
            remaining=started+env.d.time-time.perf_counter()
            if remaining>0:time.sleep(min(.02,remaining))
        completed=step==expected_steps-1
    if renderer:renderer.close()
    if view:view.close()
    result={**assess(env,rows,diagnostics,completed),"spec":spec,"policy":policy,"completed":completed,
            "expert_calls":env.expert_calls,"control_steps":step+1,"base_model_xml_sha256":env.model_hash,
            "action_clipped_fraction":clipped_count/(step+1),"scope":SCOPE[kind],
            "termination":"completed" if completed else "interrupted_or_invalid_physics"}
    if policy=="phase_bc":result["policy_scope"]="Clock-only training-reference average and interpolation; no state feedback or gait-parameter adaptation"
    data={"obs":np.array(observations),"action":np.array(actions),"next_obs":np.array(next_observations),
          "pre_qpos":np.array(pre_qpos),"next_qpos":np.array(next_qpos),"time_s":np.array(times),
          "episode_id":np.array([spec["id"]]*len(observations))}
    if output:
        output=Path(output); output.mkdir(parents=True,exist_ok=False)
        header=["time_s","x_m","y_m","z_m","height_m","tilt_rad","contacts","joint_rmse_rad","policy",
                "contacting_fingers","normal_contact_force_N",*[f"foot_{x}_contact" for x in ("FL","FR","RL","RR")],
                *[f"foot_{x}_clearance_m" for x in ("FL","FR","RL","RR")],
                *[f"joint_{i}_q" for i in range(len(env.qa))],*[f"joint_{i}_dq" for i in range(len(env.va))],
                *[f"actuator_{i}_ctrl" for i in range(env.m.nu)],*[f"joint_{i}_reference" for i in range(len(env.qa))]]
        write_csv(output/"trajectory.csv",[dict(zip(header,r)) for r in rows])
        write_csv(output/"diagnostics.csv",diagnostics)
        save_json(output/"summary.json",result)
        if frames:
            frames[0].save(output/"demo.gif",save_all=True,append_images=frames[1:],duration=100,loop=0)
            frames[len(frames)//2].save(output/"preview.png")
    return result,data


def collect(kind,specs):
    datasets,results=[],[]
    for index,spec in enumerate(specs):
        print(f"  EXPERT {kind} {spec['split']} {index+1}/{len(specs)}",flush=True)
        result,data=episode(kind,spec,record_labels=True)
        if not result["completed"]:raise RuntimeError("Incomplete expert rollout: "+spec["id"])
        datasets.append(data); results.append(result)
    return {key:np.concatenate([d[key] for d in datasets]) for key in datasets[0]},results


def report(output,summary):
    text=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>四平台模仿学习实验</title><style>body{font:17px system-ui;line-height:1.65;max-width:1000px;margin:24px auto;padding:0 18px;color:#172b3a}table{border-collapse:collapse;width:100%}td,th{border:1px solid #a9bdc5;padding:8px}img{max-width:100%}.pair{display:flex;flex-wrap:wrap;gap:12px}.pair figure{margin:0;flex:1 1 400px}a{color:#155e75}</style><h1>四平台：关节参考行为克隆</h1>']
    text.append(f'<p>模式{html.escape(summary["mode"])}；训练种子{summary["training_seeds"]}；每平台冻结测试集独立列出。PD/位置伺服保留，策略独立执行时专家查询数为0。完整协议与逐案例失败见summary.json。</p>')
    for kind,record in summary["platforms"].items():
        text.append(f'<h2>{html.escape(TITLES[kind])}</h2><p>{html.escape(SCOPE[kind])}</p><table><tr><th>控制器</th><th>ID成功率均值</th><th>OOD成功率均值</th></tr>')
        for name,aggregate in record["aggregate"].items():
            text.append(f'<tr><td>{name}</td><td>{aggregate["test"]["mean_success_rate"]:.1%}</td><td>{aggregate["ood"]["mean_success_rate"]:.1%}</td></tr>')
        text.append('</table><div class="pair">')
        for name in ("expert","bc","phase_bc"):
            path=f'{kind}/demos/{name}/demo.gif'
            text.append(f'<figure><img src="{path}" alt="{kind} {name}实际动力学回放"><figcaption>{kind} {name}：固定第一测试案例</figcaption></figure>')
        text.append('</div><p>frozen_state保留当前时钟，固定状态观测为初态；用来检验对状态反馈的依赖。各策略共享任务和判据，专家失败也保留。</p>')
    lesson=ROOT/"03-五平台模仿学习实操课.md"
    try:lesson_link=Path(os.path.relpath(lesson,output)).as_posix()
    except ValueError:lesson_link=lesson.resolve().as_uri()
    text.append(f'<p><a href="summary.json">原始结果</a> · <a href="{html.escape(lesson_link,quote=True)}">实操课</a></p></html>')
    (output/"report.html").write_text("".join(text),encoding="utf-8")


def train_suite(robot="all",output=None,mode="quick",steps=None,render=True,reuse_data=None):
    output=Path(output) if output else ROOT/"outputs/platform-runs"/datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True,exist_ok=False)
    seeds=[0] if mode=="quick" else list(range(5))
    steps=steps or (1800 if mode=="quick" else 3600)
    summary={"status":"running","mode":mode,"training_seeds":seeds,"updates":steps,"platforms":{},
             "python":sys.version,"numpy":np.__version__,"mujoco":mujoco.__version__,"menagerie_commit":lab.PIN,
             "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/"learning.py",PLATFORM_ROOT/"lab.py",PLATFORM_ROOT/"deep_practice.py")},
             "scope":"Phase-conditioned low-dimensional joint-reference BC. Full-body walking, vision, DAgger, dexterous reorientation and real robots are separately gated."}
    summary["geometry_timestamp"]="Diagnostics follow explicit mj_forward; geometry is refreshed at the reported current data.time."
    snapshots=output/"source-snapshot";snapshots.mkdir()
    for path in (Path(__file__),ROOT/"learning.py",PLATFORM_ROOT/"lab.py",PLATFORM_ROOT/"deep_practice.py"):
        shutil.copy2(path,snapshots/path.name)
    save_json(output/"summary.json",summary)
    started=time.perf_counter()
    for kind in (KINDS if robot=="all" else (robot,)):
        folder=output/kind; folder.mkdir()
        p=protocol(kind,mode); save_json(folder/"protocol.json",p)
        datasets,experts={},{}
        if reuse_data:
            source=Path(reuse_data)/kind
            assert json.loads((source/"protocol.json").read_text(encoding="utf-8"))==p, "Reused data protocol changed"
            saved_experts=json.loads((source/"expert-episodes.json").read_text(encoding="utf-8"))
        for split in p["specs"]:
            if reuse_data:
                with np.load(source/f"data-{split}.npz",allow_pickle=False) as loaded:datasets[split]={k:loaded[k].copy() for k in loaded.files}
                experts[split]=saved_experts[split]
                print(f"REUSE {kind} {split}: frozen expert dataset",flush=True)
            else:datasets[split],experts[split]=collect(kind,p["specs"][split])
            np.savez_compressed(folder/f"data-{split}.npz",**datasets[split])
        save_json(folder/"expert-episodes.json",experts)
        train,validation=datasets["train"],datasets["validation"]
        mean,std=train["obs"].mean(0),np.maximum(train["obs"].std(0),.001)
        am,astd=train["action"].mean(0),np.maximum(train["action"].std(0),.001)
        normalized={split:{"obs":data["obs"],"action":(data["action"]-am)/astd} for split,data in datasets.items()}
        record={"scope":SCOPE[kind],"observation_dim":len(mean),"action_dim":len(am),"dataset_rows":{s:len(d["obs"]) for s,d in datasets.items()},
                "dataset_sha256":{s:hashlib.sha256((folder/f"data-{s}.npz").read_bytes()).hexdigest() for s in datasets},
                "baselines":{},"results":{},"aggregate":{}}
        if reuse_data:record["reused_dataset_directory"]=str(source.resolve())
        phase_model=PhaseBC.fit(train,DURATION[kind])
        phase_model.save(folder/"phase-bc.npz")
        for name in ("expert","hold","phase_bc"):
            record["baselines"][name]={}
            for split in ("test","ood"):
                values=experts[split] if name=="expert" else [episode(kind,spec,policy=name,model=phase_model if name=="phase_bc" else None)[0] for spec in p["specs"][split]]
                if name!="expert" and any(r["expert_calls"]!=0 for r in values):raise RuntimeError("Baseline queried the expert")
                record["baselines"][name][split]=values
        for seed in seeds:
            print(f"TRAIN {kind} seed={seed}, dimensions={len(mean)}->{len(am)}",flush=True)
            timer=time.perf_counter()
            model,history,selected=fit(ReferenceMLP(seed,mean,std,am,astd),normalized["train"],normalized["validation"],steps,seed+10)
            model.save(folder/f"bc-seed-{seed}.npz")
            write_csv(folder/f"loss-seed-{seed}.csv",history)
            entry={"training_seconds":time.perf_counter()-timer,"selected_update":selected,
                   "normalized_test_action_mse":mse(model,normalized["test"]),"policies":{}}
            for name in ("bc","frozen_state"):
                entry["policies"][name]={}
                for split in ("test","ood"):
                    values=[]
                    for spec in p["specs"][split]:
                        result,_=episode(kind,spec,name,model)
                        if result["expert_calls"]!=0:raise RuntimeError("Learner rollout queried the expert")
                        values.append(result)
                    entry["policies"][name][split]=values
                    print(f"EVAL {kind} {name} seed={seed} {split}: {sum(r['passed'] for r in values)}/{len(values)}",flush=True)
            record["results"][str(seed)]=entry
        for name in ("expert","hold","bc","frozen_state","phase_bc"):
            record["aggregate"][name]={}
            for split in ("test","ood"):
                rates=[float(np.mean([r["passed"] for r in record["baselines"][name][split]]))] if name in record["baselines"] else [float(np.mean([r["passed"] for r in row["policies"][name][split]])) for row in record["results"].values()]
                record["aggregate"][name][split]={"per_seed_success_rate":rates,"mean_success_rate":float(np.mean(rates)),"sample_std":float(np.std(rates,ddof=1)) if len(rates)>1 else None}
        summary["platforms"][kind]=record
        save_json(output/"summary.json",summary)
        model=ReferenceMLP.load(folder/"bc-seed-0.npz")
        for name in ("expert","bc","phase_bc"):
            print(f"DEMO {kind} {name}: first frozen test case",flush=True)
            episode(kind,p["specs"]["test"][0],name,phase_model if name=="phase_bc" else model,folder/"demos"/name,render=render)
    report(output,summary)
    summary["status"]="completed"
    summary["elapsed_seconds"]=time.perf_counter()-started
    summary["rendered"]=render
    save_json(output/"summary.json",summary)
    (ROOT/"outputs/platform-latest.txt").write_text(str(output.resolve()),encoding="utf-8")
    print(f"COMPLETE {output.resolve()}",flush=True)
    return output


if __name__=="__main__":
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):stream.reconfigure(encoding="utf-8",errors="replace")
    parser=argparse.ArgumentParser()
    parser.add_argument("--robot",choices=["all",*KINDS],default="all")
    parser.add_argument("--mode",choices=["quick","full"],default="quick")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--steps",type=int)
    parser.add_argument("--no-render",action="store_true")
    parser.add_argument("--reuse-data",type=Path,help="Reuse saved expert data only when the entire frozen protocol matches")
    parser.add_argument("--demo-run",type=Path)
    parser.add_argument("--policy",choices=["expert","hold","bc","frozen_state","phase_bc"],default="bc")
    parser.add_argument("--split",choices=["test","ood"],default="test")
    parser.add_argument("--case",type=int,default=0)
    parser.add_argument("--viewer",action="store_true")
    args=parser.parse_args()
    if args.steps is not None and args.steps<1:parser.error("Steps must be positive")
    if args.demo_run:
        if args.robot=="all":parser.error("Choose one robot for a live demo")
        folder=args.demo_run/args.robot
        p=json.loads((folder/"protocol.json").read_text(encoding="utf-8"))
        model=PhaseBC.load(folder/"phase-bc.npz") if args.policy=="phase_bc" else ReferenceMLP.load(folder/"bc-seed-0.npz") if args.policy in ("bc","frozen_state") else None
        case=p["specs"][args.split][args.case]
        out=args.output or ROOT/"outputs/platform-runs"/datetime.datetime.now().strftime("demo-%Y%m%d-%H%M%S-%f")
        result,_=episode(args.robot,case,args.policy,model,out,render=not args.no_render,viewer=args.viewer)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else:train_suite(args.robot,args.output,args.mode,args.steps,not args.no_render,args.reuse_data)
