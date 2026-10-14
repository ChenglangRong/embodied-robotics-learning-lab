"""Train NumPy PPO placement residuals and retain matched physical rollouts."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
from dataclasses import asdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time
import sys
import subprocess
import numpy as np
import mujoco
from PIL import Image, ImageDraw
from ppo import Agent, gae
from panda_residual_env import Config, PandaResidual, ROOT, lab
from tabular_lab import save_csv, save_json


def episode(env, policy, seed, split="test", render=False, viewer=False):
    obs, _ = env.reset(seed, split)
    rng = np.random.default_rng(seed+5000000)
    total, frames = 0., []
    handle = None
    if viewer:
        import mujoco.viewer
        handle = mujoco.viewer.launch_passive(env.m, env.d)
        handle.cam.lookat[:], handle.cam.distance = [.4, -.1, .35], 1.7
        handle.cam.azimuth, handle.cam.elevation = 125, -22
    try:
        for i in range(env.cfg.horizon):
            tick = time.monotonic()
            action = policy.sample(obs[None], deterministic=True)[0][0] if isinstance(policy, Agent) else env.analytical() if policy == "analytical" else rng.uniform(-1,1,2) if policy == "random" else np.zeros(2)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if render and i % 4 == 0:
                im = Image.fromarray(env.render())
                draw = ImageDraw.Draw(im)
                draw.text((12,12), f"{info['phase']} | error {info['error_m']*1000:.1f} mm | t={info['time_s']:.2f}s", fill="white")
                frames.append(im)
            if handle:
                handle.sync()
                if not handle.is_running():
                    break
                time.sleep(max(0, env.cfg.dt-(time.monotonic()-tick)))
            if terminated or truncated:
                break
        metrics = env.assess()
        return {"seed":seed,"split":split,"filter_enabled":env.cfg.filter_enabled,"return":total,"steps":len(env.rows),
                "completed":len(env.rows)==125,"success":bool(len(env.rows)==125 and metrics["success"]),
                **{k:v for k,v in metrics.items() if k!="success"},
                "filter_interventions":sum(r["filter_intervention"] for r in env.rows),
                "violations":sum(r["violation"] for r in env.rows)}, list(env.rows), frames
    finally:
        if handle:
            handle.close()


def train(seed, steps, out):
    if steps <= 0 or steps % 1024:
        raise ValueError("positive step budget divisible by1024")
    envs = [PandaResidual() for _ in range(8)]
    validation_env = PandaResidual()
    agent = Agent(27,2,seed)
    obs = np.stack([e.reset(seed*1000+i, "train")[0] for i,e in enumerate(envs)])
    logs, best, selected, started = [], -np.inf, 0, time.perf_counter()
    try:
        for update in range(steps//1024):
            batch = {k:[] for k in ("obs","z","logp","value","reward","next_value","terminated","truncated")}
            for _ in range(128):
                actions,z,lp = agent.sample(obs)
                batch["obs"].append(obs.copy()); batch["z"].append(z); batch["logp"].append(lp); batch["value"].append(agent.value(obs))
                ts = [e.step(a) for e,a in zip(envs,actions)]
                finals = np.stack([t[0] for t in ts])
                batch["reward"].append(np.array([t[1] for t in ts])); batch["next_value"].append(agent.value(finals))
                terms,truncs = np.array([t[2] for t in ts]),np.array([t[3] for t in ts])
                batch["terminated"].append(terms); batch["truncated"].append(truncs)
                for i in np.flatnonzero(terms|truncs):
                    finals[i] = envs[i].reset(split="train")[0]
                obs = finals
            b = {k:np.asarray(v) for k,v in batch.items()}
            adv, returns = gae(b["reward"],b["value"],b["next_value"],b["terminated"],b["truncated"])
            diag = agent.optimize(b["obs"].reshape(-1,27),b["z"].reshape(-1,2),b["logp"].ravel(),adv.ravel(),returns.ravel())
            if (update+1)%4==0 or update+1==steps//1024:
                validation = [episode(validation_env,agent,s)[0] for s in range(900000,900004)]
                score = float(np.mean([v["success"] for v in validation]))-float(np.mean([v["final_error_m"] for v in validation]))
                if score>best:
                    best, selected = score, update+1
                    agent.save(out/f"ppo-seed-{seed}.npz")
                print(f"Panda seed={seed} steps={(update+1)*1024}/{steps} validation={score:.5f}",flush=True)
            logs.append({"update":update+1,"environment_steps":(update+1)*1024,**diag})
        save_csv(out/f"training-seed-{seed}.csv", logs)
        save_json(out/f"training-seed-{seed}.json", {"seed":seed,"steps":steps,"selected_update":selected,"validation_score":best,"wall_seconds":time.perf_counter()-started})
    finally:
        validation_env.close()
        for e in envs:
            e.close()


def freeze(out, steps, seeds):
    if out.exists():
        raise FileExistsError("new output folder required")
    out.mkdir(parents=True)
    source = out/"source"
    for name in ("panda_residual.py","panda_residual_env.py","ppo.py","tabular_lab.py"):
        target = source/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    target = source/"lab.py"
    target.write_bytes((ROOT.parent/"project-03-robot-platforms/lab.py").read_bytes())
    asset_dir = lab.CACHE/"franka_emika_panda"
    hashes = {p.relative_to(lab.CACHE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in asset_dir.rglob("*") if p.is_file()}
    sha = subprocess.check_output(["git","-C",str(lab.CACHE),"rev-parse","HEAD"],text=True).strip()
    if sha != lab.PIN:
        raise ValueError("pinned upstream model commit required")
    save_json(out/"protocol.json", {"date":"2026-10-14","config":asdict(Config()),"training_seeds":seeds,"steps_per_seed":steps,
             "validation_seeds":list(range(900000,900004)),"test_seeds":list(range(1000000,1000008)),"ood_seeds":list(range(2000000,2000008)),
             "model_commit":sha,"model_assets_sha256":hashes,"dependencies":{"numpy":np.__version__,"mujoco":mujoco.__version__},
             "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},
             "algorithm":"Existing NumPy PPO: gamma.97 lambda.95 clip.2 lr.0003 epochs6 minibatch256; 8x128 collection",
             "scope":"Pilot five seeds,8 test goals each; identical full-integration grasp/transport reset; ID goal shift40..65mm; OOD75..95mm; Cartesian XY residual, retained IK/position servos and scheduled gripper; state-based observations with history-dependent acceptance monitor"})


def evaluate(out):
    protocol = json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    policies = {"zero":"zero","random":"random","analytical":"analytical","untrained":Agent(27,2,777)}
    for seed in protocol["training_seeds"]:
        policies[f"ppo_seed_{seed}"] = Agent.load(out/f"ppo-seed-{seed}.npz")
    cases, traces = [], []
    for enabled in (True,False):
        env = PandaResidual(Config(filter_enabled=enabled))
        try:
            for name,policy in policies.items():
                for split in ("test","ood"):
                    for seed in protocol[f"{split}_seeds"]:
                        result,rows,_ = episode(env,policy,seed,split)
                        result["policy"] = name
                        cases.append(result)
                        if not result["success"]:
                            traces.append({"case":result,"trajectory":rows})
        finally:
            env.close()
    save_json(out/"cases.json",cases)
    with gzip.open(out/"failures.jsonl.gz","wt",encoding="utf-8") as stream:
        for t in traces:
            stream.write(json.dumps(t)+"\n")
    aggregate = {}
    for name in policies:
        rows = [c for c in cases if c["policy"]==name and c["split"]=="test" and c["filter_enabled"]]
        aggregate[name] = {"successes":sum(r["success"] for r in rows),"cases":len(rows),"mean_error_m":float(np.mean([r["final_error_m"] for r in rows])),"violations":sum(r["violations"] for r in rows)}
    env = PandaResidual()
    try:
        demos = out/"demos"; demos.mkdir()
        for name in ("zero","analytical",f"ppo_seed_{protocol['training_seeds'][0]}"):
            result, rows, frames = episode(env,policies[name],protocol["test_seeds"][0],render=True)
            save_csv(demos/f"{name}.csv",rows); save_json(demos/f"{name}.json",result)
            frames[0].save(demos/f"{name}.gif",save_all=True,append_images=frames[1:],duration=160,loop=0)
            frames[-1].save(demos/f"{name}.png")
        prefix_hash = hashlib.sha256(env.prefix_state.tobytes()).hexdigest()
    finally:
        env.close()
    save_json(out/"summary.json",{"aggregate":aggregate,"evaluation_cases":len(cases),"retained_failures":len(traces),"prefix_state_sha256":prefix_hash,
             "task_status":"Per-policy task scores; formal expanded-scene assessment and learner evidence remain separate"})
    save_json(out/"manifest.json",[{"path":p.relative_to(out).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.rglob("*")) if p.is_file() and "__pycache__" not in p.parts])
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path);p.add_argument("--steps",type=int,default=8192);p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2,3,4])
    p.add_argument("--train-seed",type=int);p.add_argument("--evaluate",action="store_true");p.add_argument("--demo",type=Path);p.add_argument("--policy",default="ppo_seed_0")
    p.add_argument("--split",choices=["test","ood"],default="test");p.add_argument("--case",type=int,default=0);p.add_argument("--filter-off",action="store_true")
    args=p.parse_args()
    if args.demo:
        if not 0<=args.case<8:p.error("case0..7")
        policy=Agent.load(args.demo/(args.policy.replace("ppo_seed_","ppo-seed-")+".npz")) if args.policy.startswith("ppo_seed_") else Agent(27,2,777) if args.policy=="untrained" else args.policy
        if isinstance(policy,str) and policy not in {"zero","random","analytical"}:p.error("unknown policy")
        env=PandaResidual(Config(filter_enabled=not args.filter_off))
        try: print(json.dumps(episode(env,policy,(1000000 if args.split=="test" else 2000000)+args.case,args.split,viewer=True)[0],indent=2))
        finally:env.close()
    elif args.output:
        if args.train_seed is not None:train(args.train_seed,args.steps,args.output)
        elif args.evaluate:evaluate(args.output)
        else:
            if len(args.seeds)!=len(set(args.seeds)) or min(args.seeds)<0:p.error("unique nonnegative seeds")
            freeze(args.output,args.steps,args.seeds)
            for s in args.seeds:train(s,args.steps,args.output)
            evaluate(args.output)
    else:p.error("--output or --demo required")
