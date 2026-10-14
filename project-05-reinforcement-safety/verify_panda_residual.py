"""Audit exact coverage, model asset hashes and frozen physical policy replay."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode=True
from verify_continuous import equal


def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(c):return (c["policy"],c["split"],c["filter_enabled"],c["seed"])


def verify(folder,skip=False,current=False):
    protocol,summary,manifest=read(folder/"protocol.json"),read(folder/"summary.json"),read(folder/"manifest.json")
    names={r["path"] for r in manifest}
    assert len(names)==len(manifest)
    assert names=={p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.name!="manifest.json" and "__pycache__" not in p.parts}
    for r in manifest:
        path=(folder/r["path"]).resolve()
        assert path.is_relative_to(folder) and sha(path)==r["sha256"],r["path"]
    for name,digest in protocol["source_sha256"].items():assert sha(folder/"source"/name)==digest
    sets=[set(protocol[k]) for k in ("validation_seeds","test_seeds","ood_seeds")]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    policies=["zero","random","analytical","untrained"]+[f"ppo_seed_{s}" for s in protocol["training_seeds"]]
    cases=read(folder/"cases.json")
    expected={(p,s,f,k) for p in policies for s in ("test","ood") for f in (True,False) for k in protocol[f"{s}_seeds"]}
    assert len(cases)==len(expected)==summary["evaluation_cases"] and {key(c) for c in cases}==expected
    with gzip.open(folder/"failures.jsonl.gz","rt",encoding="utf-8") as stream:records=[json.loads(line) for line in stream]
    traces={key(r["case"]):r for r in records}
    assert len(records)==len(traces)==summary["retained_failures"] and set(traces)=={key(c) for c in cases if not c["success"]}
    for p in policies:
        selected=[c for c in cases if c["policy"]==p and c["split"]=="test" and c["filter_enabled"]]
        equal({"successes":sum(c["success"] for c in selected),"cases":len(selected),"mean_error_m":float(np.mean([c["final_error_m"] for c in selected])),"violations":sum(c["violations"] for c in selected)},summary["aggregate"][p],p)
    for seed in protocol["training_seeds"]:
        with (folder/f"training-seed-{seed}.csv").open(encoding="utf-8",newline="") as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==protocol["steps_per_seed"]//1024 and int(rows[-1]["environment_steps"])==protocol["steps_per_seed"]
        for row in rows:assert all(v in ("True","False") or np.isfinite(float(v)) for v in row.values())
    sys.path.insert(0,str(Path(__file__).resolve().parent if current else folder/"source"))
    from panda_residual import episode
    from panda_residual_env import PandaResidual,Config,lab
    from ppo import Agent
    assert protocol["model_commit"]==lab.PIN
    for name,digest in protocol["model_assets_sha256"].items():assert sha(lab.CACHE/name)==digest,name
    replayed=0
    if not skip:
        actual={"zero":"zero","random":"random","analytical":"analytical","untrained":Agent(27,2,777)}
        actual.update({f"ppo_seed_{s}":Agent.load(folder/f"ppo-seed-{s}.npz") for s in protocol["training_seeds"]})
        for enabled in (True,False):
            env=PandaResidual(Config(filter_enabled=enabled))
            try:
                for c in cases:
                    if c["filter_enabled"]!=enabled:continue
                    result,rows,_=episode(env,actual[c["policy"]],c["seed"],c["split"])
                    result["policy"]=c["policy"]
                    equal(result,c,f"case{replayed}")
                    if key(c) in traces:
                        equal(traces[key(c)]["case"],c,f"failure{replayed}")
                        equal(rows,traces[key(c)]["trajectory"],f"trace{replayed}")
                    replayed+=1
            finally:env.close()
    return {"status":"PASS","mode":"integrity_only" if skip else "full_current_physical_replay" if current else "full_frozen_physical_replay","hashed_files":len(manifest),"covered_cases":len(cases),"replayed_cases":replayed,"retained_failure_trajectories":len(records),"current_source_differences":[name for name,digest in protocol["source_sha256"].items() if name!="lab.py" and sha(Path(__file__).resolve().parent/name)!=digest],"scope":"Task outcomes and histories checked; learner capability and expanded-scene assessment separate"}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("folder",type=Path);p.add_argument("--skip-replay",action="store_true");p.add_argument("--current-source",action="store_true");p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();result=verify(args.folder.resolve(),args.skip_replay,args.current_source)
    assert not args.output.resolve().is_relative_to(args.folder.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(result,indent=2))
