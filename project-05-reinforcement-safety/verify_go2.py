"""Exact coverage, provenance and complete Go2 state/control-trajectory replay."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from verify_continuous import equal


def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(c):return (c["controller"],c["scenario"],c["seed"])


def verify(folder,skip=False):
    protocol,summary,manifest=read(folder/"protocol.json"),read(folder/"summary.json"),read(folder/"manifest.json")
    names={r["path"] for r in manifest}
    assert len(names)==len(manifest) and names=={p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.name!="manifest.json" and "__pycache__" not in p.parts}
    for r in manifest:
        p=(folder/r["path"]).resolve();assert p.is_relative_to(folder) and sha(p)==r["sha256"],r["path"]
    for name,digest in protocol["source_sha256"].items():assert sha(folder/"source"/name)==digest
    cases=read(folder/"cases.json")
    expected={(m,s,k) for m in protocol["controllers"] for s in protocol["scenarios"] for k in protocol["seeds"]}
    assert len(cases)==len(expected)==summary["evaluation_cases"] and {key(c) for c in cases}==expected
    with gzip.open(folder/"trajectories.jsonl.gz","rt",encoding="utf-8") as stream:records=[json.loads(line) for line in stream]
    trajectories={key(r["case"]):r for r in records}
    assert len(records)==len(trajectories)==len(cases) and set(trajectories)==expected
    assert sum(not c["success"] for c in cases)==summary["retained_failure_cases"]
    for c in cases:
        equal(trajectories[key(c)]["case"],c,str(key(c)))
        assert len(trajectories[key(c)]["trajectory"])==c["steps"]
        assert c["success"]==all(c["criteria"].values())
    for m,stored in summary["aggregate"].items():
        selected=[c for c in cases if c["controller"]==m]
        equal({"successes":sum(c["success"] for c in selected),"cases":len(selected),"falls":sum(not c["criteria"]["upright"] for c in selected)},stored,m)
    sys.path.insert(0,str(folder/"source"))
    from go2_feedback import Go2,Config,episode,lab
    assert protocol["model_commit"]==lab.PIN
    for name,digest in protocol["model_assets_sha256"].items():assert sha(lab.CACHE/name)==digest,name
    replayed=0
    if not skip:
        env=Go2(Config(**protocol["config"]))
        try:
            for c in cases:
                result,rows,_=episode(env,c["controller"],c["scenario"],c["seed"])
                equal(result,c,f"case{replayed}");equal(rows,trajectories[key(c)]["trajectory"],f"trajectory{replayed}")
                replayed+=1
        finally:env.close()
    return {"status":"PASS","mode":"integrity_only" if skip else "full_frozen_physical_replay","covered_cases":len(cases),"replayed_cases":replayed,"retained_failure_cases":summary["retained_failure_cases"],"hashed_files":len(manifest),"scope":"All scenario/controller/reset keys, complete control/state/slip trajectories, raw and cycle-average metrics; no RL training or learner competence claim"}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("folder",type=Path);p.add_argument("--skip-replay",action="store_true");p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=verify(a.folder.resolve(),a.skip_replay)
    assert not a.output.resolve().is_relative_to(a.folder.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+"\n",encoding="utf-8",newline="\n");print(json.dumps(r,indent=2))
