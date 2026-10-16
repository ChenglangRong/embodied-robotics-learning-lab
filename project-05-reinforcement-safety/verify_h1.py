"""Exact coverage, provenance and complete H1 push-recovery trajectory replay."""
import argparse
from dataclasses import fields
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
def key(c):return (c["controller"],c["scenario"],c["seed"])


TRAJECTORY_RTOL=1e-7
TRAJECTORY_ATOL=1e-7


def equal_trajectory(actual,expected,label="trajectory",stats=None):
    """Compare every value; tolerate CPU roundoff only in floating trajectories."""
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and actual.keys()==expected.keys(),label
        for name in expected:equal_trajectory(actual[name],expected[name],f"{label}.{name}",stats)
    elif isinstance(expected,list):
        assert isinstance(actual,list) and len(actual)==len(expected),label
        for i,(a,b) in enumerate(zip(actual,expected)):equal_trajectory(a,b,f"{label}[{i}]",stats)
    elif isinstance(expected,float):
        assert isinstance(actual,(int,float)) and not isinstance(actual,bool),label
        assert np.isfinite(actual) and np.isfinite(expected),(label,actual,expected)
        difference=abs(actual-expected)
        tolerance=TRAJECTORY_ATOL+TRAJECTORY_RTOL*abs(expected)
        assert difference<=tolerance,(label,actual,expected,tolerance)
        if stats is not None:
            stats["compared_float_values"]+=1
            if difference>stats["maximum_absolute_difference"]:
                stats["maximum_absolute_difference"]=difference
                stats["largest_difference_path"]=label
            stats["maximum_tolerance_fraction"]=max(stats["maximum_tolerance_fraction"],difference/tolerance)
    else:
        assert type(actual) is type(expected) and actual==expected,(label,actual,expected)


def verify(folder,skip=False):
    protocol,summary,manifest=read(folder/"protocol.json"),read(folder/"summary.json"),read(folder/"manifest.json")
    names={r["path"] for r in manifest}
    assert len(names)==len(manifest) and names=={p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.name!="manifest.json" and "__pycache__" not in p.parts}
    for r in manifest:
        p=(folder/r["path"]).resolve();assert p.is_relative_to(folder) and sha(p)==r["sha256"],r["path"]
    for name,digest in protocol["source_sha256"].items():assert sha(folder/"source"/name)==digest,name
    cases=read(folder/"cases.json")
    expected={(m,s,k) for m in protocol["controllers"] for s in protocol["scenarios"] for k in protocol["seeds"]}
    assert len(cases)==len(expected)==summary["evaluation_cases"] and {key(c) for c in cases}==expected
    assert len({key(c) for c in cases})==len(cases)
    with gzip.open(folder/"trajectories.jsonl.gz","rt",encoding="utf-8") as stream:records=[json.loads(line) for line in stream]
    trajectories={key(r["case"]):r for r in records}
    assert len(records)==len(trajectories)==len(cases) and set(trajectories)==expected
    assert sum(not c["success"] for c in cases)==summary["retained_failure_cases"]
    for c in cases:
        rows=trajectories[key(c)]["trajectory"]
        assert isinstance(rows,list) and len(rows)==c["steps"],key(c)
        assert all(isinstance(r,dict) and r.keys()==rows[0].keys() for r in rows),key(c)
        equal(trajectories[key(c)]["case"],c,str(key(c)))
        assert c["success"]==all(c["criteria"].values())
    for m,stored in summary["aggregate"].items():
        selected=[c for c in cases if c["controller"]==m]
        equal({"successes":sum(c["success"] for c in selected),"cases":len(selected),"falls":sum(not c["criteria"]["upright"] for c in selected)},stored,m)
    # A fresh verifier process imports the exact historical implementation.
    sys.path.insert(0,str(folder/"source"))
    from h1_recovery import H1Stand,Config,episode,lab
    assert set(protocol["config"])=={f.name for f in fields(Config)},"frozen config keys differ from the snapshot class"
    assert protocol["model_commit"]==lab.PIN
    for name,digest in protocol["model_assets_sha256"].items():assert sha(lab.CACHE/name)==digest,name
    replayed=0
    stats={"compared_float_values":0,"maximum_absolute_difference":0.,"largest_difference_path":None,"maximum_tolerance_fraction":0.}
    if not skip:
        env=H1Stand(Config(**protocol["config"]))
        try:
            for c in cases:
                result,rows,_=episode(env,c["controller"],c["scenario"],c["seed"])
                equal(result,c,f"case{replayed}")
                equal_trajectory(rows,trajectories[key(c)]["trajectory"],f"trajectory{replayed}",stats)
                replayed+=1
        finally:env.close()
    return {"status":"PASS","mode":"integrity_only" if skip else "full_frozen_physical_replay","covered_cases":len(cases),
            "replayed_cases":replayed,"retained_failure_cases":summary["retained_failure_cases"],"hashed_files":len(manifest),
            "trajectory_comparison":{"rtol":TRAJECTORY_RTOL,"atol":TRAJECTORY_ATOL,"discrete_fields":"exact",
                                     "case_metric_rtol":1e-7,"case_metric_atol":1e-9,**stats},
            "scope":"All controller/scenario/reset keys, complete logged trajectories, frozen task metrics and per-controller "
                    "aggregates; replay imports the frozen source snapshot on the pinned H1 model. No RL training and no learner competence claim"}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("folder",type=Path);p.add_argument("--skip-replay",action="store_true");p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();folder=a.folder.resolve()
    assert not a.output.resolve().is_relative_to(folder),"keep the report outside the verified folder"
    r=verify(folder,a.skip_replay)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+"\n",encoding="utf-8",newline="\n");print(json.dumps(r,indent=2))
