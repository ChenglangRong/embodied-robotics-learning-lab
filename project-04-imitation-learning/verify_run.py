"""Fresh replay of saved policies plus dataset/protocol integrity checks."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from learning import ROOT, MLP, TwoLinkReachEnv, expert, encode, evaluate, mse, save_json


def verify(directory, replay=True, output=None):
    directory=Path(directory)
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    assert summary["status"]=="completed", "Full experiment is incomplete"
    assert hashlib.sha256((ROOT/"learning.py").read_bytes()).hexdigest()==summary["source_sha256"]
    env_source=ROOT.parent/"project-01-simulation-software/src/robot_lab/planar_arm.py"
    assert hashlib.sha256(env_source.read_bytes()).hexdigest()==summary["environment_source_sha256"]
    specs=summary["protocol"]
    assert specs==json.loads((directory/"protocol.json").read_text(encoding="utf-8"))
    assert specs["environment"]["tolerance"]==.001 and specs["environment"]["max_steps"]==600
    datasets={}
    id_sets=[]
    for split in ("train","validation","test","ood"):
        path=directory/f"data-{split}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest()==summary["dataset_sha256"][split]
        with np.load(path,allow_pickle=False) as loaded:
            data={k:loaded[k].copy() for k in loaded.files}
        datasets[split]=data
        assert len(data["obs"])==summary["data_counts"][split]
        id_set=set(data["episode_id"])
        assert id_set=={s["id"] for s in specs[split]}
        assert all(not id_set & prior for prior in id_sets)
        id_sets.append(id_set)
        np.testing.assert_allclose(data["next_q"],data["pre_q"]+.02*data["executed_action"],atol=1e-12)
        np.testing.assert_array_equal(data["action"],data["executed_action"])
    mean=datasets["train"]["obs"].mean(0)
    std=np.maximum(datasets["train"]["obs"].std(0),1e-3)
    heldout_seeds={s["reset_seed"] for split in ("validation","test","ood") for s in specs[split]}
    model_count,episode_count=0,0
    started=time.perf_counter()
    for seed in summary["training_config"]["seeds"]:
        folder=directory/f"seed-{seed}"
        ledger=json.loads((folder/"query-ledger.json").read_text(encoding="utf-8"))
        assert ledger["new_dagger_labels"]==ledger["new_fresh_expert_labels"]
        query_episodes=json.loads((folder/"query-episodes.json").read_text(encoding="utf-8"))
        for kind,filename in (("dagger","dagger-labels.npz"),("fresh","fresh-expert-labels.npz")):
            assert all(s["reset_seed"] not in heldout_seeds for s in query_episodes[kind])
            with np.load(folder/filename,allow_pickle=False) as data:
                assert len(data["obs"])==ledger["new_dagger_labels"]
                assert sum(e["labels"] for e in query_episodes[kind])==len(data["obs"])
                # Reconstruct sampled states and check expert labels independently of saved actions.
                env=TwoLinkReachEnv()
                for index in np.linspace(0,len(data["obs"])-1,50,dtype=int):
                    env.q=data["pre_q"][index].copy()
                    env.goal=data["goal"][index].copy()
                    np.testing.assert_allclose(encode(env),data["obs"][index],atol=1e-12)
                    np.testing.assert_allclose(expert(env),data["action"][index],atol=1e-12)
        for name,recorded in summary["results"][str(seed)].items():
            model=MLP.load(folder/f"{name}.npz")
            np.testing.assert_array_equal(model.mean,mean)
            np.testing.assert_array_equal(model.std,std)
            np.testing.assert_allclose(mse(model,datasets["test"]),recorded["teacher_forced_test_mse"],atol=1e-12)
            model_count+=1
            if replay:
                for split in ("test","ood"):
                    actual=evaluate(specs[split],model)
                    assert actual==recorded[split], f"Replay changed: seed={seed}, {name}, {split}"
                    episode_count+=len(specs[split])
    for name,policy in (("expert",expert),("zero",lambda env:np.zeros(2))):
        if replay:
            for split in ("test","ood"):
                actual=evaluate(specs[split],policy)
                assert actual==summary["baselines"][name][split]
                episode_count+=len(specs[split])
    for name in summary["aggregate"]:
        for split in ("test","ood"):
            rows=[r[name][split] for r in summary["results"].values()]
            aggregate=summary["aggregate"][name][split]
            rates=[row["success_rate"] for row in rows]
            np.testing.assert_array_equal(rates,aggregate["per_seed_success_rate"])
            np.testing.assert_allclose(np.mean(rates),aggregate["mean_success_rate"],atol=1e-12)
            if len(rows)>1:np.testing.assert_allclose(np.std(rates,ddof=1),aggregate["sample_std_success_rate"],atol=1e-12)
    for filename in ("overview.png","demo-test.gif","demo-ood.gif","report.html"):
        assert (directory/filename).stat().st_size>0
    result={"status":"PASS","verification_mode":"full_replay" if replay else "integrity_only",
            "checks":"source/data integrity, split isolation, normalization, expert labels, exact label budgets, aggregate values"+("; saved policy replay" if replay else ""),
            "saved_models":model_count,"replayed_episodes":episode_count,"elapsed_seconds":time.perf_counter()-started,
            "scope":"Fresh execution of stored policies using the same environment; no independent optimizer implementation or real robot validation" if replay else "Stored data/model provenance and consistency checks; no new rollout execution"}
    report_path = Path(output) if output else directory/"verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report_path,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",nargs="?",type=Path,default=ROOT/"outputs/baseline-20261012-r2")
    parser.add_argument("--skip-replay",action="store_true")
    parser.add_argument("--output",type=Path,help="Write a separate report, preserving the published verification")
    args=parser.parse_args()
    verify(args.directory,replay=not args.skip_replay,output=args.output)
