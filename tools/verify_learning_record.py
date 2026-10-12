"""Audit public learning state, evidence hashes and Git-visible secret patterns."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "records/public-evidence-manifest.json"
HISTORY = ROOT / "records/history-2026-10-12.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")


def visible_files():
    result = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT)
    return sorted({item.decode("utf-8") for item in result.split(b"\0") if item})


def archive_history():
    rows = []
    base = ROOT / "project-04-imitation-learning/outputs"
    for name in ("platform-baseline-20261012", "platform-baseline-20261012-r2"):
        folder = base / name
        if not (folder / "summary.json").exists():
            continue
        summary = read(folder / "summary.json")
        rows.append({"run_id": name, "status": summary["status"], "mode": summary["mode"],
                     "training_seeds": summary["training_seeds"], "updates": summary["updates"],
                     "source_sha256": summary["source_sha256"], "menagerie_commit": summary["menagerie_commit"],
                     "platforms": {kind: {"scope": record["scope"], "dataset_rows": record["dataset_rows"],
                                           "dataset_sha256": record["dataset_sha256"], "aggregate": record["aggregate"],
                                           "protocol": read(folder / kind / "protocol.json")}
                                   for kind, record in summary["platforms"].items()}})
    save(HISTORY, {"date": "2026-10-12", "historical_runs": rows,
                   "raw_data_policy": "Duplicate older raw datasets remain local; complete public frozen data, source snapshots, weights, metrics and failures are in platform-baseline-20261012-r3."})
    print(f"ARCHIVED {len(rows)} historical summaries")


def export_manifest():
    files = []
    for name in visible_files():
        path = ROOT / name
        if "/outputs/" in name and path.is_file():
            data = path.read_bytes()
            files.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    save(MANIFEST, {"schema_version": 1, "date": read(ROOT / "learning-status.json")["updated_date"],
                    "evidence_files": files, "scope": "Git-visible public experimental outputs; original record bytes preserved by .gitattributes."})
    print(f"MANIFEST {len(files)} evidence files")


def checked_path(name):
    path = (ROOT / name).resolve()
    assert path.is_relative_to(ROOT) and path.is_file(), f"Missing or out-of-repository evidence: {name}"
    return path


def verify():
    state = read(ROOT / "learning-status.json")
    assert state["schema_version"] == 1
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", state["updated_date"])
    ids = [stage["id"] for stage in state["stages"]]
    assert len(ids) == len(set(ids)) and state["current_stage"] in ids
    for stage in state["stages"]:
        assert stage["learner_status"] in ("review_available", "self_reported_complete", "in_progress", "planned", "independently_assessed")
        if stage["learner_status"] == "self_reported_complete":
            assert stage.get("learner_report")
        if stage["independent_assessment"] == "passed":
            assert stage.get("personal_assessment_evidence"), "Independent skill claim requires personal evidence"
        for name in stage.get("materials", [])+stage.get("evidence", [])+stage.get("personal_assessment_evidence", []):
            checked_path(name)
    manifest = read(MANIFEST)
    names = set()
    for entry in manifest["evidence_files"]:
        assert entry["path"] not in names
        names.add(entry["path"])
        data = checked_path(entry["path"]).read_bytes()
        assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"], f"Evidence changed: {entry['path']}"
    visible = visible_files()
    expected_names = {name for name in visible if "/outputs/" in name and (ROOT / name).is_file()}
    assert names == expected_names, "Refresh manifest after changing public outputs"
    secrets = re.compile(rb"(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----)")
    bad_paths = re.compile(r"(?:^|/)(?:\.env(?:\..*)?|credentials\.json|auth\.json)|(?:^|/)(?:claude-audit|\.ai-collab)/")
    for name in visible:
        assert not bad_paths.search(name), f"Local/private file is Git-visible: {name}"
        path = ROOT / name
        if path.is_file() and path.suffix.lower() in (".py", ".ps1", ".md", ".json", ".yml", ".yaml", ".toml", ".txt", ".log"):
            assert not secrets.search(path.read_bytes()), f"Credential-shaped text: {name}"
    assert read(ROOT / "project-04-imitation-learning/outputs/platform-verification-20261012.json")["status"] == "PASS"
    print(f"PASS {len(state['stages'])} stages; {len(names)} public evidence hashes; credential-pattern scan")
    print("Scope: structural/provenance checks and known credential patterns; personal skills and full numerical replay are separately assessed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-history", action="store_true")
    parser.add_argument("--export-manifest", action="store_true")
    args = parser.parse_args()
    if args.archive_history:
        archive_history()
    elif args.export_manifest:
        export_manifest()
    else:
        verify()
