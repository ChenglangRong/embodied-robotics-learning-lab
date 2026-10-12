"""Offline tests: no paid CLI/model calls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from cooperate import AgentError, Relay, parse_verdict, resolve_cli, run_process, workspace_lock

class Tests(unittest.TestCase):
    def test_valid_verdicts(self):
        for verdict in ["PASS", "CHANGES", "BLOCKED"]:
            self.assertEqual(parse_verdict(f"VERDICT: {verdict}\n证据"), verdict)

    def test_invalid_verdicts(self):
        for text in ["", "PASS", "Summary: PASS", "VERDICT: PASS EVERYTHING", "Some claim\nVERDICT: PASS"]:
            with self.subTest(text=text), self.assertRaises(AgentError):
                parse_verdict(text)

    def test_missing_executable(self):
        with self.assertRaises(AgentError):
            resolve_cli("codex", Path.cwd(), "missing-robot-cli.exe")

    def test_lock_prevents_second_writer_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with workspace_lock(root):
                with self.assertRaises(AgentError):
                    with workspace_lock(root):
                        pass
            self.assertFalse((root / "coordinator.lock").exists())

    def test_lock_releases_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                with workspace_lock(root):
                    raise ValueError("test")
            self.assertFalse((root / "coordinator.lock").exists())

    def test_utf8_prompt_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = "import sys;sys.stdout.buffer.write(sys.stdin.buffer.read())"
            output = run_process([sys.executable, "-c", code], "机械臂消息：保持原有文件", root, root / "utf8", 5)
            self.assertEqual(output.read_text(encoding="utf-8"), "机械臂消息：保持原有文件")

    def test_nonzero_exit_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(AgentError):
                run_process([sys.executable, "-c", "raise SystemExit(3)"], "", root, root / "exit", 5)

    def test_timeout_stops_own_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(AgentError):
                run_process([sys.executable, "-c", "import time;time.sleep(20)"], "", root, root / "timeout", .2)

    def make_args(self, root, mode="implement", rounds=2):
        return argparse.Namespace(workspace=str(root), codex_cli=sys.executable, claude_cli=sys.executable,
                                  mode=mode, max_rounds=rounds, timeout=5, codex_model=None, claude_model=None,
                                  smoke_test=False, check_command=[])

    def fake_run(self, mode, replies, rounds=2, test_status="not_configured"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            relay = Relay(self.make_args(root, mode, rounds))
            calls = []
            iterator = iter(replies)
            def fake(role, prompt, read_only=True, smoke=False):
                calls.append((role, prompt, read_only))
                text = next(iterator)
                relay.record(role, text)
                return text
            def checks():
                relay.result["test_status"] = test_status
                return "test evidence: " + test_status
            with patch.object(relay, "invoke", side_effect=fake), patch.object(relay, "checks", side_effect=checks):
                relay.execute("测试任务")
            return relay.result.copy(), calls

    def test_peer_message_forwarded_both_directions(self):
        result, calls = self.fake_run("implement", ["IMPLEMENTATION_A", "VERDICT: PASS\nREVIEW_B", "ack"])
        self.assertIn("IMPLEMENTATION_A", calls[1][1])
        self.assertIn("REVIEW_B", calls[2][1])
        self.assertEqual(result["status"], "review_passed_tests_unconfigured")
        self.assertTrue(calls[-1][2])

    def test_changes_trigger_repair_and_re_review(self):
        result, calls = self.fake_run("implement", ["first", "VERDICT: CHANGES\nFIX_ME", "fixed", "VERDICT: PASS\naccepted", "ack"])
        self.assertEqual([c[0] for c in calls], ["codex", "claude", "codex", "claude", "codex"])
        self.assertIn("FIX_ME", calls[2][1])
        self.assertEqual(result["rounds_completed"], 2)

    def test_round_limit_reports_remaining_changes(self):
        result, _ = self.fake_run("implement", ["first", "VERDICT: CHANGES\nfix", "ack"], rounds=1)
        self.assertEqual(result["status"], "round_limit_changes_remaining")

    def test_audit_keeps_codex_read_only(self):
        result, calls = self.fake_run("audit", ["audit", "VERDICT: CHANGES\nfix", "ack"])
        self.assertTrue(all(c[2] for c in calls))
        self.assertEqual(result["status"], "changes_requested")

    def test_blocked_review_stops_repair(self):
        result, calls = self.fake_run("implement", ["first", "VERDICT: BLOCKED\nmissing evidence", "ack"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["status"], "blocked")

    def test_failed_tests_override_review_pass(self):
        result, _ = self.fake_run("implement", ["first", "VERDICT: PASS\nreview accepted", "ack"], test_status="failed")
        self.assertEqual(result["status"], "tests_failed")

    def test_completed_requires_tests(self):
        result, _ = self.fake_run("implement", ["first", "VERDICT: PASS\nreview accepted", "ack"], test_status="passed")
        self.assertEqual(result["status"], "completed")

if __name__ == "__main__":
    unittest.main()
