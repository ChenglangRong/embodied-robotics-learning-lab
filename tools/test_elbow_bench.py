"""Software checks with constructed data; no fixture is a physical measurement."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("elbow_bench.py")


class ElbowBenchTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.file = Path(directory.name) / "bench.csv"
        self.run_tool("--template", check=True)

    def run_tool(self, *options, check=False):
        return subprocess.run([sys.executable, str(SCRIPT), str(self.file), *options],
                              capture_output=True, text=True, check=check)

    def fill(self, return_force):
        with self.file.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        for row in rows[1:]:
            row[2:] = ["6" if row[1] == "flex" else str(return_force)] * 5
        with self.file.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)

    def test_friction_both_directions_require_positive_input_work(self):
        self.fill(-6)
        output = self.run_tool(check=True).stdout
        self.assertIn("flex,5,1.2000,1.2566", output)
        self.assertIn("extend,5,1.2000,1.2566", output)

    def test_elastic_return_can_give_energy_back(self):
        self.fill(6)
        self.assertIn("extend,5,-1.2000,-1.2566", self.run_tool(check=True).stdout)

    def test_equal_median_can_hide_angle_dependence(self):
        self.fill(-6)
        with self.file.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        for row in rows[1:]:
            if row[1] == "flex":
                row[2:] = ["2", "4", "6", "8", "10"]
        with self.file.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)
        self.assertIn("flex,5,1.2000,1.2566,1.6000", self.run_tool(check=True).stdout)

    def test_unfilled_template_and_overwrite_are_rejected(self):
        self.assertNotEqual(self.run_tool("--template").returncode, 0)
        self.assertIn("fill every force cell", self.run_tool().stderr)

    def test_incomplete_cycle_is_rejected(self):
        self.fill(-6)
        self.file.write_text("\n".join(self.file.read_text().splitlines()[:-1]) + "\n")
        self.assertIn("both directions are required", self.run_tool().stderr)


if __name__ == "__main__":
    unittest.main()
