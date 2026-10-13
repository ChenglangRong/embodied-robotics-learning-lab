import unittest
import numpy as np
from robot_model_contracts import validate_chunk, validate_plan


class ContractTests(unittest.TestCase):
    def test_valid_skill_plan(self):
        self.assertEqual(validate_plan({"skill": "reach", "goal_xy": [.5, .4]})["timeout_steps"], 150)

    def test_invalid_plan(self):
        for plan in ({"skill": "shell"}, {"skill": "reach", "goal_xy": [2, 0]},
                     {"skill": "hold", "timeout_steps": 10000}, {"skill": "hold", "arbitrary_code": "x"}):
            with self.assertRaises(ValueError):
                validate_plan(plan)

    def test_receding_horizon_prefix(self):
        packet = {"actions": np.zeros((10, 2)), "units": "normalized_velocity", "period_s": .02,
                  "timestamp_s": 10., "execute_before_replan": 2}
        self.assertEqual(validate_chunk(packet, 2, 10.1).shape, (2, 2))

    def test_bad_chunk_shape_time_units(self):
        base = {"actions": [[0, 0]], "units": "normalized_velocity", "period_s": .02, "timestamp_s": 10.}
        for change in ({"actions": [[0]*8]}, {"actions": [[float("nan"), 0]]}, {"actions": [[1.1, 0]]},
                       {"timestamp_s": 9.}, {"period_s": .1}, {"units": "torque"}):
            with self.assertRaises(ValueError):
                validate_chunk({**base, **change}, 2, 10.1)


if __name__ == "__main__":
    unittest.main()
