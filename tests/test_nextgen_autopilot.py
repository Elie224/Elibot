import unittest

import nextgen_orchestrator_fr as n


class NextgenAutopilotDecisionTests(unittest.TestCase):
    def test_should_autopilot_for_eligible_case(self) -> None:
        ok, reason = n.should_autopilot(
            intent={"intent": "automation_request"},
            tool_selection={"selected": ["automation_plan"]},
            internal_scores={"risk": 0.2, "quality": 0.8, "confidence": 0.8},
            in_domain=True,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "eligible_intent")

    def test_should_not_autopilot_if_risky(self) -> None:
        ok, reason = n.should_autopilot(
            intent={"intent": "automation_request"},
            tool_selection={"selected": ["automation_plan"]},
            internal_scores={"risk": 0.9, "quality": 0.8, "confidence": 0.8},
            in_domain=True,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "risk_too_high")


if __name__ == "__main__":
    unittest.main()
