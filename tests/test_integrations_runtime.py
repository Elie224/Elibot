import tempfile
import time
import unittest
from pathlib import Path

import external_integrations_fr as ext
import integration_jobs_store_fr as jobs_store


class IntegrationRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

        self.old_state_path = ext.INTEGRATION_RUNTIME_STATE_PATH
        self.old_retries = ext.INTEGRATION_MAX_RETRIES
        self.old_threshold = ext.INTEGRATION_CIRCUIT_FAIL_THRESHOLD
        self.old_open_seconds = ext.INTEGRATION_CIRCUIT_OPEN_SECONDS

        ext.INTEGRATION_RUNTIME_STATE_PATH = self.tmp_path / "runtime_state.json"
        ext.INTEGRATION_MAX_RETRIES = 0
        ext.INTEGRATION_CIRCUIT_FAIL_THRESHOLD = 1
        ext.INTEGRATION_CIRCUIT_OPEN_SECONDS = 30

        ext._provider_health.clear()
        ext._provider_alerts.clear()
        ext._integration_metrics["total"] = 0
        ext._integration_metrics["success"] = 0
        ext._integration_metrics["error"] = 0
        ext._integration_metrics["updated_at"] = ""
        ext._integration_metrics["by_provider"] = {}

    def tearDown(self) -> None:
        ext.INTEGRATION_RUNTIME_STATE_PATH = self.old_state_path
        ext.INTEGRATION_MAX_RETRIES = self.old_retries
        ext.INTEGRATION_CIRCUIT_FAIL_THRESHOLD = self.old_threshold
        ext.INTEGRATION_CIRCUIT_OPEN_SECONDS = self.old_open_seconds
        self.tmp.cleanup()

    def test_circuit_breaker_and_alert_ack(self) -> None:
        with self.assertRaises(RuntimeError):
            ext.execute_integration("github", "create_issue", {"title": "x"}, dry_run=False)

        health = ext.get_provider_health()["github"]
        self.assertGreater(float(health.get("circuit_open_until", 0.0)), time.time())

        alerts = ext.get_provider_alerts()
        self.assertFalse(alerts["github"]["acknowledged"])
        self.assertTrue(alerts["github"]["is_open"])

        acknowledged = ext.acknowledge_provider_alert("github", acknowledged_by="unit-test")
        self.assertTrue(acknowledged["acknowledged"])
        self.assertEqual(acknowledged["acknowledged_by"], "unit-test")

    def test_jobs_purge_and_queue_counters(self) -> None:
        now_ts = 1_700_000_000.0
        jobs = {
            "done-old": {
                "principal": "p1",
                "status": "done",
                "finished_at": "2020-01-01T00:00:00+00:00",
            },
            "done-fresh": {
                "principal": "p1",
                "status": "done",
                "finished_at": "2023-11-14T22:13:20+00:00",
            },
            "queued": {
                "principal": "p1",
                "status": "queued",
                "created_at": "2023-11-14T22:13:20+00:00",
            },
            "running": {
                "principal": "p2",
                "status": "running",
                "created_at": "2023-11-14T22:13:20+00:00",
            },
        }

        removed = jobs_store.purge_expired_jobs(jobs, ttl_seconds=3600, now_ts=now_ts)
        self.assertEqual(removed, 1)
        self.assertNotIn("done-old", jobs)
        self.assertIn("queued", jobs)

        counters = jobs_store.count_jobs_by_status(jobs)
        self.assertEqual(counters["queued"], 1)
        self.assertEqual(counters["running"], 1)
        self.assertEqual(counters["finished"], 1)

        self.assertEqual(jobs_store.pending_for_principal(jobs, "p1"), 1)
        self.assertEqual(jobs_store.pending_for_principal(jobs, "p2"), 1)

        out_path = self.tmp_path / "jobs.json"
        jobs_store.save_jobs(out_path, jobs, updated_at="2026-04-16T00:00:00+00:00")
        loaded = jobs_store.load_jobs(out_path)
        self.assertEqual(set(loaded.keys()), set(jobs.keys()))


if __name__ == "__main__":
    unittest.main()
