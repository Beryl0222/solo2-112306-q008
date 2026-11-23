"""检查采集资料的身份、形状和可交叉引用性。"""

import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_fixture(name):
    return json.loads((ROOT / "fixtures" / name).read_text(encoding="utf-8"))


class FixtureContractTests(unittest.TestCase):
    def test_record_manifest(self):
        manifest = read_fixture("manifest.json")
        primary = read_fixture("primary.json")
        revision = read_fixture("revision.json")
        self.assertEqual(manifest["fixture"], "replay_sampling")
        self.assertEqual(manifest["records"], ["primary.json", "revision.json"])
        self.assertTrue(manifest["synthetic"])
        self.assertEqual(revision["replaces"], primary["capture"])
        self.assertNotEqual(revision["capture"], primary["capture"])

    def test_related_input_records(self):
        primary = read_fixture("primary.json")
        revision = read_fixture("revision.json")
        frames = {entry["id"]: entry for entry in primary["physical_frames"]}
        courses = {entry[0] for entry in primary["courses"]}
        for window in primary["windows"]:
            self.assertFalse(set(window["frames"]) - frames.keys())
            self.assertIn(window["course"], courses)
        for entry in revision["priority_events"]:
            self.assertIn(entry["id"], frames)
            self.assertTrue(np.isfinite(entry["priority"]))
            self.assertGreaterEqual(entry["priority"], 0)
        uniforms = np.asarray(primary["draw"]["uniforms"])
        self.assertTrue(((uniforms >= 0) & (uniforms < 1)).all())

    def test_training_observation_references(self):
        primary = read_fixture("primary.json")
        revision = read_fixture("revision.json")
        draw_ids = {primary["draw"]["request"], primary["peer_draw"]["request"]}
        frame_ids = {item["id"] for item in primary["physical_frames"]}
        observations = revision["training_observations"]["by_draw"]
        self.assertEqual({item["draw"] for item in observations}, draw_ids)
        for item in observations:
            self.assertFalse(set(item["td_by_frame"]) - frame_ids)
            self.assertTrue(all(np.isfinite(v) for v in item["td_by_frame"].values()))
