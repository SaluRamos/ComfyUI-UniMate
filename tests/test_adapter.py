import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
folder_paths = types.ModuleType("folder_paths")
folder_paths.base_path = str(ROOT)
folder_paths.models_dir = str(ROOT / "models")
sys.modules["folder_paths"] = folder_paths
comfy = types.ModuleType("comfy")
mm = types.ModuleType("comfy.model_management")
sys.modules["comfy"] = comfy
sys.modules["comfy.model_management"] = mm
import nodes
import model_store
import mesh_pipeline
import numpy as np


class AdapterTests(unittest.TestCase):
    def test_rig_socket_accepts_skintokens_and_native_file(self):
        optional = nodes.UniMateAnimate.INPUT_TYPES()["optional"]
        self.assertEqual(optional["mesh_path"][0], "STRING")
        self.assertTrue(optional["mesh_path"][1]["forceInput"])
        self.assertIn("FILE_3D", optional["mesh"][0])


    def test_paths_cannot_escape_output_on_windows_or_linux(self):
        for name in ("../outside", "..\\outside", "C:\\outside", "x/y", "NUL", "CON.txt", "name.", "name "):
            with self.subTest(name=name), self.assertRaises(ValueError):
                nodes.safe_name(name)
        with tempfile.TemporaryDirectory() as tmp:
            first = nodes.new_output_dir(tmp, "UniMate")
            second = nodes.new_output_dir(tmp, "UniMate")
            self.assertTrue(first.is_relative_to(Path(tmp).resolve()))
            self.assertNotEqual(first, second)

    def test_cases_cover_expansion_and_pinned_clips(self):
        cases = nodes.make_cases("motion_expand", "mixamo", "demo", "Walk.\nTurn.", "", 3, "")
        self.assertEqual(cases, {"mixamo-demo": ["Walk.", "Turn."]})
        cases = nodes.make_cases("inbetween", "unused", "unused", "unused", '{"mixamo-Squat-000":"Stand."}', 3, "")
        self.assertEqual(list(cases), ["mixamo-Squat-000"])
        for key in ("Dog-../bad", "Dog-..\\bad", "Dog-only/bad"):
            with self.assertRaises(ValueError):
                nodes.make_cases("text", "Dog", "demo", "Walk", json.dumps({key: "Walk"}), 3, "")
        with self.assertRaises(ValueError):
            nodes.make_cases("motion_edit", "mixamo", "demo", "Walk", "", 1, "Hips")
        with self.assertRaises(ValueError):
            nodes.make_cases("motion_edit", "mixamo", "demo", "Walk", "", 3, "")

    def test_selective_download_never_fetches_other_objects(self):
        clips = ["Dog-idle-000.npz", "Dog-run-000.npz", "Horse-run-000.npz"]
        self.assertEqual(model_store.select_clips(clips, "objaverse", {"Dog-demo": "Run"}, False), clips[:2])
        self.assertEqual(model_store.select_clips(["Squat-000.npz", "Walk-000.npz"], "mixamo", {"mixamo-Squat-000": "Stand"}, True), ["Squat-000.npz"])
        with self.assertRaises(FileNotFoundError):
            model_store.select_clips(clips, "objaverse", {"Dog-missing": "Run"}, True)

    def test_models_download_only_missing_artifacts_and_offline_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            def download(repo, filename, **kwargs):
                path = Path(kwargs["local_dir"]) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"artifact")
                return str(path)
            with patch.object(model_store, "hf_hub_download", side_effect=download) as download_mock:
                exp, checkpoint = model_store.ensure_model(tmp, "unimate_mixamo_f60", "test", True)
                self.assertEqual(download_mock.call_count, 3)
                self.assertEqual(checkpoint.name, "checkpoint_step_120000.pt")
                model_store.ensure_model(tmp, "unimate_mixamo_f60", "test", False)
                self.assertEqual(download_mock.call_count, 3)
                (exp / "dataset_stats.npy").unlink()
                with self.assertRaises(FileNotFoundError):
                    model_store.ensure_model(tmp, "unimate_mixamo_f60", "test", False)
                self.assertEqual(download_mock.call_count, 3)
                model_store.ensure_model(tmp, "unimate_mixamo_f60", "test", True)
                self.assertEqual(download_mock.call_count, 4)

    def test_features_download_cond_and_only_selected_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            def download(repo, filename, **kwargs):
                path = Path(kwargs["local_dir"]) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"artifact")
                return str(path)
            tree = [types.SimpleNamespace(path="features/mixamo/motions/" + name) for name in ("Squat-000.npz", "Walk-000.npz")]
            with patch.object(model_store, "HfApi") as api, patch.object(model_store, "hf_hub_download", side_effect=download) as hub:
                api.return_value.list_repo_tree.return_value = tree
                model_store.ensure_features(tmp, "mixamo", {"mixamo-Squat-000": "Stand"}, True, "test", True)
                self.assertEqual([call.args[1] for call in hub.call_args_list], ["features/mixamo/cond.npy", "features/mixamo/motions/Squat-000.npz"])
                model_store.ensure_features(tmp, "mixamo", {"mixamo-Squat-000": "Stand"}, True, "test", False)
                self.assertEqual(hub.call_count, 2)
                self.assertEqual(api.call_count, 1)

    def test_output_does_not_mix_generated_motion_and_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "motion_edit" / "motions"
            directory.mkdir(parents=True)
            for name in ("mixamo-Squat-000-rep_0-0.npy", "mixamo-Squat-000-gt_rep_0-0.npy"):
                np.save(directory / name, np.zeros((60, 22, 12), dtype=np.float32))
            result, frames, motion_json, video_json, _, mesh_file, mesh_json = nodes.collect_outputs(tmp, 0)
            self.assertEqual(len(result["motion_paths"]), 1)
            self.assertEqual(len(result["reference_paths"]), 1)
            self.assertEqual(frames.shape[0], 0)
            self.assertEqual(json.loads(video_json), [])
            self.assertEqual(len(json.loads(motion_json)), 1)
            self.assertIsNone(mesh_file)
            self.assertEqual(json.loads(mesh_json), [])

    def test_cancellation_terminates_and_waits_for_worker(self):
        class Cancelled(Exception):
            pass
        process = unittest.mock.Mock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as tmp, patch.object(nodes.subprocess, "Popen", return_value=process), patch.object(nodes.psutil, "Process") as process_info:
            process_info.return_value.children.return_value = []
            request = {"run_dir": tmp, "models_root": tmp, "auto_download": False}
            def cancel():
                raise Cancelled()
            with self.assertRaises(Cancelled):
                nodes.run_worker(request, sys.executable, cancel)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)

    def test_rest_pose_conditions_without_inventing_reference_animation(self):
        cond = {"tpos_first_frame": np.array([[0, 0, 0], [0, 1, 0]]),
                "tpos_local_rotations": np.array([[1, 0, 0, 0], [1, 0, 0, 0]])}
        with tempfile.TemporaryDirectory() as tmp:
            mesh_pipeline.add_rest_clip(tmp, cond, 60)
            with np.load(Path(tmp) / "motions" / "asset-rest-000.npz") as clip:
                self.assertEqual(clip["global_positions"].shape, (61, 2, 3))
                np.testing.assert_array_equal(clip["global_positions"][0], clip["global_positions"][-1])
                self.assertEqual(int(clip["fps"]), 30)

    def test_single_mesh_rewrites_cases_and_prevents_filename_collision(self):
        self.assertEqual(mesh_pipeline.remap_cases({"mixamo-walk": "Walk"}), {"asset-walk": "Walk"})
        with self.assertRaises(ValueError):
            mesh_pipeline.remap_cases({"Dog-walk": "Walk", "Horse-walk": "Walk"})

    def test_every_node_parameter_has_a_tooltip(self):
        for group in nodes.UniMateAnimate.INPUT_TYPES().values():
            for name, definition in group.items():
                self.assertTrue(definition[1].get("tooltip"), name)


if __name__ == "__main__":
    unittest.main()
