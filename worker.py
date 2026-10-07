"""Run official inference in a child process; VRAM is released on exit."""

import json
import os
from pathlib import Path
import shutil
import sys

# Set before importing transformers/Hub, and never enable training telemetry.
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(Path(__file__).parent / "vendor"))

import numpy as np
import torch
from device_support import initialize_device
from unimate.inference.sample import InferenceArgs, _resolve_exp_paths, main
from model_store import default_features_dir, ensure_features, ensure_model
from mesh_pipeline import export_mesh, prepare_mesh


def run(request_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    run_dir = Path(request["run_dir"])
    worker_device = initialize_device(torch, request["device"], request["device_index"])
    os.environ["UNIMATE_DEVICE"] = worker_device
    checkpoint = Path(request["checkpoint"]) if request["checkpoint"] else None
    if request["experiment_dir"]:
        exp_dir = Path(request["experiment_dir"])
        if checkpoint is None:
            _, path = _resolve_exp_paths(str(exp_dir), None)
            checkpoint = Path(path)
    else:
        exp_dir, checkpoint = ensure_model(
            request["model_cache"], request["model"], request["model_revision"],
            request["auto_download"], checkpoint,
        )
    config = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    dataset = request["dataset"]
    if dataset not in config["dataset"]["dataset_list"]:
        raise ValueError(f"O checkpoint não foi treinado com {dataset}; selecione um modelo UniML3D.")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint ausente: {checkpoint}")
    mesh_path = request.get("mesh_path", "")
    normalization_dataset = dataset
    if mesh_path:
        features_dir, canonical_mesh, blender = prepare_mesh(request, config)
        # Objaverse's loader supports arbitrary topology and prefixed clips.
        dataset = "objaverse"
        config["objaverse"]["filter_object"] = False
        config["objaverse"]["objects_num"] = -1
    else:
        features_dir = Path(request["features_dir"]) if request["features_dir"] else default_features_dir(
            request["models_root"], dataset, request["dataset_revision"],
        )
        ensure_features(features_dir, dataset, request["cases"],
                        request["mode"] in ("inbetween", "motion_edit"),
                        request["dataset_revision"], request["auto_download"])
    config["dataset"]["dataset_list"] = [dataset]
    config[dataset]["path"] = str(features_dir)
    config["sampling"]["model_path"] = None
    config["sampling"]["device"] = worker_device
    if request["num_frames"]:
        config["dataset"]["max_motion_length"] = request["num_frames"]
    config["training"]["use_ema"] = request["use_ema"]

    # Fail before allocating the model if a case references the wrong rig.
    skeletons = np.load(features_dir / "cond.npy", allow_pickle=True).item()
    unknown = {key.partition("-")[0] for key in request["cases"]} - skeletons.keys()
    if unknown:
        raise ValueError(f"Esqueleto ausente em cond.npy: {sorted(unknown)}. Exemplos disponíveis: {list(skeletons)[:12]}")
    if request["mode"] == "motion_expand":
        length = config["dataset"]["max_motion_length"]
        if not 0 < request["expand_overlap"] < length:
            raise ValueError(f"expand_overlap precisa estar entre 1 e {length - 1}.")

    runtime_exp = run_dir / "runtime"
    runtime_exp.mkdir()
    (runtime_exp / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if mesh_path:
        stats = np.load(exp_dir / "dataset_stats.npy", allow_pickle=True).item()
        np.save(runtime_exp / "dataset_stats.npy", {dataset: stats[normalization_dataset]})
    else:
        shutil.copyfile(exp_dir / "dataset_stats.npy", runtime_exp / "dataset_stats.npy")
    if request["cfg_scale"] == 1.0:
        cases_path = runtime_exp / "cases.txt"
        cases_path.write_text("\n".join(sorted({key.partition("-")[0] for key in request["cases"]})), encoding="utf-8")
        case_args = {"test_cases_txt": str(cases_path)}
    else:
        cases_path = runtime_exp / "cases.json"
        cases_path.write_text(json.dumps(request["cases"], ensure_ascii=True), encoding="utf-8")
        case_args = {"test_cases_json": str(cases_path)}

    args = InferenceArgs(
        exp_dir=str(runtime_exp), model_path=str(checkpoint),
        seed=request["seed"] % (2**32), output_dir=str(run_dir),
        num_repetitions=request["num_repetitions"], cfg_scale=request["cfg_scale"],
        batch_size=request["batch_size"], only_save_motion=request["only_save_motion"],
        save_ric=request["save_ric"], inbetween=request["mode"] == "inbetween",
        keep_frames=request["keep_frames"], motion_edit=request["mode"] == "motion_edit",
        keep_joints=request["keep_joints"],
        gt_start_frame=None if request["gt_start_frame"] < 0 else request["gt_start_frame"],
        motion_expand=request["mode"] == "motion_expand", expand_overlap=request["expand_overlap"],
        **case_args,
    )
    main(args)
    if mesh_path and request["export_animated_mesh"]:
        export_mesh(request, features_dir, canonical_mesh, blender)


if __name__ == "__main__":
    run(sys.argv[1])
