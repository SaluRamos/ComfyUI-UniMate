"""Blender entry point for rig preprocessing and animated mesh export."""

import json
from pathlib import Path
import sys

job_path = sys.argv[sys.argv.index("--") + 1]
job = json.loads(Path(job_path).read_text(encoding="utf-8"))
sys.path.insert(0, str(Path(__file__).parent / "vendor"))
sys.path.extend(p for p in job["python_sites"] if p not in sys.path)

import numpy as np
from data_process.mesh_animation.preprocess_char import preprocess_asset
from data_process.mesh_animation.animate_motion import load_animation
from data_process.mesh_animation.common import drive_and_export, load_character


if job["task"] == "prepare":
    preprocess_asset(
        job["mesh_path"], job["features_dir"], name="asset",
        face_r=job["face_right"] or None, face_l=job["face_left"] or None,
        body_axis=job["body_axis"], formats=("glb",), keep_intermediate=True,
    )
elif job["task"] == "export":
    cond = np.load(Path(job["features_dir"]) / "cond.npy", allow_pickle=True).item()["asset"].copy()
    # The baked mesh is already at canonical scale; do not undo it twice.
    cond["scale_factor"] = 1.0
    for motion in job["motion_paths"]:
        armature = load_character(job["canonical_mesh"])
        anim, rest, tpos, _ = load_animation(motion, cond, "fk")
        drive_and_export(
            armature, anim, rest, cond["joint_names"], 30,
            str(Path(job["export_dir"]) / Path(motion).stem),
            formats=tuple(job["formats"]), tpos_global_rot=tpos,
        )
else:
    raise ValueError(f"Tarefa Blender desconhecida: {job['task']}")
