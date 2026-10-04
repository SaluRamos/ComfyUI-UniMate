"""Prepare a local rig through the official pipeline and export its motion."""

import json
import os
from pathlib import Path
import re
import shutil
import site
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
MESH_FORMATS = {"glb": ("glb",), "fbx": ("fbx",), "glb+fbx": ("glb", "fbx")}


def find_blender(explicit=""):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Blender não encontrado: {path}")
        return path
    executable = shutil.which("blender")
    if executable:
        return Path(executable)
    if os.name == "nt":
        root = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Blender Foundation"
        candidates = list(root.glob("Blender */blender.exe"))
        if candidates:
            return max(candidates, key=lambda p: tuple(int(n) for n in re.findall(r"\d+", p.parent.name)))
    raise FileNotFoundError("Instale o Blender ou informe blender_executable. O pré-processamento de GLB/FBX usa o Blender local.")


def run_blender(executable, job, run_dir):
    job["python_sites"] = site.getsitepackages()
    job_path = Path(run_dir) / f"blender_{job['task']}.json"
    job_path.write_text(json.dumps(job, ensure_ascii=True, indent=2), encoding="utf-8")
    subprocess.run(
        [str(executable), "--background", "--factory-startup", "--python-exit-code", "1",
         "--python", str(HERE / "blender_worker.py"), "--", str(job_path)],
        check=True, stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def remap_cases(cases):
    mapped = {}
    for key, prompt in cases.items():
        new_key = "asset-" + key.partition("-")[2]
        if new_key in mapped:
            raise ValueError("Com uma única mesh, cada case_id precisa ser único em cases_json.")
        mapped[new_key] = prompt
    return mapped


def add_rest_clip(directory, cond, length):
    directory = Path(directory) / "motions"
    directory.mkdir(exist_ok=True)
    frames = max(2, length + 1)
    np.savez_compressed(
        directory / "asset-rest-000.npz",
        global_positions=np.repeat(np.asarray(cond["tpos_first_frame"])[None], frames, axis=0),
        local_rotations=np.repeat(np.asarray(cond["tpos_local_rotations"])[None], frames, axis=0),
        root_facing_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 1)),
        fps=np.array(30),
    )


def prepare_mesh(request, config):
    executable = find_blender(request["blender_executable"])
    directory = Path(request["run_dir"]) / "mesh_input"
    directory.mkdir()
    print(f"[UniMate] Preparando rig de {request['mesh_path']}", flush=True)
    run_blender(executable, dict(
        task="prepare", mesh_path=request["mesh_path"], features_dir=str(directory),
        face_right=request["face_right"], face_left=request["face_left"], body_axis=request["body_axis"],
    ), request["run_dir"])
    cond = np.load(directory / "cond.npy", allow_pickle=True).item()["asset"]
    joint_count = len(cond["parents"])
    max_joints = config["dataset"]["max_joints"]
    if not config["dataset"]["min_joints"] <= joint_count <= max_joints:
        raise ValueError(f"O rig processado tem {joint_count} joints; este checkpoint aceita de {config['dataset']['min_joints']} a {max_joints}. Use um checkpoint UniML3D para rigs maiores que Mixamo.")
    motions = sorted((directory / "motions").glob("*.npz"))
    if not motions:
        if request["mode"] in ("inbetween", "motion_edit"):
            raise ValueError("A mesh não contém animação de referência. Use text/motion_expand ou forneça um GLB/FBX com animação para interpolação/edição.")
        length = request["num_frames"] or config["dataset"]["max_motion_length"]
        add_rest_clip(directory, cond, length)
        print("[UniMate] Rig sem animação: usando a pose de repouso somente para condicionar o esqueleto.", flush=True)
    else:
        print(f"[UniMate] Referências disponíveis: {[p.stem.removeprefix('asset-') for p in motions]}", flush=True)
    request["cases"] = remap_cases(request["cases"])
    return directory, directory / "asset_canonical.glb", executable


def export_mesh(request, directory, canonical_mesh, executable):
    export_dir = Path(request["run_dir"]) / "animated_meshes"
    export_dir.mkdir()
    motions = sorted(str(p) for p in Path(request["run_dir"]).rglob("motions/*.npy") if "-gt_rep_" not in p.name)
    run_blender(executable, dict(
        task="export", features_dir=str(directory), canonical_mesh=str(canonical_mesh),
        motion_paths=motions, export_dir=str(export_dir), formats=list(dict.fromkeys((*MESH_FORMATS[request["mesh_export_format"]], "glb"))),
    ), request["run_dir"])
    paths = sorted(str(p) for p in export_dir.iterdir() if p.suffix.lower() in (".glb", ".fbx"))
    if len(paths) != len(motions) * len(set((*MESH_FORMATS[request["mesh_export_format"]], "glb"))):
        raise RuntimeError("O Blender não exportou todas as animações. Consulte inference.log.")
    (Path(request["run_dir"]) / "meshes.json").write_text(json.dumps({
        "animated_mesh_paths": paths, "canonical_mesh_path": str(canonical_mesh),
        "source_mesh": request["mesh_path"],
    }, indent=2), encoding="utf-8")
