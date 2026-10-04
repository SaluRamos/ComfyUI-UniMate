"""Download only the selected checkpoint and reference skeleton artifacts."""

import hashlib
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

MODEL_REPO = "Linzhan/UniMate"
DATASET_REPO = "Linzhan/UniML3D"
MODEL_REVISION = "971da7cfc1c8d99c2af6c00be9d2ed5700f99073"
DATASET_REVISION = "053e8b4b06b6a8ba717a030e675dd5b7fd5be92e"
MODELS = {
    "unimate_uniml3d_f60_v2": 100000,
    "unimate_mixamo_f60": 120000,
    "unimate_uniml3d_f60_v2_full_cross_attn": 100000,
    "unimate_uniml3d_f60_preview": 120000,
}
DATASETS = ("mixamo", "objaverse", "truebones")


def present(path):
    return path.is_file() and path.stat().st_size > 0


def ensure_model(root, variant, revision, auto_download, checkpoint=None):
    if variant not in MODELS:
        raise ValueError(f"Modelo desconhecido: {variant}")
    exp_dir = Path(root) / variant
    relative_checkpoint = f"checkpoints/checkpoint_step_{MODELS[variant]}.pt"
    artifacts = ["config.json", "dataset_stats.npy"]
    if checkpoint is None:
        artifacts.append(relative_checkpoint)
    for artifact in artifacts:
        if present(exp_dir / artifact):
            continue
        if not auto_download:
            raise FileNotFoundError(f"Falta {exp_dir / artifact}. Ative auto_download ou forneça os arquivos localmente.")
        print(f"[UniMate] Baixando {variant}/{artifact}", flush=True)
        hf_hub_download(MODEL_REPO, f"{variant}/{artifact}", revision=revision, local_dir=str(root))
    return exp_dir, Path(checkpoint) if checkpoint else exp_dir / relative_checkpoint


def default_features_dir(root, dataset, revision):
    revision_key = hashlib.sha256(revision.encode()).hexdigest()[:12]
    return Path(root) / "datasets" / revision_key / "features" / dataset


def clip_matches(filename, object_type, clip_id):
    stem = Path(filename).stem
    candidates = {clip_id, f"{object_type}-{clip_id}", f"{object_type}_{clip_id}"}
    return stem in candidates or f"{object_type}_{stem}" in candidates


def select_clips(filenames, dataset, cases, pinned):
    selected = set()
    for case_key in cases:
        object_type, _, clip_id = case_key.partition("-")
        if pinned:
            matches = [f for f in filenames if clip_matches(f, object_type, clip_id)]
        else:
            matches = [f for f in filenames if dataset == "mixamo" or Path(f).name.startswith(object_type + "-")][:3]
        if not matches:
            target = case_key if pinned else object_type
            raise FileNotFoundError(f"Nenhum clip processado encontrado para {target!r} em {dataset}.")
        selected.update(matches)
    return sorted(selected)


def ensure_features(directory, dataset, cases, pinned, revision, auto_download):
    if dataset not in DATASETS:
        raise ValueError(f"Dataset desconhecido: {dataset}")
    directory = Path(directory)
    motion_dir = directory / "motions"
    filenames = sorted(p.name for p in motion_dir.glob("*.npz") if present(p))
    if present(directory / "cond.npy"):
        try:
            select_clips(filenames, dataset, cases, pinned)
            return directory
        except FileNotFoundError:
            if not auto_download or dataset == "truebones":
                raise
    if not auto_download:
        raise FileNotFoundError(f"Features ausentes em {directory}. Ative auto_download ou informe features_dir.")
    if dataset == "truebones":
        raise FileNotFoundError("Truebones não publica os clips. Informe features_dir com cond.npy e motions/*.npz processados localmente.")

    api = HfApi()
    prefix = f"features/{dataset}"
    filenames = sorted(
        entry.path.removeprefix(prefix + "/motions/")
        for entry in api.list_repo_tree(DATASET_REPO, path_in_repo=prefix + "/motions", revision=revision, repo_type="dataset")
        if entry.path.endswith(".npz")
    )
    selected = select_clips(filenames, dataset, cases, pinned)
    # local_dir needs the Hub's features/<dataset>/ layout.
    staging = directory / ".hub"
    artifacts = ["cond.npy"] + ["motions/" + name for name in selected]
    for artifact in artifacts:
        target = directory / artifact
        if present(target):
            continue
        print(f"[UniMate] Baixando {prefix}/{artifact}", flush=True)
        downloaded = Path(hf_hub_download(DATASET_REPO, f"{prefix}/{artifact}", repo_type="dataset", revision=revision, local_dir=str(staging)))
        target.parent.mkdir(parents=True, exist_ok=True)
        # Hub downloads are resumed and verified before the file is moved.
        downloaded.replace(target)
    return directory
