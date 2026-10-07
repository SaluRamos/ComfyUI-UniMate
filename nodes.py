"""One ComfyUI output node for all official UniMate inference modes."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import psutil
import folder_paths
import comfy.model_management as model_management
import torch
try:
    from .device_support import select_device
except ImportError:
    from device_support import select_device

MODES = ("text", "inbetween", "motion_edit", "motion_expand")
MODELS = (
    "unimate_uniml3d_f60_v2", "unimate_mixamo_f60",
    "unimate_uniml3d_f60_v2_full_cross_attn", "unimate_uniml3d_f60_preview",
)
DATASETS = ("mixamo", "objaverse", "truebones")
MODEL_REVISION = "971da7cfc1c8d99c2af6c00be9d2ed5700f99073"
DATASET_REVISION = "053e8b4b06b6a8ba717a030e675dd5b7fd5be92e"
HERE = Path(__file__).resolve().parent
MESH_FORMATS = ("glb", "fbx", "glb+fbx")


def safe_name(value):
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)}
    if (not value or value in (".", "..") or re.search(r'[<>:"/\\|?*\x00-\x1f]', value)
            or value.endswith((".", " ")) or value.split(".")[0].upper() in reserved):
        raise ValueError(f"Nome de arquivo inválido: {value!r}")
    return value


def make_cases(mode, object_type, case_id, prompt, cases_json, cfg_scale, keep_joints):
    if mode not in MODES:
        raise ValueError(f"Modo desconhecido: {mode}")
    if cfg_scale < 1:
        raise ValueError("cfg_scale precisa ser >= 1. O valor 1 gera movimento sem prompt.")
    if mode != "text" and cfg_scale <= 1:
        raise ValueError("Interpolação, edição e expansão precisam de cfg_scale > 1.")
    if mode == "motion_edit" and not keep_joints.strip():
        raise ValueError("Informe keep_joints para editar o movimento.")
    if cases_json.strip():
        cases = json.loads(cases_json)
    else:
        safe_name(object_type)
        safe_name(case_id)
        if "-" in object_type:
            raise ValueError("object_type não pode conter hífen; o UniMate separa o case_id pelo primeiro hífen.")
        value = [line.strip() for line in prompt.splitlines() if line.strip()] if mode == "motion_expand" else prompt
        cases = {f"{object_type}-{case_id}": value}
    if not isinstance(cases, dict) or not cases:
        raise ValueError("cases_json precisa ser um objeto JSON com pelo menos um caso.")
    for key, value in cases.items():
        safe_name(key)
        object_name, separator, suffix = key.partition("-")
        if not separator or not suffix:
            raise ValueError(f"Caso inválido: {key!r}. Use <object_type>-<case_id>.")
        safe_name(object_name)
        if mode == "motion_expand":
            if not isinstance(value, list) or not value or any(not isinstance(p, str) or not p.strip() for p in value):
                raise ValueError(f"{key}: expansão precisa de uma lista de prompts não vazios.")
        elif not isinstance(value, str) or (cfg_scale > 1 and not value.strip()):
            raise ValueError(f"{key}: informe um prompt de texto.")
    return cases


def new_output_dir(output_root, prefix):
    safe_name(prefix)
    root = Path(output_root).resolve()
    directory = (root / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}").resolve()
    if not directory.is_relative_to(root):
        raise ValueError("A saída precisa ficar dentro da pasta output do ComfyUI.")
    directory.mkdir(parents=True)
    return directory


def run_worker(request, executable, check_interrupt):
    directory = Path(request["run_dir"])
    request_path = directory / "request.json"
    request_path.write_text(json.dumps(request, ensure_ascii=True, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env.update({"HF_HOME": str(Path(request["models_root"]) / "text_encoder"),
                "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "MPLBACKEND": "Agg"})
    if not request["auto_download"]:
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    log_path = directory / "inference.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(executable), "-u", str(HERE / "worker.py"), str(request_path)],
            cwd=str(HERE), env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            while process.poll() is None:
                check_interrupt()
                time.sleep(0.2)
            if process.returncode:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
                raise RuntimeError(f"UniMate falhou. Log: {log_path}\n{tail}\nSe faltar um módulo, instale requirements.txt no Python usado pelo node.")
        finally:
            if process.poll() is None:
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                _, alive = psutil.wait_procs(children, timeout=2)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass


def collect_outputs(directory, preview_frames):
    directory = Path(directory)
    motions = sorted(p for p in directory.rglob("motions/*.npy") if "-gt_rep_" not in p.name)
    references = sorted(p for p in directory.rglob("motions/*.npy") if "-gt_rep_" in p.name)
    videos = sorted(p for p in directory.rglob("animations/*.mp4") if "-gt_rep_" not in p.name)
    if not motions:
        raise RuntimeError(f"A inferência não salvou movimentos. Consulte {directory / 'inference.log'}.")
    frames = torch.empty((0, 64, 64, 3), dtype=torch.float32)
    if videos and preview_frames:
        # Optional decoder dependency; inference-only imports stay in the worker.
        import imageio.v2 as imageio
        preview = next((p for p in videos if p.name.endswith("_fk.mp4")), videos[0])
        reader = imageio.get_reader(str(preview), format="ffmpeg")
        images = []
        try:
            for index, frame in enumerate(reader):
                images.append(torch.from_numpy(frame.copy()).float() / 255.0)
                if index + 1 >= preview_frames:
                    break
        finally:
            reader.close()
        if images:
            frames = torch.stack(images)
    result = {"motion_paths": [str(p) for p in motions], "reference_paths": [str(p) for p in references],
              "video_paths": [str(p) for p in videos], "output_dir": str(directory), "fps": 30,
              "representation": "unimate", "features_per_joint": 12}
    mesh_manifest = directory / "meshes.json"
    mesh_paths = []
    mesh_file = None
    if mesh_manifest.is_file():
        result.update(json.loads(mesh_manifest.read_text(encoding="utf-8")))
        mesh_paths = result["animated_mesh_paths"]
        # Native 3D file type is needed only for the optional mesh output.
        from comfy_api.latest._util.geometry_types import File3D
        mesh_file = File3D(next((p for p in mesh_paths if p.endswith(".glb")), mesh_paths[0]))
    return (result, frames, json.dumps(result["motion_paths"]), json.dumps(result["video_paths"]),
            str(directory), mesh_file, json.dumps(mesh_paths))


class UniMateAnimate:
    CATEGORY = "UniMate"
    FUNCTION = "animate"
    RETURN_TYPES = ("UNIMATE_MOTION", "IMAGE", "STRING", "STRING", "STRING", "FILE_3D", "STRING")
    RETURN_NAMES = ("motion", "preview_frames", "motion_paths", "video_paths", "output_dir", "animated_mesh", "mesh_paths")
    OUTPUT_TOOLTIPS = (
        "Metadados e caminhos das animações .npy (T, J, 12), mais referências dos modos de edição.",
        "Frames do primeiro vídeo FK. Vazio quando only_save_motion está ativo ou preview_frames=0.",
        "Lista JSON com os caminhos absolutos dos movimentos gerados.",
        "Lista JSON com os caminhos absolutos dos vídeos gerados.",
        "Pasta desta execução dentro de ComfyUI/output.",
        "Primeiro GLB/FBX animado, conectável a nodes 3D. Vazio quando não há mesh de entrada ou a exportação está desativada.",
        "Lista JSON dos arquivos GLB/FBX animados exportados.",
    )
    OUTPUT_NODE = True
    DESCRIPTION = "Anima esqueletos ou meshes GLB/FBX com rig. Pré-processa o rig, gera movimento e exporta a mesh animada no mesmo node."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (list(MODELS), {"tooltip": "v2 aceita múltiplos esqueletos. mixamo_f60 é especializado no humanoide Mixamo. Baixa apenas o checkpoint final."}),
            "dataset": (list(DATASETS), {"tooltip": "Sem mesh, é a fonte do esqueleto. Com mesh, escolhe as estatísticas de normalização do checkpoint; use objaverse para rigs diversos ou mixamo para humanoides Mixamo."}),
            "object_type": ("STRING", {"default": "mixamo", "tooltip": "Chave do esqueleto em cond.npy: mixamo, espécie Truebones ou UID de Objaverse."}),
            "prompt": ("STRING", {"default": "An object walks forward at a steady pace.", "multiline": True, "tooltip": "Descreva o movimento em inglês. Em motion_expand, escreva um prompt por linha. CFG=1 ignora o prompt."}),
            "mode": (list(MODES), {"tooltip": "text: gerar; inbetween: preservar frames; motion_edit: preservar joints; motion_expand: encadear prompts."}),
            "case_id": ("STRING", {"default": "comfy", "tooltip": "Nome da geração. Em interpolação/edição, use o ID de um clip real, por exemplo Squat-000 no Mixamo."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "Semente do ruído; os 32 bits inferiores são usados pelo seed oficial do UniMate."}),
            "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100.0, "step": 0.1, "tooltip": "Força da orientação textual. 1.0 gera sem texto; os modos com restrições exigem valor maior que 1."}),
            "num_repetitions": ("INT", {"default": 1, "min": 1, "max": 1024, "tooltip": "Quantidade de animações por caso, usando ruídos diferentes."}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 1024, "tooltip": "Casos por lote. Reduza para economizar VRAM; não muda o total de repetições."}),
            "num_frames": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "0 mantém max_motion_length do checkpoint (normalmente 60 frames a 30 fps). Janelas maiores aumentam VRAM; use expansão para vídeos longos."}),
            "device": (["auto", "cuda", "cpu", "xpu"], {"tooltip": "auto usa o dispositivo do ComfyUI; xpu usa GPU Intel Arc. CPU é mais lenta."}),
            "auto_download": ("BOOLEAN", {"default": True, "tooltip": "Baixa somente os pesos, encoder e clips necessários que estiverem ausentes. Desative para execução totalmente offline."}),
            "only_save_motion": ("BOOLEAN", {"default": False, "tooltip": "Salva apenas .npy; não renderiza MP4/PNG. A saída de IMAGE fica vazia."}),
            "save_ric": ("BOOLEAN", {"default": False, "tooltip": "Renderiza também posições recuperadas por RIC, além da animação padrão FK."}),
            "use_ema": ("BOOLEAN", {"default": True, "tooltip": "Usa os pesos EMA do checkpoint quando disponíveis, conforme a inferência oficial."}),
            "keep_frames": ("STRING", {"default": "0,-1", "tooltip": "Somente inbetween: índices separados por vírgula que mantêm a pose original. -1 é o último frame da janela."}),
            "keep_joints": ("STRING", {"default": "", "tooltip": "Somente motion_edit: nomes de joints separados por vírgula. Aceita nomes originais ou limpos, sem distinguir maiúsculas."}),
            "gt_start_frame": ("INT", {"default": -1, "min": -1, "max": 10000000, "tooltip": "Primeiro frame do clip de referência em edição/interpolação. -1 mantém a janela aleatória oficial."}),
            "expand_overlap": ("INT", {"default": 10, "min": 1, "max": 16383, "tooltip": "Somente motion_expand: frames sobrepostos entre segmentos. Precisa ser menor que a janela de geração."}),
            "filename_prefix": ("STRING", {"default": "UniMate", "tooltip": "Prefixo da pasta de saída em ComfyUI/output. Cada execução cria uma pasta única."}),
        }, "optional": {
            "cases_json": ("STRING", {"default": "", "multiline": True, "tooltip": "Opcional: JSON {object_type-case_id: prompt}. Substitui object_type/case_id/prompt e permite vários casos. Expansão usa listas de prompts."}),
            "features_dir": ("STRING", {"default": "", "tooltip": "Pasta local contendo cond.npy e motions/*.npz, já processados pelo UniMate. Vazio usa o cache automático; um GLB/FBX bruto precisa de pré-processamento."}),
            "experiment_dir": ("STRING", {"default": "", "tooltip": "Experimento local com config.json, dataset_stats.npy e checkpoints/. Vazio usa o modelo selecionado e o cache automático."}),
            "checkpoint_path": ("STRING", {"default": "", "tooltip": "Checkpoint .pt específico. Vazio usa o checkpoint final oficial ou o maior step do experiment_dir local."}),
            "models_dir": ("STRING", {"default": "", "tooltip": "Pasta de cache. Vazio usa ComfyUI/models/unimate. Caminhos relativos são resolvidos a partir da pasta ComfyUI."}),
            "model_revision": ("STRING", {"default": MODEL_REVISION, "tooltip": "Revisão do repositório Linzhan/UniMate no Hub. O padrão fixa a versão consultada; cada revisão possui cache separado."}),
            "dataset_revision": ("STRING", {"default": DATASET_REVISION, "tooltip": "Revisão das features em Linzhan/UniML3D. O padrão já contém features processadas; a antiga revisão de treinamento não as publica."}),
            "preview_frames": ("INT", {"default": 60, "min": 0, "max": 16384, "tooltip": "Máximo de frames do primeiro vídeo para a saída IMAGE. 0 evita a decodificação; os arquivos completos continuam salvos."}),
            "python_executable": ("STRING", {"default": "", "tooltip": "Python de outro ambiente com as dependências instaladas. Vazio usa o mesmo Python do ComfyUI."}),
            "mesh": ("FILE_3D,FILE_3D_GLB,FILE_3D_FBX", {"forceInput": True, "tooltip": "Conecte model_3d do Load 3D (Advanced), preservando o rig do GLB/FBX. Meshes apenas de vértices/faces não carregam um esqueleto."}),
            "mesh_path": ("STRING", {"default": "", "forceInput": True, "tooltip": "GLB/FBX com rig. Use caminho absoluto ou relativo à pasta input do ComfyUI. A entrada mesh conectada tem prioridade. O rig substitui object_type/features_dir."}),
            "blender_executable": ("STRING", {"default": "", "tooltip": "Caminho para blender.exe. Vazio busca no PATH ou em Program Files/Blender Foundation. Blender importa, prepara e exporta a mesh."}),
            "face_right": ("STRING", {"default": "", "tooltip": "Bone direito usado para orientar a frente do personagem, normalmente a coxa/quadril. Opcional; informe junto com face_left."}),
            "face_left": ("STRING", {"default": "", "tooltip": "Bone esquerdo do par de orientação. Vazio com face_right vazio mantém a orientação original do rig."}),
            "body_axis": ("BOOLEAN", {"default": False, "tooltip": "O par face_right/face_left descreve eixo cabeça-cauda, para rigs serpentinos, em vez de um par bilateral."}),
            "export_animated_mesh": ("BOOLEAN", {"default": True, "tooltip": "Com mesh de entrada, aplica os movimentos gerados ao rig e salva GLB/FBX animados, mantendo a skin e os materiais."}),
            "mesh_export_format": (list(MESH_FORMATS), {"default": "glb", "tooltip": "Formato da mesh animada: GLB, FBX ou ambos. GLB embute materiais/texturas e é indicado para visualização 3D no ComfyUI."}),
        }}

    def animate(self, model, dataset, object_type, prompt, mode, case_id, seed,
                cfg_scale, num_repetitions, batch_size, num_frames, device,
                auto_download, only_save_motion, save_ric, use_ema, keep_frames,
                keep_joints, gt_start_frame, expand_overlap, filename_prefix,
                cases_json="", features_dir="", experiment_dir="", checkpoint_path="",
                models_dir="", model_revision=MODEL_REVISION, dataset_revision=DATASET_REVISION,
                preview_frames=60, python_executable="", mesh=None, mesh_path="",
                blender_executable="", face_right="", face_left="", body_axis=False,
                export_animated_mesh=True, mesh_export_format="glb"):
        if model not in MODELS or dataset not in DATASETS or device not in ("auto", "cuda", "cpu", "xpu"):
            raise ValueError("Modelo, dataset ou dispositivo inválido.")
        cases = make_cases(mode, object_type, case_id, prompt, cases_json, cfg_scale, keep_joints)
        comfy_root = Path(folder_paths.base_path)

        def local_path(value):
            path = Path(value).expanduser()
            return str((comfy_root / path).resolve())

        models_root = local_path(models_dir) if models_dir.strip() else str(Path(folder_paths.models_dir) / "unimate")
        model_cache = Path(models_root) / "releases" / hashlib.sha256(model_revision.encode()).hexdigest()[:12]
        torch_device = model_management.get_torch_device()
        selected_device, device_index = select_device(
            torch, device, torch_device, validate=not python_executable.strip())
        run_dir = new_output_dir(folder_paths.get_output_directory(), filename_prefix)
        if mesh_export_format not in MESH_FORMATS:
            raise ValueError("Formato de exportação de mesh inválido.")
        if bool(face_right.strip()) != bool(face_left.strip()):
            raise ValueError("Informe face_right e face_left juntos, ou deixe ambos vazios.")
        rig_path = ""
        if mesh is not None:
            if mesh.format.lower() not in ("glb", "fbx"):
                raise ValueError("A entrada mesh precisa ser um arquivo GLB/FBX com rig.")
            rig_path = mesh.get_source() if mesh.is_disk_backed else mesh.save_to(str(run_dir / ("asset." + mesh.format.lower())))
        elif mesh_path.strip():
            path = Path(mesh_path).expanduser()
            rig_path = str(path.resolve()) if path.is_absolute() else folder_paths.get_annotated_filepath(mesh_path)
        if rig_path:
            rig_path = str(Path(rig_path).resolve())
            if Path(rig_path).suffix.lower() not in (".glb", ".fbx") or not Path(rig_path).is_file():
                raise ValueError(f"GLB/FBX de entrada ausente ou inválido: {rig_path}")
        request = dict(
            run_dir=str(run_dir), model=model, dataset=dataset, cases=cases, mode=mode,
            seed=seed, cfg_scale=cfg_scale, num_repetitions=num_repetitions, batch_size=batch_size,
            num_frames=num_frames, device=selected_device, device_index=device_index,
            auto_download=auto_download, only_save_motion=only_save_motion, save_ric=save_ric,
            use_ema=use_ema, keep_frames=keep_frames, keep_joints=keep_joints,
            gt_start_frame=gt_start_frame, expand_overlap=expand_overlap,
            models_root=models_root, model_cache=str(model_cache), model_revision=model_revision,
            dataset_revision=dataset_revision, features_dir=local_path(features_dir) if features_dir.strip() else "",
            experiment_dir=local_path(experiment_dir) if experiment_dir.strip() else "",
            checkpoint=local_path(checkpoint_path) if checkpoint_path.strip() else "",
            mesh_path=rig_path, blender_executable=local_path(blender_executable) if blender_executable.strip() else "",
            face_right=face_right, face_left=face_left, body_axis=body_axis,
            export_animated_mesh=export_animated_mesh, mesh_export_format=mesh_export_format,
        )
        if selected_device in ("cuda", "xpu"):
            model_management.unload_all_models()
            model_management.soft_empty_cache()
        run_worker(request, local_path(python_executable) if python_executable.strip() else sys.executable,
                   model_management.throw_exception_if_processing_interrupted)
        outputs = collect_outputs(run_dir, preview_frames)
        output_root = Path(folder_paths.get_output_directory()).resolve()
        images = [{"filename": p.name, "subfolder": p.parent.relative_to(output_root).as_posix(),
                   "type": "output"} for p in run_dir.rglob("animations/*_tpos.png")]
        print(f"[UniMate] {len(outputs[0]['motion_paths'])} animação(ões) salva(s) em {run_dir}")
        previews = [{"filename": Path(f).name,
                     "subfolder": Path(f).parent.relative_to(output_root).as_posix(), "type": "output"}
                    for f in outputs[0].get("animated_mesh_paths", []) if f.lower().endswith(".glb")]
        return {"ui": {"images": images, "unimate_animation": previews}, "result": outputs}
