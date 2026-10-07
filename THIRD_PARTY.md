The inference subset in `vendor/unimate` and
`vendor/data_process/utils/plotting.py` comes from
[Friedrich-M/UniMate](https://github.com/Friedrich-M/UniMate), revision
`2c5b384715aa63d8639b1ed7eb74bfe614570c7a`; see
`vendor/UNIMATE_LICENSE` for its MIT license.

Local adaptations add CUDA/XPU device dispatch and preserve explicit device indices in T5 AMP.
The previously missing vendor/unimate/models and vendor/unimate/dataset sources were restored from the same pinned revision.

The original checkout is preserved separately in `UniMate/` and is not needed
to run this adapter. Training entry points and cloud captioning are not bundled.
The Blender executable is installed separately; mesh preprocessing helpers are bundled.

Motion is installed separately from the upstream dependency pinned in
`requirements.txt`. Model weights retain the license on
[Linzhan/UniMate](https://huggingface.co/Linzhan/UniMate) (CC BY-NC 4.0).
Dataset assets retain their source licenses; see
[Linzhan/UniML3D](https://huggingface.co/datasets/Linzhan/UniML3D).


A integração de mesh inclui também mesh_animation, motion_export, feature_extraction, joint_annotation e utilitários de rig do UniMate na mesma revisão. Mudanças locais: nome estável `asset` no preprocess_asset; criação/consulta de F-curves compatível com Blender 5 no blender_rig.py e blender_export.py. A inferência/modelo permanecem upstream.

O player inclui Three.js r128 e GLTFLoader sob licença MIT, obtidos dos arquivos locais distribuídos pelo ComfyUI-SkinTokens. Licença em web/vendor/THREE-LICENSE.txt. As bibliotecas rodam isoladas em iframe.
