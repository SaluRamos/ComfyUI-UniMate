# ComfyUI-UniMate

Um único node, **UniMate • Animate Skeleton**, adapta a inferência oficial do
[UniMate](https://github.com/Friedrich-M/UniMate) ao ComfyUI. Gera movimento de
esqueletos a partir de texto e reúne geração, interpolação de keyframes, edição
de joints e expansão de movimento. Todos os parâmetros possuem tooltips.

## Instalação

Coloque esta pasta em `ComfyUI/custom_nodes/ComfyUI-UniMate`. Instale as
dependências **com o Python do ComfyUI** e reinicie o ComfyUI. No PowerShell,
a partir da pasta do custom node:

```powershell
$python = '..\..\.venv\Scripts\python.exe'
& $python -m pip install 'setuptools<81' wheel
& $python -m pip install --no-build-isolation -r requirements.txt
```

Se o ambiente usa `uv` em vez de `pip`:

```powershell
$uv = '..\..\..\standalone-env\uv.exe'
$python = '..\..\.venv\Scripts\python.exe'
& $uv pip install --python $python 'setuptools<81' wheel
& $uv pip install --python $python --no-build-isolation -r requirements.txt
```

O pin de setuptools atende à dependência Motion, cujo instalador usa
`pkg_resources`. Não instale `UniMate/requirements.txt`: ele pertence ao ambiente
completo de treinamento e fixa outra versão de PyTorch/CUDA, além de Blender.
Este adaptador reutiliza o PyTorch instalado pelo ComfyUI.

O código de inferência e renderização necessário já está incluído em `vendor/`;
o checkout original `UniMate/` é preservado e pode continuar sendo usado
separadamente. Se preferir um ambiente separado, instale as dependências nele
e informe seu Python em `python_executable`.
Para retornar IMAGE nesse caso, instale `imageio` e `imageio-ffmpeg` também no
Python do ComfyUI, ou use preview_frames=0 para receber apenas os caminhos.

## Primeira geração

1. Adicione o node pela categoria **UniMate**.
2. Para o humanoide, selecione `unimate_mixamo_f60`, dataset `mixamo`,
   object_type `mixamo`, mode `text` e cfg_scale `3`.
3. Escreva, por exemplo: `An object walks forward at a steady pace.`
4. Execute a fila. O node baixa os arquivos ausentes e salva a animação.

`examples/unimate_workflow.json` contém um workflow de um único node para
abrir no ComfyUI. `examples/unimate_api.json` contém o equivalente em formato
API, que pode ser enviado no campo `prompt` do endpoint `/prompt`.

`auto_download` vem habilitado. São baixados apenas `config.json`,
`dataset_stats.npy`, o checkpoint final escolhido, `cond.npy`, até três clips
do esqueleto e o encoder textual necessário. Não são baixados todos os
checkpoints, vídeos de treinamento nem todo o dataset. Em edição/interpolação,
somente os clips referenciados são buscados. Arquivos existentes são reutilizados
e os downloads do Hub permitem retomada.

Os pesos ficam em `ComfyUI/models/unimate/releases/<revisão>/<modelo>/`;
as features, em `models/unimate/datasets/<revisão>/features/<dataset>/`;
o encoder usa `models/unimate/text_encoder/`. Os identificadores de cache são
hashes das revisões. `models_dir` permite trocar essa raiz. Os downloads só
acontecem quando o node executa. Desabilite `auto_download` para impedir acesso
à rede; todos os arquivos, inclusive o encoder, precisam estar em cache.

## Modos e parâmetros

| Parâmetro | Uso |
|---|---|
| `model` / `dataset` / `object_type` | Checkpoint e esqueleto alvo; object_type deve existir em cond.npy |
| `prompt` / `cases_json` | Prompt simples ou mapa JSON de vários casos |
| `seed` | Seed oficial de 32 bits, calculado a partir dos 32 bits inferiores |
| `cfg_scale` | 3 é o padrão dos checkpoints; 1 executa geração sem prompt |
| `num_repetitions` / `batch_size` | Animações por caso e número de casos por lote |
| `num_frames` | 0 mantém a janela do checkpoint; normalmente 60 frames a 30 fps |
| `device` | auto, CUDA ou CPU |
| `only_save_motion` / `save_ric` | Salvar apenas .npy ou acrescentar renderização RIC à FK |
| `use_ema` | Usar EMA quando o checkpoint tiver esses pesos |
| `keep_frames` | Frames preservados em interpolação, como `0,-1` |
| `keep_joints` | Joints preservados em edição, separados por vírgula |
| `gt_start_frame` | Início da janela de referência; -1 mantém seleção aleatória |
| `expand_overlap` | Frames sobrepostos entre segmentos de expansão |
| `features_dir` | Pasta processada local com cond.npy e motions/*.npz |
| `experiment_dir` / `checkpoint_path` | Experimento ou checkpoint local específico |
| `model_revision` / `dataset_revision` | Revisões fixadas no Hugging Face; mudar cria outro cache |
| `filename_prefix` | Prefixo da pasta única dentro de ComfyUI/output |
| `preview_frames` | Limite de frames retornados como IMAGE; 0 pula decodificação |
| `python_executable` | Python alternativo para executar a inferência |

**Interpolação (`inbetween`)**: case_id precisa identificar um clip real,
por exemplo `Squat-000` quando presente no Mixamo. Use keep_frames `0,-1`
para manter os extremos. **Edição (`motion_edit`)**: use um clip real e
keep_joints, por exemplo `Hips,Spine,Neck,Head`. Ambos exigem cfg_scale > 1;
as referências são salvas junto dos resultados.

**Expansão (`motion_expand`)**: escreva um prompt por linha; cada linha gera
um segmento. Com três segmentos de 60 frames e overlap 10, a saída tem
160 frames. O overlap deve ser menor que a janela. Também aceita:

```json
{
  "mixamo-sequence": [
    "An object stands up.",
    "An object walks forward.",
    "An object turns around."
  ]
}
```

Em geração de texto, `cases_json` usa strings:

```json
{"mixamo-walk": "An object walks forward.", "mixamo-jump": "An object jumps."}
```

## Saídas

Cada execução cria `ComfyUI/output/<prefixo>_<data>_<id>/`. A inferência oficial
salva `motions/*.npy`, `animations/*_fk.mp4`, T-pose PNG e captions.json;
os modos com restrições usam suas próprias subpastas. A configuração efetiva,
request.json e inference.log também são salvos para reprodução e diagnóstico.

| Socket | Conteúdo |
|---|---|
| `motion` (UNIMATE_MOTION) | Dicionário de caminhos, referências, fps e representação |
| `preview_frames` (IMAGE) | Frames do primeiro vídeo FK; conecte a Preview Image/Save Image |
| `motion_paths` (STRING) | Lista JSON dos caminhos absolutos de .npy gerados, sem GT |
| `video_paths` (STRING) | Lista JSON dos caminhos absolutos dos MP4 gerados, sem GT |
| `output_dir` (STRING) | Pasta da execução |
| `animated_mesh` (FILE_3D) | Primeiro arquivo com rig animado |
| `mesh_paths` (STRING) | Lista JSON dos GLB/FBX animados |

Os arrays .npy têm formato `(T, J, 12)` na representação UniMate: posição,
rotação 6D e velocidade, com a codificação própria do joint raiz. São movimentos
de esqueleto. Com uma mesh de entrada, o node também exporta GLB/FBX animados.
Quando only_save_motion está ativo ou preview_frames=0, IMAGE é um batch vazio;
use as saídas de caminhos nesse caso.

## Esqueletos locais e limites

`features_dir` aceita a saída do pré-processamento oficial. Para GLB/FBX com rig,
use mesh/mesh_path: o node faz o pré-processamento automaticamente.
Para novos rigs, consulte o [pipeline oficial](https://github.com/Friedrich-M/UniMate/tree/main/data_process).
Truebones não distribui seus clips publicamente; forneça as features locais.
Mixamo e Objaverse possuem features públicas para download seletivo.

O checkpoint Mixamo anima apenas seu rig de 22 joints. Para outras topologias,
use um checkpoint UniML3D. A duração padrão de treinamento é 60 frames;
alterar a janela não garante qualidade. Para duração maior, prefira expansão.
As features públicas fixadas aqui são uma versão posterior ao dataset usado
para treinamento; a revisão antiga não contém features prontas. Para reproduzir
exatamente o treinamento, processe aquela revisão e informe features_dir.

Cada execução ocorre em um processo separado. O node descarrega os modelos
residentes do ComfyUI antes de usar CUDA, e o encerramento do processo libera
a memória do UniMate. Cancelar a fila também encerra esse processo. Para pouca
VRAM, comece com batch_size=1 e o modelo Mixamo; CPU está disponível.

## Verificação

```powershell
& $python -m unittest discover -s tests -v
```

Os testes cobrem download seletivo, reutilização offline, parâmetros com tooltip,
validação de caminhos, separação de GT e cancelamento. Pesos e dados de teste
não fazem parte do Git. Consulte THIRD_PARTY.md para procedência e licenças:
o código UniMate é MIT, os pesos são CC BY-NC 4.0 e os dados mantêm suas
licenças de origem.

Validado com Python 3.13.12, PyTorch 2.12.1+cu130 e RTX 3060 no Windows:
geração textual `(60, 22, 12)` com MP4 e preview IMAGE; interpolação com
keyframes preservados; edição com joints preservados; expansão de dois
segmentos em `(110, 22, 12)`. Os três modos com restrições foram executados
offline, usando o cache da primeira geração.


## Mesh com rig e preview 3D

Reinicie o ComfyUI e recarregue a página depois da atualização. Adicione novamente o node **UniMate • Animate Rig** para expor as novas conexões.

No workflow SkinTokens, conecte **SkinTokens Rig Generator → output_mesh_path** diretamente a **UniMate → mesh_path**. Também pode usar o `mesh_path` do Rig Previewer. Se usar o carregador nativo do ComfyUI, conecte `model_3d` à entrada `mesh`. Não é necessário carregar o arquivo novamente. A saída de vértices/faces sem rig não substitui o arquivo GLB/FBX skinned.

Use `mode=text` para gerar animação de um rig em repouso. Para o personagem não humanoide da captura, prefira `unimate_uniml3d_f60_v2`, `dataset=objaverse`, lote e repetições em 1. O checkpoint Mixamo tem limite de 22 joints; rigs maiores exigem UniML3D. A qualidade depende do rig e da semelhança com os dados de treinamento. O rig conectado substitui object_type/features_dir; dataset passa a escolher as estatísticas de normalização do checkpoint.

O mesmo node mostra a mesh animada: botão play/pausa, barra de tempo arrastável, marcador em segundos e velocidade de 0,25× a 2×. A velocidade altera a reprodução, mantendo o arquivo a 30 fps. Arraste a área 3D para girar e use a roda para zoom. Quando há várias gerações, selecione o arquivo acima do player. O player e Three.js são locais, sem CDN e sem depender do SkinTokens instalado.

Mantenha `export_animated_mesh=true`. A saída `animated_mesh` é um arquivo 3D nativo e `mesh_paths` contém os caminhos. Mesmo com exportação FBX, um GLB adicional é salvo para o player. `only_save_motion=true` desativa o vídeo de esqueleto, mas mantém a exportação/visualização da mesh.

É necessário Blender instalado para importar, preparar e exportar o rig. No Windows, a busca automática inclui Program Files/Blender Foundation; `blender_executable` permite escolher outra instalação. Validado com Blender 5.2.1 LTS. O subprocesso usa dependências Python do ambiente ComfyUI; para esta configuração elas precisam ser compatíveis com o Python do Blender. O arquivo de origem é preservado; o GLB exportado usa escala/orientação canônicas do UniMate. Materiais e skin seguem a importação/exportação do Blender.

`face_right` e `face_left` permitem orientar a frente com um par de bones; preencha ambos ou nenhum. `body_axis` adapta esse par a rigs serpentinos. Interpolação/edição exigem uma animação real no GLB/FBX: um rig em repouso serve somente para gerar movimento/expansão. As referências disponíveis são listadas em inference.log.

Validação da integração: inferência CUDA na RTX 3060 12 GB com rig GLB de teste (22 bones), GLB exportado com 44 canais animados, e player verificado no navegador com pausa, scrubber e 2×. O rig específico do usuário não foi usado neste teste.


## Intel Arc / Intel Arc Pro B70 (XPU)

See [INTEL_XPU.md](INTEL_XPU.md) for installation, backend selection and validation limits.
