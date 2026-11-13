# Local runtime assets

Model payloads are local, excluded from Git and distribution archives. This tree
contains runtime resources; downloaded archives and retired resources are kept
under `artifacts/asset_archive/20261104/`.

| Directory | Purpose |
|---|---|
| `mano/stock/models/` | Original MANO left/right models and original bundle license/info |
| `mano/corrected/models/` | Corrected left model; right links to the original right model |
| `mano/topology/` | Preserved MANO face arrays |
| `smplh/` | Optional SMPL-H body models and body face array; no current project consumer found |
| `smplx/` | Optional SMPL-X PKL/NPZ variants and version metadata |
| `mappings/` | User-retained `vert2joint.pkl`; original provenance is not established |
| `pretrained/visual_encoder/` | Optional workspace-local DINOv3 ViT-L/16 training backbone; upstream revision and SHA-256 are recorded in the manifest |

Use `MANO_ASSETS_ROOT=assets/mano/stock` or `assets/mano/corrected`, or supply an
external directory with `models/MANO_LEFT.pkl` and `models/MANO_RIGHT.pkl`.
The shared loader `core/geometry/mano_layer.py` verifies fingerprints and applies
the requested basis once. Stock and corrected left models are distinct assets.
Update old explicit roots (`assets/mano`, `assets/mano_corrected`) to these paths.

EgoPHI points at `mano/stock/`; ContactPrompt points at the stock model files.
The two flat MANO filenames under stock are compatibility links for EgoPHI.

`manifest.json` records current locations, previous paths, sizes, SHA-256 hashes,
known consumers and provenance limits. Corrected-model evidence is under
`mano/corrected/provenance/`. Metadata/README files are tracked; payloads and links
remain ignored. Relocation preserved payload bytes. The integrity checker never deserializes pickle files.

Check the locally installed assets with `python tools/checks/check_assets.py`.
Optional absent body models are reported as skipped; present files must match
the manifest. Missing required MANO/mapping files or broken links fail the check.
