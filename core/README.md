# Shared library

`core/` provides model-independent storage, geometry, numerical and presentation
capabilities. It does not import `training`, `label_generation`, `scripts` or `tools`.

| Module | Responsibility |
|---|---|
| `io/` | File/video decoding and encoding, generic file indexes |
| `data/` | Processed manifests, episode/asset/camera tables, frame mappings, resource access |
| `geometry/` | MANO, mesh storage, camera types, projection, force coordinate frames, mass properties |
| `physics/` | Explicit-state rigid/articulated dynamics, contact solving and wrench aggregation |
| `tactile_labels/` | Label schemas, readers/writers, token matching and handcrop contracts |
| `visualization/` | Prepared scene contexts, overlays, colormaps and panels |
| `resources/` | Small versioned resources; external model assets are not bundled |

Physics consumes explicit parameters and arrays; dataset normalization and solver
scheduling live in top-level `label_generation/`. Camera types are in
`geometry/camera.py`, camera reads in `data/camera.py`, video encoding in
`io/encoding.py`. The old visualization camera exports remain compatibility aliases.
Readers do not import presentation code. Prepared scene rendering lives in
`visualization/prepared.py` and accepts explicit roots, not CLI argument objects.

Operations and experiment recipes live in `scripts/`; repository maintenance and
regressions live in `tools/`. See [architecture](../docs/architecture.md).
