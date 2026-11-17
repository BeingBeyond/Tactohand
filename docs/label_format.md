# Label format

[English](label_format.md) | [简体中文](label_format.zh-CN.md)

## Storage and indexing

`<dataset>_lerobot/meta/info.json` points to the active prepared manifest.
Integer ranges map episodes, modalities and logical rows to files and offsets;
ranges are half-open. Episodes can span files and row groups. Readers must
honor row offsets/counts and each stream's source-frame mapping.

The normalized container declares `format_version=processed_dataset_v1` and
stores `meta/`, `data/` and `assets/`. Metadata contains manifests, tables and
indexes; tactile and force streams are modalities under `data/`. Timestamps
supplement, rather than replace, solver-row, transition and token indexes.
Object identity joins instance → asset → geometry by string IDs.

## Tactile tables

| Component | Row unit | Contents |
|---|---|---|
| `samples` | Solver row | `contact`, `signed_distance` |
| `transitions` | Solver state row | World-space vertex forces, contact/environment wrench, target COM pose and velocity |
| `timeline` | Solver row | `solver_row`, `solver_time_s`, `source_position`, `tactile_flags`, `force_valid`, `target_object`, `closing_solver_row` |

Compact manifests declare `row_metadata: tactile.timeline`; readers reconstruct
logical fields from that table. The three components align by episode and
solver row. Sparse static data may declare `solver_index_column: source_position`.

Metadata-only upload publications retain the payload's timeline revision through
`upload_profile.source_revision`. A rewritten timeline declares its own
`source_revision` in the component. Readers validate against that payload revision;
statistics and physical references still bind the full published manifest hash.

- `contact` is uint8; `signed_distance` is float16 metres clipped to ±0.25 m.
  Rows contain 778 left vertices followed by 778 right vertices.
- `tactile_flags` bits 0/1 encode left/right presence; bits 2/3 encode
  contact/proximity validity. `force_valid` separately controls force supervision.
- `source_position` is a source-frame coordinate and may be fractional.
  Static samples can have independent object slots/gravity and null solver times.
- Dynamic `closing_solver_row` exists only for a next state sharing the target
  reference frame. Direct force can use the final valid row; rollout also needs
  continuous timing and a closing state.
- Forces and wrenches follow the solver's reference-mass convention. Reference
  mass, COM, inertia and `object_ids` align with target slots. Articulated data
  also store generalized wrenches and joint states.

Episode context records topology, rates, physics and content hashes.
`tactile_sha1` hashes logical `solver_sample`, `source_position`, `flags`,
`contact`, `signed_distance` in that order, independently of sharding.
`force_sha1` covers declared force/state and additional physical fields.

## Geometry and crops

Decode MANO through `core/geometry/mano_layer.py`. Installed `mano_contract`
takes precedence over `mano_source_basis`, then reviewed legacy decisions from
`MANO_BASIS_DECISIONS`. Do not infer basis from dataset names or flip it again
in consumers. The canonical parameter layout is
`[trans3, full_axis_angle48, betas10]`, in metres/radians, with
`flat_hand_mean=True`, `use_pca=False`, `center_idx=None`.

Migration records model fingerprints, source evidence and `geometry_revision`.
Installation requires finite parameters/vertices and maximum per-vertex error
≤1 mm across all valid frames; the recorded HIC exception is ≤2 mm. Invalid or
over-limit shards cannot be installed. Stored meshes with `hmr.mano=false`
and `mesh_vertices=true` take precedence over placeholder MANO parameters.

Current crop geometry uses normalized observation-plane boxes:
`crop_box_xyxy_orig`, `hand_bbox_xywh_orig`, and crop-pixel `hand_bbox_xywh`.
Do not clip negative or out-of-image windows used for padding. Derive K from
the observation camera and crop box through `frame_intrinsics`, retaining the
declared output dimensions and coordinate plane. Legacy raw-mirror coordinates
and stored K remain readable. Native raw fisheye K cannot replace rectified K.

Geometry carries episode/view/target/frame identity and independent `frame_valid`
and `crop_valid` flags. Embedded context also carries `crop_type`, `width`,
`height`, `stream_num_frames`. Restore packed offsets through the declared
mapping; never collapse missing index ranges into cumulative row counts.

## Physics references and publication

Recover multi-body references offline into `rigid_reference_v1`, not in training
loaders. Each episode has `bodies.parquet` and a manifest binding source hashes
and recovery revision. References include per-body COM state, gravity, wrenches,
vertex ownership (int16, -1 unassigned) and validity. Unsupported dynamics
remain neutral; ready publication requires all required consistency checks.

```bash
python scripts/labels/solve.py --help
python scripts/labels/publish.py --help
```

Solving writes canonical NPZ. Publication creates target slots, timeline and
supervision spans; invalid force rows are zeroed. `--install` validates payloads
and atomically activates the prepared manifest while retaining prior references
for rollback. A full-container migration requires its full dataset rollback,
not a single-info restore. Missing optional modalities remain explicitly
unavailable rather than becoming zero-valued labels.

Geometry changes mark dependent labels stale; training must reject them. A new
solve binds the HMR fingerprints and geometry revision before publication can
clear stale. Rebuild dependent statistics/references when labels change, and
verify label hashes and the actual handcrop consumer before installation.
