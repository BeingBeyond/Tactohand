# TactoHand label format (labels-v5)

Open the prepared manifest referenced by `<dataset>_lerobot/meta/info.json`.
Its integer ranges map episodes, streams, and logical rows to Parquet files
and file offsets. An episode can span several files and row groups. RGB,
HMR, camera, and handcrop streams have separate indices.

## Tactile tables

| Table | Row unit | Contents |
|---|---|---|
| `samples` | Solver row | `contact`, `signed_distance` |
| `transitions` | Solver state row | Per-vertex world-frame forces, contact and environment wrenches, and target-object center-of-mass pose and velocity |
| `timeline` | Solver row | `solver_row`, `solver_time_s`, `source_position`, `tactile_flags`, `force_valid`, `target_object`, `closing_solver_row` |

- Compact manifests declare `row_metadata: tactile.timeline`. The tables
  align by episode and solver row. Readers reconstruct label fields and
  supervision spans from the timeline and also accept earlier layouts that
  store duplicate fields. Sparse static datasets can declare
  `solver_index_column: source_position` when the original label indices
  match the source frame indices.
- Bits 0 and 1 in `tactile_flags` encode left- and right-hand presence. Bits 2
  and 3 encode contact and proximity validity. `contact` is uint8 and
  `signed_distance` is float16 in metres, clipped to ±0.25 m. Each row
  stores 778 MANO vertices per hand, with the left hand first.
- `force_valid` marks rows with valid force supervision. Forces and wrenches
  use the solver's reference mass. Mass, center of mass, inertia, and
  `object_ids` align with the target slot. Articulated episodes also store
  generalized wrenches and joint states.
- `source_position` is a source-video frame coordinate and may be fractional
  after solver resampling. For sparse static samples, use the payload row to
  read labels and the source frame position to read RGB and camera data.
  Static rows have null solver times and closing rows. Each can have its own
  object slot and gravity vector.
- Dynamic rows have a `closing_solver_row` when the next state uses the same
  target reference frame. Direct force supervision can use the last valid
  row of a clip. Rollout requires a contiguous valid interval and a closing state.
- The episode's tactile context records topology, rates, physical parameters,
  and content digests. `tactile_sha1` hashes the logical arrays in the order
  `solver_sample`, `source_position`, `flags`, `contact`, and
  `signed_distance`, independently of Parquet sharding. `force_sha1` covers
  the declared force, state, and auxiliary physics fields.

## Reading an episode

Use the shared readers instead of assuming physical parquet/video filenames.
For example, read contact and proximity labels from one DexYCB episode:

```python
from pathlib import Path
from core.tactile_labels.decode import load_label_index, read_episode_columns

root = Path("/path/to/processed/dexycb_lerobot")
episode = load_label_index(root, episode_indices=[0])[0]
labels = read_episode_columns(
    root, episode["tactile_groups"],
    ["source_position", "flags", "contact", "signed_distance"],
)
contact = labels["contact"].reshape(-1, 2, 778)
distance_m = labels["signed_distance"].reshape(-1, 2, 778)
```

Use `source_position` and the declared frame mappings to align label rows with
video. Apply the hand-presence and supervision-validity flags before computing
metrics. Compare predicted proximity with the target defined from the signed
reference distance. Contact-only datasets do not provide valid force targets.

For RGB/geometry access, use `core.data.processed_index.ProcessedIndex` and the
shared readers. For model inputs, use `training.data` so sampling, handcrop
intrinsics and force-supervision gates match training.

## Training and evaluation

Training selects clip origins from decodable source frames. Per-dataset
quotas determine the number of samples per epoch. The reader then samples
images at the model's cadence and checks RGB mapping, geometry alignment,
and force validity separately. Each visual step spans up to three 30 Hz
solver intervals. Direct force supervision uses the solver row aligned with
the input image. Rollout uses consecutive valid intervals with a closing
state for the same target. For sparse video, retain the source view's
causal hold and align geometry to the frame actually displayed.
Validation and evaluation use fixed sampler seeds and record the prepared
manifest and force scale.

The bounded window cache stores labels by episode. Single-target full-frame
windows can be indexed from the row count and video coverage. Multi-target
segments are split by target slot. The training fast path can read just the
vertex-local labels needed by the single-substep model. The corresponding
solver rows still provide camera-frame reference forces and rollout masks.
Contact-only configurations mask both direct force and rollout supervision.

## Solving and publishing

Run from the repository root. Set the solver paths, device, and worker count
through command-line arguments.

```bash
python scripts/labels/solve.py --help
python scripts/labels/publish.py --solver-run <run> --dataset <name> --processed-root <root>
```

The solver writes NPZ files. The publisher derives target slots and
supervision spans from valid solver rows. For multi-rigid-body segments, it
uses the annotated target when available and otherwise selects a loaded
object by wrench energy over the segment. It zeros invalid force rows and
sums object contributions to obtain per-vertex forces. Existing converted
data retain their published reference frames and force values.

Add `--install` to publish versioned tactile components and episode context.
The publisher checks files and ranges before atomically updating
`meta/info.json`. The RGB, HMR, camera, object, and frame indices are
preserved, and the old references are kept for rollback.
`scripts/labels/compact_tactile_release.py` verifies that redundant fields
can be recovered from the timeline before writing a candidate manifest and
validation record. Update force statistics and rigid-body references when
a new solve changes the dataset manifest on which they depend.

## Rigid-body references

The optional `rigid_reference_v1` sidecar contains a per-episode
`bodies.parquet` and `manifest.json`. The rows align with the solver and
record per-body center-of-mass position, rotation, velocity, residual
external wrench, gravity in world coordinates, vertex-to-body assignment
for both hands (int16, with `-1` for unassigned), and body validity.
`object_ids` follow the original label order. The `environment` wrench
is inferred from the published forces and reference dynamics.

Recovery checks the published target states, velocities, and aggregate
wrenches. Body-level `dynamics_valid` accounts for pose interpolation and
its Gaussian smoothing neighborhood. Bodies without observed dynamics
are neutral and contribute neither hand/environment wrench nor rollout
supervision. Other valid bodies remain eligible. The stored `valid` mask
combines force validity, dynamics validity, and recovery consistency.
Reference validation leaves the original force data and masks intact.
Training reads this sidecar directly. Label hashes, the dataset manifest,
recovery version, and payload SHA-256 identify stale references.

Vertex assignment breaks ties within 0.1 mm by preferring the target
object, then the lowest slot. The default `trust_degenerate` policy accepts
finite BVH results on degenerate geometry and uses exact float64 queries
when BVH fails. The reference preparation option `--exact-degenerate`
also uses exact queries for degenerate geometry. The manifest records
the policy in `report.ownership_policy`.

## MANO geometry

Decode MANO through `core/geometry/mano_layer.py`. Migrated HMR uses the
corrected model with `[trans3, full_axis_angle48, betas10]` in metres and
radians, `flat_hand_mean=True`, `use_pca=False`, and `center_idx=None`.
The loader uses the model fingerprint to apply any required shape-basis
correction once. Source adapters must declare the basis used to generate the
parameters.

For unmigrated parameters, `meta/info.json.mano_source_basis` can record
`version=1`, `basis=stock|corrected`, `decision_status`, `assets_sha256`,
and `decision_evidence`. Use `decision_status=user_confirmed` for a manual
selection, and record that selection as the evidence. Readers prefer the
installed `mano_contract`, then `mano_source_basis`, then a legacy
registry. The right-hand model is shared. This metadata records the decode
choice. Unrelated changes such as video paths leave it valid.

Use `source_position` to select geometry and source frames, including
static samples and repeated PROX frames. Aggregate segmented crop coverage
using `frame_offset`. When `hmr.mano=false` and `mesh_vertices=true`, read
the stored vertices. A `mesh_topology=mano_778` declaration allows reuse of
MANO faces.

For packed crop videos with `file_frame_offset_source=from_timestamp`, use
`round(from_timestamp * fps)` to preserve gaps in the index. Individually
stored segments start at timestamp zero after normalization. When native
RGB is absent, a verified `crop_coordinate_space.raw_width/raw_height`
can still define the crop coordinate canvas.

`meta/info.json.mano_contract` records the basis, pose and translation
conventions, model hashes, source evidence, shape policy, and
`geometry_revision`. Migration from stock to corrected requires a maximum
vertex error of 1 mm over present, finite frames. HIC has a 2 mm limit
recorded in `precision_acceptance`, based on a fit with a maximum error of
about 1.8983 mm. The manifest records `surface_error_limit_mm` and
`surface_error_max_mm`. Install a dataset only after all declared shards
pass. The fit allows per-frame shape coefficients.

A parameter migration or a confirmed solver-basis mismatch marks
`mano_dependent_labels.state=stale`. Training rejects stale labels.
A new solve binds the `geometry_revision` and HMR fingerprints in
`mano_geometry_contract.json`. The publisher verifies these before
clearing the flag and continues to apply the handcrop checks. For HIC,
regenerate hand–hand contact and signed distance with corrected MANO
while retaining the fractional solver-to-source timing. A corrected
declaration requires a resolved source basis and validated geometry.

### Shared full-frame payloads and explicit crop-coordinate canvases (2026-10-20)

`raw_videos` may alias `videos` when the installed video was encoded at source
resolution, with identical image plane and frame order. Declare
`full_frame_video_source.kind=native_shared`, `same_resolution_as_videos=true`,
and `capabilities.raw_videos.shares_encoded_files_with=videos`; the independent
raw observation index still resolves every episode/view and packed offset.
A calibrated/resized processed source instead declares
`full_frame_video_source={storage: videos, kind: calibrated_processed,
native_resolution: false}` and does not imply that native raw is installed.

RGB storage dimensions and encoded `*_orig` box units are independent.
`beingh_human.crop_coordinate_space` is authoritative for stored geometry,
including when a physical/shared raw mirror exists. A global canvas retains
`raw_width`/`raw_height`; mixed canvases use `by_video_key`, mapping each exact
`observation.images.*` key to its own `raw_width`/`raw_height`. The encoding is
`observation_pixels_scaled_per_axis_to_declared_canvas`. Consumers must scale
through this declaration once; they must not substitute the shared video's
physical resolution for the coordinate canvas or treat rectified processed
pixels as original fisheye pixels. Video, MANO and stored crop intrinsics need
no rewrite when restoring this declaration.

A calibrated processed crop source may coexist with a separately installed
native raw index. Restoring native raw must not change the image plane used to
render or verify existing rectified crops. `capabilities.raw_videos` describes
actual native coverage independently of `full_frame_video_source`, which remains
the declared crop reconstruction source. Raw-only audits explicitly select the
independent raw index rather than the crop source.

For ReInterHand, native ego RGB retains its original 1024x1024 fisheye plane;
native Mugsy RGB is 667x1024. The processed 384x384 ego and 250x384 Mugsy streams
and their calibrated crop geometry remain in the processed pinhole plane.
`raw_video_encoding.calibration_path` points to a metadata attachment containing
per-episode/view native focal lengths, principal points, distortion model and
coefficients. Native intrinsics must not replace processed camera intrinsics.

### Handcrop geometry storage v2 (2026-10-20)

Geometry payloads use
`crop_data/hand/observation.images.<view_id>.<target_id>/chunk-XXX/file-XXX.parquet`.
`meta/cropped_video_observations.parquet` remains the authoritative segment
index: its path, row group, offset and row count are independent of video
chunk/file numbering. Each payload contains one view/hand stream. Writers pack
small segments with targets of 32,768 rows per row group and 262,144 rows per
file (a single larger segment may exceed these targets); segment boundaries,
video timestamps and observation frame indices are unchanged.

All datasets use the same Arrow schema: non-null `episode_index:int64`,
`view_id:string`, `target_id:string`, `frame_index:int64`, three float64 fixed
lists of length four (`crop_box_xyxy_orig`, `hand_bbox_xywh_orig`,
`hand_bbox_xywh`), and independent non-null `frame_valid:bool` and
`crop_valid:bool`. `cam_intr` is a nullable typed struct of float64
`fx, fy, cx, cy` describing the crop projection. A null struct preserves a
legacy absence of stored crop intrinsics; readers derive K through the shared
camera/crop path. It does not assert that K is zero. No opaque MessagePack
intrinsics are written in v2; shared readers continue to decode v1 MessagePack.

`beingh_human.crop_data.version=v2` declares the layout and field semantics;
Parquet schema metadata also records v2. `crop_coordinate_space.kind` is
`normalized_observation`: both `*_orig` fields use unbounded normalized
coordinates in the observation camera's projection plane (x and width divided
by image width, y and height by image height). The xyxy and xywh conventions
remain distinct. Negative/out-of-image windows are retained for black padding;
normalization must never clip them. `hand_bbox_xywh` remains in crop pixels.
This encoded plane is separate from the distortion and raster size of native
raw RGB. The shared crop canvas resolver returns a unit canvas for v2.

Migration must preserve masks, frame ordering, all non-geometry index columns,
intrinsic float64 values and the previous physical crop transform. It validates
all rewritten row groups and the actual shared consumer before/after install.

### Handcrop geometry v3: derive crop intrinsics (2026-10-20)

v3 supersedes v2's duplicated `cam_intr` field. Canonical crop Parquet payloads
contain only the nine identity/frame, box and validity fields; no binary or
structured crop camera intrinsics are stored. Root layout, segment locators,
unbounded normalized `*_orig` boxes and crop-pixel hand bbox semantics remain
unchanged. Metadata declares `crop_data.version=v3` and
`intrinsics=derived_from_observation_camera_and_crop_box`.

The authoritative K is the observation camera's full-frame projection K,
resolved through `observation.camera_id` and `meta/cameras.parquet` in the
same image plane as `videos`. Denormalize the crop window using observation
width/height, then derive K with the shared `frame_intrinsics` /
`crop_adjusted_intrinsics` path and the indexed crop output dimensions. Native
raw fisheye K must not replace a rectified processed camera K. Padding outside
the image keeps the original crop-window origin.

Every valid stored legacy K is checked against this derivation before its
column is discarded (maximum component error 0.001 crop pixels). Differences
above that gate prevent publication. Legacy v1/v2 storage remains readable;
`HandcropStreamIndex.crop_intrinsics` returns no stored K for v3, so consumers
use the shared derivation. Migration preserves video, label, camera and HMR
payloads and archives the previous geometry for rollback.


### Processed dataset container v1 (2026-10-20)

The revision-5 label semantics are unchanged. A dataset whose `meta/info.json`
contains `format_version=processed_dataset_v1` stores only `meta/`, `data/`,
and `assets/`. The metadata root contains `info.json`, `provenance.json`,
`stats.parquet`, `tables/`, and `index/`. `info.json.manifest` points to the
integer index manifest; caches pin its revision. Supplemental model and
visualization outputs are preserved as registered assets without changing
their internal format. The published tree contains materialized files.

The three index families are episodes, frames, and per-modality data_ranges.
Each range has `episode, stream, start, count, path, offset`; bounds are
half-open and offset is a file-global row or video frame. Sparse camera frames
remain absent. Timestamps are auxiliary, and solver_sample, solver_transition,
and token retain their own domains plus source-frame mappings. Geometry frame
numbers follow canonical episode frames after translating packed video indices.
Crop v3 omits stored K; observation-camera K and normalized crop boxes define
crop intrinsics, independently of native raw distortion and resolution.

`core.data.processed_index` reads the normalized storage. `core.data.processed_compat`
provides legacy reader views derived from it, including row-group slices;
consumers must honor row_offset and num_rows. The container does not retain
standalone staging paths such as `meta/label_index.parquet` or `force_data`
paths. Labels and force streams are modalities under data/, with native
scientific contracts retained in provenance and the manifest. All-null optional
fixed vectors may use nullable variable-list encoding, retaining NULL values.
Predicted proximity scores retain their model-score semantics and do not become
revision-5 physical distances merely by sharing a container.

Native token contact_valid is distinct from contact-only training admission:
force-valid tokens enter force runs; contact fallback runs exclude those tokens.
Targets with object index -1 have no target run. Compatibility reconstruction
must preserve the complete original training-run partition. Object identity
continues to join instance to asset to geometry, never by numeric label alone.

Optional source sidecars whose entire payload tree is absent may be explicitly
disabled before migration, retaining original metadata in the external repair
backup. `manifest.unavailable_modalities` records `available=false` and its
reason; compatibility views preserve those explicit declarations. Missing
tactile, body-tracking or object payloads do not become zero-valued labels.
Historical prediction caches are not declared as installed coverage.


### Static object registry: object_assets_v1 (2026-10-21)

Processed object assets may use `assets/objects/assets.json` plus
`assets/objects/meshes.bin`. The manifest's `object_scene.asset_registry_format`
is `object_assets_v1`. `assets_path`, `geometries_path`, and optional
`articulation.parts_path` resolve to sections `assets`, `geometries`, and
`parts` in the JSON document's `tables` object. `mesh_store_path` resolves to
the unchanged little-endian float32 vertices/int32 triangle binary store.
Original Arrow schemas are retained for exact typed reconstruction; nonfinite
scientific values use explicit `__object_assets_float__` JSON tags.

`core.geometry.object_assets.read_asset_table` reads both the JSON registry and legacy
Parquet tables. Asset identity remains the exact string
`object_asset_id -> object_geometry_id` join. Dynamic per-frame facts,
instance membership, relationship groups and state tables remain Parquet.
This storage change does not alter any tactile/force labels or mesh arrays.


### Unified handcrop geometry (2026-10-21)

Processed `crop_geometry` payloads may declare
`context_format: embedded_handcrop_context_v1` in the manifest and
`beingh.handcrop.context=embedded_handcrop_context_v1` in their Arrow metadata.
They contain the existing nine v3 geometry/mask fields plus `crop_type`
(string), `width`, `height`, and `stream_num_frames` (int64). These fields
are joined exactly by episode/view/target and stored with every geometry row.
Independent `handcrop_context` payloads, modality declarations and range
indices are removed after validation. Pixel/normalized coordinate conventions,
frame IDs, physical row order, row-group boundaries and handcrop video offsets
remain unchanged. Legacy nine-field v3 geometry remains readable.
