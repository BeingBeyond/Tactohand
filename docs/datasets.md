# Dataset conventions

Use the installed metadata for paths, frame mappings, and splits.
`meta/provenance.json` records the data sources, and
`core.data.processed_compat` provides the shared reader.

## Shared conventions

- Join pose rows, source frames, and model tokens using the declared mappings.
- Read frame, camera, and geometry data through the shared readers. File
  numbering alone does not identify an episode.
- Interpret legacy crop boxes in the declared raw-mirror coordinate canvas and
  obtain crop intrinsics through `frame_intrinsics`.
- Decode prepared MANO parameters through `core.geometry.mano_layer`. The
  installed parameters already incorporate any coordinate transforms or
  offsets applied during preparation.
- Use the installed split, hand-presence flags, and supervision masks.

<a id="arctic"></a>

## ARCTIC

- Native ego video may use a distorted image plane, while processed ego video
  may be rectified. Use the calibration for the relevant image plane.
- Resolve calibration by subject and camera. View IDs can recur across subjects.
- Use the installed paths, dimensions, and ranges. Full-frame, native, and
  cropped streams can have different shard boundaries.
- Resampling the image plane changes intrinsics and image-space boxes. World
  MANO parameters, object poses, and force labels retain their coordinates.

<a id="dexycb"></a>

## DexYCB

- Source sequences contain 30 fps multi-camera RGB-D recordings of YCB object
  interactions. Use each sequence's annotated hand side and presence flags.
  The [official toolkit](https://github.com/NVlabs/dex-ycb-toolkit/blob/master/dex_ycb_toolkit/dex_ycb.py)
  reads the side from `mano_sides[0]`.
- The source includes MANO parameters, joints, camera intrinsics, and object
  poses. Dense contact and force labels are generated during preparation.
- Use the installed split and timeline. Resolve object geometry through
  `object_asset_id → object_geometry_id`.

<a id="h2o3d"></a>

## H2O3D

- Source preparation has used `[x, y, z] → [x, -y, -z]` for OpenGL-to-OpenCV
  conversion. Apply camera-frame rotation to the MANO root rotation only.
- Hand-specific translation offsets belong to source preparation. Use the
  installed translations directly.
- For one-hand offsets, inspect the side and source parameter convention. For
  two-hand offsets, inspect frame alignment, calibration, and projection.
- Apply the pose and translation validity flags to hand supervision.

<a id="oakink-v2"></a>

## OakInk-v2

- Verify the alignment of native RGB, canonical RGB, and labels for the
  selected camera and episode. Historical offsets and six-frame repetition
  patterns were specific to affected streams.
- Match the source sequence, camera, and annotation IDs. Use the installed
  index for the episode length within a shared shard.
- Use source timestamps to establish frame rate. Repeated pixels alone do not
  establish capture cadence. Keep labels aligned to the declared source mapping
  and exclude corrupted frames when checking geometry.

<a id="prox"></a>

## PROX

- Source acquisition runs at **5 fps**. Use its physical timeline when an
  older video container declares 30 fps. The 18 historical video chunks
  belong to a continuous recording.
- Some prepared supervision uses 5 fps-to-10 Hz frame repetition such as
  `[0, 0, 1, 1, …]`. Use the declared token-to-source mapping.
- Legacy crop boxes use the 1920 × 1080 raw-mirror canvas.
- The installed tactile labels describe hand–scene contact with the Vicon
  scene. In `solver_vertex_tracks(..., rows=...)`, `rows` indexes the
  `source_position` array.

<a id="rich"></a>

## RICH

- Source acquisition runs at **30 fps**. Historical episode chunks can belong
  to one continuous recording.
- Legacy crop boxes use the 1080 × 790 raw-mirror canvas. Read crop intrinsics
  through the shared reader.
- The installed tactile labels describe hand–scene contact. Check whether the
  installed dataset includes the corresponding scene mesh before recomputing
  geometry.
