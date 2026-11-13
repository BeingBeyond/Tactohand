"""labels-v5 storage constants (docs/labels_v5.md)."""
from __future__ import annotations

import pyarrow as pa

LABELS_SCHEMA_NAME = "beingh.labels.tactile_force"
LABELS_REVISION = 5

# Prepared payloads can omit these columns when the timeline is authoritative.
SAMPLE_TIMELINE_COLUMNS = {
    "solver_sample": "solver_row", "source_position": "source_position",
    "flags": "tactile_flags",
}
FORCE_TIMELINE_COLUMNS = {"solver_transition": "solver_row", "flags": "force_valid"}

# Vertex-force storage semantics of the written trees.  "hand_union": the
# flat per-hand force incl. forces acting on non-target objects, with
# contact/environment wrenches summed over all objects and transported to
# the target reference point (rollout should gate on has_non_target_force).
FORCE_SEMANTICS = "hand_union"

PROXIMITY_CLIP_M = 0.25
HAND_COUNT = 2
MANO_VERTICES = 778
_FLAT_VERTICES = HAND_COUNT * MANO_VERTICES          # 1556
_FLAT_FORCE = HAND_COUNT * MANO_VERTICES * 3          # 4668

# flag bits, tactile stream
FLAG_LEFT_PRESENT = 1 << 0
FLAG_RIGHT_PRESENT = 1 << 1
FLAG_CONTACT_VALID = 1 << 2
FLAG_PROXIMITY_VALID = 1 << 3
# flag bits, force stream
FLAG_FORCE_VALID = 1 << 0

GROUP_LOCATOR_STRUCT = pa.struct([("path", pa.string()), ("row_group", pa.int32())])

TACTILE_SCHEMA = pa.schema([
    ("solver_sample", pa.int32()),
    ("source_position", pa.float32()),
    ("flags", pa.uint8()),
    ("contact", pa.list_(pa.uint8(), _FLAT_VERTICES)),
    ("signed_distance", pa.list_(pa.float16(), _FLAT_VERTICES)),
])

FORCE_SCHEMA = pa.schema([
    ("solver_transition", pa.int32()),
    ("flags", pa.uint8()),
    ("vertex_force_world", pa.list_(pa.float32(), _FLAT_FORCE)),
    ("contact_wrench", pa.list_(pa.float32(), 6)),
    ("environment_wrench", pa.list_(pa.float32(), 6)),
    ("object_position", pa.list_(pa.float32(), 3)),
    ("object_rotation", pa.list_(pa.float32(), 9)),
    ("object_velocity", pa.list_(pa.float32(), 6)),
    # articulated roots only; null elsewhere
    ("generalized_wrench", pa.list_(pa.float32())),
    ("joint_state", pa.list_(pa.float32())),
    # Independent static samples may change gravity and object every row.
    # Additive metadata: older trees recover it from static targets/source ids.
    ("static_gravity_world", pa.list_(pa.float64())),
    ("static_object_index", pa.int32()),
])

LABEL_INDEX_SCHEMA = pa.schema([
    ("episode_index", pa.int64()),
    ("solver_fps", pa.float64()),
    ("source_fps", pa.float64()),
    ("n_rows", pa.int32()),
    ("topology", pa.string()),
    ("has_force", pa.bool_()),
    # true when any force-valid row carries nonzero hand force on a
    # non-target object (union semantics); rollout must gate on it
    ("has_non_target_force", pa.bool_()),
    ("dt_s", pa.float32()),
    ("substeps", pa.int32()),
    ("gravity_world", pa.list_(pa.float64(), 3)),
    ("generalized_wrench_dim", pa.int8()),
    ("object_ids", pa.list_(pa.string())),
    ("object_mass_kg", pa.list_(pa.float32())),
    ("object_com_body", pa.list_(pa.list_(pa.float32(), 3))),
    ("object_inertia_body", pa.list_(pa.list_(pa.float32(), 9))),
    ("tactile_groups", pa.list_(GROUP_LOCATOR_STRUCT)),
    ("force_groups", pa.list_(GROUP_LOCATOR_STRUCT)),
    ("canonical_view", pa.string()),
    ("tactile_sha1", pa.string()),
    ("force_sha1", pa.string()),
])

REQUIRED_TACTILE_COLUMNS = ("flags", "contact", "signed_distance",
                            "source_position")
REQUIRED_FORCE_COLUMNS = (
    "flags", "vertex_force_world", "contact_wrench", "environment_wrench",
    "object_position", "object_rotation", "object_velocity",
)

TOPOLOGY_RIGID = "rigid"
TOPOLOGY_ARTICULATED = "articulated"
TOPOLOGY_STATIC = "static"
TOPOLOGY_NONE = "none"
