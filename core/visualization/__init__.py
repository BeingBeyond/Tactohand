"""Shared visualization library (docs/architecture.md).

Layered capabilities for labels/contact/proximity/force visualization:
camera access (L0), projection (L1), drawing primitives (L2), panel
composition + H.264 encoding (L3).  Application scripts (L4) must consume
this package instead of reimplementing projection/drawing/encoding.

Conventions (§1 of the design): BGR uint8 images, float64 world meters,
``c2w`` maps camera→world (OpenCV axes: x right, y down, z forward),
intrinsics from ``meta/cameras.parquet``, videos encoded as H.264 via
ffmpeg pipes.
"""
from core.visualization.camera import (
    CameraIntrinsics, crop_intrinsics, load_cameras, view_camera,
)
from core.visualization.colormaps import (
    FORCE_COLOR, SUPPORT_COLOR, TORQUE_COLOR, force_color, proximity_color,
    target_force_color, turbo_color,
)
from core.visualization.drawing import (
    draw_arrow_world, draw_contact_markers, draw_force_legend, draw_header,
    draw_legend, draw_mesh_projection, draw_mesh_world, draw_vertex_cloud,
    put_text,
)
from core.io.encoding import encode_frames, encoder, transcode_h264
from core.visualization.mesh_render import (
    OverlayRenderer, letterbox_frame, render_mesh_overlay, scale_intrinsics,
)
from core.visualization.object_mesh import (
    draw_object_silhouette, episode_articulated_asset, instance_geometry_ids,
    load_object_meshes, load_object_parts, object_color, pose_object_vertices,
)
from core.visualization.panels import compose_row, contact_sheet
from core.geometry.projection import (
    in_frame, project_camera, project_world, world_to_camera,
)

__all__ = [
    "CameraIntrinsics", "crop_intrinsics", "load_cameras", "view_camera",
    "FORCE_COLOR", "SUPPORT_COLOR", "TORQUE_COLOR",
    "force_color", "proximity_color", "target_force_color", "turbo_color",
    "draw_arrow_world", "draw_contact_markers", "draw_force_legend",
    "draw_header", "draw_legend", "draw_mesh_projection", "draw_mesh_world",
    "draw_vertex_cloud", "put_text",
    "OverlayRenderer", "letterbox_frame", "render_mesh_overlay",
    "scale_intrinsics",
    "draw_object_silhouette", "episode_articulated_asset",
    "instance_geometry_ids", "load_object_meshes", "load_object_parts",
    "object_color", "pose_object_vertices",
    "encode_frames", "encoder", "transcode_h264",
    "compose_row", "contact_sheet",
    "in_frame", "project_camera", "project_world", "world_to_camera",
]
