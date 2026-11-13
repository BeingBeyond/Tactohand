"""Video decoder configured with the published processed-data timing contract."""
from core.io.video_reader import VideoReader
from core.data.video_timing import published_cfr_grid


class PreparedVideoReader(VideoReader):
    """Preserve exact/CFR behavior while keeping manifest access above IO."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cfr_grid_provider", published_cfr_grid)
        super().__init__(*args, **kwargs)
