"""Stream large checkpoints to disk while keeping atomic replacement."""

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from pytorch_lightning.plugins.io import TorchCheckpointIO
import torch


class StreamingCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(
        self,
        checkpoint: dict[str, Any],
        path: str | Path,
        storage_options: Any = None,
    ) -> None:
        if storage_options is not None or "://" in str(path):
            super().save_checkpoint(checkpoint, path, storage_options)
            return
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        stream = temporary.open("xb")
        try:
            with stream:
                torch.save(checkpoint, stream)
            # The temporary file is on the destination filesystem. Readers
            # see the previous complete file until serialization succeeds.
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
