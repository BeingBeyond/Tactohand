"""Locate versioned TactoHand resources without depending on the caller's cwd."""
from pathlib import Path


RESOURCE_PREFIX = "core/resources/tactohand/"
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources" / "tactohand"


def resolve_resource_path(path: str | Path) -> Path:
    """Anchor bundled config paths; preserve explicit external path semantics.

    Old run configs pointing into artifacts/ remain external paths. They are
    never silently redirected to a different statistics or curation snapshot.
    """
    value = str(path)
    if value.startswith(RESOURCE_PREFIX):
        relative = Path(value.removeprefix(RESOURCE_PREFIX))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid bundled resource path: {path}")
        return RESOURCE_ROOT / relative
    return Path(path).expanduser()
