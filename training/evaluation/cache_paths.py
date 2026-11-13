"""Persistent API result storage, including the former output-path alias."""
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONTACTPROMPT_CACHE = REPO / "artifacts/api_cache/contactprompt"
_OLD_CONTACTPROMPT_CACHE = REPO / "outputs/contactprompt_eval_cache"


def contactprompt_cache_path(value: str | Path) -> Path:
    if Path(value).expanduser() == Path("outputs/contactprompt_eval_cache"):
        return CONTACTPROMPT_CACHE
    path = Path(value).expanduser().resolve()
    if path == _OLD_CONTACTPROMPT_CACHE:
        return CONTACTPROMPT_CACHE
    return path
