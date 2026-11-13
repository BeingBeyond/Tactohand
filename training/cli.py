"""Public current-model CLI; generic research dispatch stays in train.py."""
from __future__ import annotations

import sys


def train_main() -> None:
    defaults = []
    for group, option in (("experiment", "tactohand"), ("data", "tactohand")):
        if not any(arg.lstrip("+").startswith(group + "=") for arg in sys.argv[1:]):
            defaults.append(f"{group}={option}")
    sys.argv[1:1] = defaults
    from training.train import main

    main()
