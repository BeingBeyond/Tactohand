#!/usr/bin/env python3
"""Schedule label generation using the independent solver workflow."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from label_generation.scheduler import main

if __name__ == "__main__":
    main()
