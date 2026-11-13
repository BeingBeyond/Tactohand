#!/usr/bin/env python3
"""Publish prepared labels through the shared generation workflow."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from label_generation.publish import main

if __name__ == "__main__":
    main()
