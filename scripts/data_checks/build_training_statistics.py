#!/usr/bin/env python3
"""Build versioned training-only loss statistics from prepared labels."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.data.training_statistics import prepare

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--processed-root', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[2] / 'core/resources/tactohand/split_v1.json')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.processed_root, args.manifest, args.output)
