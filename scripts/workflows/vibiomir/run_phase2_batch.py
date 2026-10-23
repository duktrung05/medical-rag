#!/usr/bin/env python3
"""CLI shim for the bounded Phase 2 migration runner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medrag.data.phase2_batch_runner import main


if __name__ == "__main__":
    main()
