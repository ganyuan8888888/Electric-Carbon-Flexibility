"""Portable archived-input root, overridable for isolated fresh computations."""
from pathlib import Path
import os
REPOSITORY=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get("ECF_DATA_ROOT",str(REPOSITORY/"data/reference"))).resolve()
