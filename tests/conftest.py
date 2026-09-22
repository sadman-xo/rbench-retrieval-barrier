import sys
from pathlib import Path

# Let `pytest tests` import the local rbench package without an install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
