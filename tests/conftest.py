"""
Shared pytest fixtures for Phase 3 modularization tests.
"""
import sys
from pathlib import Path

# Ensure project root is on the path so 'src' package is importable.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
