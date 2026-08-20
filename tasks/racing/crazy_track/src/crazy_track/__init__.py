"""DATT Figure-5 trajectory-tracking benchmark on the crazyflow simulator."""

import os

# crazyflow requires this before scipy is imported anywhere in the process.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

__version__ = "0.1.0"
