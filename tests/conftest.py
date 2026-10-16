"""Keep test temporaries in the workspace; never bulk-clean user files."""
import os
import tempfile
from pathlib import Path


def pytest_configure(config):
    root = Path(__file__).resolve().parents[1] / ".local_tmp"
    root.mkdir(exist_ok=True)
    if config.option.basetemp is None:
        # A fresh, previously nonexistent directory avoids pytest's rmtree path.
        config.option.basetemp = str(root / ("pytest-" + os.urandom(8).hex()))
    os.environ["TEMP"] = os.environ["TMP"] = str(root)
    tempfile.tempdir = str(root)
