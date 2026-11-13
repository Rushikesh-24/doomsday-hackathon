"""PS32 Campus Facility Vision Engine."""
__version__ = "0.1.0"

import os as _os

# Never let Ultralytics pip-install packages at runtime (it auto-installed `lap` once during development);
# all dependencies are pinned in requirements.txt.
_os.environ.setdefault("YOLO_AUTOINSTALL", "false")
