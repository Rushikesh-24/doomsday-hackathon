#!/usr/bin/env python
"""Pre-download and verify model weights into models_cache/ (so the API can run offline).

    python scripts/download_models.py                      # default model (+ members)
    python scripts/download_models.py --all
    python scripts/download_models.py yolo_world_l grounding_dino_tiny
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.models.registry import get_registry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    reg = get_registry()
    keys = list(reg.specs) if a.all else (a.models or [reg.default_key])
    ok = True
    for k in keys:
        try:
            m = reg.get(k)
            i = m.info()
            print(f"OK   {k:<22} device={m.device:<8} sha256={(i.weights_sha256 or '')[:16]} rev={i.revision}")
        except Exception as e:
            ok = False
            print(f"FAIL {k:<22} {e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
