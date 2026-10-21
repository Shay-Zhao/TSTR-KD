#!/usr/bin/env python3
"""Public EPFD evaluation entry point; run python test_epfd.py --help.

Keep evaluation/, models/ and utils/ beside this file. The YOLOv7 runtime is
bundled; supply weights and validation data separately. Examples use relative paths.
"""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "evaluation"))

from validate_epfd import main


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
