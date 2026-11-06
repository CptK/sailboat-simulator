#!/usr/bin/env python3
"""Let a vision-language model steer the boat: camera image + instruments in, course change out.

    python3 scripts/llm_pilot.py landgraben --target green
    python3 scripts/llm_pilot.py landgraben --target green --pilot scripted   # no API needed

The code lives in the sail_pilot package (src/sail_pilot); see
`python3 scripts/llm_pilot.py --help` for all options.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "sail_pilot"))

from sail_pilot.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
