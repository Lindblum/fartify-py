"""Rebuild samples/samples_index.json from the command line.

Run this after adding, removing, or replacing files in samples/ so the
app's cached metadata (length, volume, note frequency, envelope shape,
etc.) stays in sync -- without needing to start the Flask server.

Usage:
    python rebuild_samples.py            # rebuild only if the folder changed
    python rebuild_samples.py --force    # always rebuild from scratch
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fartify.sample_library import build_index, load_index  # noqa: E402

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild every record even if the sample files haven't changed",
    )
    args = parser.parse_args()

    if args.force:
        records = build_index(SAMPLES_DIR, force=True)
    else:
        records = load_index(SAMPLES_DIR, rebuild_if_stale=True)

    print(f"{len(records)} samples indexed in {os.path.join('samples', 'samples_index.json')}\n")
    for r in sorted(records, key=lambda r: r.filename.lower()):
        print(
            f"  {r.filename:<45} "
            f"len={r.duration_sec:5.2f}s  "
            f"eff={r.effective_duration_sec:5.2f}s  "
            f"vol={r.rms:5.3f}  "
            f"freq={r.frequency_hz:6.1f}Hz  "
            f"pitch_conf={r.pitch_confidence:4.2f}"
        )


if __name__ == "__main__":
    main()
