"""Probe the explicitly selected device; CUDA failures never fall back to CPU."""

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fame_scm.devices import device_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        result = device_report(args.device, probe=True)
    except (RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "unavailable", "device": args.device, "error": str(exc)},
                indent=2,
            )
        )
        raise SystemExit(1)
    print(json.dumps({"status": "ready", **result}, indent=2))


if __name__ == "__main__":
    main()
