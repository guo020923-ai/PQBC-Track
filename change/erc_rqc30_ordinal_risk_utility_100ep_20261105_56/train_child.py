"""Dedicated child entry point for the fixed 100-epoch GPU job."""

import json
import sys

from .training import train_epochs


def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: train_child OUTPUT CACHE")
    result = train_epochs(argv[1], argv[2])
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(sys.argv)
