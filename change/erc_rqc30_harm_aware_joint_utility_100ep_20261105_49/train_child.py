"""File-backed child process for the single authorized training run."""

from pathlib import Path
import sys

from .training import train_epochs


def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: train_child.py OUTPUT_DIR CACHE_DIR")
    output, cache = map(Path, argv[1:])
    train_epochs(output, cache)


if __name__ == "__main__":
    main(sys.argv)
