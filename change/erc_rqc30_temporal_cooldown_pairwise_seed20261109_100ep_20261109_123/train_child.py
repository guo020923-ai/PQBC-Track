import json
import sys

from .training import train_epochs


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: train_child OUTPUT CACHE")
    print(json.dumps(train_epochs(sys.argv[1], sys.argv[2]), sort_keys=True), flush=True)
