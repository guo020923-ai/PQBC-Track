import json
import sys

from .cache import build_cache


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: cache_child OUTPUT BEHAVIOR_NUMERIC")
    print(json.dumps(build_cache(sys.argv[1], sys.argv[2]), sort_keys=True), flush=True)
