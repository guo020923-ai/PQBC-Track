"""One fresh cache+30-epoch training entry; no evaluation or retry."""

import argparse
import json
from pathlib import Path

from .cache import build_cache
from .config import PROTOCOL, validate_protocol
from .io_utils import write_new_json
from .training import train_epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    authorization = json.loads(Path(args.authorization).read_text())
    if authorization.get("scope") != "one_fresh_rqc_style_erc30":
        raise PermissionError("wrong authorization scope")
    if not authorization.get("explicit_user_training_authorization"):
        raise PermissionError("explicit training authorization required")
    if authorization.get("maximum_training_launches") != 1:
        raise PermissionError("one launch required")
    if authorization.get("protocol") != validate_protocol():
        raise PermissionError("authorization protocol mismatch")
    if output.exists():
        raise FileExistsError("fresh output required: " + str(output))
    output.mkdir(parents=False)
    write_new_json(output / "authorization.json", authorization)
    write_new_json(output / "protocol.json", PROTOCOL)
    try:
        build_cache(output)
        train_epochs(output)
    except Exception as error:
        write_new_json(output / "failed.json", {
            "error": repr(error), "automatic_retry": False,
            "official_auc": None,
        })
        raise


if __name__ == "__main__":
    main()
