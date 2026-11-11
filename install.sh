#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found. Install Miniconda or Anaconda first." >&2
  exit 1
fi

conda env create -f "${repo_dir}/environment.yml"

echo
echo "PQBC-Track environment created successfully."
echo "Activate it with: conda activate pqbc-track"
