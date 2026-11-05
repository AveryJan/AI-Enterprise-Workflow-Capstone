#!/usr/bin/env sh
set -eu
python -m scripts.generate_data
python -m scripts.train
python -m scripts.visualize
sh scripts/run_tests.sh
