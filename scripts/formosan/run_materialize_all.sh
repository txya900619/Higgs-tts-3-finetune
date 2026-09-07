#!/usr/bin/env bash
# Stage 1 driver: materialize all four datasets.
#
# klokah is ~90% of the corpus (413k of ~456k rows) and the stage is
# download/decode bound rather than CPU bound, so its 42 configs are run
# PARALLEL_JOBS at a time; each config is an independent (config, split)
# shard that materialize.py resumes by skipping an existing manifest.
set -uo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python
LOG_DIR="${FORMOSAN_ROOT:-/mnt/md0/user_wayne/formosan_final}/logs"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
mkdir -p "$LOG_DIR"

run_config() {  # dataset, config
    local ds="$1" cfg="$2"
    local tag="${ds##*/}_${cfg}"
    $PY scripts/formosan/materialize.py --dataset "$ds" --config "$cfg" \
        --splits train,eval >"$LOG_DIR/mat_${tag}.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "FAIL rc=$rc $ds/$cfg (see $LOG_DIR/mat_${tag}.log)"
    else
        grep -h '"n_kept"' "$LOG_DIR/mat_${tag}.log" | tail -2
    fi
}
export -f run_config
export PY LOG_DIR

for ds in formospeech/ntu_formosan_corpus formospeech/ithuan_formosan \
          formospeech/nchc_formosan formospeech/klokah; do
    echo "=============== $ds ==============="
    mapfile -t cfgs < <($PY -c "
import sys; sys.path.insert(0,'scripts/formosan')
from common import list_configs
print('\n'.join(list_configs('$ds')))")
    echo "${#cfgs[@]} configs"
    printf '%s\n' "${cfgs[@]}" \
        | xargs -I{} -P "$PARALLEL_JOBS" bash -c 'run_config "$0" "$1"' "$ds" {}
done
echo "=============== stage 1 complete ==============="
