#!/usr/bin/env bash
# Parallel runner for the figure, table, and analysis reproducibility scripts.
#
# Each script's stdout/stderr is written to data/logs/<basename>.log with
# wall-clock timing printed at the top and bottom so the bundled logs
# document how long the live-from-CSV reproduction takes.
#
# Usage:
#   bash code/run_all.sh           # run everything in parallel
#   bash code/run_all.sh figures   # only the per-figure scripts
#   bash code/run_all.sh tables    # only the per-table scripts
#   bash code/run_all.sh analysis  # only the standalone analysis scripts

set -u
cd "$(dirname "$0")/.."

LOG_DIR=data/logs
mkdir -p "$LOG_DIR"

FIGS=(
    code/figures/fig_1.py
    code/figures/fig_4.py
    code/figures/fig_5.py
    code/figures/fig_6.py
    code/figures/supplementary_figure_4.py
    code/figures/supplementary_figure_5.py
    code/figures/supplementary_figure_6.py
    code/figures/supplementary_figure_7.py
)
TABLES=(
    code/tables/table_1.py
    code/tables/supplementary_table_2.py
    code/tables/supplementary_table_3.py
    code/tables/supplementary_table_4.py
)
ANALYSES=(
    code/analysis/interaction_power_diagnostic.py
)

case "${1:-all}" in
    figures)  SCRIPTS=("${FIGS[@]}") ;;
    tables)   SCRIPTS=("${TABLES[@]}") ;;
    analysis) SCRIPTS=("${ANALYSES[@]}") ;;
    all)      SCRIPTS=("${FIGS[@]}" "${TABLES[@]}" "${ANALYSES[@]}") ;;
    *) echo "usage: $0 [figures|tables|analysis|all]" >&2; exit 2 ;;
esac

run_one() {
    local script="$1"
    local log="$LOG_DIR/$(basename "$script" .py).log"
    {
        echo "Run started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Script:      $script"
        echo "Host CPUs:   $(nproc)"
        echo "----"
        local t0 t1 elapsed
        t0=$(date +%s.%N)
        python3 -W ignore "$script"
        t1=$(date +%s.%N)
        elapsed=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }')
        echo "----"
        echo "Run ended:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "Wall clock:  ${elapsed}s"
    } > "$log" 2>&1
    local elapsed_quiet
    elapsed_quiet=$(awk '/^Wall clock:/ {print $3}' "$log")
    echo "[$(basename "$script" .py)] ${elapsed_quiet}"
}

export -f run_one
export LOG_DIR

t0=$(date +%s.%N)
# supplementary_figure_4.py must complete before table_1.py: Table 1's Cohen's kappa row is
# read from the Supplementary Figure 4 log rather than recomputed, so running them concurrently races the
# 215 s bootstrap against the 45 s table and emits placeholders. Run it first, then the rest.
SUPPFIG4=code/figures/supplementary_figure_4.py
REST=()
for s in "${SCRIPTS[@]}"; do [ "$s" = "$SUPPFIG4" ] || REST+=("$s"); done
for s in "${SCRIPTS[@]}"; do
    if [ "$s" = "$SUPPFIG4" ]; then run_one "$SUPPFIG4"; break; fi
done
if [ ${#REST[@]} -gt 0 ]; then
    printf '%s\n' "${REST[@]}" | xargs -P 0 -I{} bash -c 'run_one "$@"' _ {}
fi
t1=$(date +%s.%N)
total=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }')
echo "----"
echo "All scripts done in ${total}s wall clock (parallel, ${#SCRIPTS[@]} jobs, $(nproc) CPUs)."
