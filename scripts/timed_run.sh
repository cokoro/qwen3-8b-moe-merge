#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 EXPERIMENT_NAME COMMAND [ARGS...]"
  exit 2
fi

NAME="$1"
shift

mkdir -p logs results
START_EPOCH=$(date +%s)
START_ISO=$(date -Is)

echo "[$START_ISO] START $NAME"

set +e
"$@"
STATUS=$?
set -e

END_EPOCH=$(date +%s)
END_ISO=$(date -Is)
ELAPSED=$((END_EPOCH - START_EPOCH))

printf '%s,%s,%s,%s,%s\n' \
  "$NAME" "$START_ISO" "$END_ISO" "$ELAPSED" "$STATUS" \
  >> results/experiment_times.csv

echo "[$END_ISO] END $NAME status=$STATUS seconds=$ELAPSED"
exit "$STATUS"
