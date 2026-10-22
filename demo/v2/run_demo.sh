#!/bin/sh
# ragaudit v2 demo: generate -> evaluate -> corpus update -> regression evaluate -> tune -> reports.
#
# Offline by default (RAGAUDIT_LLM=mock): every LLM call goes to ragaudit's
# deterministic mock, so this runs anywhere in under a minute with no key.
#
# With a real LLM:
#   export RAGAUDIT_LLM=openai RAGAUDIT_API_KEY=sk-... RAGAUDIT_MODEL=gpt-4o-mini
#   export RAGAUDIT_VERIFIER_MODEL=gpt-4o      # must differ from the generator
#   sh demo/v2/run_demo.sh
#
# Writes the store to .ragaudit/store.db (override with RAGAUDIT_STORE) and
# reports plus the best config to demo/v2/output/ (override with RAGAUDIT_DEMO_OUT).
set -e

cd "$(dirname "$0")/../.."

LLM="${RAGAUDIT_LLM:-mock}"
STORE="${RAGAUDIT_STORE:-.ragaudit/store.db}"
DEMO=demo/v2
OUT="${RAGAUDIT_DEMO_OUT:-$DEMO/output}"
ADAPTER="$DEMO/toy_adapter.py"

if command -v ragaudit >/dev/null 2>&1; then
    RAGAUDIT="ragaudit"
else
    RAGAUDIT="python3 -m ragaudit.cli"
fi

mkdir -p "$OUT"

extract_generate_run_id() {
    sed -n "s/.*(generate run \(.*\))\.$/\1/p"
}

# The toy adapter indexes whichever corpus this points at.
export RAGAUDIT_DEMO_CORPUS="$DEMO/corpus.jsonl"

echo "== 1. generate: verified, de-worded questions, split once into dev/test =="
GEN_OUTPUT=$($RAGAUDIT generate --corpus "$DEMO/corpus.jsonl" --multi-hop --store "$STORE" --llm "$LLM")
echo "$GEN_OUTPUT"
FIRST_GENERATE_RUN=$(echo "$GEN_OUTPUT" | extract_generate_run_id)

echo
echo "== 2. evaluate: the toy pipeline's default config, on the dev split =="
$RAGAUDIT evaluate --adapter "$ADAPTER" --corpus "$DEMO/corpus.jsonl" --store "$STORE" --llm "$LLM"

echo
echo "== 3. the corpus changes: 3 chunks edited, 2 deleted, 4 added =="
export RAGAUDIT_DEMO_CORPUS="$DEMO/corpus_update.jsonl"
$RAGAUDIT generate --corpus "$DEMO/corpus_update.jsonl" --multi-hop --store "$STORE" --llm "$LLM"

echo
echo "== 4. evaluate again: regression mode (all new questions + a sample of old ones) =="
$RAGAUDIT evaluate --adapter "$ADAPTER" --corpus "$DEMO/corpus_update.jsonl" --store "$STORE" --llm "$LLM"

echo
echo "== 5. tune: search configs on dev, score baseline and best once on the locked test split =="
$RAGAUDIT tune --adapter "$ADAPTER" --search-space "$DEMO/search_space.yaml" --corpus "$DEMO/corpus_update.jsonl" \
    --strategy llm --max-trials 20 --store "$STORE" --llm "$LLM" --out "$OUT/best_config.yaml"

echo
echo "== 6. evaluate the best config: results come from the cache =="
$RAGAUDIT evaluate --adapter "$ADAPTER" --config "$OUT/best_config.yaml" --store "$STORE" --llm "$LLM"

echo
echo "== 7. reports =="
$RAGAUDIT report --kind generate --run-id "$FIRST_GENERATE_RUN" --store "$STORE" --out "$OUT/generate-report.html"
$RAGAUDIT report --kind tune --store "$STORE" --out "$OUT/tune-report.html"

echo
echo "Done. Open $OUT/generate-report.html and $OUT/tune-report.html."
