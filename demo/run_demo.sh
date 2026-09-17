#!/bin/sh
set -e

SAMPLE_CORPUS_DIR=/app/demo/sample_corpus
LUCKY_CORPUS_DIR=/app/demo/lucky_corpus
HEALTHY_ADAPTER=/app/demo/example_adapter.py
WEAK_ADAPTER=/app/demo/weak_adapter.py
OUTPUT_DIR=/app/output
DB_PATH="$OUTPUT_DIR/ragaudit.db"

mkdir -p "$OUTPUT_DIR"

# Both adapters read chunks straight from the ragaudit database rather
# than re-walking the corpus, so they need to know where that lives.
export RAGAUDIT_DB_PATH="$DB_PATH"

extract_run_id() {
    sed -n "s/^Run '\(.*\)' complete.*/\1/p"
}

extract_testset_id() {
    sed -n "s/.*Saved test set '\(.*\)' to.*/\1/p"
}

# --- sample corpus: healthy vs. degraded retrieval on invented facts ---

echo "== ragaudit demo: ingest (sample corpus) =="
ragaudit ingest "$SAMPLE_CORPUS_DIR" --db "$DB_PATH"

echo "== ragaudit demo: generate (sample corpus) =="
SAMPLE_GEN_OUTPUT=$(ragaudit generate --n 1 --db "$DB_PATH")
echo "$SAMPLE_GEN_OUTPUT"
SAMPLE_TESTSET_ID=$(echo "$SAMPLE_GEN_OUTPUT" | extract_testset_id)

echo "== ragaudit demo: run (healthy adapter, example_adapter.py) =="
HEALTHY_OUTPUT=$(ragaudit run --adapter "$HEALTHY_ADAPTER" --db "$DB_PATH" --testset-id "$SAMPLE_TESTSET_ID")
echo "$HEALTHY_OUTPUT"
HEALTHY_RUN_ID=$(echo "$HEALTHY_OUTPUT" | extract_run_id)

echo "== ragaudit demo: run (degraded adapter, weak_adapter.py) =="
WEAK_OUTPUT=$(ragaudit run --adapter "$WEAK_ADAPTER" --db "$DB_PATH" --testset-id "$SAMPLE_TESTSET_ID")
echo "$WEAK_OUTPUT"
WEAK_RUN_ID=$(echo "$WEAK_OUTPUT" | extract_run_id)

echo "== ragaudit demo: report (healthy) =="
ragaudit report --db "$DB_PATH" --run-id "$HEALTHY_RUN_ID" --out "$OUTPUT_DIR/report-healthy.html"

echo "== ragaudit demo: report (degraded) =="
ragaudit report --db "$DB_PATH" --run-id "$WEAK_RUN_ID" --out "$OUTPUT_DIR/report-degraded.html"

# --- lucky corpus: well-known facts, degraded retrieval, LUCKY_PASS + ablation ---

echo "== ragaudit demo: ingest (lucky corpus) =="
ragaudit ingest "$LUCKY_CORPUS_DIR" --db "$DB_PATH"

echo "== ragaudit demo: generate (lucky corpus) =="
LUCKY_GEN_OUTPUT=$(ragaudit generate --n 1 --db "$DB_PATH")
echo "$LUCKY_GEN_OUTPUT"
LUCKY_TESTSET_ID=$(echo "$LUCKY_GEN_OUTPUT" | extract_testset_id)

echo "== ragaudit demo: run (degraded adapter, weak_adapter.py, against lucky corpus) =="
LUCKY_OUTPUT=$(ragaudit run --adapter "$WEAK_ADAPTER" --db "$DB_PATH" --testset-id "$LUCKY_TESTSET_ID")
echo "$LUCKY_OUTPUT"
LUCKY_RUN_ID=$(echo "$LUCKY_OUTPUT" | extract_run_id)

echo "== ragaudit demo: report (lucky) =="
ragaudit report --db "$DB_PATH" --run-id "$LUCKY_RUN_ID" --out "$OUTPUT_DIR/report-lucky.html"

echo ""
echo "Demo complete."
echo "Healthy-adapter report:  $OUTPUT_DIR/report-healthy.html"
echo "Degraded-adapter report: $OUTPUT_DIR/report-degraded.html"
echo "Lucky-pass report:       $OUTPUT_DIR/report-lucky.html"
