#!/bin/sh
set -e

SAMPLE_CORPUS_DIR=/app/demo/sample_corpus
LUCKY_CORPUS_DIR=/app/demo/lucky_corpus
HEALTHY_ADAPTER=/app/demo/example_adapter.py
WEAK_ADAPTER=/app/demo/weak_adapter.py
UNGROUNDED_ADAPTER=/app/demo/ungrounded_adapter.py
OUTPUT_DIR=/app/output
DB_PATH="$OUTPUT_DIR/ragaudit.db"
LUCKY_DB_PATH="$OUTPUT_DIR/ragaudit-lucky.db"

mkdir -p "$OUTPUT_DIR"

extract_run_id() {
    sed -n "s/^Run '\(.*\)' complete.*/\1/p"
}

extract_testset_id() {
    sed -n "s/.*Saved test set '\(.*\)' to.*/\1/p"
}

# --- sample corpus: healthy vs. degraded retrieval on invented facts ---

# Both adapters read chunks straight from the ragaudit database rather
# than re-walking the corpus, so they need to know where that lives.
export RAGAUDIT_DB_PATH="$DB_PATH"

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

# --- lucky corpus: well-known facts, ungrounded adapter, LUCKY_PASS + ablation ---
#
# Kept in its own database (rather than sharing $DB_PATH) so its chunks
# never overwrite the sample corpus's chunks — ingest fully replaces the
# documents/chunks tables of whatever db it's pointed at.
#
# Uses ungrounded_adapter.py, not weak_adapter.py: weak_adapter.py's
# "only use the context" prompt makes the model refuse instead of falling
# back on pretraining when retrieval is bad, which suppresses LUCKY_PASS
# entirely. ungrounded_adapter.py has the same retrieval degradation (just
# heavier — 70% random instead of 30%) but a permissive prompt that lets
# the model answer from its own knowledge, which is what actually produces
# LUCKY_PASS here.

export RAGAUDIT_DB_PATH="$LUCKY_DB_PATH"

echo "== ragaudit demo: ingest (lucky corpus) =="
ragaudit ingest "$LUCKY_CORPUS_DIR" --db "$LUCKY_DB_PATH"

echo "== ragaudit demo: generate (lucky corpus) =="
LUCKY_GEN_OUTPUT=$(ragaudit generate --n 1 --db "$LUCKY_DB_PATH")
echo "$LUCKY_GEN_OUTPUT"
LUCKY_TESTSET_ID=$(echo "$LUCKY_GEN_OUTPUT" | extract_testset_id)

echo "== ragaudit demo: run (ungrounded adapter, ungrounded_adapter.py, against lucky corpus, ablation enabled) =="
LUCKY_OUTPUT=$(ragaudit run --adapter "$UNGROUNDED_ADAPTER" --db "$LUCKY_DB_PATH" --testset-id "$LUCKY_TESTSET_ID")
echo "$LUCKY_OUTPUT"
LUCKY_RUN_ID=$(echo "$LUCKY_OUTPUT" | extract_run_id)

echo "== ragaudit demo: report (lucky) =="
ragaudit report --db "$LUCKY_DB_PATH" --run-id "$LUCKY_RUN_ID" --out "$OUTPUT_DIR/report-lucky.html"

echo ""
echo "Demo complete."
echo "Healthy-adapter report:  $OUTPUT_DIR/report-healthy.html"
echo "Degraded-adapter report: $OUTPUT_DIR/report-degraded.html"
echo "Lucky-pass report:       $OUTPUT_DIR/report-lucky.html"
