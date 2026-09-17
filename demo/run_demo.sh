#!/bin/sh
set -e

CORPUS_DIR=/app/demo/sample_corpus
ADAPTER_PATH=/app/demo/example_adapter.py
OUTPUT_DIR=/app/output
DB_PATH="$OUTPUT_DIR/ragaudit.db"
REPORT_PATH="$OUTPUT_DIR/report.html"

mkdir -p "$OUTPUT_DIR"

# example_adapter.py reads chunks straight from the ragaudit database
# rather than re-walking the corpus, so it needs to know where that
# database lives.
export RAGAUDIT_DB_PATH="$DB_PATH"

echo "== ragaudit demo: ingest =="
ragaudit ingest "$CORPUS_DIR" --db "$DB_PATH"

echo "== ragaudit demo: generate =="
ragaudit generate --n 1 --db "$DB_PATH"

echo "== ragaudit demo: run =="
ragaudit run --adapter "$ADAPTER_PATH" --db "$DB_PATH"

echo "== ragaudit demo: report =="
ragaudit report --db "$DB_PATH" --out "$REPORT_PATH"

echo ""
echo "Demo complete. Report written to $REPORT_PATH"
