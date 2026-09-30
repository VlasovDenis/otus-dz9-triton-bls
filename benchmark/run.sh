#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/perf
image='nvcr.io/nvidia/tritonserver:26.08-py3-sdk'

run_case() {
  local name="$1" input="$2" mode="${3:-standard}"
  local -a opts=()
  if [[ "$mode" == streaming ]]; then opts=(--streaming); fi
  docker run --rm --network host -v "$PWD:/work:ro" \
    "$image" perf_analyzer -m "$name" -i grpc -u 127.0.0.1:8001 \
    --input-data "/work/$input" --concurrency-range 1:4:1 \
    --measurement-interval 10000 \
    "${opts[@]}" | tee "artifacts/perf/${name}_${mode}.log"
}

run_case text_embedding benchmark/embedding.json
run_case text_classifier benchmark/classifier.json
run_case text_generator benchmark/generator.json streaming
run_case assistant_bls benchmark/bls_question.json question
run_case assistant_bls benchmark/bls_statement.json statement

