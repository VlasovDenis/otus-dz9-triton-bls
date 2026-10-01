#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/perf
image='nvcr.io/nvidia/tritonserver:26.08-py3-sdk'
failed=0

run_case() {
  local name="$1" input="$2" mode="${3:-standard}"
  local -a opts=()
  if [[ "$mode" == streaming ]]; then opts+=(--streaming); fi
  if [[ "$mode" == question ]]; then
    opts+=(--measurement-mode=count_windows --measurement-request-count=30 --stability-percentage=20)
  else
    opts+=(--measurement-interval 10000)
  fi
  if docker run --rm --network host -v "$PWD:/work:ro" \
    "$image" perf_analyzer -m "$name" -i grpc -u 127.0.0.1:8001 \
    --input-data "/work/$input" --concurrency-range 1:4:1 \
    "${opts[@]}" 2>&1 | tee "artifacts/perf/${name}_${mode}.log"; then
    :
  else
    printf "Perf Analyzer failed for %s (%s); see its log.\n" "$name" "$mode" >&2
    failed=1
  fi
}

run_case text_embedding benchmark/embedding.json
run_case text_classifier benchmark/classifier.json
run_case text_generator benchmark/generator.json streaming
run_case assistant_bls benchmark/bls_question.json question
run_case assistant_bls benchmark/bls_statement.json statement

if ((failed)); then
  printf "One or more benchmarks did not complete; inspect artifacts/perf/*.log.\n" >&2
  exit 1
fi
