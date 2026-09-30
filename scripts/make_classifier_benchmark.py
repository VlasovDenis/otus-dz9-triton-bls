"""Capture a real embedding as Perf Analyzer input for the ONNX classifier."""

import json
from pathlib import Path

import numpy as np
import tritonclient.http as http


root = Path(__file__).resolve().parents[1]
client = http.InferenceServerClient(url="127.0.0.1:8000")
data = np.asarray(["Что такое Triton Inference Server?".encode()], dtype=object)
inp = http.InferInput("TEXT", [1], "BYTES")
inp.set_data_from_numpy(data)
result = client.infer("text_embedding", [inp])
vector = result.as_numpy("EMBEDDING").reshape(-1).tolist()
if len(vector) != 384:
    raise SystemExit("Expected 384 values")
path = root / "benchmark/classifier.json"
path.write_text(json.dumps({"data": [{"EMBEDDING": vector}]}), encoding="utf-8")
print(path)

