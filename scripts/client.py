import argparse

import numpy as np
import tritonclient.http as http


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="Что такое Triton Inference Server?")
    parser.add_argument("--url", default="127.0.0.1:8000")
    args = parser.parse_args()
    client = http.InferenceServerClient(url=args.url)
    for name in ("text_embedding", "text_classifier", "text_generator", "assistant_bls"):
        if not client.is_model_ready(name):
            raise SystemExit(f"Model is not READY: {name}")
    payload = np.asarray([args.text.encode("utf-8")], dtype=object)
    inp = http.InferInput("TEXT", [1], "BYTES")
    inp.set_data_from_numpy(payload)
    result = client.infer("assistant_bls", [inp], outputs=[http.InferRequestedOutput("RESPONSE")])
    print(result.as_numpy("RESPONSE")[0].decode("utf-8"))


if __name__ == "__main__":
    main()

