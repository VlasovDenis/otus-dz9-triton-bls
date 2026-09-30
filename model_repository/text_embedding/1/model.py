import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import triton_python_backend_utils as pb_utils


MODEL_DIR = "/opt/dz9/models/embedding"


def embed_texts(tokenizer, model, texts):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=256,
                        return_tensors="pt")
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        vectors = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return vectors.cpu().numpy().astype(np.float32)


class TritonPythonModel:
    def initialize(self, args):
        if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
            raise RuntimeError("Embedding files are missing; run scripts/prepare_models.py")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        self.model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)
        self.model.eval().cpu()
        if self.model.config.hidden_size != 384:
            raise RuntimeError("Expected 384-dimensional embeddings")

    def execute(self, requests):
        result = []
        for request in requests:
            try:
                value = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
                text = value.reshape(-1)[0].decode("utf-8")
                vector = embed_texts(self.tokenizer, self.model, [text])
                result.append(pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("EMBEDDING", vector)
                ]))
            except Exception as exc:
                result.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc))))
        return result

