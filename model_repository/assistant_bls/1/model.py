import json

import numpy as np
import triton_python_backend_utils as pb_utils


def infer(model, outputs, tensors, decoupled=False):
    call = pb_utils.InferenceRequest(
        model_name=model, requested_output_names=outputs, inputs=tensors,
        timeout=60_000_000,
    )
    responses = call.exec(decoupled=decoupled)
    if not decoupled:
        if responses.has_error():
            raise RuntimeError(responses.error().message())
        return responses
    last = None
    for response in responses:
        if response.has_error():
            raise RuntimeError(response.error().message())
        if pb_utils.get_output_tensor_by_name(response, "text_output") is not None:
            last = response
    if last is None:
        raise RuntimeError("Generation returned no text")
    return last


class TritonPythonModel:
    def execute(self, requests):
        results = []
        for request in requests:
            try:
                text = pb_utils.get_input_tensor_by_name(request, "TEXT")\
                    .as_numpy().reshape(-1)[0].decode("utf-8").strip()
                if not text:
                    raise ValueError("TEXT must be nonempty")

                embedding = infer("text_embedding", ["EMBEDDING"], [
                    pb_utils.Tensor("TEXT", np.asarray([text.encode()], dtype=object))
                ])
                vector = pb_utils.get_output_tensor_by_name(
                    embedding, "EMBEDDING").as_numpy()
                classification = infer("text_classifier", ["PROBABILITIES"], [
                    pb_utils.Tensor("EMBEDDING", vector)
                ])
                probabilities = pb_utils.get_output_tensor_by_name(
                    classification, "PROBABILITIES").as_numpy()

                if float(probabilities[0, 1]) >= 0.5:
                    prompt = (
                        "<|im_start|>system\nОтвечай кратко и по-русски."
                        "<|im_end|>\n<|im_start|>user\n"
                        + text + "<|im_end|>\n<|im_start|>assistant\n"
                    )
                    generation = infer("text_generator", ["text_output"], [
                        pb_utils.Tensor("text_input", np.asarray(
                            [prompt.encode()], dtype=object)),
                        pb_utils.Tensor("sampling_parameters", np.asarray(
                            [json.dumps({"max_tokens": 64, "temperature": 0}).encode()],
                            dtype=object)),
                        pb_utils.Tensor("stream", np.asarray([False], dtype=bool)),
                        pb_utils.Tensor("exclude_input_in_output", np.asarray(
                            [True], dtype=bool)),
                    ], decoupled=True)
                    answer = pb_utils.get_output_tensor_by_name(
                        generation, "text_output").as_numpy().reshape(-1)[0]
                    if isinstance(answer, str):
                        answer = answer.encode("utf-8")
                else:
                    answer = "Сообщение принято. Задайте вопрос, чтобы получить ответ.".encode()

                results.append(pb_utils.InferenceResponse(output_tensors=[
                    pb_utils.Tensor("RESPONSE", np.asarray([answer], dtype=object))
                ]))
            except Exception as exc:
                results.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(exc))))
        return results

