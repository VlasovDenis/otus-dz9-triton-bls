"""Create an illustrative ONNX question/statement classifier from embedding centroids."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets/models/embedding"
OUTPUT = ROOT / "model_repository/text_classifier/1/model.onnx"

QUESTIONS = [
    "Что такое Triton Inference Server?",
    "Как работает динамический батчинг?",
    "Для чего нужен ONNX Runtime?",
    "Какая сегодня погода?",
    "Почему запрос выполняется долго?",
    "Объясни разницу между CPU и GPU.",
    "Где находятся файлы модели?",
    "Когда сервер будет готов?",
    "Можно ли запустить модель локально?",
    "Расскажи, что такое эмбеддинг.",
]
STATEMENTS = [
    "Сервер успешно запущен.",
    "Выполнение запроса завершено.",
    "Модель классификации загружена.",
    "Сегодня хорошая погода.",
    "Пользователь отправил сообщение.",
    "Проверка контейнера прошла успешно.",
    "Файлы модели находятся в репозитории.",
    "Я настроил динамический батчинг.",
    "Отчёт содержит показатели задержки.",
    "Это тестовое сообщение.",
]


def encode(texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).cpu().eval()
    vectors = []
    with torch.inference_mode():
        for text in texts:
            inputs = tokenizer([text], padding=True, truncation=True,
                               max_length=256, return_tensors="pt")
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            value = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            value = torch.nn.functional.normalize(value, p=2, dim=1)
            vectors.append(value[0].cpu().numpy())
    return np.asarray(vectors, dtype=np.float32)


def main():
    if not (MODEL_PATH / "config.json").is_file():
        raise SystemExit("Download embedding model first: python scripts/prepare_models.py")
    q = encode(QUESTIONS)
    s = encode(STATEMENTS)
    direction = q.mean(0) - s.mean(0)
    midpoint = (q.mean(0) + s.mean(0)) / 2
    weights = (direction * 20).astype(np.float32).reshape(384, 1)
    bias = np.asarray([-midpoint @ weights[:, 0]], dtype=np.float32)

    x = helper.make_tensor_value_info("EMBEDDING", TensorProto.FLOAT, ["batch", 384])
    y = helper.make_tensor_value_info("PROBABILITIES", TensorProto.FLOAT, ["batch", 2])
    nodes = [
        helper.make_node("MatMul", ["EMBEDDING", "W"], ["LOGIT"]),
        helper.make_node("Add", ["LOGIT", "B"], ["SHIFTED"]),
        helper.make_node("Sigmoid", ["SHIFTED"], ["QUESTION"]),
        helper.make_node("Sub", ["ONE", "QUESTION"], ["STATEMENT"]),
        helper.make_node("Concat", ["STATEMENT", "QUESTION"],
                         ["PROBABILITIES"], axis=1),
    ]
    graph = helper.make_graph(nodes, "centroid_classifier", [x], [y],
                              [numpy_helper.from_array(weights, "W"),
                               numpy_helper.from_array(bias, "B"),
                               numpy_helper.from_array(np.ones((1, 1), np.float32), "ONE")])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUTPUT)
    question_scores = 1 / (1 + np.exp(-(q @ weights + bias)))
    statement_scores = 1 / (1 + np.exp(-(s @ weights + bias)))
    print(f"Saved {OUTPUT}")
    print(f"Prototype accuracy: questions={(question_scores >= .5).mean():.2f}; "
          f"statements={(statement_scores < .5).mean():.2f}")
    print("This small classifier is for pipeline demonstration, not general NLP quality.")


if __name__ == "__main__":
    main()

