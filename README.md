# DZ9: Triton BLS pipeline

Четыре модели в одном Triton: `text_embedding` (Python, CPU), `text_classifier` (ONNX Runtime, CPU), `text_generator` (vLLM, GPU) и `assistant_bls` (Python BLS). Классификатор отличает вопрос от сообщения на маленьком демонстрационном наборе. BLS вызывает embedding, затем classifier и при классе «вопрос» вызывает генерацию.

## Подготовка в WSL

Используйте отдельное окружение Python 3.12. Весовые файлы остаются локально и не входят в Git.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-prep.txt
python scripts/prepare_models.py
python scripts/build_classifier.py
docker compose build
docker compose up -d
docker compose logs --tail=120 triton
curl --noproxy 127.0.0.1 -fsS http://127.0.0.1:8000/v2/health/ready
curl --noproxy 127.0.0.1 -fsS -X POST -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8000/v2/repository/index
python scripts/client.py 'Что такое Triton Inference Server?'
python scripts/client.py 'Сервер успешно запущен.'
```

`config.pbtxt` лежит у корня каждой модели; версия каждой модели находится в `1/`. ONNX файл создаётся `scripts/build_classifier.py` и включён в репозиторий. BLS зависимости существуют в коде его `InferenceRequest`: `config.pbtxt` Python BLS не описывает DAG как конфигурация обычного Triton ensemble. Нужен статус READY у всех четырёх моделей.

На RTX 3070 с 8 GiB embedding и ONNX classifier работают на CPU. vLLM ограничен `gpu_memory_utilization=0.55`; при проблемах с памятью закройте другие процессы GPU и уменьшите это значение или `max_model_len`.

## Нагрузочный тест

После успешных клиентских проверок:

```bash
python scripts/make_classifier_benchmark.py
bash benchmark/run.sh
nvidia-smi
curl --noproxy 127.0.0.1 -fsS http://127.0.0.1:8002/metrics > artifacts/perf/metrics.prom
```

Perf Analyzer запускается из `26.08-py3-sdk`. Для прямой нагрузки на decoupled vLLM используется `--streaming`; BLS возвращает обычный один ответ. Фактические результаты и ограничения измерений приведены в [отчёте](docs/REPORT.md); исходные логи Perf Analyzer включены в `artifacts/perf/`. Модельные веса скачиваются при подготовке и не включены в Git.

## Структура

```text
model_repository/
  text_embedding/config.pbtxt, 1/model.py
  text_classifier/config.pbtxt, 1/model.onnx (сгенерирован и включён в Git)
  text_generator/config.pbtxt, 1/model.json
  assistant_bls/config.pbtxt, 1/model.py
```

Dockerfile использует два образа одного релиза 26.08. Проверка `ldd` останавливает сборку при недостающих системных зависимостях ONNX backend. Состояние READY и корректность BLS проверяются уже на целевом Docker-хосте.

