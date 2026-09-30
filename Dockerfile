FROM nvcr.io/nvidia/tritonserver:26.08-py3 AS full

FROM nvcr.io/nvidia/tritonserver:26.08-vllm-python-py3
COPY --from=full /opt/tritonserver/backends/onnxruntime /opt/tritonserver/backends/onnxruntime
ENV LD_LIBRARY_PATH="/opt/tritonserver/lib:${LD_LIBRARY_PATH}"

# A failed linkage check means this image needs more dependencies from the
# matching full image. Do not use an image with unresolved shared libraries.
RUN test -r /opt/tritonserver/lib/libtritonserver.so \
 && ldd /opt/tritonserver/backends/onnxruntime/libtriton_onnxruntime.so \
      | tee /tmp/dz9-onnx-linkage.txt \
 && ! grep -q 'not found' /tmp/dz9-onnx-linkage.txt \
 && python3 -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'

EXPOSE 8000 8001 8002
CMD ["tritonserver", "--model-repository=/models", "--strict-readiness=true"]

