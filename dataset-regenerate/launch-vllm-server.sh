
# Serve the model
CUDA_VISIBLE_DEVICES=0,1 vllm serve unsloth/QwQ-32B-unsloth-bnb-4bit \
  --host 0.0.0.0  \
  --port  8000 \
  --tensor-parallel-size 2 \
  --max-model-len 50000 \
  --api-key API-KEY \
  --quantization=bitsandbytes \
  --load-format=bitsandbytes \
  --gpu-memory-utilization=0.75 \
  --served-model-name QwQ
