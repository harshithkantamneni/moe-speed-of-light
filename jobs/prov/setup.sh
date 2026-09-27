# Idempotent environment for the phase-4 provenance jobs (source it).
# Two uv venvs: $WORK/vv (vLLM) and $WORK/tv (the tracer: torch + transformers 5.17.0, the version the collector's
# equivalence tests pass on). Checkpoints go to $MD/<key> (root-level files only: the gpt-oss repos also hold
# original/ and metal/ copies). Exports P, MD, HF_HOME; defines dl KEY REPO and wait_dl KEY.
P=${P:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export P MD=$WORK/hfmodels HF_HOME=$WORK/hf
# the runner starts from cloud-init with no HOME: keep uv, its pythons and caches under $WORK
export HOME=${HOME:-/root} UV_INSTALL_DIR=$WORK/uvbin UV_PYTHON_INSTALL_DIR=$WORK/uvpy UV_CACHE_DIR=$WORK/uvcache UV_NO_MODIFY_PATH=1
export PATH=$WORK/uvbin:$PATH
mkdir -p $MD $HF_HOME
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh > $WORK/uv_install.log 2>&1; tail -2 $WORK/uv_install.log; }
uv --version || { echo "uv unavailable"; exit 1; }
if ! $WORK/vv/bin/python -c "import vllm" 2>/dev/null; then
  uv venv -q --python 3.12 $WORK/vv > $WORK/vv_install.log 2>&1
  timeout 20m uv pip install --python $WORK/vv/bin/python vllm "huggingface_hub[hf_xet]" --torch-backend=auto >> $WORK/vv_install.log 2>&1
  tail -3 $WORK/vv_install.log
fi
if ! $WORK/tv/bin/python -c "import torch, transformers; assert torch.cuda.is_available()" 2>/dev/null; then
  uv venv -q --python 3.12 $WORK/tv > $WORK/tv_install.log 2>&1
  timeout 20m uv pip install --python $WORK/tv/bin/python torch "transformers==5.17.0" safetensors numpy huggingface_hub requests --torch-backend=auto >> $WORK/tv_install.log 2>&1
  tail -3 $WORK/tv_install.log
fi
$WORK/vv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
$WORK/tv/bin/python -c "import transformers, torch; print('tracer: transformers', transformers.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
dl() {
  [ -f $MD/$1/.done ] && return 0
  HF_XET_HIGH_PERFORMANCE=1 timeout 45m $WORK/vv/bin/hf download "$2" --local-dir $MD/$1 \
    --include "model*.safetensors" "*.json" "tokenizer*" "*.jinja" "*.txt" "*.model" "*.tiktoken" \
    --exclude "original/*" "metal/*" > $WORK/dl_$1.log 2>&1 && touch $MD/$1/.done
}
wait_dl() { for i in $(seq 1 600); do [ -f $MD/$1/.done ] && return 0; sleep 5; done; return 1; }
