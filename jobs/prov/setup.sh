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
# vLLM: the PyPI wheel may target a newer CUDA than the driver (570 = CUDA 12.8; job 022 failed on
# libcudart.so.13), so fall back to the same release built for CUDA 12.9 / 12.8, then to vllm 0.11.0 (CUDA 12.8)
vllm_ok() { $WORK/vv/bin/python -c "import vllm, torch; torch.zeros(1).cuda(); print('vllm ok', vllm.__version__, torch.__version__)" > $WORK/vv_check.log 2>&1; }
if ! vllm_ok; then
  [ -x $WORK/vv/bin/python ] || uv venv -q --python 3.12 $WORK/vv > $WORK/vv_install.log 2>&1
  timeout 20m uv pip install --python $WORK/vv/bin/python vllm "huggingface_hub[hf_xet]" --torch-backend=auto >> $WORK/vv_install.log 2>&1
  if ! vllm_ok; then
    V=$($WORK/vv/bin/python -c "import importlib.metadata as m; print(m.version('vllm').split('+')[0])")
    for cu in 129 128; do
      timeout 20m uv pip install --python $WORK/vv/bin/python --reinstall-package vllm --reinstall-package torch \
        "https://github.com/vllm-project/vllm/releases/download/v${V}/vllm-${V}+cu${cu}-cp38-abi3-manylinux_2_28_x86_64.whl" \
        --torch-backend=cu${cu} >> $WORK/vv_install.log 2>&1 && vllm_ok && break
    done
  fi
  vllm_ok || timeout 20m uv pip install --python $WORK/vv/bin/python "vllm==0.11.0" --torch-backend=cu128 >> $WORK/vv_install.log 2>&1
  tail -3 $WORK/vv_install.log
fi
vllm_ok; cat $WORK/vv_check.log | tail -3
if ! $WORK/tv/bin/python -c "import torch, transformers; assert torch.cuda.is_available()" 2>/dev/null; then
  uv venv -q --python 3.12 $WORK/tv > $WORK/tv_install.log 2>&1
  timeout 20m uv pip install --python $WORK/tv/bin/python torch "transformers==5.17.0" safetensors numpy huggingface_hub requests --torch-backend=auto >> $WORK/tv_install.log 2>&1
  tail -3 $WORK/tv_install.log
fi
$WORK/tv/bin/python -c "import transformers, torch; print('tracer: transformers', transformers.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
have() { [ -f $MD/$1/.done ] && ls $MD/$1/model*.safetensors > /dev/null 2>&1; }
dl() {
  have $1 && return 0
  rm -f $MD/$1/.done
  # the hf CLI takes one pattern per --include / --exclude (extra words become file names: job 022's bug)
  HF_XET_HIGH_PERFORMANCE=1 timeout 45m $WORK/vv/bin/hf download "$2" --local-dir $MD/$1 \
    --include "model*.safetensors" --include "*.json" --include "tokenizer*" --include "*.jinja" --include "*.txt" \
    --exclude "original/*" --exclude "metal/*" > $WORK/dl_$1.log 2>&1 \
    && ls $MD/$1/model*.safetensors > /dev/null 2>&1 && touch $MD/$1/.done
}
wait_dl() { for i in $(seq 1 600); do have $1 && return 0; sleep 5; done; return 1; }
for d in $MD/*/; do k=$(basename $d); have $k || rm -f $MD/$k/.done; done   # drop markers of incomplete downloads
