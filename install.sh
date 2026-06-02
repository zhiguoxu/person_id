#!/bin/bash
# ==============================================================================
# Vision ID — 依赖安装脚本 (国内加速)
#
# 用法:
#   conda activate person_id
#   bash install.sh
# ==============================================================================
set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"

echo -e "${CYAN}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     🤖 Vision ID — Install (国内加速)         ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════╝${NC}"
echo ""

# --------------------------------------------------------------------------
# Step 1: 检测 CUDA 版本，选择 PyTorch wheel
# --------------------------------------------------------------------------
echo -e "${GREEN}[1/6]${NC} Detecting CUDA..."

CUDA_DRIVER=""
if command -v nvidia-smi &> /dev/null; then
    CUDA_DRIVER=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: //' | sed 's/ .*//')
    echo "  Driver CUDA: $CUDA_DRIVER"
fi

# 根据驱动版本选 PyTorch CUDA wheel
# 驱动向下兼容，选最接近的
if [ -z "$CUDA_DRIVER" ]; then
    echo -e "  ${RED}No NVIDIA GPU detected, installing CPU version${NC}"
    TORCH_INDEX=""
    TORCH_SUFFIX="cpu"
elif echo "$CUDA_DRIVER" | grep -qE "^13\.|^12\.[4-9]"; then
    TORCH_INDEX="--index-url https://download.pytorch.org/whl/cu124"
    TORCH_SUFFIX="cu124"
elif echo "$CUDA_DRIVER" | grep -qE "^12\.[1-3]"; then
    TORCH_INDEX="--index-url https://download.pytorch.org/whl/cu121"
    TORCH_SUFFIX="cu121"
elif echo "$CUDA_DRIVER" | grep -qE "^12\."; then
    TORCH_INDEX="--index-url https://download.pytorch.org/whl/cu121"
    TORCH_SUFFIX="cu121"
elif echo "$CUDA_DRIVER" | grep -qE "^11\.8"; then
    TORCH_INDEX="--index-url https://download.pytorch.org/whl/cu118"
    TORCH_SUFFIX="cu118"
else
    TORCH_INDEX="--index-url https://download.pytorch.org/whl/cu124"
    TORCH_SUFFIX="cu124"
fi

echo "  PyTorch wheel: $TORCH_SUFFIX"

# --------------------------------------------------------------------------
# Step 2: 安装 PyTorch (pip + 官方 CUDA wheel)
# --------------------------------------------------------------------------
echo ""
echo -e "${GREEN}[2/6]${NC} Installing PyTorch ($TORCH_SUFFIX)..."

# 检查是否已有正确的 CUDA 版本
NEED_INSTALL=true
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CURRENT=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo -e "  ${GREEN}PyTorch $CURRENT (CUDA) already installed, skipping${NC}"
    NEED_INSTALL=false
fi

if [ "$NEED_INSTALL" = true ]; then
    echo "  Removing CPU version (if any)..."
    pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
    echo "  Installing CUDA version (nvidia deps via 清华镜像)..."
    pip install torch torchvision $TORCH_INDEX \
        --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        --trusted-host pypi.tuna.tsinghua.edu.cn
fi

# --------------------------------------------------------------------------
# Step 3: 其余 pip 依赖 (清华镜像)
# --------------------------------------------------------------------------
echo ""
echo ""
echo -e "${GREEN}[3/6]${NC} Installing pip dependencies (清华镜像)..."

# 先卸载 CPU 版 onnxruntime, 避免与 GPU 版冲突
pip uninstall onnxruntime onnxruntime-gpu -y 2>/dev/null || true

# 单独安装 onnxruntime-gpu (需要 CUDA 12 专用源)
pip install "onnxruntime-gpu>=1.18.0" \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

pip install $PIP_MIRROR \
    "ultralytics>=8.3.0" \
    "insightface>=0.7.3" \
    "opencv-python>=4.9.0" \
    "numpy>=1.26.0" \
    "scipy>=1.12.0" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.6.0" \
    "websockets>=12.0" \
    "openai>=1.14.0" \
    "aiosqlite>=0.20.0" \
    "pillow>=10.2.0" \
    "scikit-learn>=1.4.0" \
    "loguru>=0.7.0"

# --------------------------------------------------------------------------
# Step 4: boxmot + torchreid (特殊处理)
# --------------------------------------------------------------------------
echo ""
echo -e "${GREEN}[4/6]${NC} Installing boxmot + torchreid..."

# boxmot: --no-deps 跳过 torchvision/numpy 严格版本限制, 手动装运行时依赖
pip install $PIP_MIRROR --no-deps boxmot
pip install $PIP_MIRROR filterpy ftfy gitpython lapx pandas regex yacs

# torchreid + 缺失依赖
pip install $PIP_MIRROR torchreid gdown tensorboard

# --------------------------------------------------------------------------
# Step 5: 修复 libstdc++ (GLIBCXX 版本问题)
# --------------------------------------------------------------------------
echo ""
echo -e "${GREEN}[5/6]${NC} Fixing libstdc++ (GLIBCXX)..."
conda install -c conda-forge libstdcxx-ng -y 2>/dev/null || echo "  (skipped)"

# --------------------------------------------------------------------------
# Step 6: 验证
# --------------------------------------------------------------------------
echo ""
echo -e "${GREEN}[6/6]${NC} Verifying..."
mkdir -p data models

# 设置 LD_LIBRARY_PATH 确保 conda 的 libstdc++ 生效
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/home/zhiguo/miniconda3/envs/person_id}/lib:${LD_LIBRARY_PATH:-}"

python3 -c "
import torch
cuda_ok = torch.cuda.is_available()
print(f'  torch:        {torch.__version__}  CUDA={cuda_ok}')
if cuda_ok:
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
else:
    print('  ⚠ CUDA not available — check PyTorch installation')
import torchvision; print(f'  torchvision:  {torchvision.__version__}')
import ultralytics; print(f'  ultralytics:  {ultralytics.__version__}')
import fastapi; print(f'  fastapi:      {fastapi.__version__}')
try: import boxmot; print(f'  boxmot:       OK')
except Exception as e: print(f'  boxmot:       ⚠ {e}')
try: import insightface; print(f'  insightface:  {insightface.__version__}')
except Exception as e: print(f'  insightface:  ⚠ {e}')
try: import torchreid; print(f'  torchreid:    OK')
except Exception as e: print(f'  torchreid:    ⚠ {e}')
"

echo ""
echo -e "${GREEN}✅ Done!${NC}"
echo -e "  1. Download models: ${CYAN}bash download_models.sh${NC}"
echo -e "  2. Start server:    ${CYAN}bash deploy.sh${NC}"
