#!/bin/bash
# ==============================================================================
# Vision ID — CUDA 服务器部署启动脚本
#
# 架构: 前端(本地浏览器) ←WebSocket→ 后端(此服务器 CUDA 推理)
# 服务器: 8.145.38.125:10003
#
# 用法:  bash deploy.sh  (自动激活 conda 环境)
# ==============================================================================
set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --------------------------------------------------------------------------
# 0. 自动激活 conda 环境
# --------------------------------------------------------------------------
CONDA_ENV="person_id"
CONDA_BASE="${HOME}/miniconda3"

# 如果当前不在目标 conda 环境中, 自动激活
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        . "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
        printf "${YELLOW}  Auto-activated conda env: %s${NC}\n" "$CONDA_ENV"
    elif [ -d "$CONDA_BASE/envs/$CONDA_ENV/bin" ]; then
        export PATH="$CONDA_BASE/envs/$CONDA_ENV/bin:$PATH"
        export CONDA_PREFIX="$CONDA_BASE/envs/$CONDA_ENV"
        printf "${YELLOW}  Manually set conda env: %s${NC}\n" "$CONDA_ENV"
    else
        printf "${RED}ERROR: conda env '%s' not found${NC}\n" "$CONDA_ENV"
        echo "  Create it first: conda create -n $CONDA_ENV python=3.13"
        exit 1
    fi
fi

printf "${CYAN}╔═══════════════════════════════════════════════╗${NC}\n"
printf "${CYAN}║     🤖 Vision ID — CUDA Backend Server       ║${NC}\n"
printf "${CYAN}╚═══════════════════════════════════════════════╝${NC}\n"
echo ""

# --------------------------------------------------------------------------
# 1. 查找 Python
# --------------------------------------------------------------------------
PYTHON="python3"
if ! command -v python3 &> /dev/null; then
    printf "${RED}ERROR: python3 not found${NC}\n"
    exit 1
fi

# --------------------------------------------------------------------------
# 2. 环境检查
# --------------------------------------------------------------------------
printf "${GREEN}[1/3]${NC} Checking environment...\n"
echo "  Python: $($PYTHON --version 2>&1) ($PYTHON)"

$PYTHON -c "
import torch
if torch.cuda.is_available():
    print(f'  CUDA:   ✅ {torch.cuda.get_device_name(0)}')
else:
    print('  CUDA:   ❌ Not available (will use CPU)')
" 2>/dev/null || printf "  ${YELLOW}CUDA:   PyTorch not installed yet${NC}\n"

# --------------------------------------------------------------------------
# 3. 依赖安装
# --------------------------------------------------------------------------
echo ""
printf "${GREEN}[2/3]${NC} Checking dependencies...\n"

if ! $PYTHON -c "import fastapi" 2>/dev/null; then
    printf "  ${YELLOW}Dependencies missing. Run install.sh first:${NC}\n"
    echo "    bash install.sh"
    exit 1
else
    echo "  Dependencies OK"
fi

mkdir -p data models

# --------------------------------------------------------------------------
# 4. 启动
# --------------------------------------------------------------------------
echo ""
printf "${GREEN}[3/3]${NC} Starting backend server...\n"
echo ""
printf "  ${CYAN}════════════════════════════════════════════════════════${NC}\n"
printf "  ${CYAN}  Backend API: http://0.0.0.0:10003${NC}\n"
printf "  ${CYAN}  WebSocket:   ws://8.145.38.125:10003/ws/vision${NC}\n"
printf "  ${CYAN}${NC}\n"
printf "  ${CYAN}  Frontend: 在本地浏览器打开 frontend/index.html${NC}\n"
printf "  ${CYAN}  (摄像头在本地采集, JPEG 帧通过 WebSocket 发到此服务器处理)${NC}\n"
printf "  ${CYAN}════════════════════════════════════════════════════════${NC}\n"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

# 确保使用 conda 环境的 libstdc++ (解决 GLIBCXX 版本问题)
CONDA_LIB="${CONDA_PREFIX:-/home/zhiguo/miniconda3/envs/person_id}/lib"
export LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
exec $PYTHON -m src.api.server
