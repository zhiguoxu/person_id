#!/bin/bash
# ==============================================================================
# Vision ID — CUDA 服务器部署启动脚本
#
# 架构: 前端(本地浏览器) ←WebSocket→ 后端(此服务器 CUDA 推理)
# 服务器: 123.206.174.158:10003 (GPU 服务机, 内网 172.17.48.17; 2026-08 迁自 1.15.11.133)
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
# conda 安装位置因机器而异: 优先用户目录, 回退系统目录 (GPU 服务机为 /opt/miniconda3)
CONDA_BASE="${HOME}/miniconda3"
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || CONDA_BASE="/opt/miniconda3"

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
printf "  ${CYAN}  WebSocket:   ws://123.206.174.158:10003/ws/vision${NC}\n"
printf "  ${CYAN}${NC}\n"
printf "  ${CYAN}  Frontend: web 控制台「视觉识别」页 (web/src/vision, 经 /vision 代理访问)${NC}\n"
printf "  ${CYAN}════════════════════════════════════════════════════════${NC}\n"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

# 确保使用 conda 环境的 libstdc++ (解决 GLIBCXX 版本问题)
CONDA_LIB="${CONDA_PREFIX:-$CONDA_BASE/envs/$CONDA_ENV}/lib"
export LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}"

# onnxruntime-gpu 独立 dlopen CUDA 库 (libcublasLt 等): torch 先导入时会把
# pip 装的 nvidia 库预加载进进程, 但为不依赖导入顺序, 显式挂上这些目录
# (缺失时 ORT 会静默回退 CPU, Tier2 推理慢一个数量级)
NV_LIBS="$(ls -d "$CONDA_LIB"/python3*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
[ -n "$NV_LIBS" ] && export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH}"

# voice_agent_common + session_store 以 PYTHONPATH 源码副本方式引入
# (配置在线编辑等共享逻辑及其 DB 存储): 远端部署时 deploy_mac.sh 把
# packages/{common,session_store} 同步进本目录; 从工程仓库直跑时它们在
# 上级 packages/ 下, 两组路径都挂上, 不存在的目录 Python 会自动忽略。
export PYTHONPATH="$SCRIPT_DIR/packages/common:$SCRIPT_DIR/packages/session_store:$SCRIPT_DIR/../packages/common:$SCRIPT_DIR/../packages/session_store:$SCRIPT_DIR:$PYTHONPATH"
exec $PYTHON -m src.main
