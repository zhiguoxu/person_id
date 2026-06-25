#!/bin/bash
# ============================================================
# 在【Mac】上执行: 从 Linux 开发机拉取 person_id 前端到本地副本
#
# 方向: 远程开发机(Linux)  ──rsync over SSH──▶  本地 Mac
#
# 用法 (在 Mac 终端):
#   bash pull_frontend.sh                   # 推荐: 用 bash 调用, 无需执行权限
#   ./pull_frontend.sh                      # 需先 chmod +x pull_frontend.sh
#   bash pull_frontend.sh --delete          # 镜像(删除本地多余文件)
#   REMOTE_HOST=64.83.11.248 bash pull_frontend.sh
# ============================================================
set -euo pipefail

# ── 远程开发机 (源, 按需修改) ─────────────────────────────
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-64.83.11.248}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/root/workspace/voice_agent/person_id/frontend/}"

# ── 本地 Mac (目标) ───────────────────────────────────────
LOCAL_DIR="${LOCAL_DIR:-/Users/xuzhiguo/workspace/python/lx/person_id/frontend/}"

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

# 解析参数: --delete 镜像; 第一个非选项参数当作远程地址覆盖
# 注: 用普通字符串而非数组, 兼容 macOS 自带 bash 3.2 在 set -u 下对空数组的 bug
DELETE=""
for arg in "$@"; do
    case "$arg" in
        --delete) DELETE="--delete" ;;
        -*) printf "${RED}未知选项: %s${NC}\n" "$arg"; exit 1 ;;
        *) REMOTE_HOST="$arg" ;;
    esac
done

if [ -z "$REMOTE_HOST" ]; then
    printf "${RED}[ERROR]${NC} 未设置远程开发机地址 REMOTE_HOST\n"; exit 1
fi

mkdir -p "$LOCAL_DIR"

printf "${CYAN}── 拉取前端: 开发机 → 本机 Mac ──${NC}\n"
printf "  源:   %s@%s:%s\n" "$REMOTE_USER" "$REMOTE_HOST" "$REMOTE_DIR"
printf "  目标: %s\n" "$LOCAL_DIR"
[ -n "$DELETE" ] && printf "  ${YELLOW}模式: --delete (镜像, 会删除本地多余文件)${NC}\n"

# 注: macOS 自带 rsync 较老, 使用通用标志 (-az / --stats, 不用 --info)
# $DELETE 不加引号: 空时展开为无, 有值时为单个 --delete token
rsync -az --stats $DELETE \
    -e "ssh -p ${REMOTE_PORT}" \
    --exclude '.DS_Store' --exclude 'node_modules' --exclude '.git' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}" "$LOCAL_DIR"

printf "${GREEN}[OK]${NC} 前端已拉到本地。浏览器硬刷新(Cmd+Shift+R)使 config.js 生效。\n"
printf "  打开: file://%sindex.html?camera_id=EU0125MH00100015056\n" "$LOCAL_DIR"
