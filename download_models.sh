#!/bin/bash
# ==============================================================================
# 模型下载脚本 (国内加速)
#
# 用法:  bash download_models.sh
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔═══════════════════════════════════════════════╗"
echo "║     📦 Download All Models (CN Proxy)         ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# --------------------------------------------------------------------------
# GitHub 代理下载
# --------------------------------------------------------------------------
download_github() {
    url="$1"
    filepath="$2"

    if [ -f "$filepath" ]; then
        echo "[SKIP] $filepath already exists"
        return 0
    fi

    mkdir -p "$(dirname "$filepath")"
    echo "[DOWN] $(basename "$filepath")"

    for proxy in "https://ghfast.top/" "https://gh-proxy.com/" "https://ghproxy.cc/"; do
        proxy_url="${proxy}${url}"
        echo "  Trying: ${proxy}..."
        if curl -fSL --connect-timeout 15 --progress-bar -o "$filepath" "$proxy_url"; then
            echo "  ✅ OK"
            return 0
        fi
        rm -f "$filepath"
    done

    echo "  Trying direct..."
    if curl -fSL --progress-bar -o "$filepath" "$url"; then
        echo "  ✅ OK"
        return 0
    fi

    echo "  ❌ Failed: $filepath"
    return 1
}

# --------------------------------------------------------------------------
# Google Drive 下载 (gdown)
# --------------------------------------------------------------------------
download_gdrive() {
    file_id="$1"
    filepath="$2"
    filename="$(basename "$filepath")"

    if [ -f "$filepath" ]; then
        echo "[SKIP] $filename already exists"
        return 0
    fi

    mkdir -p "$(dirname "$filepath")"
    echo "[DOWN] $filename (Google Drive)"

    # 方法 1: gdown (Python, 支持大文件确认)
    if command -v gdown &> /dev/null; then
        echo "  Trying gdown..."
        if gdown "$file_id" -O "$filepath" 2>/dev/null; then
            if [ -f "$filepath" ] && [ -s "$filepath" ]; then
                echo "  ✅ OK (gdown)"
                return 0
            fi
        fi
        rm -f "$filepath"
    fi

    # 方法 2: curl 直连 (有时可行)
    echo "  Trying curl direct..."
    gdrive_url="https://drive.usercontent.google.com/download?id=${file_id}&confirm=t"
    if curl -fSL --connect-timeout 30 --progress-bar -o "$filepath" "$gdrive_url"; then
        if [ -f "$filepath" ] && [ -s "$filepath" ]; then
            echo "  ✅ OK (curl)"
            return 0
        fi
    fi
    rm -f "$filepath"

    # 方法 3: 代理
    for proxy in "https://ghfast.top/" "https://gh-proxy.com/"; do
        echo "  Trying proxy: ${proxy}..."
        proxy_url="${proxy}https://drive.google.com/uc?id=${file_id}"
        if curl -fSL --connect-timeout 15 --progress-bar -o "$filepath" "$proxy_url"; then
            if [ -f "$filepath" ] && [ -s "$filepath" ]; then
                echo "  ✅ OK (proxy)"
                return 0
            fi
        fi
        rm -f "$filepath"
    done

    echo "  ❌ Failed: $filename"
    echo "     Manual download: https://drive.google.com/uc?id=${file_id}"
    echo "     Then place at: $filepath"
    return 1
}

# ==========================================================================
# 1. YOLO Pose
# ==========================================================================
echo "=== [1/4] YOLO11-Pose ==="
download_github \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt" \
    "yolo11n-pose.pt"
echo ""
download_github \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x-pose.pt" \
    "yolo11x-pose.pt"

# ==========================================================================
# 2. InsightFace buffalo_l
# ==========================================================================
echo ""
echo "=== [2/4] InsightFace (buffalo_l) ==="
INSIGHTFACE_DIR="$HOME/.insightface/models"
BUFFALO_DIR="$INSIGHTFACE_DIR/buffalo_l"

if [ -f "$BUFFALO_DIR/det_10g.onnx" ] && [ -f "$BUFFALO_DIR/w600k_r50.onnx" ]; then
    echo "[SKIP] buffalo_l already extracted"
else
    BUFFALO_ZIP="$INSIGHTFACE_DIR/buffalo_l.zip"
    download_github \
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" \
        "$BUFFALO_ZIP"

    if [ -f "$BUFFALO_ZIP" ]; then
        echo "  Extracting to $BUFFALO_DIR ..."
        mkdir -p "$BUFFALO_DIR"
        unzip -qoj "$BUFFALO_ZIP" "*.onnx" -d "$BUFFALO_DIR"
        echo "  ✅ Extracted"
        rm -f "$BUFFALO_ZIP"
    fi
fi

# ==========================================================================
# 3. torchreid OSNet-AIN x1.0
# ==========================================================================
echo ""
echo "=== [3/4] torchreid (osnet_ain_x1_0) ==="

# torchreid 缓存路径
TORCH_CACHE="${TORCH_HOME:-$HOME/.cache/torch}/checkpoints"
OSNET_FILE="$TORCH_CACHE/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64.pth.tar"

# Google Drive file ID
OSNET_GDRIVE_ID="1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo"

download_gdrive "$OSNET_GDRIVE_ID" "$OSNET_FILE"

# ==========================================================================
# 4. BoT-SORT (boxmot 自动下载, 此处预下载加速)
# ==========================================================================
echo ""
echo "=== [4/4] BoT-SORT ReID weights (optional) ==="
echo "[SKIP] boxmot runs without ReID weights (with_reid=False)"

# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo "══════════════════════════════════════════════"
echo "  Summary"
echo "══════════════════════════════════════════════"
echo ""
echo "YOLO Pose:"
ls -lh "$SCRIPT_DIR"/yolo11*-pose.pt 2>/dev/null || echo "  ❌ not found"
echo ""
echo "InsightFace:"
ls -lh "$BUFFALO_DIR"/*.onnx 2>/dev/null || echo "  ❌ not found"
echo ""
echo "torchreid:"
if [ -f "$OSNET_FILE" ]; then
    ls -lh "$OSNET_FILE"
else
    echo "  ❌ not found"
    echo "  Manual: download from Google Drive and place at:"
    echo "  $OSNET_FILE"
fi
