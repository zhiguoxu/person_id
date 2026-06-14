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

# --------------------------------------------------------------------------
# HuggingFace 下载
# --------------------------------------------------------------------------
download_huggingface() {
    repo="$1"
    filename="$2"
    filepath="$3"

    if [ -f "$filepath" ]; then
        echo "[SKIP] $(basename "$filepath") already exists"
        return 0
    fi

    mkdir -p "$(dirname "$filepath")"
    echo "[DOWN] $(basename "$filepath") (HuggingFace)"

    url="https://huggingface.co/${repo}/resolve/main/${filename}"

    # 国内镜像 + 直连
    for mirror in \
        "https://hf-mirror.com" \
        "https://huggingface-mirror.com" \
        "https://huggingface.co"; do
        mirror_url="${mirror}/${repo}/resolve/main/${filename}"
        echo "  Trying: ${mirror}..."
        if curl -fSL --connect-timeout 10 --max-time 120 --progress-bar -o "$filepath" "$mirror_url"; then
            if [ -f "$filepath" ] && [ -s "$filepath" ]; then
                echo "  ✅ OK"
                return 0
            fi
        fi
        rm -f "$filepath"
    done

    # GitHub 代理 (部分 HF 模型在 GitHub 有镜像)
    for proxy in "https://ghfast.top/" "https://gh-proxy.com/"; do
        proxy_url="${proxy}https://huggingface.co/${repo}/resolve/main/${filename}"
        echo "  Trying proxy: ${proxy}..."
        if curl -fSL --connect-timeout 10 --max-time 120 --progress-bar -o "$filepath" "$proxy_url"; then
            if [ -f "$filepath" ] && [ -s "$filepath" ]; then
                echo "  ✅ OK"
                return 0
            fi
        fi
        rm -f "$filepath"
    done

    echo "  ❌ Failed: $(basename "$filepath")"
    echo "     Manual download: $url"
    echo "     Then place at: $filepath"
    return 1
}

# ==========================================================================
# 1. YOLO Pose
# ==========================================================================
echo "=== [1/7] YOLO11-Pose ==="
download_github \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt" \
    "yolo11n-pose.pt"
echo ""
download_github \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x-pose.pt" \
    "yolo11x-pose.pt"

# ==========================================================================
# 2. InsightFace buffalo_l (SCRFD 检测 + ArcFace 识别)
# ==========================================================================
echo ""
echo "=== [2/7] InsightFace (buffalo_l) ==="
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

# 复制 ArcFace 模型到 PROJECT_ROOT/models/
ARCFACE_DST="$SCRIPT_DIR/models/w600k_r50.onnx"
if [ -f "$ARCFACE_DST" ]; then
    echo "[SKIP] ArcFace model already in models/"
elif [ -f "$BUFFALO_DIR/w600k_r50.onnx" ]; then
    mkdir -p "$SCRIPT_DIR/models"
    cp "$BUFFALO_DIR/w600k_r50.onnx" "$ARCFACE_DST"
    echo "  ✅ Copied ArcFace (w600k_r50.onnx) to models/"
fi

# ==========================================================================
# 3. AdaFace IR-101 (人脸识别 — 自动下载 + ONNX 转换)
# ==========================================================================
echo ""
echo "=== [3/7] AdaFace IR-101 (face recognition) ==="
ADAFACE_DST="$SCRIPT_DIR/models/adaface_ir101.onnx"
if [ -f "$ADAFACE_DST" ]; then
    echo "[SKIP] AdaFace model already exists"
else
    # 自动调用 Python 脚本: 下载权重 + 转换为 ONNX
    CONVERT_SCRIPT="$SCRIPT_DIR/scripts/convert_adaface_to_onnx.py"
    if [ -f "$CONVERT_SCRIPT" ]; then
        echo "  Running: python $CONVERT_SCRIPT --cleanup"
        if python "$CONVERT_SCRIPT" --cleanup; then
            echo "  ✅ AdaFace ONNX conversion complete"
        else
            echo "  ⚠️ Auto-conversion failed. Manual steps:"
            echo "     pip install torch onnxruntime"
            echo "     python scripts/convert_adaface_to_onnx.py"
        fi
    else
        echo "  ❌ Conversion script not found: $CONVERT_SCRIPT"
    fi
fi
# ==========================================================================
# 4. torchreid OSNet-AIN x1.0
# ==========================================================================
echo ""
echo "=== [4/7] torchreid (osnet_ain_x1_0) ==="

# torchreid 缓存路径
TORCH_CACHE="${TORCH_HOME:-$HOME/.cache/torch}/checkpoints"
OSNET_FILE="$TORCH_CACHE/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64.pth.tar"

# Google Drive file ID
OSNET_GDRIVE_ID="1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo"

download_gdrive "$OSNET_GDRIVE_ID" "$OSNET_FILE"

# ==========================================================================
# 5. BoT-SORT (boxmot 自动下载, 此处预下载加速)
# ==========================================================================
echo ""
echo "=== [5/7] BoT-SORT ReID weights (optional) ==="
echo "[SKIP] boxmot runs without ReID weights (with_reid=False)"

# ==========================================================================
# 6. eDifFIQA Tiny (人脸质量评估)
# ==========================================================================
echo ""
echo "=== [6/7] eDifFIQA Tiny (face quality) ==="
download_huggingface \
    "opencv/face_image_quality_assessment_ediffiqa" \
    "ediffiqa_tiny_jun2024.onnx" \
    "models/edifiqa_tiny.onnx"

echo ""
echo "══════════════════════════════════════════════"
echo "  Summary"
echo "══════════════════════════════════════════════"
echo ""
echo "YOLO Pose:"
ls -lh "$SCRIPT_DIR"/yolo11*-pose.pt 2>/dev/null || echo "  ❌ not found"
echo ""
echo "InsightFace (buffalo_l, SCRFD_10G 检测):"
ls -lh "$BUFFALO_DIR"/det_10g.onnx 2>/dev/null || echo "  ❌ not found"
echo ""
echo "Face Recognition (models/):"
echo "  ArcFace (w600k_r50):"
ls -lh "$SCRIPT_DIR"/models/w600k_r50.onnx 2>/dev/null || echo "    ❌ not found"
echo "  AdaFace (IR-101):"
ls -lh "$SCRIPT_DIR"/models/adaface_ir101.onnx 2>/dev/null || echo "    ❌ not found (run: python scripts/convert_adaface_to_onnx.py)"
echo ""
echo "torchreid:"
if [ -f "$OSNET_FILE" ]; then
    ls -lh "$OSNET_FILE"
else
    echo "  ❌ not found"
    echo "  Manual: download from Google Drive and place at:"
    echo "  $OSNET_FILE"
fi
echo ""
echo "eDifFIQA:"
ls -lh "$SCRIPT_DIR"/models/edifiqa_tiny.onnx 2>/dev/null || echo "  ❌ not found"
