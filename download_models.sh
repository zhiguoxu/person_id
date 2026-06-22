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
# 4. SOLIDER Swin-Small ReID (全身重识别)
# ==========================================================================
echo ""
echo "=== [4/7] SOLIDER Swin-Small (ReID) ==="

SOLIDER_DST="$SCRIPT_DIR/models/solider_swin_small_reid.pth"

# SOLIDER-REID finetuned on Market-1501 (Swin-Small)
# Source: https://github.com/tinyvision/SOLIDER-REID Performance table, row 2 (Swin-Small), Market-1501
SOLIDER_REID_GDRIVE_ID="1C-aIZdFyjFsZX4W4feG-Ex39RU2Qvu3b"

# SOLIDER pretrained backbone (Swin-Small, fallback)
# Source: https://github.com/tinyvision/SOLIDER Models table
SOLIDER_PRETRAINED_GDRIVE_ID="1oyEgASqDHc7YUPsQUMxuo2kBZyi2Tzfv"

if [ -f "$SOLIDER_DST" ]; then
    echo "[SKIP] SOLIDER Swin-Small already exists"
else
    echo "[DOWN] SOLIDER Swin-Small ReID (Market-1501 finetuned)"
    # 优先下载 SOLIDER-REID 微调版 (精度更高)
    download_gdrive "$SOLIDER_REID_GDRIVE_ID" "$SOLIDER_DST"

    if [ ! -f "$SOLIDER_DST" ] || [ ! -s "$SOLIDER_DST" ]; then
        echo "  ⚠️ SOLIDER-REID finetuned download failed, trying SOLIDER pretrained..."
        rm -f "$SOLIDER_DST"
        # 下载 SOLIDER 原始预训练权重 (需要提取 teacher 键)
        SOLIDER_RAW="$SCRIPT_DIR/models/_solider_swin_small_raw.pth"
        download_gdrive "$SOLIDER_PRETRAINED_GDRIVE_ID" "$SOLIDER_RAW"
        if [ -f "$SOLIDER_RAW" ] && [ -s "$SOLIDER_RAW" ]; then
            echo "  Converting SOLIDER pretrained → ReID format..."
            python -c "
import torch
ckpt = torch.load('$SOLIDER_RAW', map_location='cpu')
if 'teacher' in ckpt:
    torch.save(ckpt['teacher'], '$SOLIDER_DST')
else:
    torch.save(ckpt, '$SOLIDER_DST')
print('  ✅ Converted')
"
            rm -f "$SOLIDER_RAW"
        fi
    fi

    if [ -f "$SOLIDER_DST" ] && [ -s "$SOLIDER_DST" ]; then
        echo "  ✅ SOLIDER weights ready"
    else
        echo "  ❌ All downloads failed."
        echo "  Manual download options:"
        echo "    1. SOLIDER-REID finetuned (推荐): https://drive.google.com/file/d/$SOLIDER_REID_GDRIVE_ID/view"
        echo "    2. SOLIDER pretrained (备选):     https://drive.google.com/file/d/$SOLIDER_PRETRAINED_GDRIVE_ID/view"
        echo "  Place at: $SOLIDER_DST"
    fi
fi


# ==========================================================================
# 5. BoT-SORT (boxmot 自动下载, 此处预下载加速)
# ==========================================================================
echo ""
echo "=== [5/7] BoT-SORT ReID weights (optional) ==="
echo "[SKIP] boxmot runs without ReID weights (with_reid=False)"

# ==========================================================================
# 6. eDifFIQA (人脸质量评估 — 4 个变体)
# ==========================================================================
echo ""
echo "=== [6/7] eDifFIQA (face quality, all variants) ==="

# Tiny — from HuggingFace (OpenCV Zoo)
download_huggingface \
    "opencv/face_image_quality_assessment_ediffiqa" \
    "ediffiqa_tiny_jun2024.onnx" \
    "models/ediffiqa_tiny.onnx"

# Small / Medium / Large — from GitHub releases (yakhyo/face-image-quality-assessment)
EDIFIQA_RELEASE="https://github.com/yakhyo/face-image-quality-assessment/releases/download/weights"
# 上游文件名用双 f (ediffiqa), 项目统一用单 f (edifiqa)
for size in s m l; do
    download_github \
        "${EDIFIQA_RELEASE}/ediffiqa_${size}.onnx" \
        "models/ediffiqa_${size}.onnx"
done

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
echo "SOLIDER ReID:"
if [ -f "$SOLIDER_DST" ]; then
    ls -lh "$SOLIDER_DST"
else
    echo "  ❌ not found"
    echo "  Manual: download from https://github.com/tinyvision/SOLIDER-REID"
    echo "  Then place at: $SOLIDER_DST"
fi
echo ""
echo "eDifFIQA (face quality):"
for f in ediffiqa_tiny.onnx ediffiqa_s.onnx ediffiqa_m.onnx ediffiqa_l.onnx; do
    if [ -f "$SCRIPT_DIR/models/$f" ]; then
        printf "  ✅ %-20s " "$f"
        ls -lh "$SCRIPT_DIR/models/$f" | awk '{print $5}'
    else
        echo "  ❌ $f not found"
    fi
done

