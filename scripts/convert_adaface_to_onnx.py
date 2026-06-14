#!/usr/bin/env python3
"""
AdaFace IR-101 模型自动下载 + ONNX 转换 — 一键脚本

自动完成以下步骤:
  1. 从 HuggingFace 下载 AdaFace IR-101 (WebFace4M) 预训练权重
  2. 从 HuggingFace 下载 IResNet 网络架构代码
  3. 构建 PyTorch 模型并加载权重
  4. 导出为 ONNX 格式
  5. 验证 ONNX 模型输出

用法:
  python scripts/convert_adaface_to_onnx.py
  python scripts/convert_adaface_to_onnx.py --output models/adaface_ir101.onnx
  python scripts/convert_adaface_to_onnx.py --cleanup   # 转换后清理临时文件

前置依赖:
  pip install torch onnxruntime

输出:
  models/adaface_ir101.onnx — 可直接被 FaceExtractor 加载的 ONNX 模型
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


# ==============================================================================
# HuggingFace 文件 URL (从 minchul/cvlface_adaface_ir101_webface4m)
# ==============================================================================
HF_REPO = "minchul/cvlface_adaface_ir101_webface4m"
HF_MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface-mirror.com",
    "https://huggingface.co",
]

# 需要下载的文件 (仅 model.py 用于构建网络, model.pt 用于加载权重)
FILES_TO_DOWNLOAD = {
    "pretrained_model/model.pt": "权重文件 (~250MB)",
    "models/iresnet/model.py": "IResNet 网络架构",
}


def _download_hf_file(repo: str, filename: str, dest: Path,
                       desc: str = "") -> bool:
    """从 HuggingFace 下载单个文件, 支持国内镜像 fallback。"""
    if dest.exists():
        print(f"  [SKIP] {dest.name} already exists")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or filename

    for mirror in HF_MIRRORS:
        url = f"{mirror}/{repo}/resolve/main/{filename}"
        print(f"  [{label}] Trying {mirror} ...")
        try:
            _download_with_progress(url, dest)
            if dest.exists() and dest.stat().st_size > 0:
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"  ✅ OK ({size_mb:.1f} MB)")
                return True
        except Exception as e:
            print(f"     Failed: {e}")
            dest.unlink(missing_ok=True)

    print(f"  ❌ Failed to download: {filename}")
    return False


def _download_with_progress(url: str, dest: Path) -> None:
    """下载文件, 显示进度。"""
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB

        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    print(f"\r     {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)",
                          end="", flush=True)
        if total > 0:
            print()  # newline after progress


def _build_model_and_export(model_py_path: Path, weights_path: Path,
                             output_path: Path, opset: int) -> bool:
    """构建 IResNet 模型, 加载权重, 导出 ONNX。

    直接使用 HuggingFace repo 中的 model.py (IResNet Backbone),
    不依赖 CVLFace 完整框架 (避免 fvcore, omegaconf 等额外依赖)。
    """
    import torch
    import numpy as np

    # --- 动态加载 model.py (patch 掉仅 __main__ 使用的依赖) ---
    print("\n[3/5] Building IResNet-101 model ...")

    # model.py 顶部有 `from fvcore.nn import flop_count` 和 `import numpy`,
    # 这些只在 __main__ 的 FLOPs 计算中使用, 推理完全不需要。
    # 读取源码并去掉这些 import, 避免安装 fvcore。
    patched_path = model_py_path.parent / "_iresnet_patched.py"
    source = model_py_path.read_text(encoding="utf-8")
    lines_out = []
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("from fvcore") or stripped.startswith("import fvcore"):
            lines_out.append("# " + line)  # 注释掉
        elif stripped == "import numpy as np":
            lines_out.append("# " + line)  # 注释掉 (仅 __main__ 使用)
        else:
            lines_out.append(line)
    patched_path.write_text("".join(lines_out), encoding="utf-8")

    import importlib.util
    spec = importlib.util.spec_from_file_location("iresnet_model", str(patched_path))
    iresnet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(iresnet)

    model = iresnet.IR_101(input_size=(112, 112), output_dim=512)
    print(f"  ✅ Model built: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # --- 加载权重 ---
    print("\n[4/5] Loading pretrained weights ...")
    state_dict = torch.load(str(weights_path), map_location="cpu")

    # CVLFace 的 model.pt 可能有不同的 key 前缀格式
    # 情况1: 直接 state_dict
    # 情况2: state_dict 嵌套在 'state_dict' key 下
    # 情况3: key 带 'net.' 前缀 (CVLFace wrapper)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # 尝试直接加载
    try:
        model.load_state_dict(state_dict, strict=True)
        print("  ✅ Weights loaded (exact match)")
    except RuntimeError:
        # 尝试去掉常见前缀: 'net.', 'model.', 'model.net.'
        for prefix in ["net.", "model.net.", "model."]:
            cleaned = {}
            matched = False
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    cleaned[k[len(prefix):]] = v
                    matched = True
                else:
                    cleaned[k] = v
            if matched:
                try:
                    model.load_state_dict(cleaned, strict=True)
                    print(f"  ✅ Weights loaded (stripped prefix: '{prefix}')")
                    break
                except RuntimeError:
                    continue
        else:
            # 最后尝试 non-strict 加载
            result = model.load_state_dict(cleaned, strict=False)
            missing = len(result.missing_keys)
            unexpected = len(result.unexpected_keys)
            if missing > 0:
                print(f"  ⚠️ Loaded with {missing} missing / {unexpected} unexpected keys")
                print(f"     Missing: {result.missing_keys[:5]} ...")
            else:
                print(f"  ✅ Weights loaded ({unexpected} extra keys ignored)")

    model.eval()

    # --- 导出 ONNX ---
    print(f"\n[5/5] Exporting to ONNX (opset={opset}) ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, 112, 112)

    # PyTorch 2.6+ 默认使用 dynamo exporter, 可能导致权重丢失 (输出仅 <1MB)。
    # 强制使用 legacy TorchScript exporter 确保权重正确嵌入。
    export_kwargs = dict(
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    # dynamo=False 在 PyTorch >= 2.6 可用
    try:
        torch.onnx.export(
            model, dummy_input, str(output_path),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        # PyTorch < 2.6 不支持 dynamo 参数
        torch.onnx.export(
            model, dummy_input, str(output_path),
            **export_kwargs,
        )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size_mb:.1f} MB)")

    # 文件大小校验: IR-101 应该 > 100MB, 太小说明权重丢失
    if file_size_mb < 100:
        print(f"  ❌ ERROR: File too small ({file_size_mb:.1f} MB), expected ~250 MB.")
        print("     This usually means the ONNX exporter failed to embed weights.")
        print("     Try: pip install --upgrade torch onnx onnxscript")
        output_path.unlink(missing_ok=True)
        return False

    print(f"  ✅ Size OK ({file_size_mb:.1f} MB)")

    # --- 验证 ONNX ---
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(output_path), providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        dummy_np = np.random.randn(1, 3, 112, 112).astype(np.float32)
        result = session.run(None, {input_name: dummy_np})[0]

        assert result.shape[1] == 512, f"Expected 512D, got {result.shape[1]}D"
        norm = np.linalg.norm(result[0])
        print(f"  ✅ ONNX verified: shape={result.shape}, norm={norm:.4f}")
    except ImportError:
        print("  ⚠️ onnxruntime not installed, skipping verification")
    except Exception as e:
        print(f"  ⚠️ ONNX verification warning: {e}")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AdaFace IR-101 and convert to ONNX (fully automated)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output ONNX file path (default: models/adaface_ir101.onnx)",
    )
    parser.add_argument(
        "--opset", type=int, default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Remove temporary downloaded files after conversion",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download and re-convert even if ONNX already exists",
    )
    args = parser.parse_args()

    # 确定项目根目录和输出路径
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_path = Path(args.output) if args.output else project_root / "models" / "adaface_ir101.onnx"

    # 如果已存在且不强制, 跳过
    if output_path.exists() and not args.force:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ AdaFace ONNX model already exists: {output_path} ({size_mb:.1f} MB)")
        print("   Use --force to re-convert")
        return

    print("╔═══════════════════════════════════════════════╗")
    print("║  AdaFace IR-101 Download + ONNX Conversion    ║")
    print("╚═══════════════════════════════════════════════╝")
    print()

    # 临时目录
    tmp_dir = project_root / ".cache" / "adaface_convert"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- 下载文件 ---
    print("[1/5] Downloading model architecture ...")
    model_py = tmp_dir / "model.py"
    if not _download_hf_file(HF_REPO, "models/iresnet/model.py", model_py,
                              desc="IResNet model.py"):
        sys.exit(1)

    print("\n[2/5] Downloading pretrained weights (this may take a few minutes) ...")
    weights_pt = tmp_dir / "model.pt"
    if not _download_hf_file(HF_REPO, "pretrained_model/model.pt", weights_pt,
                              desc="model.pt"):
        sys.exit(1)

    # --- 检查 torch 是否可用 ---
    try:
        import torch
    except ImportError:
        print("\n❌ PyTorch is not installed.")
        print("   Install with: pip install torch")
        print(f"\n   Downloaded files are cached in: {tmp_dir}")
        print("   Re-run this script after installing torch.")
        sys.exit(1)

    # --- 构建并导出 ---
    success = _build_model_and_export(model_py, weights_pt, output_path, args.opset)

    if not success:
        print("\n❌ Conversion failed")
        sys.exit(1)

    # --- 清理 ---
    if args.cleanup:
        print(f"\nCleaning up: {tmp_dir}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("  ✅ Cleaned")
    else:
        print(f"\n💡 Cached files in: {tmp_dir}")
        print("   Use --cleanup to remove after conversion")

    print()
    print("══════════════════════════════════════════════")
    print("  Done! AdaFace IR-101 is ready to use.")
    print("══════════════════════════════════════════════")
    print()
    print("  To use AdaFace, set in config:")
    print('    face.recognition_backend = "adaface"')
    print()
    print("  To switch back to ArcFace:")
    print('    face.recognition_backend = "arcface"')
    print()


if __name__ == "__main__":
    main()
