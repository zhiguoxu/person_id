/**
 * Image Lightbox — 全局图片预览 + 下载
 *
 * 使用事件委托自动拦截所有 <img> 点击,
 * 弹出大图预览遮罩层, 支持下载按钮。
 * 如果图片有 data-overlay-bbox 属性, 在预览时叠加框线。
 * 下载始终为原图 (不含框线)。
 */
(function () {
    'use strict';

    // =========================================================================
    // 创建 Lightbox DOM (只创建一次)
    // =========================================================================
    const overlay = document.createElement('div');
    overlay.id = 'image-lightbox';
    overlay.className = 'lightbox-overlay hidden';
    overlay.innerHTML = `
        <div class="lightbox-backdrop"></div>
        <div class="lightbox-body">
            <div class="lightbox-img-container">
                <img class="lightbox-img" alt="Preview" />
                <canvas class="lightbox-canvas"></canvas>
            </div>
            <div class="lightbox-toolbar">
                <a class="lightbox-btn lightbox-download" title="Download" download="image.jpg">
                    ⬇ Download
                </a>
                <button class="lightbox-btn lightbox-close" title="Close">✕</button>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);

    const lightboxImg = overlay.querySelector('.lightbox-img');
    const lightboxCanvas = overlay.querySelector('.lightbox-canvas');
    const downloadBtn = overlay.querySelector('.lightbox-download');

    // =========================================================================
    // 打开 / 关闭
    // =========================================================================
    function open(src, overlayBbox, boxColor) {
        lightboxImg.src = src;
        downloadBtn.href = src;

        // 生成文件名
        const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        downloadBtn.download = `vision-id-${ts}.jpg`;

        // 清除旧的 canvas
        lightboxCanvas.width = 0;
        lightboxCanvas.height = 0;
        lightboxCanvas.style.display = 'none';

        // 如果有 overlay_bbox, 图片加载后画框
        if (overlayBbox) {
            lightboxImg.onload = () => {
                _drawOverlayBbox(overlayBbox, boxColor || '#00e5ff');
                lightboxImg.onload = null;
            };
        }

        overlay.classList.remove('hidden');
    }

    function _drawOverlayBbox(bbox, color) {
        const imgW = lightboxImg.naturalWidth;
        const imgH = lightboxImg.naturalHeight;
        const displayW = lightboxImg.offsetWidth;
        const displayH = lightboxImg.offsetHeight;

        if (!displayW || !displayH) return;

        lightboxCanvas.width = displayW;
        lightboxCanvas.height = displayH;
        lightboxCanvas.style.display = 'block';

        // object-fit: contain 缩放计算
        const scale = Math.min(displayW / imgW, displayH / imgH);
        const renderedW = imgW * scale;
        const renderedH = imgH * scale;
        const offsetX = (displayW - renderedW) / 2;
        const offsetY = (displayH - renderedH) / 2;

        const [x1, y1, x2, y2] = bbox;

        const ctx = lightboxCanvas.getContext('2d');
        ctx.clearRect(0, 0, displayW, displayH);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.shadowColor = color.replace(')', ', 0.5)').replace('rgb', 'rgba');
        ctx.shadowBlur = 6;
        ctx.strokeRect(
            x1 * scale + offsetX, y1 * scale + offsetY,
            (x2 - x1) * scale, (y2 - y1) * scale
        );
    }

    function close() {
        overlay.classList.add('hidden');
        lightboxImg.src = '';
        lightboxImg.onload = null;
        lightboxCanvas.style.display = 'none';
    }

    // 点击遮罩关闭
    overlay.querySelector('.lightbox-backdrop').addEventListener('click', close);
    overlay.querySelector('.lightbox-close').addEventListener('click', close);

    // ESC 关闭
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !overlay.classList.contains('hidden')) {
            close();
        }
    });

    // =========================================================================
    // 全局事件委托: 拦截所有 img 点击
    // =========================================================================
    document.addEventListener('click', (e) => {
        const img = e.target.closest('img');
        if (!img) return;

        // 排除: video 元素、不含 base64 src 的图片、overlay canvas 等
        if (!img.src || !img.src.startsWith('data:image')) return;

        // 排除某些容器中的图片 (如 confirm modal 的候选头像等不需要放大的)
        if (img.closest('#video-container')) return;
        if (img.closest('.candidate-card')) return;
        // 画廊卡片头像: 点击应打开个人详情面板, 不弹图片预览
        if (img.closest('.person-avatar')) return;

        e.stopPropagation();

        // 检查是否有 overlay_bbox 数据
        let overlayBbox = null;
        let boxColor = null;
        if (img.dataset.overlayBbox) {
            try {
                overlayBbox = JSON.parse(img.dataset.overlayBbox);
                boxColor = img.dataset.boxColor || '#00e5ff';
            } catch { /* ignore */ }
        }

        open(img.src, overlayBbox, boxColor);
    });
})();
