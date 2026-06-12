/**
 * Image Lightbox — 全局图片预览 + 下载
 *
 * 使用事件委托自动拦截所有 <img> 点击,
 * 弹出大图预览遮罩层, 支持下载按钮。
 * 排除 video, canvas, overlay 等非图片元素。
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
            <img class="lightbox-img" alt="Preview" />
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
    const downloadBtn = overlay.querySelector('.lightbox-download');

    // =========================================================================
    // 打开 / 关闭
    // =========================================================================
    function open(src) {
        lightboxImg.src = src;
        downloadBtn.href = src;

        // 生成文件名
        const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        downloadBtn.download = `vision-id-${ts}.jpg`;

        overlay.classList.remove('hidden');
    }

    function close() {
        overlay.classList.add('hidden');
        lightboxImg.src = '';
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

        e.stopPropagation();
        open(img.src);
    });

    // 暴露全局 API (供其他模块主动调用)
    window.imageLightbox = { open, close };
})();
