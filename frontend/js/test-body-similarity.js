/**
 * Test Body Similarity — 全身 ReID 相似度测试 (SOLIDER vs OSNet 对比)
 */
(function () {
    'use strict';

    document.addEventListener('DOMContentLoaded', () => {
        const btnOpen = document.getElementById('btn-open-body-sim');
        if (!btnOpen) return;

        const modal = document.getElementById('body-sim-modal');
        const btnClose = document.getElementById('body-sim-modal-close');
        const backdrop = modal.querySelector('.modal-backdrop');

        const file1Input = document.getElementById('body-sim-file1');
        const file2Input = document.getElementById('body-sim-file2');
        const fileName1 = document.getElementById('body-sim-filename1');
        const fileName2 = document.getElementById('body-sim-filename2');

        const preview1Container = document.getElementById('body-sim-preview1-container');
        const preview1Img = document.getElementById('body-sim-preview1-img');
        const preview1Canvas = document.getElementById('body-sim-preview1-canvas');
        const info1Container = document.getElementById('body-sim-info1');

        const preview2Container = document.getElementById('body-sim-preview2-container');
        const preview2Img = document.getElementById('body-sim-preview2-img');
        const preview2Canvas = document.getElementById('body-sim-preview2-canvas');
        const info2Container = document.getElementById('body-sim-info2');

        const resultContainer = document.getElementById('body-sim-result');

        let file1 = null;
        let file2 = null;

        // Open/Close
        btnOpen.addEventListener('click', () => {
            modal.classList.remove('hidden');
        });

        const closeModal = () => {
            modal.classList.add('hidden');
            file1Input.value = '';
            file2Input.value = '';
            fileName1.textContent = 'No file selected';
            fileName2.textContent = 'No file selected';
            preview1Container.classList.add('hidden');
            preview2Container.classList.add('hidden');
            info1Container.classList.add('hidden');
            info2Container.classList.add('hidden');
            resultContainer.classList.add('hidden');
            preview1Img.src = '';
            preview2Img.src = '';
            _clearCanvas(preview1Canvas);
            _clearCanvas(preview2Canvas);
            file1 = null;
            file2 = null;
        };

        btnClose.addEventListener('click', closeModal);
        backdrop.addEventListener('click', closeModal);
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !modal.classList.contains('hidden')) closeModal();
        });

        // File selection handlers
        file1Input.addEventListener('change', (e) => {
            file1 = e.target.files[0] || null;
            fileName1.textContent = file1 ? file1.name : 'No file selected';
            if (file1) {
                _showPreview(file1, preview1Img, preview1Container, preview1Canvas);
            }
            _tryCompare();
        });

        file2Input.addEventListener('change', (e) => {
            file2 = e.target.files[0] || null;
            fileName2.textContent = file2 ? file2.name : 'No file selected';
            if (file2) {
                _showPreview(file2, preview2Img, preview2Container, preview2Canvas);
            }
            _tryCompare();
        });

        function _showPreview(file, imgEl, container, canvas) {
            const reader = new FileReader();
            reader.onload = (ev) => {
                imgEl.src = ev.target.result;
                container.classList.remove('hidden');
                _clearCanvas(canvas);
            };
            reader.readAsDataURL(file);
        }

        function _clearCanvas(canvas) {
            if (canvas.width) {
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        }

        async function _tryCompare() {
            if (!file1 || !file2) return;

            // Show loading
            resultContainer.classList.remove('hidden');
            resultContainer.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 20px;"><div class="face-sim-spinner"></div>Analyzing body features (SOLIDER + OSNet)...</div>';
            info1Container.classList.add('hidden');
            info2Container.classList.add('hidden');

            const formData = new FormData();
            formData.append('file1', file1);
            formData.append('file2', file2);
            const undistortCb = document.getElementById('body-sim-undistort');
            if (undistortCb && undistortCb.checked) {
                formData.append('undistort', 'true');
            }

            try {
                const apiUrl = window.BACKEND_CONFIG ? window.BACKEND_CONFIG.apiUrl : '/api';
                const response = await fetch(`${apiUrl}/test_reid_compare`, {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();

                if (data.error) {
                    resultContainer.innerHTML = `<div style="color: var(--accent-red); text-align: center; padding: 15px;">Error: ${data.error}</div>`;
                    return;
                }

                // 如果有矫正后的图片, 替换预览
                if (data.corrected_image1_b64) {
                    preview1Img.src = 'data:image/jpeg;base64,' + data.corrected_image1_b64;
                }
                if (data.corrected_image2_b64) {
                    preview2Img.src = 'data:image/jpeg;base64,' + data.corrected_image2_b64;
                }

                // Render body info panels
                _renderBodyInfo(data.body1, info1Container, preview1Img, preview1Canvas, 'Image 1');
                _renderBodyInfo(data.body2, info2Container, preview2Img, preview2Canvas, 'Image 2');

                // Render comparison result
                _renderCompareResult(data);

            } catch (error) {
                console.error('Body Similarity Error:', error);
                resultContainer.innerHTML = `<div style="color: var(--accent-red); text-align: center; padding: 15px;">Failed to connect to API: ${error.message}</div>`;
            }
        }

        function _renderBodyInfo(bodyInfo, infoEl, imgEl, canvasEl, label) {
            infoEl.classList.remove('hidden');

            if (!bodyInfo.has_body) {
                infoEl.innerHTML = `<div style="color: var(--accent-orange); font-size: 13px;">⚠️ No person detected in ${label}</div>`;
                return;
            }

            let html = '<div class="body-sim-body-info">';

            // Body crop thumbnail
            if (bodyInfo.body_crop_b64) {
                html += `<img class="body-sim-crop-thumb" src="data:image/jpeg;base64,${bodyInfo.body_crop_b64}" title="Body crop 128×384">`;
            }

            html += `<div class="body-sim-label">✅ Person detected</div>`;
            html += '</div>';
            infoEl.innerHTML = html;

            // Draw person bbox on preview
            const tryDraw = () => {
                if (imgEl.naturalWidth) {
                    _drawBbox(bodyInfo, imgEl, canvasEl);
                } else {
                    setTimeout(tryDraw, 50);
                }
            };
            tryDraw();
        }

        function _drawBbox(bodyInfo, imgEl, canvasEl) {
            const imgW = imgEl.naturalWidth;
            const imgH = imgEl.naturalHeight;
            const displayW = imgEl.offsetWidth;
            const displayH = imgEl.offsetHeight;

            if (!displayW || !displayH || !imgW || !imgH) return;

            canvasEl.width = displayW;
            canvasEl.height = displayH;

            const scale = Math.min(displayW / imgW, displayH / imgH);
            const renderedW = imgW * scale;
            const renderedH = imgH * scale;
            const offsetX = (displayW - renderedW) / 2;
            const offsetY = (displayH - renderedH) / 2;

            const ctx = canvasEl.getContext('2d');
            ctx.clearRect(0, 0, displayW, displayH);

            // Draw person bbox (green, dashed)
            if (bodyInfo.person_bbox) {
                const [px1, py1, px2, py2] = bodyInfo.person_bbox;
                ctx.strokeStyle = '#76ff03';
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 3]);
                ctx.strokeRect(
                    px1 * scale + offsetX, py1 * scale + offsetY,
                    (px2 - px1) * scale, (py2 - py1) * scale
                );
                ctx.setLineDash([]);
                ctx.fillStyle = 'rgba(118, 255, 3, 0.7)';
                ctx.font = '11px Inter, sans-serif';
                ctx.fillText('Person', px1 * scale + offsetX + 3, py1 * scale + offsetY - 4);
            }
        }

        function _makeGaugeHtml(sim, modelName, dim, barColor, interpText, interpColor) {
            const pct = Math.max(0, Math.min(100, sim * 100));
            return `
                <div style="flex: 1; min-width: 280px;">
                    <div class="face-sim-score-label">${modelName} (${dim}D)</div>
                    <div class="face-sim-score-value" style="color: ${barColor}; font-size: 28px;">${sim.toFixed(4)}</div>
                    <div class="face-sim-gauge-track">
                        <div class="face-sim-gauge-fill" style="width: ${pct}%; background: ${barColor};"></div>
                        <div class="face-sim-gauge-marker" style="left: ${pct}%;"></div>
                    </div>
                    <div class="face-sim-gauge-labels">
                        <span>0.0</span>
                        <span>0.5</span>
                        <span>1.0</span>
                    </div>
                    <div class="face-sim-interp" style="color: ${interpColor}; font-size: 12px; margin-top: 6px;">${interpText}</div>
                </div>
            `;
        }

        function _getInterpAndColor(sim) {
            if (sim >= 0.85) return { text: '✅ Very High — Same person', color: 'var(--accent-green)' };
            if (sim >= 0.70) return { text: '🟢 High — Very likely same', color: 'var(--accent-green)' };
            if (sim >= 0.50) return { text: '🟡 Medium — Possibly same', color: 'var(--accent-orange)' };
            if (sim >= 0.30) return { text: '🟠 Low — Unlikely same', color: 'var(--accent-orange)' };
            return { text: '🔴 Very Low — Different', color: 'var(--accent-red)' };
        }

        function _getBarColor(sim) {
            if (sim >= 0.70) return 'var(--accent-green)';
            if (sim >= 0.40) return 'var(--accent-orange)';
            return 'var(--accent-red)';
        }

        function _renderCompareResult(data) {
            if (!data.body1.has_body || !data.body2.has_body) {
                let msg = '';
                if (!data.body1.has_body && !data.body2.has_body) {
                    msg = 'No person detected in either image.';
                } else if (!data.body1.has_body) {
                    msg = 'No person detected in Image 1.';
                } else {
                    msg = 'No person detected in Image 2.';
                }
                resultContainer.innerHTML = `<div style="color: var(--accent-orange); text-align: center; padding: 20px; font-size: 14px;">⚠️ ${msg}</div>`;
                return;
            }

            let html = '<div style="display: flex; gap: 20px; flex-wrap: wrap;">';

            // SOLIDER panel
            if (data.solider_similarity != null) {
                const sim = data.solider_similarity;
                const interp = _getInterpAndColor(sim);
                html += _makeGaugeHtml(sim, '🔷 SOLIDER Swin-Small', data.solider_dim || 768, _getBarColor(sim), interp.text, interp.color);
            }

            // OSNet panel
            if (data.osnet_similarity != null) {
                const sim = data.osnet_similarity;
                const interp = _getInterpAndColor(sim);
                html += _makeGaugeHtml(sim, '🔶 OSNet-AIN x1.0', data.osnet_dim || 512, _getBarColor(sim), interp.text, interp.color);
            }

            html += '</div>';

            // Comparison summary
            if (data.solider_similarity != null && data.osnet_similarity != null) {
                const diff = data.solider_similarity - data.osnet_similarity;
                const diffAbs = Math.abs(diff).toFixed(4);
                const winner = diff > 0 ? 'SOLIDER higher' : diff < 0 ? 'OSNet higher' : 'Equal';
                html += `
                    <div style="margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.03); border-radius: 6px; text-align: center; font-size: 13px; color: var(--text-muted);">
                        Δ = ${diffAbs} (${winner})
                    </div>
                `;
            }

            resultContainer.innerHTML = html;
        }
    });
})();
