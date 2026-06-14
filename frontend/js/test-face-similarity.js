/**
 * Test Face Similarity UI Logic
 */
(function () {
    'use strict';

    document.addEventListener('DOMContentLoaded', () => {
        const btnOpen = document.getElementById('btn-open-face-sim');
        if (!btnOpen) return;

        const modal = document.getElementById('face-sim-modal');
        const btnClose = document.getElementById('face-sim-modal-close');
        const backdrop = modal.querySelector('.modal-backdrop');

        const file1Input = document.getElementById('face-sim-file1');
        const file2Input = document.getElementById('face-sim-file2');
        const fileName1 = document.getElementById('face-sim-filename1');
        const fileName2 = document.getElementById('face-sim-filename2');

        const preview1Container = document.getElementById('face-sim-preview1-container');
        const preview1Img = document.getElementById('face-sim-preview1-img');
        const preview1Canvas = document.getElementById('face-sim-preview1-canvas');
        const info1Container = document.getElementById('face-sim-info1');

        const preview2Container = document.getElementById('face-sim-preview2-container');
        const preview2Img = document.getElementById('face-sim-preview2-img');
        const preview2Canvas = document.getElementById('face-sim-preview2-canvas');
        const info2Container = document.getElementById('face-sim-info2');

        const resultContainer = document.getElementById('face-sim-result');

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
            resultContainer.innerHTML = '<div style="text-align: center; color: var(--text-muted); padding: 20px;"><div class="face-sim-spinner"></div>Analyzing faces...</div>';
            info1Container.classList.add('hidden');
            info2Container.classList.add('hidden');

            const formData = new FormData();
            formData.append('file1', file1);
            formData.append('file2', file2);

            try {
                const apiUrl = window.BACKEND_CONFIG ? window.BACKEND_CONFIG.apiUrl : '/api';
                const response = await fetch(`${apiUrl}/test_face_similarity`, {
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

                // Render face info panels
                _renderFaceInfo(data.face1, info1Container, preview1Img, preview1Canvas, 'Image 1');
                _renderFaceInfo(data.face2, info2Container, preview2Img, preview2Canvas, 'Image 2');

                // Render similarity result
                _renderSimilarityResult(data);

            } catch (error) {
                console.error('Face Similarity Error:', error);
                resultContainer.innerHTML = `<div style="color: var(--accent-red); text-align: center; padding: 15px;">Failed to connect to API: ${error.message}</div>`;
            }
        }

        function _renderFaceInfo(faceInfo, infoEl, imgEl, canvasEl, label) {
            infoEl.classList.remove('hidden');

            if (!faceInfo.has_face) {
                infoEl.innerHTML = `<div style="color: var(--accent-orange); font-size: 13px;">⚠️ No face detected in ${label}</div>`;
                return;
            }

            let html = '<div class="face-sim-face-info">';

            // Aligned face thumbnail
            if (faceInfo.aligned_face_b64) {
                html += `<img class="face-sim-aligned-thumb" src="data:image/jpeg;base64,${faceInfo.aligned_face_b64}" title="Aligned 112×112 face">`;
            }

            // Quality badge
            if (faceInfo.face_quality != null) {
                const q = faceInfo.face_quality;
                const qColor = q >= 0.5 ? 'var(--accent-green)' : q >= 0.3 ? 'var(--accent-orange)' : 'var(--accent-red)';
                html += `<div class="face-sim-quality">Quality: <span style="color: ${qColor}; font-weight: 600;">${q.toFixed(4)}</span></div>`;
            }

            html += '</div>';
            infoEl.innerHTML = html;

            // Draw bboxes on preview
            const tryDraw = () => {
                if (imgEl.naturalWidth) {
                    _drawBboxes(faceInfo, imgEl, canvasEl);
                } else {
                    setTimeout(tryDraw, 50);
                }
            };
            tryDraw();
        }

        function _drawBboxes(faceInfo, imgEl, canvasEl) {
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
            if (faceInfo.person_bbox) {
                const [px1, py1, px2, py2] = faceInfo.person_bbox;
                ctx.strokeStyle = '#76ff03';
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 3]);
                ctx.strokeRect(
                    px1 * scale + offsetX, py1 * scale + offsetY,
                    (px2 - px1) * scale, (py2 - py1) * scale
                );
                // Label
                ctx.setLineDash([]);
                ctx.fillStyle = 'rgba(118, 255, 3, 0.7)';
                ctx.font = '11px Inter, sans-serif';
                ctx.fillText('Person', px1 * scale + offsetX + 3, py1 * scale + offsetY - 4);
            }

            // Draw face bbox (cyan, solid)
            if (faceInfo.face_bbox) {
                const [fx1, fy1, fx2, fy2] = faceInfo.face_bbox;
                ctx.strokeStyle = '#00e5ff';
                ctx.lineWidth = 2;
                ctx.setLineDash([]);
                ctx.strokeRect(
                    fx1 * scale + offsetX, fy1 * scale + offsetY,
                    (fx2 - fx1) * scale, (fy2 - fy1) * scale
                );
                // Label
                ctx.fillStyle = 'rgba(0, 229, 255, 0.8)';
                ctx.font = '11px Inter, sans-serif';
                ctx.fillText('Face', fx1 * scale + offsetX + 3, fy1 * scale + offsetY - 4);
            }
        }

        function _renderSimilarityResult(data) {
            if (!data.face1.has_face || !data.face2.has_face) {
                let msg = '';
                if (!data.face1.has_face && !data.face2.has_face) {
                    msg = 'No face detected in either image.';
                } else if (!data.face1.has_face) {
                    msg = 'No face detected in Image 1.';
                } else {
                    msg = 'No face detected in Image 2.';
                }
                resultContainer.innerHTML = `<div style="color: var(--accent-orange); text-align: center; padding: 20px; font-size: 14px;">⚠️ ${msg}</div>`;
                return;
            }

            const sim = data.similarity;
            const pct = Math.max(0, Math.min(100, ((sim + 1) / 2) * 100)); // map [-1,1] to [0,100]

            // Interpretation
            let interp, interpColor;
            if (sim >= 0.5) {
                interp = '✅ Very High — Very likely the same person';
                interpColor = 'var(--accent-green)';
            } else if (sim >= 0.4) {
                interp = '🟢 High — Likely the same person';
                interpColor = 'var(--accent-green)';
            } else if (sim >= 0.3) {
                interp = '🟡 Medium — Possibly the same person';
                interpColor = 'var(--accent-orange)';
            } else if (sim >= 0.2) {
                interp = '🟠 Low — Unlikely the same person';
                interpColor = 'var(--accent-orange)';
            } else {
                interp = '🔴 Very Low — Different persons';
                interpColor = 'var(--accent-red)';
            }

            // Gauge bar color
            let barColor;
            if (sim >= 0.4) barColor = 'var(--accent-green)';
            else if (sim >= 0.25) barColor = 'var(--accent-orange)';
            else barColor = 'var(--accent-red)';

            resultContainer.innerHTML = `
                <div class="face-sim-score-panel">
                    <div class="face-sim-score-label">Face Similarity Score (Cosine)</div>
                    <div class="face-sim-score-value" style="color: ${barColor};">${sim.toFixed(4)}</div>
                    <div class="face-sim-gauge-track">
                        <div class="face-sim-gauge-fill" style="width: ${pct}%; background: ${barColor};"></div>
                        <div class="face-sim-gauge-marker" style="left: ${pct}%;"></div>
                    </div>
                    <div class="face-sim-gauge-labels">
                        <span>-1.0</span>
                        <span>0.0</span>
                        <span>+1.0</span>
                    </div>
                    <div class="face-sim-interp" style="color: ${interpColor};">${interp}</div>
                    <div class="face-sim-thresholds">
                        <span>System thresholds — Definite: ≥0.85 | Confident: ≥0.72 | Suspected: ≥0.55</span>
                    </div>
                </div>
            `;
        }
    });
})();
