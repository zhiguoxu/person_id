/**
 * Test Body Quality UI Logic
 */
(function () {
    'use strict';

    document.addEventListener('DOMContentLoaded', () => {
        const btnOpen = document.getElementById('btn-open-test-quality');
        if (!btnOpen) return;

        const modal = document.getElementById('test-quality-modal');
        const btnClose = document.getElementById('test-quality-modal-close');
        const backdrop = modal.querySelector('.modal-backdrop');
        
        const fileInput = document.getElementById('test-quality-file');
        const fileNameSpan = document.getElementById('test-quality-filename');
        
        const previewContainer = document.getElementById('test-quality-preview-container');
        const previewImg = document.getElementById('test-quality-preview-img');
        const previewCanvas = document.getElementById('test-quality-preview-canvas');
        
        const resultContainer = document.getElementById('test-quality-result');

        // Open/Close logic
        btnOpen.addEventListener('click', () => {
            modal.classList.remove('hidden');
        });

        const closeModal = () => {
            modal.classList.add('hidden');
            // Reset state
            fileInput.value = '';
            fileNameSpan.textContent = 'No file selected';
            previewContainer.classList.add('hidden');
            resultContainer.classList.add('hidden');
            previewImg.src = '';
            if(previewCanvas.width) {
                const ctx = previewCanvas.getContext('2d');
                ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
            }
        };

        btnClose.addEventListener('click', closeModal);
        backdrop.addEventListener('click', closeModal);

        // File Selection logic
        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) {
                fileNameSpan.textContent = 'No file selected';
                previewContainer.classList.add('hidden');
                resultContainer.classList.add('hidden');
                return;
            }

            fileNameSpan.textContent = file.name;
            resultContainer.classList.remove('hidden');
            resultContainer.innerHTML = '<div style="text-align: center; color: var(--text-muted);">Testing...</div>';
            
            // Show preview
            const reader = new FileReader();
            reader.onload = (ev) => {
                previewImg.src = ev.target.result;
                previewContainer.classList.remove('hidden');
                const ctx = previewCanvas.getContext('2d');
                if (previewCanvas.width) {
                    ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
                }
            };
            reader.readAsDataURL(file);

            // Send API request
            const formData = new FormData();
            formData.append('file', file);

            try {
                const apiUrl = window.BACKEND_CONFIG ? window.BACKEND_CONFIG.apiUrl : '/api';
                const response = await fetch(`${apiUrl}/test_body_quality`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                
                if (data.error) {
                    resultContainer.innerHTML = `<div style="color: var(--color-danger);">Error: ${data.error}</div>`;
                    return;
                }

                if (!data.has_person) {
                    resultContainer.innerHTML = `<div style="color: var(--color-warning);">No person detected in the image.</div>`;
                    return;
                }

                // Render results
                resultContainer.innerHTML = `
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div style="background: var(--bg-surface); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 12px; color: var(--text-muted);">Final Quality</div>
                            <div style="font-size: 20px; font-weight: bold; color: var(--color-primary);">${data.quality.toFixed(3)}</div>
                        </div>
                        <div style="background: var(--bg-surface); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 12px; color: var(--text-muted);">Formula</div>
                            <div style="font-size: 13px; margin-top: 4px;">0.75 * Hint + 0.25 * Sharpness</div>
                        </div>
                        <div style="background: var(--bg-surface); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 12px; color: var(--text-muted);">Quality Hint</div>
                            <div style="font-size: 16px; font-weight: 500;">${data.quality_hint.toFixed(3)}</div>
                        </div>
                        <div style="background: var(--bg-surface); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 12px; color: var(--text-muted);">Sharpness</div>
                            <div style="font-size: 16px; font-weight: 500;">${data.sharpness.toFixed(3)}</div>
                        </div>
                    </div>
                `;

                // Draw bbox
                if (data.bbox) {
                    const tryDraw = () => {
                        if (previewImg.naturalWidth) {
                            drawBbox(data.bbox);
                        } else {
                            setTimeout(tryDraw, 50);
                        }
                    };
                    tryDraw();
                }

            } catch (error) {
                console.error('Test Body Quality Error:', error);
                resultContainer.innerHTML = `<div style="color: var(--color-danger);">Failed to connect to API: ${error.message}</div>`;
            }
        });

        function drawBbox(bbox) {
            const imgW = previewImg.naturalWidth;
            const imgH = previewImg.naturalHeight;
            const displayW = previewImg.offsetWidth;
            const displayH = previewImg.offsetHeight;

            if (!displayW || !displayH || !imgW || !imgH) return;

            previewCanvas.width = displayW;
            previewCanvas.height = displayH;

            const scale = Math.min(displayW / imgW, displayH / imgH);
            const renderedW = imgW * scale;
            const renderedH = imgH * scale;
            // Handle centering offset since max-width/max-height might leave padding depending on CSS layout, 
            // though standard img without object-fit matches rendered dimensions directly.
            const offsetX = (displayW - renderedW) / 2;
            const offsetY = (displayH - renderedH) / 2;
            
            const [x1, y1, x2, y2] = bbox;
            const ctx = previewCanvas.getContext('2d');
            ctx.clearRect(0, 0, displayW, displayH);
            ctx.strokeStyle = '#76ff03'; // body color
            ctx.lineWidth = 2;
            ctx.strokeRect(x1 * scale + offsetX, y1 * scale + offsetY, (x2 - x1) * scale, (y2 - y1) * scale);
        }
    });
})();
