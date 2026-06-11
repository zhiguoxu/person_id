/**
 * Pipeline Panel — 算法流程可视化面板
 * 
 * 显示算法各阶段的状态、耗时、缩略图、匹配分数条形图。
 */
class PipelinePanel {
    constructor() {
        this.stages = ['detection', 'face_detect', 'face_assess', 'reid'];
        this._lastTimeMs = {};  // 缓存每个阶段的最后执行时长
        this._bindExpanders();
    }

    _bindExpanders() {
        this.stages.forEach(stage => {
            const el = document.querySelector(`.pipeline-stage[data-stage="${stage}"] .stage-header`);
            if (el) {
                el.addEventListener('click', () => {
                    const parent = el.closest('.pipeline-stage');
                    parent.classList.toggle('expanded');
                });
            }
        });
    }

    /**
     * 更新流水线调试面板
     */
    update(debug) {
        if (!debug) return;

        let totalMs = 0;

        this.stages.forEach(stage => {
            const data = debug[stage];
            if (!data) return;

            const el = document.querySelector(`.pipeline-stage[data-stage="${stage}"]`);
            if (!el) return;

            const statusEl = el.querySelector('.stage-status');
            const timeEl = el.querySelector('.stage-time');
            const detailsEl = el.querySelector('.stage-details');

            // 状态
            const status = data.status || 'pending';
            statusEl.className = `stage-status ${status}`;
            statusEl.textContent = this._statusIcon(status);

            // 耗时
            if (data.time_ms !== undefined && data.time_ms > 0) {
                timeEl.textContent = `${data.time_ms.toFixed(1)}ms`;
                timeEl.classList.remove('stale');
                totalMs += data.time_ms;
                this._lastTimeMs[stage] = data.time_ms;
            } else if (this._lastTimeMs[stage] > 0) {
                // pending/skipped 时显示上次的时长 (变灰)
                timeEl.textContent = `${this._lastTimeMs[stage].toFixed(1)}ms`;
                timeEl.classList.add('stale');
            } else {
                timeEl.textContent = '—';
                timeEl.classList.remove('stale');
            }

            // 高亮当前处理中的阶段
            el.classList.toggle('active', status === 'running');

            // 详情
            detailsEl.innerHTML = this._renderDetails(stage, data);
        });

        // 总耗时
        const totalEl = document.getElementById('pipeline-total-time');
        if (totalEl) totalEl.textContent = `${totalMs.toFixed(1)}ms`;
    }

    _statusIcon(status) {
        switch (status) {
            case 'done': return '✅';
            case 'running': return '⏳';
            case 'skipped': return '⏭️';
            case 'error': return '❌';
            default: return '—';
        }
    }

    _renderDetails(stage, data) {
        switch (stage) {
            case 'detection':
                return this._renderDetectionDetails(data);
            case 'face_detect':
                return this._renderFaceDetectDetails(data);
            case 'face_assess':
                return this._renderFaceAssessDetails(data);
            case 'reid':
                return this._renderReIDDetails(data);
            default:
                return '';
        }
    }

    _renderDetectionDetails(data) {
        const details = data.details || {};
        const count = details.count || 0;
        let html = `<div class="detail-line">${count} person(s) detected</div>`;

        if (details.thumbnails_base64) {
            html += '<div class="detail-thumbnails">';
            details.thumbnails_base64.forEach((thumb, i) => {
                html += `<img src="data:image/jpeg;base64,${thumb}" class="detail-thumb" alt="Person ${i}" />`;
            });
            html += '</div>';
        }
        return html;
    }

    _renderFaceDetectDetails(data) {
        const details = data.details || {};
        const count = details.detected || 0;
        const total = details.total || 0;
        if (total === 0) return '';
        return `<div class="detail-line">${count}/${total} faces detected (SCRFD)</div>`;
    }

    _renderFaceAssessDetails(data) {
        const results = data.details?.results || [];
        if (results.length === 0) return '';

        let html = '';
        results.forEach(r => {
            const quality = r.quality != null ? r.quality.toFixed(2) : 'N/A';
            const icon = r.extracted ? (r.quality > 0.7 ? '✅' : '⚠️') : '❌';
            html += `<div class="detail-line">
                Track #${r.track_id}: ${icon} quality=${quality}
            </div>`;
        });
        return html;
    }

    _renderReIDDetails(data) {
        const results = data.details?.results || [];
        if (results.length === 0) return '';

        let html = '';
        results.forEach(r => {
            html += `<div class="detail-line">
                Track #${r.track_id}: ${r.feature_dim || 2048}-d extracted
            </div>`;
        });
        return html;
    }
}

// 全局实例
window.pipelinePanel = new PipelinePanel();
