/**
 * Pipeline Panel — 算法流程可视化面板
 * 
 * 显示算法各阶段的状态、耗时、缩略图、匹配分数条形图。
 */
class PipelinePanel {
    constructor() {
        this.stages = ['detection', 'pose', 'face', 'reid', 'matching', 'identity'];
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
            if (data.time_ms !== undefined) {
                timeEl.textContent = `${data.time_ms.toFixed(1)}ms`;
                totalMs += data.time_ms;
            } else {
                timeEl.textContent = '—';
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
            case 'pose':
                return this._renderPoseDetails(data);
            case 'face':
                return this._renderFaceDetails(data);
            case 'reid':
                return this._renderReIDDetails(data);
            case 'matching':
                return this._renderMatchingDetails(data);
            case 'identity':
                return this._renderIdentityDetails(data);
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

    _renderPoseDetails(data) {
        const results = data.details?.results || [];
        if (results.length === 0) return '';

        let html = '';
        results.forEach(r => {
            const bucketEmoji = { frontal: '👤', left: '◀', right: '▶', back: '🔙', unknown: '❓' };
            html += `<div class="detail-line">
                Track #${r.track_id}: ${bucketEmoji[r.bucket] || '?'} ${r.bucket}
            </div>`;
        });
        return html;
    }

    _renderFaceDetails(data) {
        const results = data.details?.results || [];
        if (results.length === 0) return '';

        let html = '';
        results.forEach(r => {
            const quality = r.quality !== null ? r.quality.toFixed(2) : 'N/A';
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

    _renderMatchingDetails(data) {
        const results = data.details?.results || [];
        if (results.length === 0) return '';

        let html = '';
        results.forEach(r => {
            html += `<div class="match-person-header">Track #${r.track_id}</div>`;

            if (r.candidates && r.candidates.length > 0) {
                html += this._renderScoreBars(r.candidates, r.thresholds_used || {});
            }

            const decisionColor = {
                confident: '#00ff88',
                suspected: '#ffa500',
                conflict: '#ff6b6b',
                stranger: '#6b7280'
            };
            const color = decisionColor[r.decision] || '#6b7280';
            html += `<div class="detail-line" style="color: ${color}; font-weight: 600;">
                → ${r.decision?.toUpperCase() || 'UNKNOWN'}
                ${r.matched_id ? `= ${r.matched_id}` : ''}
            </div>`;
        });
        return html;
    }

    _renderScoreBars(candidates, thresholds) {
        const xThresh = thresholds.X || 0.72;
        const yThresh = thresholds.Y || 0.55;

        let html = '<div class="match-candidates">';
        candidates.forEach(c => {
            const score = c.fused_score || 0;
            const pct = (score * 100).toFixed(0);
            const barColor = score >= xThresh ? '#00ff88' :
                             score >= yThresh ? '#ffa500' : '#6b7280';

            html += `
                <div class="match-candidate">
                    <span class="candidate-name" title="${c.person_id}">${c.person_id || '?'}</span>
                    <div class="candidate-bar-container">
                        <div class="candidate-bar" style="width: ${pct}%; background: ${barColor};"></div>
                        <div class="threshold-line x-line" style="left: ${xThresh * 100}%;"></div>
                        <div class="threshold-line y-line" style="left: ${yThresh * 100}%;"></div>
                    </div>
                    <span class="candidate-score" style="color: ${barColor}">${score.toFixed(2)}</span>
                </div>`;
        });
        html += '</div>';
        return html;
    }

    _renderIdentityDetails(data) {
        const details = data.details || {};
        let html = '';
        if (details.confirmed !== undefined) {
            html += `<div class="detail-line">✅ Confirmed: ${details.confirmed}</div>`;
        }
        if (details.identifying !== undefined) {
            html += `<div class="detail-line">⏳ Identifying: ${details.identifying}</div>`;
        }
        if (details.vlm_pending !== undefined && details.vlm_pending > 0) {
            html += `<div class="detail-line">🧠 VLM Pending: ${details.vlm_pending}</div>`;
        }
        return html;
    }
}

// 全局实例
window.pipelinePanel = new PipelinePanel();
