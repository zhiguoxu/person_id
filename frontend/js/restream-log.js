/**
 * RestreamLog — 自动重推流日志查看
 *
 * 拉流连续失败触发自动重推流后, 服务端把每次恢复尝试(触发原因、设备在线
 * 检查、ISS 调用结果、每一步错误日志)记在 data/restream_log/ 下。
 * 本模块从 GET /api/{camera_id}/device_stream/restream_log 拉取并渲染,
 * 入口是「📡 设备推流」下拉菜单里的「🧾 重推日志」。
 */
(function () {
    'use strict';

    const OUTCOME_BADGES = {
        restreamed: { text: '✅ 重推成功', cls: 'ok' },
        device_offline: { text: '📴 设备不在线', cls: 'warn' },
        iss_start_failed: { text: '❌ ISS 开启推流失败', cls: 'err' },
        error: { text: '❌ 恢复流程异常', cls: 'err' },
    };

    function fmtTime(epochSec) {
        if (!epochSec) return '-';
        const d = new Date(epochSec * 1000);
        const pad = (n) => String(n).padStart(2, '0');
        return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} `
            + `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    }

    function escapeHtml(s) {
        return String(s ?? '')
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function renderAttempt(a, index) {
        const badge = OUTCOME_BADGES[a.outcome] || { text: a.outcome, cls: 'err' };
        const onlineText = a.device_online === true ? '在线'
            : a.device_online === false ? '不在线' : '检查失败';
        const logsHtml = (a.logs || []).map((l) => `
            <div class="restream-log-line restream-log-${escapeHtml(l.level)}">
                <span class="restream-log-line-time">${fmtTime(l.time).slice(11)}</span>
                <span>${escapeHtml(l.message)}</span>
            </div>`).join('');

        return `
        <details class="restream-attempt" ${index === 0 ? 'open' : ''}>
            <summary>
                <span class="restream-attempt-time">${fmtTime(a.started_at)}</span>
                <span class="restream-badge ${badge.cls}">${badge.text}</span>
                <span class="restream-attempt-meta">连续失败 ${a.trigger_fail_count} 次 · 设备${onlineText} · ${escapeHtml(a.env)} 环境</span>
            </summary>
            <div class="restream-attempt-detail">
                <div class="restream-kv"><span>触发错误</span><code>${escapeHtml(a.trigger_error || '无')}</code></div>
                <div class="restream-kv"><span>旧地址</span><code>${escapeHtml(a.old_url || '-')}</code></div>
                ${a.new_url ? `<div class="restream-kv"><span>新地址</span><code>${escapeHtml(a.new_url)}</code></div>` : ''}
                <div class="restream-log-lines">${logsHtml || '<div class="restream-log-line">无过程日志</div>'}</div>
            </div>
        </details>`;
    }

    async function load() {
        const listEl = document.getElementById('restream-log-list');
        const summaryEl = document.getElementById('restream-log-summary');
        if (!listEl) return;
        const camId = window.BACKEND_CONFIG.cameraId;
        if (!camId) {
            listEl.innerHTML = '<div class="restream-log-empty">请先填写设备 SN</div>';
            return;
        }
        listEl.innerHTML = '<div class="restream-log-empty">加载中...</div>';
        try {
            const resp = await fetch(
                `${window.BACKEND_CONFIG.apiUrl}/${encodeURIComponent(camId)}/device_stream/restream_log?limit=100`);
            if (!resp.ok) {
                listEl.innerHTML = `<div class="restream-log-empty">加载失败: HTTP ${resp.status}</div>`;
                return;
            }
            const data = await resp.json();
            const attempts = data.attempts || [];
            if (summaryEl) {
                summaryEl.textContent = `设备 ${camId} · 最近 ${attempts.length} 次自动重推流记录 (新的在前)`;
            }
            if (!attempts.length) {
                listEl.innerHTML = '<div class="restream-log-empty">暂无记录: 该设备还没有触发过自动重推流</div>';
                return;
            }
            listEl.innerHTML = attempts.map(renderAttempt).join('');
        } catch (e) {
            listEl.innerHTML = `<div class="restream-log-empty">加载失败: ${escapeHtml(e.message)}</div>`;
        }
    }

    function init() {
        const modal = document.getElementById('restream-log-modal');
        document.getElementById('btn-restream-log')?.addEventListener('click', () => {
            document.getElementById('stream-menu')?.classList.add('hidden');
            modal?.classList.remove('hidden');
            load();
        });
        document.getElementById('restream-log-modal-close')?.addEventListener('click', () => {
            modal?.classList.add('hidden');
        });
        modal?.querySelector('.modal-backdrop')?.addEventListener('click', () => {
            modal.classList.add('hidden');
        });
        document.getElementById('btn-restream-log-refresh')?.addEventListener('click', load);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    window.restreamLog = { load };
})();
