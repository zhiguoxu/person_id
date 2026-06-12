/**
 * Events Timeline — 事件时间线面板 (Debug-friendly)
 *
 * 显示每次 Tier2 匹配结果，包括 identity status + fused score。
 * - 连续相同 status 的事件合并为一行，水平追加 score badge
 * - 点击 score badge 弹出匹配候选详情 popover
 * - 支持按 track 过滤: All / Active Tracks
 */
class EventsTimeline {
    constructor() {
        this.container = document.getElementById('events-timeline');
        this.events = [];
        this.maxEvents = 50;
        this._trackCards = new Map();   // trackId → DOM card
        this._activePopover = null;     // 当前打开的 popover
        this._filterMode = 'all';       // 'all' | 'active'
        this._activeTrackIds = new Set();

        this._bindFilterTabs();
    }

    // =========================================================================
    // Filter Tabs
    // =========================================================================
    _bindFilterTabs() {
        document.querySelectorAll('.events-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.events-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                this._filterMode = tab.dataset.filter;
                this._fullRender();
            });
        });
    }

    /**
     * 更新当前活跃的 track IDs (由 app.js 每帧调用)
     */
    updateActiveTracks(trackIds) {
        const newSet = new Set(trackIds);
        if (this._sameTrackSet(this._activeTrackIds, newSet)) return;
        this._activeTrackIds = newSet;
        if (this._filterMode === 'active') {
            this._fullRender();
        }
    }

    _sameTrackSet(a, b) {
        if (a.size !== b.size) return false;
        for (const id of a) {
            if (!b.has(id)) return false;
        }
        return true;
    }

    /**
     * 添加事件
     */
    addEvent(event) {
        this.events.unshift(event);
        if (this.events.length > this.maxEvents) {
            this.events = this.events.slice(0, this.maxEvents);
        }

        if (this._filterMode === 'active' && !this._isEventVisible(event)) {
            return;
        }

        const trackKey = `${event.track_id}`;
        const existingCard = this._trackCards.get(trackKey);

        if (existingCard && existingCard.parentNode === this.container) {
            // 合并到已有卡片，并置顶
            this._appendScore(existingCard, event);
            this.container.insertBefore(existingCard, this.container.firstChild);
        } else {
            // 新建卡片
            this._renderEvent(event);
        }
    }

    /**
     * 检查事件是否在当前过滤器下可见
     */
    _isEventVisible(event) {
        if (this._filterMode === 'all') return true;
        return event.track_id != null && this._activeTrackIds.has(event.track_id);
    }

    /**
     * 完整重渲染 (切换过滤器或活跃 tracks 变化时调用)
     */
    _fullRender() {
        this.container.innerHTML = '';
        this._trackCards.clear();

        const filtered = this.events.filter(e => this._isEventVisible(e));

        if (filtered.length === 0) {
            const emptyMsg = this._filterMode === 'active'
                ? 'No events for active tracks.'
                : 'No events yet. Start the camera to begin detection.';
            this.container.innerHTML = `<div class="timeline-empty">${emptyMsg}</div>`;
            return;
        }

        // 倒序遍历 (events[0] 是最新的，需要最后渲染才能在顶部)
        for (let i = filtered.length - 1; i >= 0; i--) {
            const event = filtered[i];
            const trackKey = `${event.track_id}`;
            const existingCard = this._trackCards.get(trackKey);

            if (existingCard) {
                this._appendScore(existingCard, event);
            } else {
                this._renderEvent(event);
            }
        }
    }

    /**
     * 向已有卡片追加 score (如果状态变了，先追加状态标签)
     */
    _appendScore(card, event) {
        const scoresContainer = card.querySelector('.event-scores');
        if (!scoresContainer) return;

        // data_stale → 追加暂停标记，不追加分数
        if (event.event_type === 'data_stale') {
            // 避免连续追加多个 stale 标记
            const lastChild = scoresContainer.lastElementChild;
            if (lastChild && lastChild.classList.contains('event-stale-marker')) return;
            const marker = document.createElement('span');
            marker.className = 'event-stale-marker';
            marker.textContent = '⏸';
            marker.title = 'No new data';
            scoresContainer.appendChild(marker);
            scoresContainer.scrollLeft = scoresContainer.scrollWidth;
            return;
        }

        const currentStatus = event.message || event.event_type;
        const prevStatus = card.dataset.lastStatus;

        // 状态变化 → 追加新状态标签
        if (currentStatus !== prevStatus) {
            const statusTag = document.createElement('span');
            statusTag.className = `event-status-inline event-status--${this._statusClass(currentStatus)}`;
            statusTag.textContent = this._statusShort(currentStatus);
            scoresContainer.appendChild(statusTag);
            card.dataset.lastStatus = currentStatus;

            // 更新卡片头部状态标签
            const headerStatus = card.querySelector('.event-status');
            if (headerStatus) {
                headerStatus.className = `event-status event-status--${this._statusClass(currentStatus)}`;
                headerStatus.textContent = this._statusShort(currentStatus);
            }
            // 更新名称
            const nameSpan = card.querySelector('.event-name');
            const name = event.display_name || event.person_id || '';
            if (nameSpan && name) nameSpan.textContent = name;
        }

        const badge = this._createScoreBadge(event);
        scoresContainer.appendChild(badge);

        // 滚动到最右
        scoresContainer.scrollLeft = scoresContainer.scrollWidth;
    }

    _renderEvent(event) {
        // 移除空状态提示
        const empty = this.container.querySelector('.timeline-empty');
        if (empty) empty.remove();

        const card = document.createElement('div');
        card.className = 'event-card';
        card.dataset.type = event.event_type;
        card.dataset.trackId = event.track_id != null ? event.track_id : '';
        card.dataset.lastStatus = event.message || event.event_type;

        // 状态圆点
        const dot = document.createElement('span');
        dot.className = 'event-dot';

        // 时间
        const time = document.createElement('span');
        time.className = 'event-time';
        time.textContent = this._formatTime(event.timestamp);

        // Track badge
        const trackBadge = document.createElement('span');
        trackBadge.className = 'event-track';
        trackBadge.textContent = event.track_id != null ? `#${event.track_id}` : '';

        // Status 标签
        const statusLabel = document.createElement('span');
        statusLabel.className = `event-status event-status--${this._statusClass(event.message)}`;
        statusLabel.textContent = this._statusShort(event.message || event.event_type);

        // 显示名称
        const nameSpan = document.createElement('span');
        nameSpan.className = 'event-name';
        const name = event.display_name || event.person_id || '';
        nameSpan.textContent = name ? name : '';

        // Scores 容器 (水平滚动，可追加多个 score badge)
        const scoresContainer = document.createElement('span');
        scoresContainer.className = 'event-scores';

        if (event.event_type === 'data_stale') {
            const marker = document.createElement('span');
            marker.className = 'event-stale-marker';
            marker.textContent = '⏸';
            marker.title = 'No new data';
            scoresContainer.appendChild(marker);
        } else {
            const badge = this._createScoreBadge(event);
            scoresContainer.appendChild(badge);
        }

        card.appendChild(dot);
        card.appendChild(time);
        card.appendChild(trackBadge);
        card.appendChild(statusLabel);
        card.appendChild(nameSpan);
        card.appendChild(scoresContainer);

        // 清理缓存按钮 (所有有 track_id 的卡片)
        if (event.track_id != null) {
            const clearBtn = document.createElement('button');
            clearBtn.className = 'btn-clear-cache';
            clearBtn.textContent = '🗑';
            clearBtn.title = 'Clear quality cache';
            clearBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                clearBtn.disabled = true;
                clearBtn.textContent = '⏳';
                try {
                    const cameraId = window.BACKEND_CONFIG?.cameraId || 'default';
                    const resp = await fetch(
                        `${window.BACKEND_CONFIG.apiUrl}/${cameraId}/track/${event.track_id}/quality_cache`,
                        { method: 'DELETE' }
                    );
                    if (resp.ok) {
                        // 清空这一行的所有 score 数据
                        const scores = card.querySelector('.event-scores');
                        if (scores) scores.innerHTML = '';
                    }
                } catch { /* ignore */ }
                setTimeout(() => {
                    clearBtn.textContent = '🗑';
                    clearBtn.disabled = false;
                }, 800);
            });
            card.appendChild(clearBtn);
        }

        // 插入到最前面
        this.container.insertBefore(card, this.container.firstChild);

        // 注册到 track → card 映射
        const trackKey = `${event.track_id}`;
        this._trackCards.set(trackKey, card);

        // 限制 DOM 节点数
        while (this.container.children.length > this.maxEvents) {
            const removed = this.container.lastChild;
            // 同步清理 map
            if (removed.dataset?.trackId) {
                const key = removed.dataset.trackId;
                if (this._trackCards.get(key) === removed) {
                    this._trackCards.delete(key);
                }
            }
            this.container.removeChild(removed);
        }
    }

    /**
     * 创建可点击的 fused score badge
     */
    _createScoreBadge(event) {
        const badge = document.createElement('span');
        badge.className = `event-score-badge event-score-badge--${this._statusClass(event.message)}`;
        const score = event.fused_score != null ? event.fused_score.toFixed(2) : '—';
        badge.textContent = score;
        badge.title = 'Click to view match details';

        // 点击显示候选详情 popover
        if (event.candidates && event.candidates.length > 0) {
            badge.classList.add('clickable');
            badge.addEventListener('click', (e) => {
                e.stopPropagation();
                this._showCandidatesPopover(badge, event);
            });
        }
        return badge;
    }

    /**
     * 显示匹配候选详情 popover
     */
    _showCandidatesPopover(anchor, event) {
        // 关闭已打开的
        this._closePopover();

        const popover = document.createElement('div');
        popover.className = 'event-popover';

        // Header (含 margin 信息)
        const header = document.createElement('div');
        header.className = 'event-popover-header';
        let headerText = `Match Details — Track #${event.track_id}`;
        if (event.candidates && event.candidates.length >= 2) {
            const top1 = event.candidates[0].fused_score || 0;
            const top2 = event.candidates[1].fused_score || 0;
            headerText += ` | margin: ${(top1 - top2).toFixed(3)}`;
        }
        header.innerHTML = `
            <span>${headerText}</span>
            <button class="event-popover-close">&times;</button>
        `;
        popover.appendChild(header);

        // Candidate table
        const table = document.createElement('table');
        table.className = 'event-popover-table';
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Fused</th>
                    <th>Face</th>
                    <th>Body</th>
                    <th>Prop</th>
                    <th>FQ</th>
                    <th>BQ</th>
                    <th>FW</th>
                    <th>BW</th>
                </tr>
            </thead>
        `;

        const tbody = document.createElement('tbody');
        for (const c of event.candidates) {
            const tr = document.createElement('tr');
            const fused = c.fused_score != null ? c.fused_score.toFixed(3) : '—';
            const face = c.face_score != null ? c.face_score.toFixed(3) : '—';
            const body = c.body_score != null ? c.body_score.toFixed(3) : '—';
            const prop = c.proportion_score != null ? c.proportion_score.toFixed(3) : '—';
            const fq = c.face_match_quality != null ? c.face_match_quality.toFixed(2) : '—';
            const bq = c.body_match_quality != null ? c.body_match_quality.toFixed(2) : '—';
            const fw = c.face_weight != null ? c.face_weight.toFixed(2) : '—';
            const bw = c.body_weight != null ? c.body_weight.toFixed(2) : '—';

            tr.innerHTML = `
                <td class="pop-name">${c.display_name || c.person_id || '?'}</td>
                <td class="pop-fused">${fused}</td>
                <td>${face}</td>
                <td>${body}</td>
                <td>${prop}</td>
                <td>${fq}</td>
                <td>${bq}</td>
                <td>${fw}</td>
                <td>${bw}</td>
            `;
            tbody.appendChild(tr);
        }
        table.appendChild(tbody);
        popover.appendChild(table);

        // Quality Cache section (placeholder, async load)
        const cacheSection = document.createElement('div');
        cacheSection.className = 'pop-cache-section';
        cacheSection.innerHTML = '<div class="pop-cache-loading">Loading quality cache…</div>';
        popover.appendChild(cacheSection);

        // 异步加载 quality cache
        this._loadQualityCache(event.track_id, cacheSection);

        // 定位
        document.body.appendChild(popover);
        const anchorRect = anchor.getBoundingClientRect();
        popover.style.left = `${anchorRect.left}px`;
        popover.style.top = `${anchorRect.bottom + 4}px`;

        // 确保不超出视口
        requestAnimationFrame(() => {
            const popRect = popover.getBoundingClientRect();
            if (popRect.right > window.innerWidth - 8) {
                popover.style.left = `${window.innerWidth - popRect.width - 8}px`;
            }
            if (popRect.bottom > window.innerHeight - 8) {
                popover.style.top = `${anchorRect.top - popRect.height - 4}px`;
            }
        });

        // 关闭事件
        popover.querySelector('.event-popover-close').addEventListener('click', () => this._closePopover());
        setTimeout(() => {
            document.addEventListener('click', this._onDocClick = () => this._closePopover(), { once: true });
        }, 0);

        this._activePopover = popover;
    }

    /**
     * 异步加载并渲染 quality cache 图片
     */
    async _loadQualityCache(trackId, container) {
        const cameraId = window.BACKEND_CONFIG?.cameraId;
        if (!cameraId || trackId == null) {
            container.innerHTML = '<div class="pop-cache-empty">—</div>';
            return;
        }

        try {
            const resp = await fetch(
                `${window.BACKEND_CONFIG.apiUrl}/${cameraId}/track/${trackId}/quality_cache`
            );
            if (!resp.ok) {
                container.innerHTML = '<div class="pop-cache-empty">Cache not available</div>';
                return;
            }
            const data = await resp.json();
            container.innerHTML = '';

            const facePool = data.face_pool || [];
            const bodyPool = data.body_pool || [];

            if (facePool.length === 0 && bodyPool.length === 0) {
                container.innerHTML = '<div class="pop-cache-empty">Quality cache empty</div>';
                return;
            }

            if (facePool.length > 0) {
                container.appendChild(this._renderCachePool('👤 Face Pool', facePool, 'face'));
            }
            if (bodyPool.length > 0) {
                container.appendChild(this._renderCachePool('🏃 Body Pool', bodyPool, 'body'));
            }

            // 重新定位 popover (内容变化后可能溢出)
            if (this._activePopover) {
                requestAnimationFrame(() => {
                    const popRect = this._activePopover.getBoundingClientRect();
                    if (popRect.bottom > window.innerHeight - 8) {
                        this._activePopover.style.top = `${window.innerHeight - popRect.height - 8}px`;
                    }
                    if (popRect.right > window.innerWidth - 8) {
                        this._activePopover.style.left = `${window.innerWidth - popRect.width - 8}px`;
                    }
                });
            }
        } catch (e) {
            container.innerHTML = `<div class="pop-cache-empty">Failed: ${e.message}</div>`;
        }
    }

    /**
     * 渲染单个 cache pool (face 或 body)
     */
    _renderCachePool(title, pool, poolType = 'face') {
        const section = document.createElement('div');
        section.className = 'pop-cache-pool';

        const header = document.createElement('div');
        header.className = 'pop-cache-pool-header';
        header.textContent = `${title} (${pool.length})`;
        section.appendChild(header);

        const grid = document.createElement('div');
        grid.className = 'pop-cache-grid';

        // 从服务端配置获取阈值
        const thresholds = window.QUALITY_THRESHOLDS || { face: 0.3, body: 0.2 };
        const minQ = poolType === 'body' ? thresholds.body : thresholds.face;

        for (const item of pool) {
            const card = document.createElement('div');
            card.className = `pop-cache-card${item.enrolled ? ' enrolled' : ''}`;

            const img = document.createElement('img');
            img.className = 'pop-cache-img';
            img.src = `data:image/jpeg;base64,${item.image_b64}`;
            img.alt = item.pose_bucket;
            card.appendChild(img);

            const info = document.createElement('div');
            info.className = 'pop-cache-info';

            const qClass = item.quality < minQ ? 'low' : 'high';
            info.innerHTML = `
                <span class="pop-cache-quality pop-cache-q-${qClass}">Q: ${item.quality.toFixed(2)}</span>
                <span class="pop-cache-pose">${item.pose_bucket}</span>
                <span class="pop-cache-time">${this._formatTime(item.timestamp)}</span>
                ${item.enrolled ? '<span class="pop-cache-enrolled">✓</span>' : ''}
            `;
            card.appendChild(info);
            grid.appendChild(card);
        }

        section.appendChild(grid);
        return section;
    }

    _closePopover() {
        if (this._activePopover) {
            this._activePopover.remove();
            this._activePopover = null;
        }
    }

    _formatTime(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp * 1000);
        return date.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
        });
    }

    _statusShort(status) {
        const map = {
            // IdentityStatus values (message 字段)
            'definite': 'DEF',
            'confident': 'CONF',
            'suspected': 'SUSP',
            'conflict': 'CNFL',
            'stranger': 'STR',
            // EventType values (event_type 字段 fallback)
            'new_person': 'NEW',
            'identity_definite': 'DEF',
            'identity_confident': 'CONF',
            'identity_suspected': 'SUSP',
            'identity_conflict': 'CNFL',
            'human_confirmed': 'HUMAN',
            'vlm_result': 'VLM',
            'vlm_invoked': 'VLM',
            'track_lost': 'LOST',
            'track_recovered': 'RECV',
            'data_stale': 'STALE',
        };
        return map[status] || status?.toUpperCase()?.slice(0, 4) || '?';
    }

    _statusClass(status) {
        const map = {
            // IdentityStatus values
            'definite': 'definite',
            'confident': 'confident',
            'suspected': 'suspected',
            'conflict': 'conflict',
            'stranger': 'stranger',
            // EventType values
            'new_person': 'stranger',
            'identity_definite': 'definite',
            'identity_confident': 'confident',
            'identity_suspected': 'suspected',
            'identity_conflict': 'conflict',
            'human_confirmed': 'definite',
            'vlm_result': 'confident',
            'vlm_invoked': 'confident',
            'track_lost': 'stranger',
            'track_recovered': 'suspected',
            'data_stale': 'stale',
        };
        return map[status] || 'default';
    }

    /**
     * 清空事件
     */
    clear() {
        this.events = [];
        this._lastCard = null;
        this._lastEventKey = null;
        this.container.innerHTML = '<div class="timeline-empty">No events yet. Start the camera to begin detection.</div>';
    }
}

// 全局实例
window.eventsTimeline = new EventsTimeline();
