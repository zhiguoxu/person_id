/**
 * Events Timeline — 事件时间线面板 (Debug-friendly)
 *
 * 显示每次 Tier2 匹配结果，包括 identity status + fused score。
 * - 连续相同 status 的事件合并为一行，水平追加 score badge
 * - 点击 score badge 弹出匹配候选详情 popover
 */
class EventsTimeline {
    constructor() {
        this.container = document.getElementById('events-timeline');
        this.events = [];
        this.maxEvents = 50;
        this._lastCard = null;        // 上一个 event card DOM
        this._lastEventKey = null;    // 上一个事件的合并 key
        this._activePopover = null;   // 当前打开的 popover
    }

    /**
     * 添加事件
     */
    addEvent(event) {
        this.events.unshift(event);
        if (this.events.length > this.maxEvents) {
            this.events = this.events.slice(0, this.maxEvents);
        }

        // 合并 key: 同一 track + 同一 status (message 字段存 status 值)
        const eventKey = `${event.track_id}:${event.message || event.event_type}`;

        if (this._lastCard && this._lastEventKey === eventKey) {
            // 合并: 追加 score badge 到已有卡片
            this._appendScore(this._lastCard, event);
        } else {
            // 新建卡片
            this._renderEvent(event);
            this._lastEventKey = eventKey;
        }
    }

    /**
     * 向已有卡片追加一个 fused score badge
     */
    _appendScore(card, event) {
        const scoresContainer = card.querySelector('.event-scores');
        if (!scoresContainer) return;

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

        const badge = this._createScoreBadge(event);
        scoresContainer.appendChild(badge);

        card.appendChild(dot);
        card.appendChild(time);
        card.appendChild(trackBadge);
        card.appendChild(statusLabel);
        card.appendChild(nameSpan);
        card.appendChild(scoresContainer);

        // 插入到最前面
        this.container.insertBefore(card, this.container.firstChild);

        // 限制 DOM 节点数
        while (this.container.children.length > this.maxEvents) {
            this.container.removeChild(this.container.lastChild);
        }

        this._lastCard = card;
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

            tr.innerHTML = `
                <td class="pop-name">${c.display_name || c.person_id || '?'}</td>
                <td class="pop-fused">${fused}</td>
                <td>${face}</td>
                <td>${body}</td>
                <td>${prop}</td>
                <td>${fq}</td>
                <td>${bq}</td>
            `;
            tbody.appendChild(tr);
        }
        table.appendChild(tbody);
        popover.appendChild(table);

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
