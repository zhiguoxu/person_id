/**
 * Events Timeline — 事件时间线面板
 * 
 * 显示最近的系统事件（新人物检测、身份确认、追踪丢失等）
 */
class EventsTimeline {
    constructor() {
        this.container = document.getElementById('events-timeline');
        this.events = [];
        this.maxEvents = 50;
    }

    /**
     * 添加事件
     */
    addEvent(event) {
        this.events.unshift(event);
        if (this.events.length > this.maxEvents) {
            this.events = this.events.slice(0, this.maxEvents);
        }
        this._renderEvent(event);
    }

    _renderEvent(event) {
        // 移除空状态提示
        const empty = this.container.querySelector('.timeline-empty');
        if (empty) empty.remove();

        const card = document.createElement('div');
        card.className = 'event-card';
        card.dataset.type = event.event_type;

        const dot = document.createElement('span');
        dot.className = 'event-dot';

        const time = document.createElement('span');
        time.className = 'event-time';
        time.textContent = this._formatTime(event.timestamp);

        const trackBadge = document.createElement('span');
        trackBadge.className = 'event-track';
        trackBadge.textContent = event.track_id != null ? `#${event.track_id}` : '';

        const msg = document.createElement('span');
        msg.className = 'event-message';
        msg.textContent = event.message || this._defaultMessage(event);

        card.appendChild(dot);
        card.appendChild(time);
        card.appendChild(trackBadge);
        card.appendChild(msg);

        // 插入到最前面
        this.container.insertBefore(card, this.container.firstChild);

        // 限制 DOM 节点数
        while (this.container.children.length > this.maxEvents) {
            this.container.removeChild(this.container.lastChild);
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

    _defaultMessage(event) {
        const type = event.event_type;
        const name = event.display_name || event.person_id || `Track #${event.track_id}`;
        const conf = event.confidence ? ` (${(event.confidence * 100).toFixed(0)}%)` : '';

        switch (type) {
            case 'new_person': return `New person detected: ${name}`;
            case 'identity_confirmed': return `${name} confirmed${conf} via ${event.source}`;
            case 'identity_conflict': return `Conflict: ${name}`;
            case 'vlm_invoked': return `VLM arbitration for ${name}`;
            case 'vlm_result': return `VLM result: ${name}${conf}`;
            case 'track_lost': return `Track lost: ${name}`;
            case 'track_recovered': return `Track recovered: ${name}`;
            case 'outfit_updated': return `Outfit updated: ${name}`;
            case 'human_confirmed': return `Admin confirmed: ${name}`;
            case 'gallery_updated': return `Gallery updated: ${name}`;
            default: return type;
        }
    }

    /**
     * 清空事件
     */
    clear() {
        this.events = [];
        this.container.innerHTML = '<div class="timeline-empty">No events yet. Start the camera to begin detection.</div>';
    }
}

// 全局实例
window.eventsTimeline = new EventsTimeline();
