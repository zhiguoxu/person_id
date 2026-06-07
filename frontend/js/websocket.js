/**
 * WebSocket Manager — 管理与后端的 WebSocket 连接
 * 
 * 功能:
 * - 二进制帧发送 (JPEG)
 * - JSON 结果接收
 * - 配置更新发送
 * - 操作命令发送
 * - 自动重连
 * - 背压控制
 */
class WebSocketManager {
    constructor() {
        this.ws = null;
        this.url = '';
        this.connected = false;
        this.pendingFrame = false;
        this.frameInterval = 100; // 初始 10 FPS
        this.minInterval = 33;    // 最高 30 FPS
        this.maxInterval = 200;   // 最低 5 FPS
        this.reconnectDelay = 2000;
        this.reconnectTimer = null;
        this.frameTimer = null;

        // 回调
        this.onResult = null;      // (result: Object) => void
        this.onEvent = null;       // (event: Object) => void
        this.onConnected = null;   // () => void
        this.onDisconnected = null;// () => void

        // 统计
        this.lastResultTime = 0;
        this.fpsHistory = [];
        this.latencyHistory = [];
    }

    /**
     * 建立 WebSocket 连接 (连接到远程 CUDA 后端)
     */
    connect() {
        this.url = window.BACKEND_CONFIG.wsUrl;
        this._createConnection();
    }

    _createConnection() {
        if (this.ws) {
            this.ws.close();
        }

        this.ws = new WebSocket(this.url);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
            console.log('[WS] Connected to', this.url);
            this.connected = true;
            this.pendingFrame = false;
            if (this.reconnectTimer) {
                clearTimeout(this.reconnectTimer);
                this.reconnectTimer = null;
            }
            this._updateStatusUI(true);
            if (this.onConnected) this.onConnected();
        };

        this.ws.onmessage = (event) => {
            if (typeof event.data === 'string') {
                this._handleTextMessage(event.data);
            }
        };

        this.ws.onclose = (event) => {
            console.log('[WS] Disconnected:', event.code, event.reason);
            this.connected = false;
            this.pendingFrame = false;
            this._updateStatusUI(false);
            if (this.onDisconnected) this.onDisconnected();
            this._scheduleReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('[WS] Error:', error);
        };
    }

    _handleTextMessage(data) {
        try {
            const msg = JSON.parse(data);

            if (msg.type === 'frame_result') {
                this.pendingFrame = false; // 允许发送下一帧
                this._updateStats(msg);
                if (this.onResult) this.onResult(msg);
            } else if (msg.type === 'event') {
                // 事件字段直接在 msg 顶层 (event_type, track_id, ...)
                if (this.onEvent) this.onEvent(msg);
            } else if (msg.type === 'config_updated' || msg.type === 'config_ack') {
                console.log('[WS] Config updated:', msg.updated_keys || msg.message);
            } else if (msg.type === 'connected') {
                console.log('[WS] Server acknowledged connection');
            }
        } catch (e) {
            console.error('[WS] Failed to parse message:', e);
        }
    }

    /**
     * 发送视频帧 (JPEG 二进制)
     */
    sendFrame(blob) {
        if (!this.connected || this.pendingFrame) return false;

        this.pendingFrame = true;
        blob.arrayBuffer().then(buffer => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(buffer);
            } else {
                this.pendingFrame = false;
            }
        });
        return true;
    }

    /**
     * 发送配置更新
     */
    sendConfigUpdate(params) {
        this._sendJSON({
            type: 'config_update',
            payload: params
        });
    }

    /**
     * 发送身份确认
     */
    sendConfirmIdentity(trackId, personId, name) {
        this._sendJSON({
            type: 'confirm_identity',
            track_id: trackId,
            person_id: personId || null,
            name: name || ''
        });
    }

    _sendJSON(obj) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(obj));
        }
    }

    /**
     * 自适应帧率调整
     */
    _updateStats(result) {
        const now = performance.now();
        if (this.lastResultTime > 0) {
            const dt = now - this.lastResultTime;
            this.fpsHistory.push(1000 / dt);
            if (this.fpsHistory.length > 30) this.fpsHistory.shift();
        }
        this.lastResultTime = now;

        if (result.processing_time_ms) {
            this.latencyHistory.push(result.processing_time_ms);
            if (this.latencyHistory.length > 30) this.latencyHistory.shift();

            // 自适应帧率
            if (result.processing_time_ms < 50) {
                this.frameInterval = Math.max(this.frameInterval - 5, this.minInterval);
            } else if (result.processing_time_ms > 100) {
                this.frameInterval = Math.min(this.frameInterval + 10, this.maxInterval);
            }
        }

        this._updateCounters();
    }

    get currentFPS() {
        if (this.fpsHistory.length === 0) return 0;
        return this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length;
    }

    get currentLatency() {
        if (this.latencyHistory.length === 0) return 0;
        return this.latencyHistory.reduce((a, b) => a + b, 0) / this.latencyHistory.length;
    }

    _updateCounters() {
        const fpsEl = document.querySelector('#fps-counter .fps-value');
        const latEl = document.querySelector('#latency-counter .latency-value');
        if (fpsEl) fpsEl.textContent = this.currentFPS.toFixed(1);
        if (latEl) latEl.textContent = this.currentLatency.toFixed(0);
    }

    _updateStatusUI(connected) {
        const el = document.getElementById('connection-status');
        if (!el) return;
        el.classList.toggle('connected', connected);
        el.classList.toggle('disconnected', !connected);
        el.querySelector('.status-text').textContent = connected ? 'Connected' : 'Disconnected';
    }

    _scheduleReconnect() {
        if (this.reconnectTimer) return;
        console.log(`[WS] Reconnecting in ${this.reconnectDelay}ms...`);
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this._createConnection();
        }, this.reconnectDelay);
    }

    disconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.connected = false;
    }
}

// 全局实例
window.wsManager = new WebSocketManager();
