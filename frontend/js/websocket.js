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

        // 回调
        this.onResult = null;      // (result: Object) => void
        this.onEvent = null;       // (event: Object) => void
        this.onConnected = null;   // () => void

        // 统计
        // - 服务端拉流观看: FPS 由 StreamViewer 上屏回调刷新 (见 refreshFpsFromViewer)
        // - 本地上传: 仍按 frame_result 到达间隔
        this.lastResultTime = 0;
        this.fpsHistory = [];
        this.latencyHistory = [];
    }

    /**
     * 建立 WebSocket 连接 (连接到远程 CUDA 后端)
     */
    connect() {
        if (!window.BACKEND_CONFIG.cameraId) {
            console.warn('[WS] No camera_id, connection skipped');
            return;
        }
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
            } else if (event.data instanceof ArrayBuffer) {
                // 服务端拉流模式: 后端推送的 JPEG 帧 → StreamViewer 渲染
                // (onFrame 内会在未激活时自动 start, 避免画面卡死而角标空转)
                window.streamViewer?.onFrame(event.data);
            }
        };

        this.ws.onclose = (event) => {
            console.log('[WS] Disconnected:', event.code, event.reason);
            this.connected = false;
            this.pendingFrame = false;
            this._updateStatusUI(false);
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
                // 服务端拉流模式: 同步识别坐标基准 (处理帧尺寸)
                if (msg.frame_w && msg.frame_h) {
                    window.streamViewer?.setFrameSize(msg.frame_w, msg.frame_h);
                }
                // 立即触发下一帧发送 (在 UI 更新之前, 避免被 DOM 操作阻塞)
                window.videoCapture?.onResultReceived();
                this._updateStats(msg);
                if (this.onResult) this.onResult(msg);
            } else if (msg.type === 'event') {
                // 事件字段直接在 msg 顶层 (event_type, track_id, ...)
                if (this.onEvent) this.onEvent(msg);
            } else if (msg.type === 'identity_confirmed') {
                alert(`身份已确认: ${msg.name}`);
            } else if (msg.type === 'error') {
                // 服务端错误反馈
                if (msg.code === 'confirm_error') {
                    alert(`确认身份失败: ${msg.message}`);
                } else if (msg.code === 'consumer_active') {
                    // 服务端拉流消费中, 本地上传帧被拒绝 → 停止本地采集
                    console.warn('[WS] Server-side consumer active, stopping local capture');
                    this.pendingFrame = false;
                    if (window.videoCapture?.capturing) {
                        window.videoCapture.stop();
                        window.app?.resetCameraButton?.();
                    }
                } else {
                    console.warn('[WS] Server error:', msg.code, msg.message);
                }
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
     * 发送配置更新 (REST PUT)
     */
    async sendConfigUpdate(params) {
        try {
            const resp = await fetch(`${window.BACKEND_CONFIG.apiUrl}/config`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ updates: params }),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                console.error('[Config] Update failed:', err.detail || resp.statusText);
            }
        } catch (e) {
            console.error('[Config] Update failed:', e.message);
        }
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
     * 更新延迟 / 本地采集自适应帧率; 服务端拉流时 FPS 由 StreamViewer 负责
     */
    _updateStats(result) {
        // 本地上传模式才用收包间隔计 FPS; 拉流观看时收包突发会虚高到上百
        if (!window.streamViewer?.active) {
            const now = performance.now();
            if (this.lastResultTime > 0) {
                const dt = now - this.lastResultTime;
                if (dt >= 8 && dt < 5000) {
                    this.fpsHistory.push(1000 / dt);
                    if (this.fpsHistory.length > 30) this.fpsHistory.shift();
                }
            }
            this.lastResultTime = now;
        }

        if (result.processing_ms) {
            this.latencyHistory.push(result.processing_ms);
            if (this.latencyHistory.length > 30) this.latencyHistory.shift();

            // 自适应帧率 (仅本地上传模式用 frameInterval)
            if (result.processing_ms < 50) {
                this.frameInterval = Math.max(this.frameInterval - 5, this.minInterval);
            } else if (result.processing_ms > 100) {
                this.frameInterval = Math.min(this.frameInterval + 10, this.maxInterval);
            }
        }

        this._updateCounters();
    }

    /** 服务端拉流观看: 按 canvas 实际绘制间隔刷新角标 */
    refreshFpsFromViewer() {
        this._updateCounters();
    }

    get currentFPS() {
        if (window.streamViewer?.active) {
            return window.streamViewer.currentFPS || 0;
        }
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
