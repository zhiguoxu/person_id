/**
 * Stream Viewer — 服务端拉流观看模式
 *
 * 服务端 StreamConsumer 后台拉流并识别后, 通过 WebSocket 推送:
 * - 二进制 JPEG 帧 → 本模块解码并画到 canvas (取代 <video> 元素)
 * - frame_result JSON → 仍走原有 overlay 渲染逻辑画框
 *
 * 提供与 VideoCapture.getVideoRect() 相同结构的坐标基准,
 * overlay-renderer 据此把检测框映射到显示坐标。
 */
class StreamViewer {
    constructor() {
        this.canvas = document.getElementById('server-stream-canvas');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.active = false;

        // 预览帧 (canvas 位图) 尺寸
        this.frameW = 0;
        this.frameH = 0;

        // 识别坐标基准 = 服务端处理帧尺寸 (随 frame_result 下发)。
        // 预览图可能为省带宽被缩小, 画框必须以处理帧尺寸为基准。
        this.coordW = 0;
        this.coordH = 0;

        // 解码背压: 解码中收到新帧则只保留最新的
        this._decoding = false;
        this._pendingBuf = null;

        // 绘制帧率 (角标用真实上屏速率, 不用 WS 收包间隔 — 收包突发会虚高到上百)
        this._lastDrawTime = 0;
        this._fpsHistory = [];
        this.onDraw = null; // () => void, 每成功画一帧回调
    }

    /**
     * 进入观看模式: 显示 canvas, 隐藏 <video>
     */
    start() {
        if (!this.canvas) return;
        if (this.active) return;
        this.active = true;
        this.canvas.classList.remove('hidden');
        document.getElementById('webcam')?.classList.add('hidden');
        document.getElementById('no-camera-message')?.classList.add('hidden');
        console.log('[StreamViewer] Started');
    }

    /**
     * 退出观看模式: 隐藏 canvas, 恢复 <video>
     */
    stop() {
        this.active = false;
        this._pendingBuf = null;
        this._fpsHistory = [];
        this._lastDrawTime = 0;
        if (this.canvas) {
            this.canvas.classList.add('hidden');
            if (this.canvas.width) {
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            }
        }
        document.getElementById('webcam')?.classList.remove('hidden');
        if (!window.videoCapture?.capturing) {
            document.getElementById('no-camera-message')?.classList.remove('hidden');
        }
        // 清掉残留的 overlay 框
        window.overlayRenderer?.update([]);
        console.log('[StreamViewer] Stopped');
    }

    /**
     * 记录服务端处理帧尺寸 (由 websocket.js 从 frame_result 中提取)
     */
    setFrameSize(w, h) {
        if (w > 0 && h > 0) {
            this.coordW = w;
            this.coordH = h;
        }
    }

    get currentFPS() {
        if (this._fpsHistory.length === 0) return 0;
        return this._fpsHistory.reduce((a, b) => a + b, 0) / this._fpsHistory.length;
    }

    /**
     * 收到服务端推送的二进制 JPEG 帧
     * @param {ArrayBuffer} buf
     */
    onFrame(buf) {
        if (!this.canvas) return;
        // 只要服务端在推预览帧就进入观看模式 — 避免「别处开了拉流 / 重启恢复后」
        // 本页还没点按钮, active=false 把 JPEG 全丢了, 只剩 JSON 把角标刷到上百 FPS、画面卡死。
        if (!this.active) {
            this.start();
            window.app?.onServerStreamFrames?.();
        }
        if (this._decoding) {
            this._pendingBuf = buf;
            return;
        }
        this._decodeAndDraw(buf);
    }

    async _decodeAndDraw(buf) {
        this._decoding = true;
        try {
            const blob = new Blob([buf], { type: 'image/jpeg' });
            const bitmap = await createImageBitmap(blob);

            this.frameW = bitmap.width;
            this.frameH = bitmap.height;
            if (this.canvas.width !== bitmap.width || this.canvas.height !== bitmap.height) {
                this.canvas.width = bitmap.width;
                this.canvas.height = bitmap.height;
            }
            this.ctx.drawImage(bitmap, 0, 0);
            bitmap.close();

            const now = performance.now();
            if (this._lastDrawTime > 0) {
                const dt = now - this._lastDrawTime;
                // 过滤异常突发 (<8ms ≈ >125fps, 多半是积压连画), 避免角标虚高
                if (dt >= 8 && dt < 5000) {
                    this._fpsHistory.push(1000 / dt);
                    if (this._fpsHistory.length > 30) this._fpsHistory.shift();
                }
            }
            this._lastDrawTime = now;
            if (this.onDraw) this.onDraw();
        } catch (e) {
            console.error('[StreamViewer] Frame decode failed:', e);
        } finally {
            this._decoding = false;
            if (this._pendingBuf) {
                const next = this._pendingBuf;
                this._pendingBuf = null;
                // 必须 await, 否则 finally 里同步间隙会与新的 onFrame 并发解码
                await this._decodeAndDraw(next);
            }
        }
    }

    /**
     * 坐标基准 (与 VideoCapture.getVideoRect 同构, 供 overlay-renderer 使用)
     */
    getVideoRect() {
        const container = document.getElementById('video-container');
        if (!container || !this.frameW || !this.frameH) return null;

        const containerW = container.clientWidth;
        const containerH = container.clientHeight;

        // canvas 使用 object-fit: contain, 计算实际显示区域
        const scale = Math.min(containerW / this.frameW, containerH / this.frameH);
        const displayW = this.frameW * scale;
        const displayH = this.frameH * scale;

        return {
            offsetX: (containerW - displayW) / 2,
            offsetY: (containerH - displayH) / 2,
            displayW, displayH, scale,
            // 坐标基准: 优先用服务端处理帧尺寸 (预览图可能缩小过)
            videoW: this.coordW || this.frameW,
            videoH: this.coordH || this.frameH,
        };
    }
}

// 全局实例
window.streamViewer = new StreamViewer();
