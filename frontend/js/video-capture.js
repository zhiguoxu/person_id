/**
 * Video Capture — 摄像头采集管理
 * 
 * 功能:
 * - 枚举可用摄像头
 * - 开始/停止采集
 * - 定时抓帧并通过 WebSocket 发送
 */
class VideoCapture {
    constructor() {
        this.videoEl = document.getElementById('webcam');
        this.stream = null;
        this.capturing = false;
        this.captureCanvas = document.createElement('canvas');
        this.captureCtx = this.captureCanvas.getContext('2d');
        this.captureCanvas.width = 640;
        this.captureCanvas.height = 480;
        this.maxCaptureWidth = 640;  // 限制最大发送宽度
        this.frameTimer = null;
    }

    /**
     * 枚举可用摄像头设备
     */
    async enumerateDevices() {
        try {
            // 先请求权限
            const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
            tempStream.getTracks().forEach(t => t.stop());

            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(d => d.kind === 'videoinput');

            const select = document.getElementById('camera-select');
            select.innerHTML = '<option value="">Select Camera...</option>';
            videoDevices.forEach((device, idx) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.textContent = device.label || `Camera ${idx + 1}`;
                select.appendChild(option);
            });

            // 默认选择第一个
            if (videoDevices.length > 0) {
                select.value = videoDevices[0].deviceId;
            }

            return videoDevices;
        } catch (e) {
            console.error('[Camera] Failed to enumerate devices:', e);
            return [];
        }
    }

    /**
     * 开始摄像头采集
     */
    async start(deviceId = null) {
        if (this.capturing) return;

        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30 },
            },
            audio: false,
        };

        if (deviceId) {
            constraints.video.deviceId = { exact: deviceId };
        }

        try {
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoEl.srcObject = this.stream;
            await this.videoEl.play();

            this.capturing = true;
            document.getElementById('no-camera-message').classList.add('hidden');

            // 开始帧发送循环
            this._startFrameLoop();

            console.log('[Camera] Started');
        } catch (e) {
            console.error('[Camera] Failed to start:', e);
            alert('Camera access failed: ' + e.message);
        }
    }

    /**
     * 停止摄像头采集
     */
    stop() {
        this.capturing = false;

        if (this.frameTimer) {
            clearInterval(this.frameTimer);
            this.frameTimer = null;
        }

        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }

        this.videoEl.srcObject = null;
        document.getElementById('no-camera-message').classList.remove('hidden');
        console.log('[Camera] Stopped');
    }

    /**
     * 帧发送循环 (受背压控制)
     */
    _startFrameLoop() {
        if (this.frameTimer) clearInterval(this.frameTimer);

        this.frameTimer = setInterval(() => {
            if (!this.capturing || !window.wsManager.connected) return;

            this._captureAndSend();
        }, window.wsManager.frameInterval);

        // 动态调整发送频率
        setInterval(() => {
            if (this.frameTimer && this.capturing) {
                clearInterval(this.frameTimer);
                this.frameTimer = setInterval(() => {
                    if (!this.capturing || !window.wsManager.connected) return;
                    this._captureAndSend();
                }, window.wsManager.frameInterval);
            }
        }, 2000);
    }

    _captureAndSend() {
        if (this.videoEl.readyState < 2) return; // HAVE_CURRENT_DATA

        // 动态检测摄像头的原始宽高比并适配 (防止图像形变)
        const vw = this.videoEl.videoWidth;
        const vh = this.videoEl.videoHeight;
        
        if (vw > 0 && vh > 0) {
            const targetW = Math.min(this.maxCaptureWidth, vw);
            const targetH = Math.round(targetW * (vh / vw));
            
            // 只有尺寸变化时才重新设置 canvas
            if (this.captureCanvas.width !== targetW || this.captureCanvas.height !== targetH) {
                this.captureCanvas.width = targetW;
                this.captureCanvas.height = targetH;
            }
        }

        // 绘制当前画面
        this.captureCtx.drawImage(this.videoEl, 0, 0, this.captureCanvas.width, this.captureCanvas.height);

        this.captureCanvas.toBlob((blob) => {
            if (blob) {
                window.wsManager.sendFrame(blob);
            }
        }, 'image/jpeg', 0.7);
    }

    /**
     * 获取当前视频尺寸 (用于 Canvas overlay 坐标映射)
     */
    getVideoRect() {
        const video = this.videoEl;
        const container = document.getElementById('video-container');
        if (!container || !video.videoWidth) return null;

        const containerW = container.clientWidth;
        const containerH = container.clientHeight;
        
        // 1. 计算浏览器实际渲染的视频区域 (保持原始比例)
        const origVideoW = video.videoWidth;
        const origVideoH = video.videoHeight;
        const scaleX = containerW / origVideoW;
        const scaleY = containerH / origVideoH;
        const scale = Math.min(scaleX, scaleY);

        const displayW = origVideoW * scale;
        const displayH = origVideoH * scale;
        const offsetX = (containerW - displayW) / 2;
        const offsetY = (containerH - displayH) / 2;

        // 2. 导出 capture 尺寸 (640x480) 供 overlay_renderer 用作坐标基准
        return {
            offsetX, offsetY, 
            displayW, displayH, scale,
            videoW: this.captureCanvas.width,   // 640
            videoH: this.captureCanvas.height,  // 480
        };
    }
}

// 全局实例
window.videoCapture = new VideoCapture();
