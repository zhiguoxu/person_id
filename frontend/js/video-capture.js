/**
 * Video Capture — 摄像头采集管理
 * 
 * 功能:
 * - 枚举可用本地摄像头
 * - 支持网络摄像头 (FLV / HLS / MJPEG)
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
        this._adjustTimer = null;  // 动态帧率调整定时器

        // --- 摄像头源类型: 'local' | 'network' ---
        this.sourceType = 'local';
        this.streamUrl = '';

        // --- 网络流播放器实例 ---
        this._flvPlayer = null;
        this._hlsPlayer = null;
    }

    /**
     * 设置摄像头源类型
     * @param {'local'|'network'} type
     */
    setSourceType(type) {
        this.sourceType = type;
    }

    /**
     * 设置网络流地址
     * @param {string} url
     */
    setStreamUrl(url) {
        this.streamUrl = url;
    }

    /**
     * 枚举可用本地摄像头设备
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
     * 开始采集 (根据 sourceType 决定本地或网络)
     */
    async start(deviceId = null) {
        if (this.capturing) return;

        if (this.sourceType === 'network') {
            await this._startNetworkStream();
        } else {
            await this._startLocalCamera(deviceId);
        }
    }

    /**
     * 开始本地摄像头采集
     */
    async _startLocalCamera(deviceId = null) {
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

            console.log('[Camera] Local camera started');
        } catch (e) {
            console.error('[Camera] Failed to start local camera:', e);
            alert('Camera access failed: ' + e.message);
        }
    }

    /**
     * 开始网络摄像头流播放
     */
    async _startNetworkStream() {
        const url = this.streamUrl.trim();
        if (!url) {
            alert('请输入网络摄像头的直播地址');
            return;
        }

        // 检测流类型
        const streamType = this._detectStreamType(url);
        console.log(`[Camera] Starting network stream: type=${streamType}, url=${url}`);

        try {
            if (streamType === 'flv') {
                await this._startFlvStream(url);
            } else if (streamType === 'hls') {
                await this._startHlsStream(url);
            } else {
                // 尝试直接作为视频源播放 (适用于 MJPEG 或其他浏览器原生支持的格式)
                await this._startDirectStream(url);
            }

            this.capturing = true;
            document.getElementById('no-camera-message').classList.add('hidden');

            // 开始帧发送循环
            this._startFrameLoop();

            console.log('[Camera] Network stream started');
        } catch (e) {
            console.error('[Camera] Failed to start network stream:', e);
            alert('网络流连接失败: ' + e.message);
        }
    }

    /**
     * 检测网络流类型
     */
    _detectStreamType(url) {
        const lowerUrl = url.toLowerCase().split('?')[0];
        if (lowerUrl.endsWith('.flv') || url.includes('.flv?') || url.includes('.live.flv')) {
            return 'flv';
        }
        if (lowerUrl.endsWith('.m3u8')) {
            return 'hls';
        }
        return 'direct';
    }

    /**
     * FLV 流播放 (使用 flv.js)
     */
    async _startFlvStream(url) {
        if (typeof flvjs === 'undefined') {
            throw new Error('flv.js library not loaded. Cannot play FLV streams.');
        }
        if (!flvjs.isSupported()) {
            throw new Error('Your browser does not support FLV playback.');
        }

        this._flvPlayer = flvjs.createPlayer({
            type: 'flv',
            url: url,
            isLive: true,
            hasAudio: false,
            hasVideo: true,
        }, {
            enableWorker: false,
            enableStashBuffer: false,
            stashInitialSize: 128,
            lazyLoad: false,
            autoCleanupSourceBuffer: true,
            autoCleanupMaxBackwardDuration: 5,
            autoCleanupMinBackwardDuration: 3,
        });

        this._flvPlayer.attachMediaElement(this.videoEl);
        this._flvPlayer.load();

        // 监听错误
        this._flvPlayer.on(flvjs.Events.ERROR, (errType, errDetail) => {
            console.error('[FLV] Error:', errType, errDetail);
        });

        await this.videoEl.play();
    }

    /**
     * HLS 流播放 (原生支持或 hls.js)
     */
    async _startHlsStream(url) {
        // Safari 原生支持 HLS
        if (this.videoEl.canPlayType('application/vnd.apple.mpegurl')) {
            this.videoEl.src = url;
            await this.videoEl.play();
            return;
        }

        // 非 Safari: 尝试 hls.js (需额外加载)
        if (typeof Hls !== 'undefined' && Hls.isSupported()) {
            this._hlsPlayer = new Hls();
            this._hlsPlayer.loadSource(url);
            this._hlsPlayer.attachMedia(this.videoEl);
            this._hlsPlayer.on(Hls.Events.MANIFEST_PARSED, () => {
                this.videoEl.play();
            });
            return;
        }

        // Fallback: 直接设置 src
        this.videoEl.src = url;
        await this.videoEl.play();
    }

    /**
     * 直接视频源播放 (MJPEG 等浏览器原生支持的格式)
     */
    async _startDirectStream(url) {
        this.videoEl.src = url;
        await this.videoEl.play();
    }

    /**
     * 停止采集
     */
    stop() {
        this.capturing = false;

        if (this.frameTimer) {
            clearInterval(this.frameTimer);
            this.frameTimer = null;
        }
        if (this._adjustTimer) {
            clearInterval(this._adjustTimer);
            this._adjustTimer = null;
        }

        // 停止本地摄像头流
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }

        // 销毁 FLV 播放器
        if (this._flvPlayer) {
            try {
                this._flvPlayer.pause();
                this._flvPlayer.unload();
                this._flvPlayer.detachMediaElement();
                this._flvPlayer.destroy();
            } catch (e) {
                console.warn('[Camera] FLV player cleanup error:', e);
            }
            this._flvPlayer = null;
        }

        // 销毁 HLS 播放器
        if (this._hlsPlayer) {
            try {
                this._hlsPlayer.destroy();
            } catch (e) {
                console.warn('[Camera] HLS player cleanup error:', e);
            }
            this._hlsPlayer = null;
        }

        this.videoEl.srcObject = null;
        this.videoEl.removeAttribute('src');
        this.videoEl.load(); // 重置 video 元素
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

        // 动态调整发送频率 (每 2s 检查一次 frameInterval 是否变化)
        if (this._adjustTimer) clearInterval(this._adjustTimer);
        this._adjustTimer = setInterval(() => {
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

        // 动态检测视频的原始宽高比并适配 (防止图像形变)
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

        // 2. 导出 capture 尺寸 供 overlay_renderer 用作坐标基准
        return {
            offsetX, offsetY, 
            displayW, displayH, scale,
            videoW: this.captureCanvas.width,
            videoH: this.captureCanvas.height,
        };
    }
}

// 全局实例
window.videoCapture = new VideoCapture();
