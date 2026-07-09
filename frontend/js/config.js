/**
 * 后端服务器配置
 * 
 * 前端运行在本地浏览器，通过 WebSocket 连接远程 CUDA 服务器。
 * 修改此处的地址以指向你的后端服务器。
 */
const BACKEND_CONFIG = {
    // 远程 CUDA 服务器地址
    host: '1.15.11.133',
    port: 10003,
    // 摄像头 ID (约定 camera_id = 设备 device_sn)。无默认值:
    // 未填写设备号时不连 WebSocket, 也不允许推流/拉流/本地采集。
    // 优先级: URL 参数 ?camera_id=xxx > 页面上次填写的设备号(localStorage)。
    // 页面顶部的「设备 SN」输入框修改后 (失焦或回车) 立即生效并记住。
    cameraId: (() => {
        const fromUrl = new URLSearchParams(location.search).get('camera_id');
        if (fromUrl) {
            try { localStorage.setItem('vision_camera_id', fromUrl); } catch (e) { }
            return fromUrl;
        }
        try {
            return localStorage.getItem('vision_camera_id') || '';
        } catch (e) { }
        return '';
    })(),

    // 自动构建 URL
    get baseUrl() {
        return `http://${this.host}:${this.port}`;
    },
    get wsUrl() {
        return `ws://${this.host}:${this.port}/ws/vision?camera_id=${encodeURIComponent(this.cameraId)}`;
    },
    get apiUrl() {
        return `${this.baseUrl}/api`;
    },
};

// 导出为全局变量
window.BACKEND_CONFIG = BACKEND_CONFIG;
