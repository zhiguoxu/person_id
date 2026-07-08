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
    // 摄像头 ID: 优先取 URL 查询参数 ?camera_id=xxx, 缺省回退 'zhiguo'。
    // 约定 camera_id = 设备 device_sn, 例: index.html?camera_id=EU0125MH00100015056
    cameraId: new URLSearchParams(location.search).get('camera_id') || 'zhiguo',

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
