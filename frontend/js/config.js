/**
 * 后端服务器配置
 * 
 * 前端运行在本地浏览器，通过 WebSocket 连接远程 CUDA 服务器。
 * 修改此处的地址以指向你的后端服务器。
 */
const BACKEND_CONFIG = {
    // 远程 CUDA 服务器地址
    host: '8.145.38.125',
    port: 10003,
    
    // 自动构建 URL
    get baseUrl() {
        return `http://${this.host}:${this.port}`;
    },
    get wsUrl() {
        return `ws://${this.host}:${this.port}/ws/vision`;
    },
    get apiUrl() {
        return `${this.baseUrl}/api`;
    },
};

// 导出为全局变量
window.BACKEND_CONFIG = BACKEND_CONFIG;
