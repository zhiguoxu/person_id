/**
 * App.js — 应用入口
 * 
 * 初始化所有模块，连接 WebSocket，绑定事件。
 */
(function () {
    'use strict';

    // =========================================================================
    // 初始化
    // =========================================================================
    async function init() {
        console.log('[App] Initializing Vision ID Dashboard...');

        // 1. 初始化控制面板 (使用默认值，等连接后从服务端加载)
        window.controlsPanel.initialize();

        // 3. 设置 WebSocket 回调
        setupWebSocketCallbacks();

        // 4. 绑定 UI 事件
        bindUIEvents();

        // 5. 连接 WebSocket (未填设备 SN 时不连, 等用户填写后页面会重载)
        if (window.BACKEND_CONFIG.cameraId) {
            window.wsManager.connect();
        } else {
            const msg = document.querySelector('#no-camera-message p');
            if (msg) msg.textContent = '请先在上方输入框填写设备 SN';
            document.getElementById('device-sn-input')?.focus();
            console.log('[App] No camera_id yet, WebSocket not connected');
        }

        console.log('[App] Initialization complete');
    }

    // =========================================================================
    // WebSocket 回调
    // =========================================================================
    function setupWebSocketCallbacks() {
        // 帧处理结果
        window.wsManager.onResult = (result) => {
            // 将嵌套的 TrackedPersonResponse 展平适配前端原有逻辑
            let persons = result.tracked_persons || result.persons || [];
            persons = persons.map(p => {
                if (p.person && p.identity_result) {
                    return {
                        track_id: p.person.track_id,
                        bbox: p.person.detection?.bbox,
                        keypoints: p.person.detection?.keypoints,
                        pose_bucket: p.person.detection?.pose_bucket,
                        attention_score: p.person.attention_score,
                        trail: p.person.trail,
                        person_id: p.identity_result.person_id,
                        display_name: p.identity_result.display_name,
                        identity_status: p.identity_result.status,
                        confidence: p.identity_result.confidence,
                        face_quality: p.identity_result.face_quality,
                        is_current_target: p.is_current_target
                    };
                }
                return p;
            });
            result.tracked_persons = persons;

            // 清除已删除用户的身份标记
            const deletedIds = window.personGallery?._deletedIds;
            if (deletedIds && deletedIds.size > 0) {
                for (const p of persons) {
                    if (p.person_id && deletedIds.has(p.person_id)) {
                        p.person_id = null;
                        p.display_name = null;
                        p.identity_status = 'identifying';
                    }
                }
            }

            // 更新 Canvas 叠加层 (后端用 tracked_persons)
            window.overlayRenderer.update(persons);

            // 更新事件时间线的活跃 track IDs
            const activeTrackIds = persons
                .map(p => p.track_id)
                .filter(id => id != null);
            window.eventsTimeline.updateActiveTracks(activeTrackIds);

            // 更新流水线调试面板
            if (result.pipeline_debug) {
                window.pipelinePanel.update(result.pipeline_debug);
            }

            // 更新人物画廊
            window.personGallery.updateFromResult(result);
        };

        // 异步事件
        window.wsManager.onEvent = (event) => {
            if (event && event.event_type) {
                window.eventsTimeline.addEvent(event);
            }
        };

        // 连接成功
        window.wsManager.onConnected = () => {
            loadServerConfig();
            // WS 重连后后端状态已刷新, 清除前端的删除标记
            window.personGallery._deletedIds.clear();
            // 同步服务端拉流状态 (刷新页面 / 重连后恢复观看模式)
            window.app?.syncConsumeState?.();
        };
    }

    // =========================================================================
    // UI 事件绑定
    // =========================================================================
    function bindUIEvents() {
        // 摄像头开关
        const cameraBtn = document.getElementById('btn-toggle-camera');
        const streamUrlInput = document.getElementById('stream-url-input');

        // 恢复上次的 stream URL
        if (streamUrlInput) {
            const savedUrl = localStorage.getItem('vision_stream_url');
            if (savedUrl) streamUrlInput.value = savedUrl;
        }

        // --- 本地摄像头选择 (Start Camera 的下拉, 多摄像头时才显示) ---
        const cameraMenuBtn = document.getElementById('btn-camera-menu');
        const cameraMenu = document.getElementById('camera-menu');
        let localCameras = [];
        let selectedCameraId = localStorage.getItem('vision_local_camera_id') || '';

        function validCameraId() {
            return localCameras.some(d => d.deviceId === selectedCameraId)
                ? selectedCameraId : null;
        }

        function updateCameraMenu() {
            if (!cameraMenuBtn || !cameraMenu) return;
            if (localCameras.length <= 1) {
                cameraMenuBtn.classList.add('hidden');
                return;
            }
            cameraMenuBtn.classList.remove('hidden');
            const currentId = validCameraId() || localCameras[0].deviceId;
            cameraMenu.innerHTML = '';
            localCameras.forEach((d, idx) => {
                const item = document.createElement('button');
                const checked = d.deviceId === currentId;
                item.className = 'split-menu-item' + (checked ? ' checked' : '');
                item.textContent = `${checked ? '✓' : '\u3000'} ${d.label || 'Camera ' + (idx + 1)}`;
                item.addEventListener('click', async () => {
                    cameraMenu.classList.add('hidden');
                    if (d.deviceId === currentId) return;
                    selectedCameraId = d.deviceId;
                    localStorage.setItem('vision_local_camera_id', selectedCameraId);
                    updateCameraMenu();
                    // 正在本地采集 → 直接热切换摄像头
                    if (window.videoCapture.capturing && window.videoCapture.sourceType !== 'network') {
                        window.videoCapture.stop();
                        await window.videoCapture.start(selectedCameraId);
                    }
                });
                cameraMenu.appendChild(item);
            });
        }

        if (cameraMenuBtn && cameraMenu) {
            cameraMenuBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                cameraMenu.classList.toggle('hidden');
            });
            document.addEventListener('click', () => cameraMenu.classList.add('hidden'));
            cameraMenu.addEventListener('click', (e) => e.stopPropagation());
        }

        function resetCameraButton() {
            if (!cameraBtn) return;
            cameraBtn.innerHTML = '<span class="btn-icon">▶</span> Start Camera';
            cameraBtn.classList.remove('active');
        }

        // --- Camera Start/Stop ---
        if (cameraBtn) {
            cameraBtn.addEventListener('click', async () => {
                if (!requireDeviceSn()) return;
                if (consumeActive && !window.videoCapture.capturing) {
                    alert('服务端拉流进行中，无需本地采集。如需切换请先停止服务端拉流。');
                    return;
                }
                if (window.videoCapture.capturing) {
                    window.videoCapture.stop();
                    cameraBtn.innerHTML = '<span class="btn-icon">▶</span> Start Camera';
                    cameraBtn.classList.remove('active');
                } else {
                    const url = streamUrlInput?.value?.trim();
                    if (url) {
                        // 有 URL → 网络流, 保存到 localStorage
                        localStorage.setItem('vision_stream_url', url);
                        window.videoCapture.setSourceType('network');
                        window.videoCapture.setStreamUrl(url);
                        await window.videoCapture.start();
                    } else {
                        // 无 URL → 本地摄像头 (首次使用时枚举设备并请求权限)
                        localStorage.removeItem('vision_stream_url');
                        window.videoCapture.setSourceType('local');
                        if (!window.videoCapture._devicesEnumerated) {
                            localCameras = await window.videoCapture.enumerateDevices();
                            window.videoCapture._devicesEnumerated = true;
                            updateCameraMenu(); // 多摄像头时显示下拉箭头
                        }
                        await window.videoCapture.start(validCameraId());
                    }
                    cameraBtn.innerHTML = '<span class="btn-icon">⏹</span> Stop Camera';
                    cameraBtn.classList.add('active');
                }
            });
        }

        // --- URL 输入框回车启动 ---
        if (streamUrlInput) {
            streamUrlInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    cameraBtn?.click();
                }
            });
        }

        // --- 设备 SN 输入框: 失焦或回车即生效并记住 (无默认设备号) ---
        const deviceSnInput = document.getElementById('device-sn-input');

        /** 校验设备号已填写, 未填写则拦截操作并聚焦输入框 */
        function requireDeviceSn() {
            if (window.BACKEND_CONFIG.cameraId) return true;
            alert('请先在左侧输入框填写设备 SN');
            deviceSnInput?.focus();
            return false;
        }

        function applyDeviceSn() {
            const sn = deviceSnInput.value.trim();
            if (sn === window.BACKEND_CONFIG.cameraId) return; // 未变化
            // 保存并带着新 camera_id 重载页面 (WS/底库/拉流观看状态全部干净重建);
            // 清空则移除绑定, 页面回到"未选设备"状态
            try {
                if (sn) localStorage.setItem('vision_camera_id', sn);
                else localStorage.removeItem('vision_camera_id');
            } catch (err) { }
            localStorage.removeItem('vision_stream_url'); // 流地址跟设备走, 换设备后作废
            const u = new URL(location.href);
            if (sn) u.searchParams.set('camera_id', sn);
            else u.searchParams.delete('camera_id');
            location.href = u.toString();
        }

        if (deviceSnInput) {
            deviceSnInput.value = window.BACKEND_CONFIG.cameraId;
            deviceSnInput.addEventListener('change', applyDeviceSn); // 失焦且值有变化
            deviceSnInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    applyDeviceSn();
                }
            });
        }

        // --- 轻量 toast 提示 (成功绿色 / 失败红色, 自动消失) ---
        function showToast(message, type = 'success', duration = 3000) {
            let box = document.getElementById('toast-box');
            if (!box) {
                box = document.createElement('div');
                box.id = 'toast-box';
                document.body.appendChild(box);
            }
            const item = document.createElement('div');
            item.className = `toast toast-${type}`;
            item.textContent = message;
            box.appendChild(item);
            setTimeout(() => {
                item.classList.add('toast-out');
                setTimeout(() => item.remove(), 300);
            }, duration);
        }

        // --- 按钮 1: 开启设备推流 (ISS start_stream → FLV 地址填入输入框) ---
        const deviceStreamBtn = document.getElementById('btn-device-stream');
        if (deviceStreamBtn) {
            const deviceStreamBtnHtml = deviceStreamBtn.innerHTML;
            deviceStreamBtn.addEventListener('click', async () => {
                if (!requireDeviceSn()) return;

                // 进入加载态: 按钮与 URL 输入框都锁住, 输入框显示加载提示
                deviceStreamBtn.disabled = true;
                deviceStreamBtn.innerHTML = '⏳ 获取中...';
                const prevUrl = streamUrlInput?.value || '';
                if (streamUrlInput) {
                    streamUrlInput.value = '';
                    streamUrlInput.placeholder = '正在开启设备推流, 获取直播地址...';
                    streamUrlInput.readOnly = true;
                    streamUrlInput.classList.add('loading');
                }

                let ok = false;
                try {
                    const camId = encodeURIComponent(window.BACKEND_CONFIG.cameraId);
                    const resp = await fetch(`${window.BACKEND_CONFIG.apiUrl}/${camId}/device_stream/start`, {
                        method: 'POST',
                    });
                    if (!resp.ok) {
                        const err = await resp.json().catch(() => ({}));
                        showToast(`❌ 开启设备推流失败: ${err.detail || resp.statusText}`, 'error', 6000);
                        return;
                    }
                    const data = await resp.json();
                    if (data.flv_url) {
                        ok = true;
                        streamUrlInput.value = data.flv_url;
                        localStorage.setItem('vision_stream_url', data.flv_url);
                        showToast('✅ 设备推流已开启, 地址已填入。可点击「服务端拉流」开始识别');
                    } else {
                        showToast('❌ 开启设备推流失败: 未返回直播地址', 'error', 6000);
                    }
                } catch (e) {
                    showToast(`❌ 开启设备推流失败: ${e.message}`, 'error', 6000);
                } finally {
                    deviceStreamBtn.disabled = false;
                    deviceStreamBtn.innerHTML = deviceStreamBtnHtml;
                    if (streamUrlInput) {
                        streamUrlInput.readOnly = false;
                        streamUrlInput.classList.remove('loading');
                        streamUrlInput.placeholder = 'Stream URL (留空用本地摄像头)';
                        if (!ok) streamUrlInput.value = prevUrl; // 失败时恢复原值
                    }
                }
            });
        }

        // --- 设备推流的下拉菜单: 停止推流 ---
        const streamMenuBtn = document.getElementById('btn-stream-menu');
        const streamMenu = document.getElementById('stream-menu');
        const deviceStreamStopBtn = document.getElementById('btn-device-stream-stop');

        if (streamMenuBtn && streamMenu) {
            streamMenuBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                streamMenu.classList.toggle('hidden');
            });
            // 点击页面其他位置关闭菜单
            document.addEventListener('click', () => streamMenu.classList.add('hidden'));
            streamMenu.addEventListener('click', (e) => e.stopPropagation());
        }

        if (deviceStreamStopBtn) {
            deviceStreamStopBtn.addEventListener('click', async () => {
                streamMenu?.classList.add('hidden');
                if (!requireDeviceSn()) return;
                deviceStreamStopBtn.disabled = true;
                try {
                    const camId = encodeURIComponent(window.BACKEND_CONFIG.cameraId);
                    // 服务端还在消费该流的话先停消费, 避免消费器对着死流反复重连
                    if (consumeActive) {
                        await fetch(`${window.BACKEND_CONFIG.apiUrl}/${camId}/consume/stop`, { method: 'POST' })
                            .catch(() => { });
                        setConsumeUI(false);
                    }
                    const resp = await fetch(`${window.BACKEND_CONFIG.apiUrl}/${camId}/device_stream/stop`, {
                        method: 'POST',
                    });
                    if (!resp.ok) {
                        const err = await resp.json().catch(() => ({}));
                        showToast(`❌ 停止推流失败: ${err.detail || resp.statusText}`, 'error', 6000);
                        return;
                    }
                    // 推流已停, 旧地址作废
                    if (streamUrlInput) streamUrlInput.value = '';
                    localStorage.removeItem('vision_stream_url');
                    showToast('✅ 设备推流已停止');
                } catch (e) {
                    showToast(`❌ 停止推流失败: ${e.message}`, 'error', 6000);
                } finally {
                    deviceStreamStopBtn.disabled = false;
                }
            });
        }

        // --- 按钮 2: 服务端拉流消费开关 ---
        const consumeBtn = document.getElementById('btn-toggle-consume');
        let consumeActive = false;

        // --- 拉流状态徽章 (视频区左上角): 连接中/拉流中/失败详情 ---
        let consumeStatusTimer = null;

        function renderConsumeStatus(st) {
            const el = document.getElementById('consume-status');
            if (!el) return;
            if (!st || !st.running) {
                el.classList.add('hidden');
                return;
            }
            el.classList.remove('hidden', 'ok', 'warn', 'err');
            el.title = st.last_error || ''; // 截断时悬停可看完整错误
            if (st.connected) {
                el.classList.add('ok');
                const viewers = st.viewers > 1 ? ` · ${st.viewers} 人观看` : '';
                el.textContent = `🟢 拉流中 ${st.stream_width}×${st.stream_height} @ ${(st.process_fps || 0).toFixed(1)}fps${viewers}`;
            } else if (st.last_error) {
                el.classList.add('err');
                el.textContent = `🔴 ${st.last_error}`;
            } else {
                el.classList.add('warn');
                el.textContent = '🟡 正在连接视频流...';
            }
        }

        async function pollConsumeStatus() {
            try {
                const camId = encodeURIComponent(window.BACKEND_CONFIG.cameraId);
                const resp = await fetch(`${window.BACKEND_CONFIG.apiUrl}/${camId}/consume/status`);
                if (!resp.ok) return;
                const st = await resp.json();
                renderConsumeStatus(st);
                // 服务端已停止 (服务重启/他人停止) → 同步本页 UI
                if (!st.running && consumeActive) {
                    setConsumeUI(false);
                    showToast('⚠️ 服务端拉流已停止', 'error', 5000);
                }
            } catch (e) { /* 网络抖动忽略, 下个周期重试 */ }
        }

        function setConsumeUI(active) {
            consumeActive = active;
            if (consumeBtn) {
                consumeBtn.innerHTML = active
                    ? '<span class="btn-icon">⏹</span> 停止拉流'
                    : '<span class="btn-icon">▶</span> 服务端拉流';
                consumeBtn.classList.toggle('active', active);
            }
            if (active) {
                window.streamViewer.start();
                pollConsumeStatus(); // 立即刷一次 (显示"连接中")
                if (!consumeStatusTimer) {
                    consumeStatusTimer = setInterval(pollConsumeStatus, 3000);
                }
            } else {
                window.streamViewer.stop();
                if (consumeStatusTimer) {
                    clearInterval(consumeStatusTimer);
                    consumeStatusTimer = null;
                }
                renderConsumeStatus(null);
            }
        }

        if (consumeBtn) {
            consumeBtn.addEventListener('click', async () => {
                if (!requireDeviceSn()) return;
                const camId = encodeURIComponent(window.BACKEND_CONFIG.cameraId);
                consumeBtn.disabled = true;
                try {
                    if (!consumeActive) {
                        const url = streamUrlInput?.value?.trim();
                        if (!url) {
                            alert('请先填写视频流地址（可点击「📡 设备推流」自动获取）');
                            return;
                        }
                        // 与本地采集互斥: 先停掉浏览器端采集
                        if (window.videoCapture.capturing) {
                            window.videoCapture.stop();
                            resetCameraButton();
                        }
                        localStorage.setItem('vision_stream_url', url);
                        const resp = await fetch(`${window.BACKEND_CONFIG.apiUrl}/${camId}/consume/start`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ url }),
                        });
                        if (!resp.ok) {
                            const err = await resp.json().catch(() => ({}));
                            alert(`开启服务端拉流失败: ${err.detail || resp.statusText}`);
                            return;
                        }
                        setConsumeUI(true);
                    } else {
                        const resp = await fetch(`${window.BACKEND_CONFIG.apiUrl}/${camId}/consume/stop`, {
                            method: 'POST',
                        });
                        if (!resp.ok) {
                            const err = await resp.json().catch(() => ({}));
                            console.warn('[App] Stop consume failed:', err.detail || resp.statusText);
                        }
                        setConsumeUI(false);
                    }
                } catch (e) {
                    alert(`操作失败: ${e.message}`);
                } finally {
                    consumeBtn.disabled = false;
                }
            });
        }

        // 查询服务端拉流状态并同步 UI (页面加载 / WS 重连后调用, 恢复观看模式)
        async function syncConsumeState() {
            if (!window.BACKEND_CONFIG.cameraId) return;
            try {
                const camId = encodeURIComponent(window.BACKEND_CONFIG.cameraId);
                const resp = await fetch(`${window.BACKEND_CONFIG.apiUrl}/${camId}/consume/status`);
                if (!resp.ok) return;
                const st = await resp.json();
                if (st.running && !consumeActive) {
                    if (st.url && streamUrlInput) streamUrlInput.value = st.url;
                    setConsumeUI(true);
                } else if (!st.running && consumeActive) {
                    setConsumeUI(false);
                }
            } catch (e) {
                // 服务端不可达时忽略, 等下次重连再同步
            }
        }

        // 暴露给 websocket.js / onConnected 回调使用
        window.app = { resetCameraButton, syncConsumeState };

        // --- 镜头畸变矫正开关 ---
        const correctionToggle = document.getElementById('toggle-correction');
        if (correctionToggle) {
            correctionToggle.addEventListener('change', () => {
                window.wsManager.sendConfigUpdate({
                    IMAGE_CORRECTION_ENABLED: correctionToggle.checked,
                });
            });
        }

        // 清空事件
        const clearBtn = document.getElementById('btn-clear-events');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                window.eventsTimeline.clear();
            });
        }

        // 窗口大小变化时同步 canvas
        window.addEventListener('resize', () => {
            window.overlayRenderer.update(window.overlayRenderer.persons);
        });

        // 绑定 Canvas 人物点击事件，打开确认身份弹窗
        const confirmModal = document.getElementById('confirm-modal');
        const confirmForm = document.getElementById('confirm-identity-form');
        const inputTrackId = document.getElementById('confirm-track-id');
        const inputPersonId = document.getElementById('confirm-person-id');
        const inputName = document.getElementById('confirm-name');
        const candidatesSection = document.getElementById('confirm-candidates-section');
        const candidatesList = document.getElementById('confirm-candidates-list');

        /**
         * 从服务端加载 gallery 人物列表，填充候选卡片
         */
        async function populateCandidates() {
            candidatesList.innerHTML = '';

            // 拉取服务端 gallery 数据
            let galleryPersons = [];
            try {
                const cameraId = window.BACKEND_CONFIG.cameraId;
                const resp = await fetch(`${window.BACKEND_CONFIG.apiUrl}/${cameraId}/gallery/persons`);
                if (resp.ok) {
                    const data = await resp.json();
                    galleryPersons = data.persons || data || [];
                }
            } catch (e) {
                console.log('[App] Could not load gallery persons for candidates');
            }

            candidatesSection.classList.remove('empty');

            // 添加 "+ New Person" 卡片 (默认选中)
            const newCard = document.createElement('div');
            newCard.className = 'candidate-card new-person selected';
            newCard.dataset.personId = '';
            newCard.dataset.displayName = '';
            newCard.innerHTML = `
                <div class="candidate-avatar">＋</div>
                <div class="candidate-info">
                    <span class="candidate-display-name">New Person</span>
                    <span class="candidate-person-id">Create new gallery entry</span>
                </div>
            `;
            candidatesList.appendChild(newCard);

            // 默认选中 New Person: 清空 personId, 让用户输入新名字
            inputPersonId.value = '';
            inputName.value = '';
            inputName.focus();

            // 添加 gallery 人物卡片
            for (const person of galleryPersons) {
                const initial = (person.display_name || '?')[0].toUpperCase();
                const card = document.createElement('div');
                card.className = 'candidate-card';
                card.dataset.personId = person.person_id;
                card.dataset.displayName = person.display_name;
                card.innerHTML = `
                    <div class="candidate-avatar">${initial}</div>
                    <div class="candidate-info">
                        <span class="candidate-display-name">${person.display_name}</span>
                        <span class="candidate-person-id">${person.person_id}</span>
                    </div>
                `;
                candidatesList.appendChild(card);
            }

            // 点击候选卡片 → 选中并填充表单 (事件委托, 只绑一次)
            if (!candidatesList._bindDone) {
                candidatesList.addEventListener('click', (e) => {
                    const card = e.target.closest('.candidate-card');
                    if (!card) return;

                    candidatesList.querySelectorAll('.candidate-card').forEach(c => c.classList.remove('selected'));
                    card.classList.add('selected');

                    inputPersonId.value = card.dataset.personId || '';
                    inputName.value = card.dataset.displayName || '';
                    if (!card.dataset.personId) {
                        inputName.value = '';
                        inputName.focus();
                    }
                });
                candidatesList._bindDone = true;
            }
        }

        window.overlayRenderer.onPersonClicked = (person) => {
            inputTrackId.value = person.track_id || '';
            inputPersonId.value = person.person_id || '';
            inputName.value = person.display_name || '';
            confirmModal.classList.remove('hidden');

            // 异步加载候选列表
            populateCandidates();
            inputName.focus();
        };

        // 关闭确认弹窗
        document.getElementById('confirm-modal-close')?.addEventListener('click', () => {
            confirmModal.classList.add('hidden');
        });
        document.getElementById('confirm-modal-cancel')?.addEventListener('click', () => {
            confirmModal.classList.add('hidden');
        });

        // 提交确认身份
        confirmForm?.addEventListener('submit', (e) => {
            e.preventDefault();
            const trackId = parseInt(inputTrackId.value, 10);
            const personId = inputPersonId.value.trim();
            const name = inputName.value.trim();
            
            if (trackId && name) {
                window.wsManager.sendConfirmIdentity(trackId, personId, name);
                confirmModal.classList.add('hidden');
            }
        });

        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                // 关闭模态框
                document.getElementById('person-modal')?.classList.add('hidden');
                confirmModal?.classList.add('hidden');
            }
            if (e.key === ' ' && e.target.tagName !== 'INPUT') {
                // 空格切换摄像头
                e.preventDefault();
                cameraBtn?.click();
            }
        });
    }

    // =========================================================================
    // 从服务端加载配置
    // =========================================================================
    async function loadServerConfig() {
        try {
            const response = await fetch(`${window.BACKEND_CONFIG.apiUrl}/config`);
            if (response.ok) {
                const data = await response.json();
                if (data.params) {
                    window.controlsPanel.initialize(data.params);
                }
                // flags: 非滑块控制的配置项
                if (data.flags) {
                    window.QUALITY_THRESHOLDS = {
                        face: data.flags.AGG_MIN_FACE_QUALITY ?? 0.3,
                        body: data.flags.AGG_MIN_BODY_QUALITY ?? 0.2,
                    };
                    // 同步畸变矫正开关状态
                    const correctionToggle = document.getElementById('toggle-correction');
                    if (correctionToggle && data.flags.IMAGE_CORRECTION_ENABLED != null) {
                        correctionToggle.checked = !!data.flags.IMAGE_CORRECTION_ENABLED;
                    }
                }
                console.log('[App] Config loaded from server:', window.BACKEND_CONFIG.baseUrl);
            }
        } catch (e) {
            console.log('[App] Could not load config from server, using defaults');
        }
    }

    // =========================================================================
    // 启动
    // =========================================================================
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
