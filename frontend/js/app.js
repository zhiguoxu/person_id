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

        // 1. 枚举摄像头
        await window.videoCapture.enumerateDevices();

        // 2. 初始化控制面板 (使用默认值，等连接后从服务端加载)
        window.controlsPanel.initialize();

        // 3. 设置 WebSocket 回调
        setupWebSocketCallbacks();

        // 4. 绑定 UI 事件
        bindUIEvents();

        // 5. 连接 WebSocket
        window.wsManager.connect();

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
                        is_current_target: p.is_current_target,
                        thumbnail_b64: p.thumbnail_b64
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
        };
    }

    // =========================================================================
    // UI 事件绑定
    // =========================================================================
    function bindUIEvents() {
        // 摄像头开关
        const cameraBtn = document.getElementById('btn-toggle-camera');
        const cameraSelect = document.getElementById('camera-select');

        if (cameraBtn) {
            cameraBtn.addEventListener('click', async () => {
                if (window.videoCapture.capturing) {
                    window.videoCapture.stop();
                    cameraBtn.innerHTML = '<span class="btn-icon">▶</span> Start Camera';
                    cameraBtn.classList.remove('active');
                } else {
                    const deviceId = cameraSelect?.value || null;
                    await window.videoCapture.start(deviceId);
                    cameraBtn.innerHTML = '<span class="btn-icon">⏹</span> Stop Camera';
                    cameraBtn.classList.add('active');
                }
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

            // 如果 gallery 为空则隐藏候选区
            if (galleryPersons.length === 0) {
                candidatesSection.classList.add('empty');
                return;
            }
            candidatesSection.classList.remove('empty');

            // 添加 "+ New Person" 卡片
            const newCard = document.createElement('div');
            newCard.className = 'candidate-card new-person';
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
