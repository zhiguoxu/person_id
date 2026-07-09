/**
 * Person Gallery — 人物画廊面板
 *
 * 两种显示模式:
 *   - active: 仅显示当前帧匹配到的人 (实时)
 *   - all:    显示数据库中所有已知用户
 */
const _HIGH_CONFIDENCE = new Set(['confident', 'definite']);

class PersonGallery {
    constructor() {
        this.container = document.getElementById('person-gallery');
        this.persons = new Map(); // person_id -> person data (session active)
        this.allPersons = new Map(); // person_id -> person data (from DB)
        this.activeTracks = new Map(); // person_id -> track_id (当前在场)
        this._deletedIds = new Set(); // 已删除 person_id (防止 WS 帧数据重新添加)
        this.mode = 'active'; // 'active' | 'all'

        this._bindTabs();
    }

    // =========================================================================
    // Tab 切换
    // =========================================================================
    _bindTabs() {
        document.querySelectorAll('.gallery-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.gallery-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                this.mode = tab.dataset.mode;

                if (this.mode === 'all') {
                    this._fetchAllPersons();
                } else {
                    this._render();
                }
            });
        });
    }

    /**
     * 从底库补齐实时在场人物的头像 (质量最高的人脸特征图), 5s 节流
     */
    async _refreshAvatars() {
        if (this._avatarRefreshing) return;
        this._avatarRefreshing = true;
        setTimeout(() => { this._avatarRefreshing = false; }, 5000);
        try {
            const cameraId = window.BACKEND_CONFIG?.cameraId;
            if (!cameraId) return;
            const resp = await fetch(`${window.BACKEND_CONFIG.apiUrl}/${cameraId}/gallery/persons`);
            if (!resp.ok) return;
            const data = await resp.json();
            let updated = false;
            for (const p of (data.persons || [])) {
                if (!p.avatar_b64) continue;
                const entry = this.persons.get(p.person_id);
                if (entry && !entry.thumbnail) {
                    entry.thumbnail = p.avatar_b64;
                    updated = true;
                }
                const allEntry = this.allPersons.get(p.person_id);
                if (allEntry && !allEntry.thumbnail) {
                    allEntry.thumbnail = p.avatar_b64;
                    updated = true;
                }
            }
            if (updated) this._render();
        } catch (e) { /* 静默, 下次创建条目时再试 */ }
    }

    /**
     * 从 API 拉取全部 gallery 用户
     */
    async _fetchAllPersons() {
        try {
            const cameraId = window.BACKEND_CONFIG?.cameraId;
            if (!cameraId) return;

            const resp = await fetch(`${window.BACKEND_CONFIG.apiUrl}/${cameraId}/gallery/persons`);
            if (!resp.ok) return;

            const data = await resp.json();
            const persons = data.persons || data || [];
            this.allPersons.clear();
            for (const p of persons) {

                // 合并 session 中的实时数据
                const sessionData = this.persons.get(p.person_id);
                this.allPersons.set(p.person_id, {
                    person_id: p.person_id,
                    display_name: p.display_name || p.person_id,
                    confidence: sessionData?.confidence ?? 0,
                    thumbnail: p.avatar_b64 || null,
                    first_seen: sessionData?.first_seen ?? Date.now(),
                    last_seen: sessionData?.last_seen ?? Date.now(),
                    present: sessionData?.present ?? false,
                });
            }
            this._render();
        } catch (e) {
            console.error('[Gallery] Failed to fetch all persons:', e);
        }
    }

    // =========================================================================
    // 帧结果更新
    // =========================================================================
    updateFromResult(result) {
        const activePersonIds = new Set();
        const persons = result.tracked_persons || result.persons || [];

        if (persons.length > 0) {
            persons.forEach(p => {
                if (p.person_id && !this._deletedIds.has(p.person_id)
                    && _HIGH_CONFIDENCE.has(p.identity_status)) {
                    activePersonIds.add(p.person_id);
                    this.activeTracks.set(p.person_id, p.track_id);

                    // 更新或添加人物
                    if (!this.persons.has(p.person_id)) {
                        this.persons.set(p.person_id, {
                            person_id: p.person_id,
                            display_name: p.display_name || p.person_id,
                            confidence: p.confidence,
                            thumbnail: null,
                            first_seen: Date.now(),
                            last_seen: Date.now(),
                        });
                        this._refreshAvatars(); // 从底库补头像 (节流)
                    } else {
                        const person = this.persons.get(p.person_id);
                        person.display_name = p.display_name || person.display_name;
                        person.confidence = p.confidence;
                        person.last_seen = Date.now();
                    }
                }
            });
        }

        // 更新在场/离开状态
        this.persons.forEach((person, id) => {
            person.present = activePersonIds.has(id);
        });

        // 在场但还没头像的, 持续尝试从底库补齐 (内部 5s 节流)
        for (const [id, person] of this.persons) {
            if (person.present && !person.thumbnail) {
                this._refreshAvatars();
                break;
            }
        }

        // 同步到 allPersons (更新已有 + 添加新增)
        if (this.mode === 'all') {
            this.persons.forEach((sp, id) => {
                if (this.allPersons.has(id)) {
                    const ap = this.allPersons.get(id);
                    ap.present = sp.present;
                    ap.confidence = sp.confidence;
                    ap.display_name = sp.display_name;
                    ap.last_seen = sp.last_seen;
                } else {
                    // 新确认的用户，同步添加到 allPersons
                    this.allPersons.set(id, { ...sp });
                }
            });
        }

        this._render();
    }

    // =========================================================================
    // 渲染
    // =========================================================================
    _render() {
        if (!this.container) return;

        // 根据模式选择数据源
        const source = this.mode === 'all' ? this.allPersons : this.persons;

        // 在 active 模式下，只显示 present 的人
        let displayList;
        if (this.mode === 'active') {
            displayList = new Map([...source].filter(([, p]) => p.present));
        } else {
            displayList = source;
        }

        // 更新计数
        const countEl = document.getElementById('gallery-count');
        if (countEl) countEl.textContent = `${displayList.size} persons`;

        if (displayList.size === 0) {
            const emptyMsg = this.mode === 'active'
                ? 'No active matches.'
                : 'Gallery is empty.';
            this.container.innerHTML = `<div class="gallery-empty">${emptyMsg}</div>`;
            return;
        }

        // 移除空状态提示
        const empty = this.container.querySelector('.gallery-empty');
        if (empty) empty.remove();

        // 同步卡片
        const existingCards = new Map();
        this.container.querySelectorAll('.person-card').forEach(card => {
            existingCards.set(card.dataset.personId, card);
        });

        displayList.forEach((person, id) => {
            let card = existingCards.get(id);

            if (!card) {
                card = this._createCard(person);
                this.container.appendChild(card);
            } else {
                this._updateCard(card, person);
                existingCards.delete(id);
            }
        });

        // 移除不存在的卡片
        existingCards.forEach(card => card.remove());
    }

    _createCard(person) {
        const card = document.createElement('div');
        card.className = 'person-card';
        card.dataset.personId = person.person_id;

        card.innerHTML = `
            <div class="person-avatar">${this._getAvatar(person)}</div>
            <span class="person-name" title="${person.person_id}">${person.display_name}</span>
            <span class="person-status-label ${person.present ? 'present' : 'absent'}">
                ${person.present ? '● Present' : '○ Away'}
            </span>
        `;

        card.addEventListener('click', () => this._showDetail(person));
        card.classList.toggle('active', person.present);

        return card;
    }

    _updateCard(card, person) {
        const nameEl = card.querySelector('.person-name');
        const statusEl = card.querySelector('.person-status-label');

        // 卡片创建时头像可能还没到 (thumbnail=null 显示首字母), 到货后补渲染
        const avatarEl = card.querySelector('.person-avatar');
        if (avatarEl && person.thumbnail && !avatarEl.querySelector('img')) {
            avatarEl.innerHTML = this._getAvatar(person);
        }

        if (nameEl) nameEl.textContent = person.display_name;
        if (statusEl) {
            statusEl.className = `person-status-label ${person.present ? 'present' : 'absent'}`;
            statusEl.textContent = person.present ? '● Present' : '○ Away';
        }
        card.classList.toggle('active', person.present);
    }

    _getAvatar(person) {
        if (person.thumbnail) {
            return `<img src="data:image/jpeg;base64,${person.thumbnail}" alt="${person.display_name}" />`;
        }
        const initial = (person.display_name || '?')[0].toUpperCase();
        return initial;
    }

    // =========================================================================
    // Gallery Detail Modal
    // =========================================================================

    async _showDetail(person) {
        const modal = document.getElementById('person-modal');
        const title = document.getElementById('modal-person-name');
        const body = document.getElementById('modal-body');

        title.textContent = person.display_name || person.person_id;
        body.innerHTML = '<div class="gd-loading">Loading gallery data…</div>';
        modal.classList.remove('hidden');

        // Close handlers
        modal.querySelector('.modal-close').onclick = () => modal.classList.add('hidden');
        modal.querySelector('.modal-backdrop').onclick = () => modal.classList.add('hidden');

        // Fetch detail from API
        const cameraId = window.BACKEND_CONFIG?.cameraId;
        if (!cameraId) {
            body.innerHTML = '<div class="gd-empty">Camera not connected</div>';
            return;
        }

        let detail;
        try {
            const resp = await fetch(
                `${window.BACKEND_CONFIG.apiUrl}/${cameraId}/gallery/person/${person.person_id}`
            );
            if (!resp.ok) throw new Error(resp.statusText);
            detail = await resp.json();
        } catch (e) {
            body.innerHTML = `<div class="gd-empty">Failed to load: ${e.message}</div>`;
            return;
        }

        // Render full detail
        body.innerHTML = '';
        body.appendChild(this._renderDetailContent(detail, person));
    }

    _renderDetailContent(detail, person) {
        const frag = document.createDocumentFragment();

        // --- Info Grid ---
        frag.appendChild(this._renderInfoGrid(detail));

        // --- Face Features ---
        const faceCount = this._countFeatures(detail.face_features);
        frag.appendChild(this._renderFeatureSection(
            '👤', 'Face Features', faceCount, detail.face_features, 'face'
        ));

        // --- Body Features ---
        const bodyCount = this._countFeatures(detail.body_features);
        frag.appendChild(this._renderFeatureSection(
            '🏃', 'Body Features', bodyCount, detail.body_features, 'body'
        ));

        // --- Wardrobe ---
        frag.appendChild(this._renderWardrobeSection(detail.wardrobe || []));

        // --- Body Proportions ---
        frag.appendChild(this._renderProportionsSection(detail.body_proportions));

        // --- VLM Description ---
        frag.appendChild(this._renderVLMSection(detail.vlm_description));

        // --- Actions ---
        frag.appendChild(this._renderActions(person, detail));

        return frag;
    }

    // --- Info Grid ---
    _renderInfoGrid(detail) {
        const div = document.createElement('div');
        div.className = 'gd-info-grid';
        div.innerHTML = `
            <div class="gd-info-item">
                <label>Person ID</label>
                <span>${detail.person_id}</span>
            </div>
            <div class="gd-info-item">
                <label>Display Name</label>
                <span>${detail.display_name}</span>
            </div>
            <div class="gd-info-item">
                <label>Created</label>
                <span>${this._fmtTime(detail.created_at)}</span>
            </div>
            <div class="gd-info-item">
                <label>Last Updated</label>
                <span>${this._fmtTime(detail.last_updated)}</span>
            </div>
            <div class="gd-info-item">
                <label>Update Count</label>
                <span>${detail.update_count}</span>
            </div>
            <div class="gd-info-item">
                <label>Face / Body / Wardrobe</label>
                <span>${this._countFeatures(detail.face_features)} / ${this._countFeatures(detail.body_features)} / ${(detail.wardrobe || []).length}</span>
            </div>
        `;
        return div;
    }

    // --- Feature Section (face or body) ---
    _renderFeatureSection(icon, title, count, features, featureType) {
        const section = document.createElement('div');
        section.className = 'gd-section';

        const header = document.createElement('div');
        header.className = 'gd-section-header';
        header.innerHTML = `
            <span class="gd-section-icon">${icon}</span>
            <span class="gd-section-title">${title}</span>
            <span class="gd-section-badge">${count}</span>
            <span class="gd-section-toggle">▼</span>
        `;
        header.addEventListener('click', () => section.classList.toggle('collapsed'));
        section.appendChild(header);

        const body = document.createElement('div');
        body.className = 'gd-section-body';

        if (!features || Object.keys(features).length === 0) {
            body.innerHTML = '<div class="gd-empty">No features enrolled</div>';
        } else {
            const bucketOrder = ['frontal', 'left', 'right', 'back'];
            const sortedKeys = Object.keys(features).sort((a, b) => {
                const ai = bucketOrder.indexOf(a);
                const bi = bucketOrder.indexOf(b);
                return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
            });

            for (const bucket of sortedKeys) {
                const entries = features[bucket];
                if (!entries || entries.length === 0) continue;

                const group = document.createElement('div');
                group.className = 'gd-bucket-group';

                const label = document.createElement('div');
                label.className = 'gd-bucket-label';
                label.textContent = `${bucket} (${entries.length})`;
                group.appendChild(label);

                const grid = document.createElement('div');
                grid.className = 'gd-feature-grid';

                for (const entry of entries) {
                    grid.appendChild(this._renderFeatureCard(entry, featureType));
                }

                group.appendChild(grid);
                body.appendChild(group);
            }
        }

        section.appendChild(body);
        return section;
    }

    _renderFeatureCard(entry, featureType) {
        const card = document.createElement('div');
        card.className = 'gd-feature-card';

        // Image or placeholder
        if (entry.source_image_b64) {
            if (entry.overlay_bbox) {
                // 用 canvas 叠加框线 (face: 青色, body: 亮绿色)
                const wrapper = document.createElement('div');
                wrapper.className = 'gd-feature-img-wrapper';

                const img = document.createElement('img');
                img.className = 'gd-feature-img';
                // body 全帧图用 contain 确保框线位置正确
                if (featureType === 'body') {
                    img.style.objectFit = 'contain';
                    img.style.backgroundColor = '#1a1a2e';
                }
                img.src = `data:image/png;base64,${entry.source_image_b64}`;
                img.alt = entry.pose_bucket;
                img.dataset.overlayBbox = JSON.stringify(entry.overlay_bbox);
                img.dataset.boxColor = featureType === 'body' ? '#76ff03' : '#00e5ff';

                const canvas = document.createElement('canvas');
                canvas.className = 'gd-feature-canvas';

                const boxColor = featureType === 'body' ? '#76ff03' : '#00e5ff';
                const lineWidth = featureType === 'body' ? 2.5 : 2;

                img.onload = () => {
                    const naturalW = img.naturalWidth;
                    const naturalH = img.naturalHeight;
                    const displayW = img.offsetWidth;
                    const displayH = img.offsetHeight;

                    canvas.width = displayW;
                    canvas.height = displayH;

                    // 根据 object-fit 模式计算缩放
                    let scale, offsetX, offsetY;
                    if (featureType === 'body') {
                        // contain: 取较小缩放比, 图像完整显示
                        scale = Math.min(displayW / naturalW, displayH / naturalH);
                    } else {
                        // cover: 取较大缩放比, 图像居中裁切
                        scale = Math.max(displayW / naturalW, displayH / naturalH);
                    }
                    const renderedW = naturalW * scale;
                    const renderedH = naturalH * scale;
                    offsetX = (displayW - renderedW) / 2;
                    offsetY = (displayH - renderedH) / 2;

                    const [x1, y1, x2, y2] = entry.overlay_bbox;

                    const ctx = canvas.getContext('2d');
                    ctx.strokeStyle = boxColor;
                    ctx.lineWidth = lineWidth;
                    ctx.strokeRect(
                        x1 * scale + offsetX, y1 * scale + offsetY,
                        (x2 - x1) * scale, (y2 - y1) * scale
                    );
                };

                wrapper.appendChild(img);
                wrapper.appendChild(canvas);
                card.appendChild(wrapper);
            } else {
                const img = document.createElement('img');
                img.className = 'gd-feature-img';
                img.src = `data:image/png;base64,${entry.source_image_b64}`;
                img.alt = entry.pose_bucket;
                card.appendChild(img);
            }
        } else {
            const ph = document.createElement('div');
            ph.className = 'gd-feature-placeholder';
            ph.textContent = '📷';
            card.appendChild(ph);
        }

        // Quality + timestamp
        const info = document.createElement('div');
        info.className = 'gd-feature-info';

        const q = entry.quality_score;
        const thresholds = window.QUALITY_THRESHOLDS || { face: 0.3, body: 0.2 };
        const minQ = featureType === 'body' ? thresholds.body : thresholds.face;
        const qClass = q < minQ ? 'gd-quality-low' : 'gd-quality-high';

        info.innerHTML = `
            <div class="gd-feature-quality ${qClass}">Q: ${q.toFixed(2)}</div>
            <div class="gd-feature-time">${this._fmtTime(entry.timestamp)}</div>
        `;
        card.appendChild(info);

        return card;
    }

    // --- Wardrobe ---
    _renderWardrobeSection(wardrobe) {
        const section = document.createElement('div');
        section.className = 'gd-section';

        const header = document.createElement('div');
        header.className = 'gd-section-header';
        header.innerHTML = `
            <span class="gd-section-icon">👔</span>
            <span class="gd-section-title">Wardrobe</span>
            <span class="gd-section-badge">${wardrobe.length}</span>
            <span class="gd-section-toggle">▼</span>
        `;
        header.addEventListener('click', () => section.classList.toggle('collapsed'));
        section.appendChild(header);

        const body = document.createElement('div');
        body.className = 'gd-section-body';

        if (wardrobe.length === 0) {
            body.innerHTML = '<div class="gd-empty">No wardrobe records</div>';
        } else {
            const table = document.createElement('table');
            table.className = 'gd-wardrobe-table';
            table.innerHTML = `
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Quality</th>
                        <th>First Seen</th>
                        <th>Last Seen</th>
                        <th>Count</th>
                    </tr>
                </thead>
            `;

            const tbody = document.createElement('tbody');
            wardrobe.forEach((outfit, i) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${i + 1}</td>
                    <td>${outfit.quality_score.toFixed(3)}</td>
                    <td>${this._fmtTime(outfit.first_seen)}</td>
                    <td>${this._fmtTime(outfit.last_seen)}</td>
                    <td>${outfit.seen_count}</td>
                `;
                tbody.appendChild(tr);
            });
            table.appendChild(tbody);
            body.appendChild(table);
        }

        section.appendChild(body);
        return section;
    }

    // --- Body Proportions ---
    _renderProportionsSection(proportions) {
        const section = document.createElement('div');
        section.className = 'gd-section';

        const header = document.createElement('div');
        header.className = 'gd-section-header';
        header.innerHTML = `
            <span class="gd-section-icon">📐</span>
            <span class="gd-section-title">Body Proportions</span>
            <span class="gd-section-badge">${proportions ? `${proportions.samples} samples` : '—'}</span>
            <span class="gd-section-toggle">▼</span>
        `;
        header.addEventListener('click', () => section.classList.toggle('collapsed'));
        section.appendChild(header);

        const body = document.createElement('div');
        body.className = 'gd-section-body';

        if (!proportions) {
            body.innerHTML = '<div class="gd-empty">No body proportion data</div>';
        } else {
            const props = [
                { label: 'Torso / Leg', value: proportions.torso_leg_ratio, max: 1.5 },
                { label: 'Shoulder / Hip', value: proportions.shoulder_hip_ratio, max: 2.0 },
                { label: 'Arm / Torso', value: proportions.arm_torso_ratio, max: 2.0 },
                { label: 'Head / Body', value: proportions.head_body_ratio, max: 0.5 },
            ];

            for (const prop of props) {
                const pct = Math.min(100, (prop.value / prop.max) * 100);
                const item = document.createElement('div');
                item.className = 'gd-prop-item';
                item.innerHTML = `
                    <span class="gd-prop-label">${prop.label}</span>
                    <div class="gd-prop-bar-bg">
                        <div class="gd-prop-bar" style="width: ${pct}%"></div>
                    </div>
                    <span class="gd-prop-value">${prop.value.toFixed(3)}</span>
                `;
                body.appendChild(item);
            }

            // Height
            const heightItem = document.createElement('div');
            heightItem.className = 'gd-prop-item';
            heightItem.innerHTML = `
                <span class="gd-prop-label">Height (px)</span>
                <div class="gd-prop-bar-bg">
                    <div class="gd-prop-bar" style="width: ${Math.min(100, proportions.relative_height_px / 5)}%"></div>
                </div>
                <span class="gd-prop-value">${proportions.relative_height_px.toFixed(0)}</span>
            `;
            body.appendChild(heightItem);
        }

        section.appendChild(body);
        return section;
    }

    // --- VLM Description ---
    _renderVLMSection(vlmDescription) {
        const section = document.createElement('div');
        section.className = 'gd-section';

        const header = document.createElement('div');
        header.className = 'gd-section-header';
        header.innerHTML = `
            <span class="gd-section-icon">🧠</span>
            <span class="gd-section-title">VLM Description</span>
            <span class="gd-section-badge">${vlmDescription ? 'Available' : '—'}</span>
            <span class="gd-section-toggle">▼</span>
        `;
        header.addEventListener('click', () => section.classList.toggle('collapsed'));
        section.appendChild(header);

        const body = document.createElement('div');
        body.className = 'gd-section-body';

        if (!vlmDescription) {
            body.innerHTML = '<div class="gd-empty">No VLM description available</div>';
        } else {
            body.innerHTML = `<div class="gd-vlm-text">${this._escapeHtml(vlmDescription)}</div>`;
        }

        section.appendChild(body);
        return section;
    }

    // --- Actions ---
    _renderActions(person, detail) {
        const div = document.createElement('div');
        div.className = 'gd-actions';

        // Rename
        const renameBtn = document.createElement('button');
        renameBtn.className = 'btn';
        renameBtn.textContent = '✏️ Rename';
        renameBtn.addEventListener('click', async () => {
            const newName = prompt('Enter new name:', person.display_name);
            if (!newName || !newName.trim()) return;

            const trimmed = newName.trim();
            const cameraId = window.BACKEND_CONFIG?.cameraId;
            if (!cameraId) {
                alert('Camera not connected');
                return;
            }

            try {
                const resp = await fetch(
                    `${window.BACKEND_CONFIG.apiUrl}/${cameraId}/gallery/person/${person.person_id}`,
                    {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ display_name: trimmed }),
                    }
                );
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({}));
                    alert(`重命名失败: ${err.detail || resp.statusText}`);
                    return;
                }

                // 更新本地数据
                person.display_name = trimmed;
                document.getElementById('modal-person-name').textContent = trimmed;

                // 同步到 session 和 allPersons
                const sessionPerson = this.persons.get(person.person_id);
                if (sessionPerson) sessionPerson.display_name = trimmed;
                const allPerson = this.allPersons.get(person.person_id);
                if (allPerson) allPerson.display_name = trimmed;

                this._render();
            } catch (e) {
                alert(`重命名失败: ${e.message}`);
            }
        });
        div.appendChild(renameBtn);

        // Confirm (if present)
        if (person.present) {
            const confirmBtn = document.createElement('button');
            confirmBtn.className = 'btn';
            confirmBtn.textContent = '✅ Confirm';
            confirmBtn.addEventListener('click', () => {
                const trackId = this.activeTracks.get(person.person_id);
                if (trackId !== undefined) {
                    window.wsManager.sendConfirmIdentity(trackId, person.person_id, person.display_name);
                }
            });
            div.appendChild(confirmBtn);
        }

        // Delete
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn btn-danger';
        deleteBtn.textContent = '🗑️ Delete';
        deleteBtn.addEventListener('click', async () => {
            const confirmed = confirm(
                `确定要删除 "${person.display_name}" 吗？\n\nPerson ID: ${person.person_id}\n此操作不可恢复。`
            );
            if (!confirmed) return;

            try {
                const cameraId = window.BACKEND_CONFIG.cameraId;
                const resp = await fetch(
                    `${window.BACKEND_CONFIG.apiUrl}/${cameraId}/gallery/person/${person.person_id}`,
                    { method: 'DELETE' }
                );
                if (resp.ok) {
                    this._deletedIds.add(person.person_id);
                    this.persons.delete(person.person_id);
                    this.allPersons.delete(person.person_id);
                    this.activeTracks.delete(person.person_id);
                    this._render();
                    document.getElementById('person-modal').classList.add('hidden');
                } else {
                    const err = await resp.json().catch(() => ({}));
                    alert(`删除失败: ${err.detail || resp.statusText}`);
                }
            } catch (e) {
                alert(`删除失败: ${e.message}`);
            }
        });
        div.appendChild(deleteBtn);

        return div;
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    _countFeatures(features) {
        if (!features) return 0;
        let count = 0;
        for (const entries of Object.values(features)) {
            count += entries.length;
        }
        return count;
    }

    _fmtTime(timestamp) {
        if (!timestamp) return '—';
        const d = new Date(timestamp * 1000);
        return d.toLocaleString('zh-CN', {
            month: '2-digit', day: '2-digit',
            hour: '2-digit', minute: '2-digit', second: '2-digit',
            hour12: false,
        });
    }

    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// 全局实例
window.personGallery = new PersonGallery();
