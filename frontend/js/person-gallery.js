/**
 * Person Gallery — 人物画廊面板
 *
 * 两种显示模式:
 *   - active: 仅显示当前帧匹配到的人 (实时)
 *   - all:    显示数据库中所有已知用户
 */
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
                    thumbnail: null,
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
                if (p.person_id && !this._deletedIds.has(p.person_id)) {
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

    _showDetail(person) {
        const modal = document.getElementById('person-modal');
        const title = document.getElementById('modal-person-name');
        const body = document.getElementById('modal-body');

        title.textContent = person.display_name || person.person_id;

        const confText = person.confidence != null ? `${(person.confidence * 100).toFixed(1)}%` : '—';

        body.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div>
                    <h4 style="color: var(--text-muted); margin-bottom: 8px;">Info</h4>
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">
                        <p><strong>ID:</strong> ${person.person_id}</p>
                        <p><strong>Name:</strong> ${person.display_name}</p>
                        <p><strong>Confidence:</strong> ${confText}</p>
                        <p><strong>Status:</strong> ${person.present ? '🟢 Present' : '⚪ Away'}</p>
                        <p><strong>First seen:</strong> ${new Date(person.first_seen).toLocaleString()}</p>
                        <p><strong>Last seen:</strong> ${new Date(person.last_seen).toLocaleString()}</p>
                    </div>
                </div>
                <div>
                    <h4 style="color: var(--text-muted); margin-bottom: 8px;">Actions</h4>
                    <div style="display: flex; flex-direction: column; gap: 8px;">
                        <button class="btn btn-primary" id="modal-btn-rename">✏️ Rename</button>
                        ${person.present ? `<button class="btn" id="modal-btn-confirm">✅ Confirm Identity</button>` : ''}
                        <button class="btn btn-danger" id="modal-btn-delete">🗑️ Delete from Gallery</button>
                    </div>
                </div>
            </div>
        `;

        // Rename handler
        const renameBtn = body.querySelector('#modal-btn-rename');
        if (renameBtn) {
            renameBtn.addEventListener('click', () => {
                const newName = prompt('Enter new name:', person.display_name);
                if (newName && newName.trim()) {
                    person.display_name = newName.trim();
                    title.textContent = newName.trim();
                    this._render();
                }
            });
        }

        // Confirm handler
        const confirmBtn = body.querySelector('#modal-btn-confirm');
        if (confirmBtn) {
            confirmBtn.addEventListener('click', () => {
                const trackId = this.activeTracks.get(person.person_id);
                if (trackId !== undefined) {
                    window.wsManager.sendConfirmIdentity(trackId, person.person_id, person.display_name);
                    alert(`Identity confirmed: ${person.display_name}`);
                }
            });
        }

        // Delete handler
        const deleteBtn = body.querySelector('#modal-btn-delete');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', async () => {
                const confirmed = confirm(`确定要删除 "${person.display_name}" 吗？\n\nPerson ID: ${person.person_id}\n此操作不可恢复。`);
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
                        modal.classList.add('hidden');
                        console.log(`[Gallery] Deleted ${person.person_id}`);
                    } else {
                        const err = await resp.json().catch(() => ({}));
                        alert(`删除失败: ${err.detail || resp.statusText}`);
                    }
                } catch (e) {
                    alert(`删除失败: ${e.message}`);
                }
            });
        }

        // Show modal
        modal.classList.remove('hidden');

        // Close handlers
        modal.querySelector('.modal-close').onclick = () => modal.classList.add('hidden');
        modal.querySelector('.modal-backdrop').onclick = () => modal.classList.add('hidden');
    }
}

// 全局实例
window.personGallery = new PersonGallery();
