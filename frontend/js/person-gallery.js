/**
 * Person Gallery — 人物画廊面板
 * 
 * 显示已识别的人物卡片，支持点击查看详情和手动确认身份。
 */
class PersonGallery {
    constructor() {
        this.container = document.getElementById('person-gallery');
        this.persons = new Map(); // person_id -> person data
        this.activeTracks = new Map(); // person_id -> track_id (当前在场)
    }

    /**
     * 从帧结果更新在场状态
     */
    updateFromResult(result) {
        const activePersonIds = new Set();

        const persons = result.tracked_persons || result.persons || [];

        if (persons.length > 0) {
            persons.forEach(p => {
                if (p.person_id) {
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

        this._render();
    }

    _render() {
        if (!this.container) return;

        // 更新计数
        const countEl = document.getElementById('gallery-count');
        if (countEl) countEl.textContent = `${this.persons.size} persons`;

        if (this.persons.size === 0) {
            this.container.innerHTML = '<div class="gallery-empty">No persons identified yet.</div>';
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

        this.persons.forEach((person, id) => {
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
        // 基于名称生成首字母头像
        const initial = (person.display_name || '?')[0].toUpperCase();
        return initial;
    }

    _showDetail(person) {
        const modal = document.getElementById('person-modal');
        const title = document.getElementById('modal-person-name');
        const body = document.getElementById('modal-body');

        title.textContent = person.display_name || person.person_id;

        body.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div>
                    <h4 style="color: var(--text-muted); margin-bottom: 8px;">Info</h4>
                    <div style="font-size: 0.85rem; color: var(--text-secondary);">
                        <p><strong>ID:</strong> ${person.person_id}</p>
                        <p><strong>Name:</strong> ${person.display_name}</p>
                        <p><strong>Confidence:</strong> ${(person.confidence * 100).toFixed(1)}%</p>
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

        // Show modal
        modal.classList.remove('hidden');

        // Close handlers
        modal.querySelector('.modal-close').onclick = () => modal.classList.add('hidden');
        modal.querySelector('.modal-backdrop').onclick = () => modal.classList.add('hidden');
    }
}

// 全局实例
window.personGallery = new PersonGallery();
