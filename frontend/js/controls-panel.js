/**
 * Controls Panel — 阈值控制面板
 * 
 * 动态创建滑块，实时发送配置更新到后端。
 * 支持预设（保守/均衡/激进）。
 */
class ControlsPanel {
    constructor() {
        this.container = document.getElementById('threshold-sliders');
        this.params = {};
        this.presets = {
            conservative: {
                A_THRESHOLD: 0.85,
                B_THRESHOLD: 0.75,
                C_THRESHOLD: 0.60,
                FACE_QUALITY_ENROLL_THRESHOLD: 0.65,
                BODY_QUALITY_ENROLL_THRESHOLD: 0.55,
                OUTFIT_MATCH_THRESHOLD: 0.90,
            },
            balanced: {
                A_THRESHOLD: 0.78,
                B_THRESHOLD: 0.68,
                C_THRESHOLD: 0.50,
                FACE_QUALITY_ENROLL_THRESHOLD: 0.55,
                BODY_QUALITY_ENROLL_THRESHOLD: 0.40,
                OUTFIT_MATCH_THRESHOLD: 0.85,
            },
            aggressive: {
                A_THRESHOLD: 0.70,
                B_THRESHOLD: 0.60,
                C_THRESHOLD: 0.40,
                FACE_QUALITY_ENROLL_THRESHOLD: 0.45,
                BODY_QUALITY_ENROLL_THRESHOLD: 0.30,
                OUTFIT_MATCH_THRESHOLD: 0.75,
            },
        };

        this._bindPresets();
    }

    /**
     * 初始化滑块 (从服务端获取配置)
     */
    initialize(tunableParams) {
        this.params = tunableParams || this._defaultParams();
        this._render();
    }

    _defaultParams() {
        return {
            A_THRESHOLD: { value: 0.78, min: 0, max: 1, step: 0.01, group: 'reid', label: 'A Threshold (笃定)' },
            B_THRESHOLD: { value: 0.68, min: 0, max: 1, step: 0.01, group: 'reid', label: 'B Threshold (确定)' },
            C_THRESHOLD: { value: 0.50, min: 0, max: 1, step: 0.01, group: 'reid', label: 'C Threshold (怀疑)' },
            FACE_QUALITY_ENROLL_THRESHOLD: { value: 0.55, min: 0, max: 1, step: 0.05, group: 'quality', label: '人脸入库质量门槛' },
            BODY_QUALITY_ENROLL_THRESHOLD: { value: 0.40, min: 0, max: 1, step: 0.05, group: 'quality', label: '人体入库质量门槛' },
            MIN_FACE_SIZE: { value: 60, min: 0, max: 200, step: 5, group: 'quality', label: '入库人脸最小像素' },
            MIN_PERSON_HEIGHT_PX: { value: 120, min: 0, max: 400, step: 10, group: 'quality', label: '最小人体像素高度' },
            AGG_MIN_FACE_QUALITY: { value: 0.20, min: 0, max: 1, step: 0.05, group: 'quality', label: '人脸聚合最低质量' },
            AGG_MIN_BODY_QUALITY: { value: 0.30, min: 0, max: 1, step: 0.05, group: 'quality', label: '人体聚合最低质量' },
            OUTFIT_MATCH_THRESHOLD: { value: 0.85, min: 0, max: 1, step: 0.01, group: 'matching', label: '衣橱匹配阈值' },
        };
    }

    _render() {
        if (!this.container) return;
        this.container.innerHTML = '';

        // 按 group 分组
        const groups = {};
        Object.entries(this.params).forEach(([key, param]) => {
            const group = param.group || 'other';
            if (!groups[group]) groups[group] = [];
            groups[group].push({ key, ...param });
        });

        const groupLabels = {
            reid: '🔍 ReID',
            quality: '📊 Quality',
            matching: '🔗 Matching',
        };

        const groupColors = {
            reid: 'group-reid',
            quality: 'group-quality',
            matching: 'group-matching',
        };

        Object.entries(groups).forEach(([group, items]) => {
            const groupDiv = document.createElement('div');
            groupDiv.className = 'slider-group';

            const label = document.createElement('div');
            label.className = 'slider-group-label';
            label.textContent = groupLabels[group] || group;
            groupDiv.appendChild(label);

            items.forEach(item => {
                const sliderItem = document.createElement('div');
                sliderItem.className = 'slider-item';

                const sliderLabel = document.createElement('span');
                sliderLabel.className = 'slider-label';
                sliderLabel.textContent = item.label;

                const sliderInput = document.createElement('input');
                sliderInput.type = 'range';
                sliderInput.className = `slider-input ${groupColors[group] || ''}`;
                sliderInput.min = item.min;
                sliderInput.max = item.max;
                sliderInput.step = item.step;
                sliderInput.value = item.value;
                sliderInput.id = `slider-${item.key}`;

                const valueDisplay = document.createElement('span');
                valueDisplay.className = 'slider-value';
                valueDisplay.textContent = parseFloat(item.value).toFixed(2);
                valueDisplay.id = `value-${item.key}`;

                // 实时更新
                sliderInput.addEventListener('input', () => {
                    const val = parseFloat(sliderInput.value);
                    valueDisplay.textContent = val.toFixed(2);
                    this.params[item.key].value = val;
                });

                // 防抖发送
                let debounceTimer;
                sliderInput.addEventListener('input', () => {
                    clearTimeout(debounceTimer);
                    debounceTimer = setTimeout(() => {
                        this._sendUpdate(item.key, parseFloat(sliderInput.value));
                    }, 150);
                });

                sliderItem.appendChild(sliderLabel);
                sliderItem.appendChild(sliderInput);
                sliderItem.appendChild(valueDisplay);
                groupDiv.appendChild(sliderItem);
            });

            this.container.appendChild(groupDiv);
        });
    }

    _sendUpdate(key, value) {
        window.wsManager.sendConfigUpdate({ [key]: value });
    }

    _bindPresets() {
        document.querySelectorAll('[data-preset]').forEach(btn => {
            btn.addEventListener('click', () => {
                const presetName = btn.dataset.preset;
                this.applyPreset(presetName);

                // 切换 active 样式
                document.querySelectorAll('[data-preset]').forEach(b => b.classList.remove('btn-active'));
                btn.classList.add('btn-active');
            });
        });
    }

    applyPreset(name) {
        const preset = this.presets[name];
        if (!preset) return;

        Object.entries(preset).forEach(([key, value]) => {
            const slider = document.getElementById(`slider-${key}`);
            const display = document.getElementById(`value-${key}`);
            if (slider) {
                slider.value = value;
                if (display) display.textContent = value.toFixed(2);
                if (this.params[key]) this.params[key].value = value;
            }
        });

        // 批量发送
        window.wsManager.sendConfigUpdate(preset);
    }
}

// 全局实例
window.controlsPanel = new ControlsPanel();
