/**
 * Overlay Renderer — Canvas 叠加层渲染
 * 
 * 在 video 元素上方的 canvas 绘制:
 * - 检测框 (颜色编码: 确认/识别中/陌生)
 * - 骨骼关键点 + 连线
 * - 人物标签 (ID, 名称, 置信度)
 * - 追踪轨迹
 * - 注意力目标高亮
 * - 姿态标签
 */
class OverlayRenderer {
    constructor() {
        this.canvas = document.getElementById('overlay-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.persons = [];
        this.options = {
            showBbox: true,
            showSkeleton: true,
            showTrail: true,
            showLabels: true,
        };

        // COCO 骨骼连线拓扑
        this.SKELETON_PAIRS = [
            [5, 6],    // 左肩-右肩
            [5, 11], [6, 12],   // 肩-髋
            [11, 12],  // 左髋-右髋
            [5, 7], [7, 9],     // 左臂
            [6, 8], [8, 10],    // 右臂
            [11, 13], [13, 15], // 左腿
            [12, 14], [14, 16], // 右腿
            [0, 1], [0, 2],     // 鼻-眼
            [1, 3], [2, 4],     // 眼-耳
        ];

        // 颜色方案
        this.COLORS = {
            confirmed: '#00ff88',
            identifying: '#ffa500',
            suspected: '#ff6b6b',
            stranger: '#6b7280',
            spatial_inferred: '#ffeb3b',
            target_glow: '#00e5ff',
            skeleton_high: 'rgba(0, 255, 136, 0.8)',
            skeleton_low: 'rgba(100, 100, 100, 0.4)',
        };

        this.onPersonClicked = null;

        this._bindToggles();
        this._bindEvents();
    }

    _bindEvents() {
        this.canvas.addEventListener('click', (e) => {
            if (!this.onPersonClicked || !this.persons.length) return;

            const rect = window.videoCapture.getVideoRect();
            if (!rect) return;

            // Get click coordinates relative to canvas
            const canvasRect = this.canvas.getBoundingClientRect();
            const clickX = e.clientX - canvasRect.left;
            const clickY = e.clientY - canvasRect.top;

            // Find clicked person (iterate backwards to click top-most first)
            for (let i = this.persons.length - 1; i >= 0; i--) {
                const person = this.persons[i];
                const bbox = this._normToPixel(person.bbox, rect);
                
                const [x1, y1, x2, y2] = bbox;
                // Add a small padding (e.g., 10px) to make clicking easier
                const padding = 10;
                
                if (clickX >= x1 - padding && clickX <= x2 + padding && 
                    clickY >= y1 - padding && clickY <= y2 + padding) {
                    this.onPersonClicked(person);
                    break;
                }
            }
        });
    }

    _bindToggles() {
        const toggles = {
            'toggle-bbox': 'showBbox',
            'toggle-skeleton': 'showSkeleton',
            'toggle-trail': 'showTrail',
            'toggle-labels': 'showLabels',
        };

        Object.entries(toggles).forEach(([id, key]) => {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener('change', () => {
                    this.options[key] = el.checked;
                });
            }
        });
    }

    /**
     * 更新并重绘叠加层
     */
    update(persons) {
        this.persons = persons || [];
        this._render();
    }

    _render() {
        const rect = window.videoCapture.getVideoRect();
        if (!rect) return;

        // 同步 canvas 尺寸
        const container = document.getElementById('video-container');
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        for (const person of this.persons) {
            const color = this._getColor(person);

            // 坐标转换: 归一化 [0,1] → 像素
            const bbox = this._normToPixel(person.bbox, rect);

            if (this.options.showTrail && person.trail && person.trail.length > 1) {
                this._drawTrail(person.trail, rect, color);
            }

            if (this.options.showBbox) {
                this._drawBbox(bbox, color, person.is_current_target);
            }

            if (this.options.showSkeleton && person.keypoints) {
                this._drawSkeleton(person.keypoints, rect);
            }

            if (this.options.showLabels) {
                this._drawLabel(bbox, person, color);
                this._drawPoseBadge(bbox, person.pose_bucket);
            }
        }
    }

    _getColor(person) {
        const status = person.identity_status || person.status || 'identifying';
        return this.COLORS[status] || this.COLORS.identifying;
    }

    _normToPixel(bbox, rect) {
        if (!bbox || bbox.length < 4) return [0, 0, 0, 0];

        // 检测坐标基于发送帧 (rect.videoW x rect.videoH)
        const vw = rect.videoW;
        const vh = rect.videoH;
        const scaleX = rect.displayW / vw;
        const scaleY = rect.displayH / vh;
        return [
            rect.offsetX + bbox[0] * scaleX,
            rect.offsetY + bbox[1] * scaleY,
            rect.offsetX + bbox[2] * scaleX,
            rect.offsetY + bbox[3] * scaleY,
        ];
    }

    _drawBbox(bbox, color, isTarget) {
        const [x1, y1, x2, y2] = bbox;
        const ctx = this.ctx;

        if (isTarget) {
            // 发光外框
            ctx.strokeStyle = this.COLORS.target_glow;
            ctx.lineWidth = 4;
            ctx.setLineDash([]);
            ctx.strokeRect(x1 - 3, y1 - 3, x2 - x1 + 6, y2 - y1 + 6);

            // 角落装饰
            const cornerLen = 15;
            ctx.lineWidth = 3;
            ctx.strokeStyle = this.COLORS.target_glow;
            this._drawCorners(x1 - 3, y1 - 3, x2 - x1 + 6, y2 - y1 + 6, cornerLen);
        }

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }

    _drawCorners(x, y, w, h, len) {
        const ctx = this.ctx;
        ctx.beginPath();
        // Top-left
        ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y);
        // Top-right
        ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
        // Bottom-left
        ctx.moveTo(x, y + h - len); ctx.lineTo(x, y + h); ctx.lineTo(x + len, y + h);
        // Bottom-right
        ctx.moveTo(x + w - len, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - len);
        ctx.stroke();
    }

    _drawSkeleton(keypoints, rect) {
        const ctx = this.ctx;

        const vw = rect.videoW;
        const vh = rect.videoH;
        const scaleX = rect.displayW / vw;
        const scaleY = rect.displayH / vh;

        const toX = (v) => rect.offsetX + v * scaleX;
        const toY = (v) => rect.offsetY + v * scaleY;

        // 绘制骨骼连线
        for (const [i, j] of this.SKELETON_PAIRS) {
            const kp1 = keypoints[i];
            const kp2 = keypoints[j];
            if (!kp1 || !kp2) continue;

            const conf = Math.min(kp1[2] || 0, kp2[2] || 0);
            if (conf < 0.2) continue;

            ctx.beginPath();
            ctx.moveTo(toX(kp1[0]), toY(kp1[1]));
            ctx.lineTo(toX(kp2[0]), toY(kp2[1]));
            ctx.strokeStyle = conf > 0.5 ? this.COLORS.skeleton_high : this.COLORS.skeleton_low;
            ctx.lineWidth = conf > 0.5 ? 2 : 1;
            ctx.stroke();
        }

        // 绘制关键点
        for (let i = 0; i < keypoints.length; i++) {
            const kp = keypoints[i];
            if (!kp || (kp[2] || 0) < 0.2) continue;

            const x = toX(kp[0]);
            const y = toY(kp[1]);
            const r = kp[2] > 0.5 ? 3 : 2;

            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fillStyle = kp[2] > 0.5 ? this.COLORS.skeleton_high : this.COLORS.skeleton_low;
            ctx.fill();
        }
    }

    _drawLabel(bbox, person, color) {
        const ctx = this.ctx;
        const status = person.identity_status || person.status || 'identifying';

        // 根据身份状态决定显示名称
        let name;
        if (status === 'stranger') {
            name = 'Unknown';
        } else if (status === 'suspected') {
            const rawName = person.display_name || person.person_id || '?';
            name = `Suspect (${rawName})`;
        } else {
            name = person.display_name || person.person_id || 'Unknown';
        }

        const conf = person.confidence ? ` ${(person.confidence * 100).toFixed(0)}%` : '';
        const trackId = person.track_id !== undefined ? `[#${person.track_id}] ` : '';
        const text = `${trackId}${name}${conf}`;

        ctx.font = '600 12px Inter, sans-serif';
        const metrics = ctx.measureText(text);
        const textW = metrics.width + 12;
        const textH = 20;
        const gap = 4;

        let x = bbox[0];
        let y = bbox[1] - textH - gap;

        // 边界保护: 上方放不下 → 放到框内顶部
        if (y < 0) {
            y = bbox[1] + gap;
        }
        // 右侧超出 → 左移
        if (x + textW > this.canvas.width) {
            x = this.canvas.width - textW;
        }
        // 左侧超出
        if (x < 0) x = 0;

        // 背景
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.beginPath();
        ctx.roundRect(x, y, textW, textH, 4);
        ctx.fill();

        // 左侧色条
        ctx.fillStyle = color;
        ctx.fillRect(x, y, 3, textH);

        // 文字
        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, x + 8, y + 14);
    }

    _drawPoseBadge(bbox, poseBucket) {
        if (!poseBucket || poseBucket === 'unknown') return;

        const ctx = this.ctx;
        const badgeMap = {
            frontal: '👤',
            left: '◀',
            right: '▶',
            back: '🔙',
        };

        const emoji = badgeMap[poseBucket] || '?';
        const badgeW = 22;
        const badgeH = 16;

        let x = bbox[2] + 4;
        let y = bbox[1] + 4;

        // 右侧超出 → 放到框内右上角
        if (x + badgeW > this.canvas.width) {
            x = bbox[2] - badgeW - 4;
        }
        // 上方超出
        if (y < 0) y = 0;
        // 下方超出
        if (y + badgeH > this.canvas.height) {
            y = this.canvas.height - badgeH;
        }

        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.beginPath();
        ctx.roundRect(x, y, badgeW, badgeH, 3);
        ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.font = '10px sans-serif';
        ctx.fillText(emoji, x + 4, y + 12);
    }

    _drawTrail(trail, rect, color) {
        if (trail.length < 2) return;
        const ctx = this.ctx;

        const scaleX = rect.displayW / rect.videoW;
        const scaleY = rect.displayH / rect.videoH;

        ctx.beginPath();
        for (let i = 0; i < trail.length; i++) {
            const x = rect.offsetX + trail[i][0] * scaleX;
            const y = rect.offsetY + trail[i][1] * scaleY;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.5;
        ctx.stroke();
        ctx.globalAlpha = 1.0;
    }
}

// 全局实例
window.overlayRenderer = new OverlayRenderer();
