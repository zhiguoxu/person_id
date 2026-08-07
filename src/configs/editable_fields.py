"""
web 控制台配置在线编辑的标注表与锁定表（person_id）。

编辑范围：配置模型的**全部叶子字段默认可编辑**（编辑需口令，见
voice_agent_common.config_override_api），本文件只做两件事：

1. FIELD_ANNOTATIONS 标注表：为常用字段提供中文说明和 hot（热生效）标记。
   未标注的字段一样可编辑，只是没有说明、且按 hot=False 保守提示「重启生效」。
   hot 语义（诚实原则，不做假热更新，与 voice/agent server 同一口径）：
     True  = 消费点调用时现读 config 单例（如 `cfg = config.matching` 每次
             匹配现取），改完立即生效
     False = 启动/首次使用时消费掉（模型加载、客户端构造），写库成功但需
             重启本服务才生效
   person_id 没有设备级配置解析链路（不存在 current_config(device_sn) 消费点），
   所有字段一律 device_editable=False（文件末尾统一补齐），设备级覆盖端点
   实际不开放任何字段——放出来也不生效，就是骗人。

2. LOCKED_PATHS 锁定表：彻底禁止在线编辑的字段，分两类：
   - 结构性死旋钮：在 DB 覆盖套用之前就被消费掉的（uvicorn 绑定参数在
     main() 里、早于 lifespan 的 load_and_apply），改了永远不生效；
   - 危险开关：改错即数据"丢失"或服务假死的项（如底库路径，换路径等于
     整库人物凭空消失）。
   敏感字段（vlm.api_key / redis.password）不在锁定之列——可编辑（重启后
   生效），展示和响应里一律脱敏为 ***，由 config_dump 的字段名规则自动识别。

注意：本服务另有一套滑块实时调参（config.update_from_dict，web 控制台
vision 页 Controls 面板经 REST /api/params 调用），它直接改内存不落库、重启
即失。两套机制作用于同一个 config 单例：滑块改过的值会在本页显示为
"当前值 ≠ yaml 原值"但没有「已修改」徽标（徽标只跟踪 DB 覆盖）。

启动时 ConfigOverrideManager 会校验两张表的路径真实存在（写错直接启动失败）。
"""

from voice_agent_common.config_override import EditableField

# 结构性死旋钮 + 危险开关，禁止在线编辑（GET 列表里也不出现）
LOCKED_PATHS: frozenset[str] = frozenset({
    "version",                  # 服务版本号是代码/发布的事实描述, 在线改了只会骗人
    "db_url",                   # init_db 在覆盖套用之前执行(覆盖本身存在这个库里), 改了永远不生效
    "server.host",              # uvicorn 绑定发生在 main(), 早于覆盖套用, 改了永远不生效
    "server.port",              # 同上
    "server.log_level",         # 同上 (import 期 LogManager.setup/intercept 消费)
    "server.ws_max_frame_size",  # 同上 (uvicorn.Config 的 ws_max_size 参数)
    "server.gallery_db_path",   # 换路径 = 整库人物凭空消失, 不上网页
})

FIELD_ANNOTATIONS: dict[str, EditableField] = {
    # ── 匹配与融合（消费点全部 `cfg = config.matching` 调用时现读, 改完下一次匹配即生效） ──
    "matching.A_threshold": {
        "description": "笃定判定阈值（唯一终态 A 级）。相似度超过它且 margin 达标即锁定身份",
        "hot": True,
    },
    "matching.A_margin": {
        "description": "笃定所需的最小 margin（Top1 与 Top2 的相似度差）",
        "hot": True,
    },
    "matching.B_threshold": {
        "description": "确定判定阈值（B 级）",
        "hot": True,
    },
    "matching.B_margin": {
        "description": "确定所需的最小 margin",
        "hot": True,
    },
    "matching.C_threshold": {
        "description": "怀疑/陌生分界线（低于它按陌生人处理）",
        "hot": True,
    },
    "matching.blend_alpha": {
        "description": "Body Top-K Blend 的 peak 权重（1-α 为 depth 权重）",
        "hot": True,
    },
    "matching.cross_pose_discount": {
        "description": "跨姿态投票权折扣（同姿态=1.0）",
        "hot": True,
    },
    "matching.wardrobe_boost_gamma": {
        "description": "衣橱命中提升因子（贝叶斯 lift）",
        "hot": True,
    },
    "matching.face_base_weight": {
        "description": "多模态融合的人脸基础权重",
        "hot": True,
    },
    "matching.body_base_weight": {
        "description": "多模态融合的全身基础权重",
        "hot": True,
    },
    "matching.proportion_base_weight": {
        "description": "多模态融合的体型比例基础权重",
        "hot": True,
    },

    # ── 检测 / 入库质量门槛（Tier1 逐帧现读; 与控制台 Controls 滑块是同一批字段） ──
    "detection.min_person_height_px": {
        "description": "最小人体像素高度——Tier1 硬门槛，过小的人直接不追踪/不入库。调高可收紧注册",
        "hot": True,
    },
    "detection.yolo_confidence": {
        "description": "YOLO 检测置信度阈值",
        "hot": False,  # 检测器构造时消费(每摄像头初始化), 保守按重启生效提示
    },
    "face.min_face_size": {
        "description": "入库人脸最小像素边长（人脸 bbox 短边），小于此值不入库",
        "hot": True,
    },
    "gallery.face_quality_enroll_threshold": {
        "description": "人脸入库最低质量分（eDifFIQA 评估）。人脸是强标识，糊脸入库会污染质心",
        "hot": True,
    },
    "gallery.body_quality_enroll_threshold": {
        "description": "人体/衣橱入库最低质量分（清晰度+完整度）",
        "hot": True,
    },
    "gallery.outfit_match_threshold": {
        "description": "衣橱匹配阈值（判定是否同一套衣服）",
        "hot": True,
    },
    "gallery.face_match_half_life_days": {
        "description": "人脸匹配端质心权重半衰期（天）",
        "hot": True,  # data_models 每次算质心现读
    },
    "gallery.ediffiqa_enroll_variant": {
        "description": "入库质量评估模型变体（tiny/small/medium/large），模型在启动预热时加载",
        "hot": False,
    },
    "multiframe.agg_min_face_quality": {
        "description": "参与身份聚合的人脸帧最低质量（影响识别，间接影响入库特征来源）",
        "hot": True,
    },
    "multiframe.agg_min_body_quality": {
        "description": "参与身份聚合的人体帧最低质量",
        "hot": True,
    },

    # ── VLM 仲裁 ──
    "vlm.enabled": {
        "description": "是否启用 Tier3 VLM 仲裁（调度器每次决策现读，关掉即当场停用）",
        "hot": True,
    },
    "vlm.model": {
        "description": "VLM 模型名（客户端在仲裁器初始化时构造）",
        "hot": False,
    },
    "vlm.api_key": {
        "description": "VLM API Key（脱敏不回显，输入新值整体替换）",
        "hot": False,
    },
    "vlm.base_url": {
        "description": "VLM 服务地址",
        "hot": False,
    },

    # ── 服务端拉流（StreamConsumer 逐帧/逐次重连现读） ──
    "server.stream_max_fps": {
        "description": "拉流处理帧率上限（拉到的多余帧直接丢弃）",
        "hot": True,
    },
    "server.stream_proc_max_width": {
        "description": "识别路径分辨率上限：0 = 按视频流原生分辨率处理（无损），算力不足时才设正数换速度",
        "hot": True,
    },
    "server.stream_preview_max_width": {
        "description": "前端预览帧最大宽度（只影响网页观看带宽/清晰度，不影响识别）",
        "hot": True,
    },
    "server.stream_preview_jpeg_quality": {
        "description": "前端预览帧 JPEG 质量（1-100，只影响网页观看）",
        "hot": True,
    },
    "server.stream_reconnect_delay": {
        "description": "拉流断开后的重连间隔（秒）",
        "hot": True,
    },
    "server.stream_restream_fail_threshold": {
        "description": "连续拉流失败达到该次数后自动重新开启设备推流并切换新地址",
        "hot": True,
    },
    "server.image_correction_enabled": {
        "description": "是否启用镜头畸变矫正（逐帧现读）",
        "hot": True,
    },

    # ── 需重启的重资产选择（模型/硬件在启动或首次使用时加载） ──
    "hardware.device": {
        "description": "统一计算设备（如 cuda:0 / cpu），模型加载时消费",
        "hot": False,
    },
    "face.recognition_backend": {
        "description": "人脸识别模型后端（arcface / adaface），启动时加载",
        "hot": False,
    },
    "redis.host": {
        "description": "Redis 地址（拉流期望状态持久化 + 设备在线标记查询）。留空则拉流状态不持久化",
        "hot": False,
    },
}

# person_id 没有按设备(device_sn)解析配置的消费链路, 设备级覆盖一律不开放
# (见文件头): 统一补 device_editable=False, 避免 hot 字段按缺省规则被放出来。
for _spec in FIELD_ANNOTATIONS.values():
    _spec.setdefault("device_editable", False)
