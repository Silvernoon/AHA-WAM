# AHA-WAM 待解决问题

## P0：正确部署契约下 Teacher 闭环失败

**状态：未解决，当前模型尚未通过闭环验收。**

即使使用与训练一致的历史/偏移配置，16 步 Teacher 仍在 RoboTwin `grab_roller` 上失败。因此不能把当前问题仅归因于一步蒸馏；在 Teacher 闭环基线建立之前，不应继续扩大 Student 成功率评测或将模型标记为可用。

## 已确认事实

### Teacher 配置

- Checkpoint：`ahawam_da3_3view_history_offset_stage2_17000/checkpoints/weights/step_017000.pt`
- 部署 task：`robotwin_ahawam_da3_history_offset`
- 三视角共享视觉主干：启用
- `num_history_frames: 6`
- `action_horizon: 80`
- `action_chunk_size: 16`
- `max_action_offset: 15`
- `action_video_read_mode: current_only`
- Teacher action diffusion：16 步

部署配置已确认解析为：

```text
shared=True, views=3, history=6, horizon=80, offset=15, chunk=16
```

### 闭环结果

- 任务：`grab_roller`
- 场景：`demo_clean`
- Seed：`42`（RoboTwin 内部有效 seed `4300000`）
- 结果：`0/1`
- 视频：`/home/sivn/Downloads/ahawam-eval-videos/grab_roller_teacher_history6_step017000_seed42.mp4`
- SHA-256：`56e6238688be8e1f1de284e0bc201e8152fe94a67c05b0ec0ff1eafaa595e580`

视频编码和三相机拼接已经修复并验证：H.264、`640x480`、400 帧、10 FPS、40 秒。Teacher 的连续运动比一步 Student 更平滑，但没有成功抓起 roller，任务仍然失败。

### 渲染和运行环境

- RTX 4090 的 SAPIEN Vulkan/OIDN 路径可以输出无条纹的三相机 RGB。
- 必须使用：`VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json`。
- CuRobo 已用 `TORCH_CUDA_ARCH_LIST=8.9` 重编，`CuroboPlanner` 可以正常初始化。
- 评估画面仍明显偏白、低对比度，与训练视频存在视觉域差异。

### 本轮已确认的部署偏差

- Teacher 实际训练产物 `config.yaml` 的 `model.video_rope_frame_stride` 为 `8`，而此前评估配置硬编码为 `1`。这会让每次 32-action video prefill 的时序位置前进 `32`，训练契约应前进 `32 / 8 = 4`。评估配置已改为 `${data.train.action_video_freq_ratio}`，当前解析值为 `8`。
- Teacher 训练启用 `preserve_camera_views`，每路图像先变为 `240x320`，再按 `video_size=[384,320]` 做保持比例 resize 和中心裁剪；此前部署直接使用 `256x320` 全视野图像，既改变了视场也改变了 VAE latent 网格。部署预处理已改为复现训练的两阶段 resize、BICUBIC resize 和中心裁剪，warmup 尺寸同步为 `384x320`。
- 训练与部署 `dataset_stats.json` SHA-256 均为 `7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095`。
- 数据集元数据显示动作和状态均为 50 FPS、14D，顺序为左臂 7D、右臂 7D；与 RoboTwin `get_obs()` 的 `left_arm + left_gripper + right_arm + right_gripper` 顺序一致。
- 修正后 A800 真实 checkpoint smoke 通过：输入尺寸 `384x320`、RoPE stride `8`、输出 `(14,)`、全部 finite；这只证明推理链可运行，不证明闭环成功。
- 新 4090 渲染机 `connect.bjb3.seetacloud.com:49223` 已验证：RTX 4090、Driver `580.142`、密钥 SSH 和 `my_nvidia_icd.json` 下 SAPIEN 正常。
- 重新闭环时确认 RoboTwin 默认仍把每个 50 FPS qpos 当作独立 TOPP 终点；单回合超过一小时仍未完成，不能作为有效控制实验。已增加可选 `ROBOTWIN_QPOS_CONTROL_MODE=fixed_rate`，用 5 个 250 Hz physics step 对应一个 50 Hz 数据动作。
- 但 16 步 Teacher 首个远程 action 请求仍出现分钟级无 GPU 利用率的等待；将 timeout 从 180 秒提高到 900 秒后仍未形成完整结果。因此当前还有模型服务请求阻塞/推理延迟问题，不能声称已经完成修正后的闭环复测。
- 分阶段日志已定位旧“Teacher 推理卡住”为网络传输问题：每个三相机 observation 经 base64 JSON 后约 `1.232 MB`，通过双层 SSH 转发接收耗时 `85–152 s`；真实首个 Teacher method 只需 `1.013 s`，后续 deque 命中为 `0.000 s`。GPU 低利用率是服务端在等请求字节，不是 16 步扩散卡死。
- zlib level-1 将请求压到约 `207–308 KB`，客户端编码约 `0.018–0.021 s`，首个请求总耗时降至 `46.565 s`，后续降至约 `15–40 s`。当前链路仍受双层 SSH 的高延迟/吞吐影响，下一步应取消本地中继，改为渲染机到 A800 的单跳 TCP/SSH，或换二进制图像协议。
- A800 已通过其自身 SSH 密钥直连公网 4090；已建立 `A800 -> 4090` 单跳反向隧道，移除本机中继。
- RGB 已改为 JPEG quality 90 二进制附件协议，不再走 base64 JSON。单请求从约 `1.232 MB` 降为 header `~3.3 KB` + JPEG `~31–45 KB`；客户端编码约 `6–17 ms`，服务端接收+解码约 `0.10–0.13 s`。
- 修正协议后 `grab_roller` seed `4300000` 完整闭环在数十秒内结束，不再超时；结果仍为 `0/1`。失败视频已下载为 `/home/sivn/Downloads/ahawam-eval-videos/grab_roller_teacher_jpeg_direct_q90_fixed_seed4300000.mp4`，SHA-256 `9addcabb438c993330271877aae940d75b9bbc417db5c35746f92b49ee68a07d`。这确认网络和控制时序问题已解除，但 Teacher 策略本身仍未成功抓取。

## 优先排查项

### 1. 视觉域偏移

- 保存评估时三路原始 RGB，不只检查拼接视频。
- 与训练集同任务三视角帧比较亮度、对比度、颜色直方图和目标尺寸。
- 检查 SAPIEN 灯光、材质、曝光、tone mapping、ray-tracing shader 和 OIDN 设置。
- 在输入模型前保存 resize/normalize 后 tensor，确认数值范围和通道顺序。
- 必要时先让新渲染机复现训练数据的视觉风格，再评估策略。

### 2. Action/State 语义和归一化

- 校验评估使用的 `dataset_stats.json` 与训练时完全相同，并记录校验和。
- 核对 14D action/state 的字段顺序、左右臂顺序、夹爪位置、单位和符号。
- 同时记录归一化前后 action、反归一化 action、当前 qpos 和最终发送给 RoboTwin 的 qpos。
- 检查预测是否越过训练分布、机器人关节限制或夹爪范围。

### 3. 控制频率和 TOPP 执行语义

RoboTwin 当前对每个预测 qpos 都从当前关节状态重新执行一次 TOPP。需要核对：

- 训练数据实际 action 频率与仿真每次 `take_action` 的语义是否一致。
- 相邻 qpos 是否被错误地当作独立终点，而不是连续轨迹采样点。
- 每次 TOPP 重置速度边界是否造成停顿、反向或抖动。
- 记录每步实际执行时长、内部 250 Hz 物理步数、速度、加速度和 jerk。

### 4. 异步历史和缓存时序

- 记录每次 video prefill、action chunk、`chunk_index`、`next_chunk_index`。
- 记录历史缓存长度、历史帧索引、当前帧索引和失效/重置事件。
- 确认 `chunks_per_video_prefill=2` 与训练时 offset/history 分布兼容。
- 确认 episode reset 会清空旧 episode 状态，但 observation refresh 不会误删需要保留的历史。

### 5. Teacher 本身的离线质量

- 从训练集取真实 episode，用正确 context/proprio 运行 16 步 Teacher。
- 将 Teacher action 与 GT action 对齐，报告位置误差、速度误差、chunk 边界误差和夹爪误差。
- 对 offset `0..15`、history length `0..6` 分桶评估。
- 检查最终 `step_017000` 是否优于中间 checkpoint，而不是只按最后一步选择。
- 确认训练 loss/REPA loss 下降是否对应 action validation 指标改善。

## 下一轮实验顺序

1. 用同一 seed 跑 RoboTwin expert，保存正常成功轨迹，确认环境和任务可解。
2. 保存 Teacher 全部输入、14D 输出、chunk/cache 状态和实际执行 qpos。
3. 对比 expert、GT 数据和 Teacher 的轨迹频率与 action 语义。
4. 修正视觉域或控制接口后，只复测一个 `grab_roller` seed。
5. 单 seed 成功且无异常抖动后，再做 Teacher 10 次测试。
6. Teacher 基线通过后，才恢复 Student 对照和一步蒸馏诊断。

## 验收条件

- 三路原始 RGB 与训练视觉分布基本一致，无过曝、条纹或错误通道。
- action/state 顺序、单位、归一化和数据频率全部有日志证据。
- Teacher 输出不越界，chunk 边界连续，无明显高频反向和异常 jerk。
- 同 seed expert 能成功，Teacher 至少能在预先约定的小规模任务/seed 集上得到非零成功率。
- Teacher 闭环基线未通过前，不宣称 Student 或完整 Y 形模型闭环有效。
