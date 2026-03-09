# 实时手掌骨架识别（Python + OpenCV）

这个项目使用 **OpenCV + MediaPipe** 调用摄像头进行：

- 实时手掌骨架识别
- 距离粗略估算（cm）
- 下一帧骨架预测（常速度模型）
- 实时 FPS 显示

## 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 运行

```bash
python hand_skeleton_realtime.py
```

常用参数：

```bash
python hand_skeleton_realtime.py --camera 0 --max-hands 2 --focal-length-px 950
```

## 3. 说明

- 绿色骨架：MediaPipe 当前帧识别结果
- 橙色骨架：根据上一帧与当前帧做的“下一帧预测”
- 距离估算：基于手掌宽度（index_mcp 到 pinky_mcp）与相机焦距的近似模型

> 距离是粗略值，若想更准，请对你的摄像头做标定并调整 `--focal-length-px`。

## 4. 退出

按 `q` 或 `ESC` 退出。
