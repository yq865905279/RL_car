# 机器人状态监控与OCR识别服务
这是一个完整的机器人状态监控系统，包含两个FastAPI服务：

1. **PaddleOCR服务** - 图像文字识别
2. **机器人状态监控服务** - 实时监控机器人运行状态

## 系统架构

```
Webots机器人
    ├── 图像识别请求 → FastAPI OCR服务 (端口8008)
    └── 状态数据发送 → 机器人状态监控服务 (端口8009)
```

## 快速启动

### 1. 启动PaddleOCR服务

```bash
cd ./paddle_ocr_ros
python fastapi_ocr_service.py
```

服务地址：http://127.0.0.1:8008

### 2. 启动机器人状态监控服务

```bash
cd ./robot_status_backend
python robot_status_service.py
```

服务地址：http://127.0.0.1:8009
监控仪表板：http://127.0.0.1:8009/

### 3. 运行机器人控制器

在Webots中运行：
```bash
cd ../rosbot_navigation
python test_model_webots.py
```

## 功能特性

### PaddleOCR服务 (端口8008)

- **POST /recognize** - 图像文字识别
- **GET /stats** - 服务统计信息
- **GET /health** - 健康检查

### 机器人状态监控服务 (端口8009)

- **POST /status** - 接收机器人状态更新
- **GET /status** - 获取当前机器人状态
- **GET /status/history** - 获取状态历史记录
- **GET /stats** - 获取服务统计信息
- **GET /** - 实时监控仪表板

## 机器人控制器功能

### 键盘控制
- **W** - 前进
- **S** - 后退
- **A** - 左转
- **D** - 右转
- **S** - 手动发送状态
- **C** - 拍照并发送OCR识别
- **Q** - 退出

### 自动功能
- 实时发送机器人状态到监控后端
- 显示摄像头图像
- 打印传感器数据

## 监控仪表板

访问 http://127.0.0.1:8009/ 可以看到：

- 📊 **服务统计** - 总更新次数、历史记录数、最后更新时间
- 🚀 **当前状态** - 机器人ID、线速度、角速度、详细速度信息
- 📈 **历史记录** - 状态变化历史表格，自动刷新

仪表板每5秒自动刷新，实时显示机器人状态。

## API使用示例

### 发送图像进行OCR识别

```python
import requests

# 发送图像文件
with open('image.jpg', 'rb') as f:
    files = {'file': ('image.jpg', f, 'image/jpeg')}
    response = requests.post('http://127.0.0.1:8008/recognize', files=files)
    result = response.json()
    print(f"识别结果: {result['total_text']}")
```

### 发送机器人状态

```python
import requests

status_data = {
    "linear_velocity": {"x": 0.5, "y": 0.0, "z": 0.0},
    "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.1},
    "robot_id": "robot_01",
    "status_message": "正在前进"
}

response = requests.post('http://127.0.0.1:8009/status', json=status_data)
result = response.json()
print(f"状态更新: {result['message']}")
```

## 数据格式

### 机器人状态数据结构

```json
{
    "timestamp": 1640995200.0,
    "linear_velocity": {"x": 0.5, "y": 0.0, "z": 0.0},
    "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.1},
    "position": {"x": 1.0, "y": 2.0, "z": 0.0},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.5, "w": 0.866},
    "battery_level": 85.5,
    "status_message": "正在执行任务",
    "robot_id": "robot_01"
}
```

### OCR识别结果结构

```json
{
    "success": true,
    "message": "recognized_lines=2, time=1.23s",
    "texts": ["识别到的文字1", "识别到的文字2"],
    "confidences": [0.95, 0.87],
    "total_text": "识别到的文字1 识别到的文字2",
    "processing_time": 1.23,
    "request_id": 1
}
```