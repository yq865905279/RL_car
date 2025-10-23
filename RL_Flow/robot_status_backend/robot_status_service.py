#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time
import logging
from typing import List, Dict, Any, Optional
import uvicorn
from datetime import datetime
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Robot Status Monitor", description="机器人状态监控服务", version="1.0.0")

# 数据模型
class RobotStatus(BaseModel):
    timestamp: Optional[float] = None
    linear_velocity: Dict[str, float]  # {"x": 0.0, "y": 0.0, "z": 0.0}
    angular_velocity: Dict[str, float]  # {"x": 0.0, "y": 0.0, "z": 0.0}
    position: Optional[Dict[str, float]] = None  # {"x": 0.0, "y": 0.0, "z": 0.0}
    orientation: Optional[Dict[str, float]] = None  # {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    battery_level: Optional[float] = None
    status_message: Optional[str] = None
    robot_id: Optional[str] = "robot_01"

class RobotStatusService:
    def __init__(self):
        """初始化机器人状态服务"""
        self.status_history: List[RobotStatus] = []
        self.current_status: Optional[RobotStatus] = None
        self.max_history_size = 1000  # 最多保存1000条历史记录
        self.total_updates = 0
        
        logger.info("机器人状态监控服务初始化完成")

    def update_status(self, status: RobotStatus) -> Dict[str, Any]:
        """更新机器人状态"""
        # 设置时间戳
        if not status.timestamp:
            status.timestamp = time.time()
        
        # 更新当前状态
        self.current_status = status
        self.total_updates += 1
        
        # 添加到历史记录
        self.status_history.append(status)
        
        # 限制历史记录大小
        if len(self.status_history) > self.max_history_size:
            self.status_history.pop(0)
        
        logger.info(f"收到机器人状态更新 #{self.total_updates} - "
                   f"线速度: ({status.linear_velocity['x']:.2f}, {status.linear_velocity['y']:.2f}, {status.linear_velocity['z']:.2f}), "
                   f"角速度: ({status.angular_velocity['x']:.2f}, {status.angular_velocity['y']:.2f}, {status.angular_velocity['z']:.2f})")
        
        return {
            "success": True,
            "message": f"状态更新成功 #{self.total_updates}",
            "timestamp": status.timestamp,
            "robot_id": status.robot_id
        }

    def get_current_status(self) -> Optional[RobotStatus]:
        """获取当前状态"""
        return self.current_status

    def get_status_history(self, limit: int = 100) -> List[RobotStatus]:
        """获取状态历史记录"""
        return self.status_history[-limit:] if limit > 0 else self.status_history

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.current_status:
            return {
                "total_updates": self.total_updates,
                "history_count": len(self.status_history),
                "last_update": None,
                "status": "no_data"
            }
        
        # 计算最后更新时间
        last_update_time = datetime.fromtimestamp(self.current_status.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        time_since_last = time.time() - self.current_status.timestamp
        
        return {
            "total_updates": self.total_updates,
            "history_count": len(self.status_history),
            "last_update": last_update_time,
            "time_since_last_update": f"{time_since_last:.1f}秒",
            "current_linear_speed": (
                self.current_status.linear_velocity['x']**2 + 
                self.current_status.linear_velocity['y']**2 + 
                self.current_status.linear_velocity['z']**2
            )**0.5,
            "current_angular_speed": (
                self.current_status.angular_velocity['x']**2 + 
                self.current_status.angular_velocity['y']**2 + 
                self.current_status.angular_velocity['z']**2
            )**0.5,
            "robot_id": self.current_status.robot_id,
            "status": "active" if time_since_last < 10 else "inactive"
        }

# 创建全局服务实例
robot_service = RobotStatusService()

@app.post("/status")
async def update_robot_status(status: RobotStatus):
    """
    更新机器人状态接口
    
    接收机器人的速度和状态信息
    """
    try:
        result = robot_service.update_status(status)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"更新机器人状态时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.get("/status")
async def get_current_status():
    """
    获取当前机器人状态
    """
    current = robot_service.get_current_status()
    if not current:
        return JSONResponse(content={"message": "暂无机器人状态数据"})
    
    return JSONResponse(content=current.dict())

@app.get("/status/history")
async def get_status_history(limit: int = 100):
    """
    获取机器人状态历史记录
    """
    history = robot_service.get_status_history(limit)
    return JSONResponse(content=[status.dict() for status in history])

@app.get("/stats")
async def get_service_stats():
    """
    获取服务统计信息
    """
    return JSONResponse(content=robot_service.get_statistics())

@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {"status": "healthy", "service": "Robot Status Monitor"}

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """
    机器人状态监控仪表板
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>机器人状态监控</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                text-align: center;
            }
            .card {
                background: white;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .status-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }
            .status-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 0;
                border-bottom: 1px solid #eee;
            }
            .status-item:last-child {
                border-bottom: none;
            }
            .status-label {
                font-weight: bold;
                color: #333;
            }
            .status-value {
                color: #666;
                font-family: monospace;
            }
            .active {
                color: #28a745;
            }
            .inactive {
                color: #dc3545;
            }
            .refresh-btn {
                background: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 10px 5px;
            }
            .refresh-btn:hover {
                background: #0056b3;
            }
            #history-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            #history-table th, #history-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            #history-table th {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 机器人状态监控仪表板</h1>
                <p>实时监控机器人运行状态和性能数据</p>
            </div>
            
            <div class="status-grid">
                <div class="card">
                    <h3>📊 服务统计</h3>
                    <div id="service-stats">
                        <div class="status-item">
                            <span class="status-label">加载中...</span>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🚀 当前状态</h3>
                    <div id="current-status">
                        <div class="status-item">
                            <span class="status-label">加载中...</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>📈 历史记录</h3>
                <button class="refresh-btn" onclick="loadHistory()">刷新历史</button>
                <button class="refresh-btn" onclick="clearHistory()">清空显示</button>
                <div style="max-height: 400px; overflow-y: auto;">
                    <table id="history-table">
                        <thead>
                            <tr>
                                <th>时间</th>
                                <th>线速度 (m/s)</th>
                                <th>角速度 (rad/s)</th>
                                <th>状态消息</th>
                            </tr>
                        </thead>
                        <tbody id="history-body">
                            <tr><td colspan="4">加载中...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            async function loadServiceStats() {
                try {
                    const response = await fetch('/stats');
                    const stats = await response.json();
                    
                    const statsDiv = document.getElementById('service-stats');
                    statsDiv.innerHTML = `
                        <div class="status-item">
                            <span class="status-label">总更新次数:</span>
                            <span class="status-value">${stats.total_updates}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">历史记录数:</span>
                            <span class="status-value">${stats.history_count}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">最后更新:</span>
                            <span class="status-value">${stats.last_update || '无数据'}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">距离上次更新:</span>
                            <span class="status-value">${stats.time_since_last_update || '无数据'}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">服务状态:</span>
                            <span class="status-value ${stats.status}">${stats.status === 'active' ? '活跃' : '非活跃'}</span>
                        </div>
                    `;
                } catch (error) {
                    console.error('加载服务统计失败:', error);
                }
            }

            async function loadCurrentStatus() {
                try {
                    const response = await fetch('/status');
                    const status = await response.json();
                    
                    const statusDiv = document.getElementById('current-status');
                    if (status.message) {
                        statusDiv.innerHTML = `<div class="status-item"><span class="status-label">${status.message}</span></div>`;
                        return;
                    }
                    
                    const linearSpeed = Math.sqrt(
                        status.linear_velocity.x**2 + 
                        status.linear_velocity.y**2 + 
                        status.linear_velocity.z**2
                    ).toFixed(3);
                    
                    const angularSpeed = Math.sqrt(
                        status.angular_velocity.x**2 + 
                        status.angular_velocity.y**2 + 
                        status.angular_velocity.z**2
                    ).toFixed(3);
                    
                    statusDiv.innerHTML = `
                        <div class="status-item">
                            <span class="status-label">机器人ID:</span>
                            <span class="status-value">${status.robot_id}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">线速度:</span>
                            <span class="status-value">${linearSpeed} m/s</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">角速度:</span>
                            <span class="status-value">${angularSpeed} rad/s</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">线速度详细:</span>
                            <span class="status-value">x:${status.linear_velocity.x.toFixed(3)}, y:${status.linear_velocity.y.toFixed(3)}, z:${status.linear_velocity.z.toFixed(3)}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">角速度详细:</span>
                            <span class="status-value">x:${status.angular_velocity.x.toFixed(3)}, y:${status.angular_velocity.y.toFixed(3)}, z:${status.angular_velocity.z.toFixed(3)}</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">状态消息:</span>
                            <span class="status-value">${status.status_message || '无'}</span>
                        </div>
                    `;
                } catch (error) {
                    console.error('加载当前状态失败:', error);
                }
            }

            async function loadHistory() {
                try {
                    const response = await fetch('/status/history?limit=50');
                    const history = await response.json();
                    
                    const tbody = document.getElementById('history-body');
                    if (history.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="4">暂无历史数据</td></tr>';
                        return;
                    }
                    
                    tbody.innerHTML = history.reverse().map(status => {
                        const time = new Date(status.timestamp * 1000).toLocaleString();
                        const linearSpeed = Math.sqrt(
                            status.linear_velocity.x**2 + 
                            status.linear_velocity.y**2 + 
                            status.linear_velocity.z**2
                        ).toFixed(3);
                        const angularSpeed = Math.sqrt(
                            status.angular_velocity.x**2 + 
                            status.angular_velocity.y**2 + 
                            status.angular_velocity.z**2
                        ).toFixed(3);
                        
                        return `
                            <tr>
                                <td>${time}</td>
                                <td>${linearSpeed}</td>
                                <td>${angularSpeed}</td>
                                <td>${status.status_message || '-'}</td>
                            </tr>
                        `;
                    }).join('');
                } catch (error) {
                    console.error('加载历史记录失败:', error);
                }
            }

            function clearHistory() {
                document.getElementById('history-body').innerHTML = '<tr><td colspan="4">已清空显示</td></tr>';
            }

            // 自动刷新
            function autoRefresh() {
                loadServiceStats();
                loadCurrentStatus();
                loadHistory();
            }

            // 初始加载
            autoRefresh();
            
            // 每5秒自动刷新
            setInterval(autoRefresh, 5000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("机器人状态监控服务启动中...")
    logger.info("服务地址: http://127.0.0.1:8009")
    logger.info("状态更新接口: POST http://127.0.0.1:8009/status")
    logger.info("监控仪表板: http://127.0.0.1:8009/")
    logger.info("="*60)
    
    uvicorn.run(app, host="127.0.0.1", port=8009)
