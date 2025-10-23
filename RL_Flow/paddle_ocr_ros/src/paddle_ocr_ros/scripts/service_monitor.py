#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PaddleOCR服务监控脚本
监控服务状态和性能指标
"""

import rospy
import json
import time
from std_msgs.msg import String
from paddle_ocr_ros.srv import OCRService

class ServiceMonitor:
    def __init__(self):
        """初始化服务监控器"""
        rospy.init_node('paddle_ocr_service_monitor', anonymous=True)
        
        # 监控参数
        self.monitor_interval = rospy.get_param('~monitor_interval', 10.0)
        self.service_name = '/ocr_service'
        self.stats_topic = '/ocr_service_stats'
        
        # 监控数据
        self.last_stats = None
        self.service_available = False
        self.last_check_time = 0
        
        # 订阅统计信息
        self.stats_sub = rospy.Subscriber(self.stats_topic, String, self.stats_callback)
        
        rospy.loginfo("PaddleOCR服务监控器已启动")
        rospy.loginfo(f"监控间隔: {self.monitor_interval}秒")
        
    def stats_callback(self, msg):
        """处理统计信息回调"""
        try:
            self.last_stats = json.loads(msg.data)
            self.last_check_time = time.time()
        except Exception as e:
            rospy.logwarn(f"解析统计信息失败: {str(e)}")
    
    def check_service_health(self):
        """检查服务健康状态"""
        try:
            # 检查服务是否可用
            rospy.wait_for_service(self.service_name, timeout=2.0)
            self.service_available = True
            return True
        except rospy.ROSException:
            self.service_available = False
            return False
    
    def print_monitor_report(self):
        """打印监控报告"""
        print("\n" + "="*60)
        print("PaddleOCR服务监控报告")
        print("="*60)
        
        # 服务状态
        service_status = "🟢 正常" if self.service_available else "🔴 异常"
        print(f"服务状态: {service_status}")
        print(f"服务名称: {self.service_name}")
        
        # 统计信息
        if self.last_stats:
            stats = self.last_stats
            print(f"\n📊 服务统计:")
            print(f"  总请求数: {stats.get('total_requests', 0)}")
            print(f"  成功请求: {stats.get('successful_requests', 0)}")
            print(f"  失败请求: {stats.get('failed_requests', 0)}")
            print(f"  成功率: {stats.get('success_rate', 0):.1f}%")
            print(f"  最后更新: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats.get('timestamp', 0)))}")
        else:
            print("\n📊 服务统计: 暂无数据")
        
        # 系统信息
        print(f"\n🖥️ 系统信息:")
        print(f"  节点运行时间: {rospy.get_time():.1f}秒")
        print(f"  监控间隔: {self.monitor_interval}秒")
        
        print("="*60)
    
    def run(self):
        """运行监控器"""
        rospy.loginfo("开始监控PaddleOCR服务...")
        
        while not rospy.is_shutdown():
            try:
                # 检查服务健康状态
                self.check_service_health()
                
                # 打印监控报告
                self.print_monitor_report()
                
                # 等待下次检查
                rospy.sleep(self.monitor_interval)
                
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                rospy.logerr(f"监控过程中发生错误: {str(e)}")
                rospy.sleep(5.0)
        
        rospy.loginfo("服务监控器已停止")

if __name__ == '__main__':
    try:
        monitor = ServiceMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("服务监控器已关闭")
    except Exception as e:
        rospy.logerr(f"监控器运行错误: {str(e)}")
