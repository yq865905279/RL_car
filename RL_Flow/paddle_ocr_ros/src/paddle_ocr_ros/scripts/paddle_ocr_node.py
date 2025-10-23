#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from paddleocr import PaddleOCR
from paddleocr import TextRecognition
from paddle_ocr_ros.srv import OCRService, OCRServiceResponse, OCRServiceRequest
import json
from PIL import Image

class PaddleOCRServiceNode:
    def __init__(self):
        """初始化PaddleOCR服务节点"""
        rospy.init_node('paddle_ocr_service_node', anonymous=True)
        
        # 初始化cv_bridge
        self.bridge = CvBridge()
        
        # 服务统计信息
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        
        # 创建统计信息发布器
        self.stats_pub = rospy.Publisher('/ocr_service_stats', String, queue_size=10)

        # 读取并固定OCR初始化参数，仅在启动时初始化一次
        self.language = rospy.get_param('~language', 'ch')
        self.use_angle_cls = rospy.get_param('~use_angle_cls', True)

        rospy.loginfo("正在初始化PaddleOCR引擎...")
        self.ocr = TextRecognition(model_name="PP-OCRv5_mobile_rec")
        # self.ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.language)
        rospy.loginfo(f"PaddleOCR引擎初始化完成，语言: {self.language}, 角度分类器: {self.use_angle_cls}")

        # 创建OCR服务（在引擎初始化完成后再提供服务）
        self.ocr_service = rospy.Service('/ocr_service', OCRService, self.handle_ocr_request)
        
        rospy.loginfo("="*60)
        rospy.loginfo("PaddleOCR服务端已启动")
        rospy.loginfo("服务名称: /ocr_service")
        rospy.loginfo("等待远端客户端连接...")
        rospy.loginfo("="*60)
        
    def handle_ocr_request(self, req):
        """处理OCR服务请求"""
        start_time = rospy.Time.now()
        self.request_count += 1
        
        try:
            rospy.loginfo(f"[请求 #{self.request_count}] 收到远端OCR识别请求")
            rospy.loginfo(f"图像尺寸: {req.input_image.width}x{req.input_image.height}")
            
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(req.input_image, "bgr8")
            # 请求中携带的参数仅用于记录；引擎仅在启动时初始化一次
            language = req.language if req.language else self.language
            use_angle_cls = req.use_angle_cls if hasattr(req, 'use_angle_cls') else self.use_angle_cls
            rospy.loginfo(f"识别参数(请求) - 语言: {language}, 角度分类器: {use_angle_cls}")
            if language != self.language or use_angle_cls != self.use_angle_cls:
                rospy.logwarn(f"请求参数与已初始化参数不一致，将使用已初始化参数 - 语言: {self.language}, 角度分类器: {self.use_angle_cls}")
            
            # 使用PaddleOCR进行文字识别
            rospy.loginfo("正在进行文字识别...")
            # 将OpenCV BGR图像转换为PIL RGB图像
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.ocr.predict(rgb_image, batch_size=1)
            # 处理识别结果，统一为数组类型
            recognized_texts = []
            confidences = []
            if isinstance(results, dict):
                # 兼容 { 'res': {...} } 或直接扁平字典
                res_obj = results.get('res', results)
                if isinstance(res_obj, dict):
                    text_val = res_obj.get('rec_text', '')
                    score_val = res_obj.get('rec_score', 0.0)
                    if isinstance(text_val, str) and len(text_val) > 0:
                        recognized_texts.append(text_val)
                        try:
                            confidences.append(float(score_val))
                        except Exception:
                            confidences.append(0.0)
            elif isinstance(results, list) and len(results) > 0:
                first = results[0]
                if isinstance(first, dict):
                    text_val = first.get('rec_text', '')
                    score_val = first.get('rec_score', 0.0)
                    if isinstance(text_val, str) and len(text_val) > 0:
                        recognized_texts.append(text_val)
                        try:
                            confidences.append(float(score_val))
                        except Exception:
                            confidences.append(0.0)
            else:
                rospy.logwarn("Unrecognized TextRecognition result structure")

            
            # 计算处理时间
            processing_time = (rospy.Time.now() - start_time).to_sec()
            
            # 创建响应
            response = OCRServiceResponse()
            response.success = True
            # 使用 ASCII 安全消息，避免 ROS Noetic 序列化报错
            response.message = f"recognized_lines={len(recognized_texts)}, time={processing_time:.2f}s"
            response.texts = recognized_texts
            response.confidences = confidences
            response.bboxes = []
            response.total_text = ' '.join(recognized_texts)
            
            # 更新统计信息
            self.success_count += 1
            
            # 打印识别结果到控制台
            if recognized_texts:
                rospy.loginfo(f"✓ 识别成功 - 共 {len(recognized_texts)} 行文字:")
                for i, (text, conf) in enumerate(zip(recognized_texts, confidences)):
                    rospy.loginfo(f"  {i+1}. {text} (置信度: {conf:.2f})")
                rospy.loginfo(f"完整文本: {response.total_text}")
            else:
                rospy.loginfo("⚠ 未识别到任何文字")
                response.message = "未识别到任何文字"
            
            rospy.loginfo(f"处理完成，耗时: {processing_time:.2f}秒")
            self.publish_stats()
            
            return response
                
        except Exception as e:
            processing_time = (rospy.Time.now() - start_time).to_sec()
            rospy.logerr(f"✗ OCR服务处理错误: {str(e)}")
            
            # 更新错误统计
            self.error_count += 1
            
            # 返回错误响应
            response = OCRServiceResponse()
            response.success = False
            # ASCII-only error message
            response.message = f"ocr_failed: {str(e)}"
            response.texts = []
            response.confidences = []
            response.bboxes = []
            response.total_text = ""
            
            rospy.logerr(f"处理失败，耗时: {processing_time:.2f}秒")
            self.publish_stats()
            
            return response
    
    def publish_stats(self):
        """发布服务统计信息"""
        try:
            stats = {
                'timestamp': rospy.Time.now().to_sec(),
                'total_requests': self.request_count,
                'successful_requests': self.success_count,
                'failed_requests': self.error_count,
                'success_rate': (self.success_count / self.request_count * 100) if self.request_count > 0 else 0
            }
            
            stats_msg = String()
            stats_msg.data = json.dumps(stats, ensure_ascii=False)
            self.stats_pub.publish(stats_msg)
            
            # 每10个请求打印一次统计信息
            if self.request_count % 10 == 0:
                rospy.loginfo(f"📊 服务统计 - 总请求: {self.request_count}, 成功: {self.success_count}, 失败: {self.error_count}, 成功率: {stats['success_rate']:.1f}%")
                
        except Exception as e:
            rospy.logwarn(f"发布统计信息失败: {str(e)}")
    
    def run(self):
        """运行节点"""
        rospy.loginfo("服务端运行中，按 Ctrl+C 停止...")
        rospy.spin()

if __name__ == '__main__':
    try:
        node = PaddleOCRServiceNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("PaddleOCR服务节点已关闭")
    except Exception as e:
        rospy.logerr(f"节点运行错误: {str(e)}")
