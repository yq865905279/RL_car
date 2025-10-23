#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PaddleOCR服务客户端测试脚本
用于测试OCR服务的功能
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from paddle_ocr_ros.srv import OCRService, OCRServiceRequest
import sys
import os

class OCRClientTest:
    def __init__(self):
        """初始化OCR客户端"""
        rospy.init_node('ocr_client_test', anonymous=True)
        
        # 初始化cv_bridge
        self.bridge = CvBridge()
        
        # 等待服务可用
        rospy.loginfo("等待OCR服务...")
        rospy.wait_for_service('/ocr_service')
        self.ocr_service = rospy.ServiceProxy('/ocr_service', OCRService)
        rospy.loginfo("OCR服务已连接")
        
    def test_with_image_file(self, image_path, language='ch', use_angle_cls=True):
        """使用图像文件测试OCR服务"""
        try:
            # 读取图像文件
            if not os.path.exists(image_path):
                rospy.logerr(f"图像文件不存在: {image_path}")
                return False
                
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                rospy.logerr(f"无法读取图像文件: {image_path}")
                return False
            
            # 转换为ROS图像消息
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            
            # 创建服务请求
            request = OCRServiceRequest()
            request.input_image = ros_image
            request.language = language
            request.use_angle_cls = use_angle_cls
            
            rospy.loginfo(f"发送OCR请求 - 语言: {language}, 角度分类器: {use_angle_cls}")
            
            # 调用服务
            response = self.ocr_service(request)
            
            # 处理响应
            if response.success:
                rospy.loginfo(f"OCR识别成功: {response.message}")
                rospy.loginfo(f"识别到 {len(response.texts)} 行文字:")
                
                for i, (text, confidence) in enumerate(zip(response.texts, response.confidences)):
                    rospy.loginfo(f"  {i+1}. {text} (置信度: {confidence:.2f})")
                
                rospy.loginfo(f"完整文本: {response.total_text}")
                return True
            else:
                rospy.logerr(f"OCR识别失败: {response.message}")
                return False
                
        except Exception as e:
            rospy.logerr(f"测试过程中发生错误: {str(e)}")
            return False
    
    def test_with_camera(self, language='ch', use_angle_cls=True):
        """使用摄像头测试OCR服务"""
        try:
            # 初始化摄像头
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                rospy.logerr("无法打开摄像头")
                return False
            
            rospy.loginfo("摄像头已打开，按 'q' 退出，按 's' 进行OCR识别")
            
            while not rospy.is_shutdown():
                ret, frame = cap.read()
                if not ret:
                    rospy.logerr("无法读取摄像头数据")
                    break
                
                # 显示图像
                cv2.imshow('Camera Feed - Press s for OCR, q to quit', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 进行OCR识别
                    ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    
                    request = OCRServiceRequest()
                    request.input_image = ros_image
                    request.language = language
                    request.use_angle_cls = use_angle_cls
                    
                    rospy.loginfo("正在进行OCR识别...")
                    response = self.ocr_service(request)
                    
                    if response.success:
                        rospy.loginfo(f"识别结果: {response.total_text}")
                    else:
                        rospy.logerr(f"识别失败: {response.message}")
            
            cap.release()
            cv2.destroyAllWindows()
            return True
            
        except Exception as e:
            rospy.logerr(f"摄像头测试过程中发生错误: {str(e)}")
            return False
    
    def test_with_ros_image_topic(self, topic_name='/camera/image_raw', language='ch', use_angle_cls=True):
        """从ROS图像话题获取图像进行测试"""
        try:
            rospy.loginfo(f"等待图像话题: {topic_name}")
            
            # 等待图像话题
            msg = rospy.wait_for_message(topic_name, Image, timeout=10.0)
            
            # 创建服务请求
            request = OCRServiceRequest()
            request.input_image = msg
            request.language = language
            request.use_angle_cls = use_angle_cls
            
            rospy.loginfo("从ROS话题获取图像，正在进行OCR识别...")
            
            # 调用服务
            response = self.ocr_service(request)
            
            # 处理响应
            if response.success:
                rospy.loginfo(f"OCR识别成功: {response.message}")
                rospy.loginfo(f"识别到 {len(response.texts)} 行文字:")
                
                for i, (text, confidence) in enumerate(zip(response.texts, response.confidences)):
                    rospy.loginfo(f"  {i+1}. {text} (置信度: {confidence:.2f})")
                
                rospy.loginfo(f"完整文本: {response.total_text}")
                return True
            else:
                rospy.logerr(f"OCR识别失败: {response.message}")
                return False
                
        except rospy.ROSException as e:
            rospy.logerr(f"等待图像话题超时: {str(e)}")
            return False
        except Exception as e:
            rospy.logerr(f"ROS话题测试过程中发生错误: {str(e)}")
            return False

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python3 ocr_client_test.py image <图像路径> [语言] [角度分类器]")
        print("  python3 ocr_client_test.py camera [语言] [角度分类器]")
        print("  python3 ocr_client_test.py topic [话题名称] [语言] [角度分类器]")
        print("")
        print("参数说明:")
        print("  语言: ch(中文), en(英文), ch_en(中英文混合) - 默认: ch")
        print("  角度分类器: true/false - 默认: true")
        print("")
        print("示例:")
        print("  python3 ocr_client_test.py image /path/to/image.jpg ch true")
        print("  python3 ocr_client_test.py camera ch_en false")
        print("  python3 ocr_client_test.py topic /camera/image_raw")
        return
    
    test_type = sys.argv[1]
    client = OCRClientTest()
    
    # 解析参数
    language = sys.argv[3] if len(sys.argv) > 3 else 'ch'
    use_angle_cls = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else True
    
    success = False
    
    if test_type == 'image':
        if len(sys.argv) < 3:
            rospy.logerr("请提供图像文件路径")
            return
        image_path = sys.argv[2]
        success = client.test_with_image_file(image_path, language, use_angle_cls)
        
    elif test_type == 'camera':
        success = client.test_with_camera(language, use_angle_cls)
        
    elif test_type == 'topic':
        topic_name = sys.argv[2] if len(sys.argv) > 2 else '/camera/image_raw'
        success = client.test_with_ros_image_topic(topic_name, language, use_angle_cls)
        
    else:
        rospy.logerr(f"未知的测试类型: {test_type}")
        return
    
    if success:
        rospy.loginfo("测试完成")
    else:
        rospy.logerr("测试失败")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("客户端测试已关闭")
    except Exception as e:
        rospy.logerr(f"客户端测试错误: {str(e)}")
