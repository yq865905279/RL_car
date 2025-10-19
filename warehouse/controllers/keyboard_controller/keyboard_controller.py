#!/usr/bin/env python3
"""
Rosbot控制器 - 符合Webots R2023b标准
功能：
1. 正确初始化Rosbot的电机
2. 确保机器人初始时保持静止
3. 提供键盘控制功能
"""

from ast import Pass
from controller import Robot, Motor, Camera, Lidar, GPS, Compass
import numpy as np
import requests
import cv2
import base64

class RosbotController:
    def __init__(self):
        # 初始化机器人
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # 初始化电机 - 使用Webots文档中的正确名称
        self.init_motors()
        
        # 初始化传感器
        self.init_sensors()
        
        # 运动参数
        self.max_speed = 20.0  # 增加最大速度（Rosbot PROTO中定义的最大速度为26.0）
        self.current_left_speed = 0.0
        self.current_right_speed = 0.0
        
        print("RosbotController initialized!")
        print("==========================================")
        print("Control instructions:")
        print("W: Move forward")
        print("S: Move backward")
        print("A: Turn left")
        print("D: Turn right")
        print("Space: Stop")
        print("Q: Quit")
        print("C: Capture and send image for recognition")
        print("==========================================")
        
    def init_motors(self):
        """根据Rosbot PROTO文件初始化电机"""
        try:
            # Rosbot的电机名称（根据PROTO文件中的定义）
            motor_names = {
                'front_left': 'fl_wheel_joint',
                'rear_left': 'rl_wheel_joint',
                'front_right': 'fr_wheel_joint',
                'rear_right': 'rr_wheel_joint'
            }
            
            # 获取电机设备
            self.motors = {}
            for key, name in motor_names.items():
                motor = self.robot.getDevice(name)
                if motor is None:
                    print(f"Error: Motor not found: {name}")
                else:
                    # 设置位置为无穷大以启用速度控制
                    motor.setPosition(float('inf'))
                    # 初始速度设为0
                    motor.setVelocity(0.0)
                    self.motors[key] = motor
                    print(f"Motor initialized successfully: {name}")
                    
            if len(self.motors) == 4:
                print("All motors initialized")
            else:
                print(f"Warning: Only {len(self.motors)}/4 motors initialized successfully")
                
        except Exception as e:
            print(f"Error: Motor initialization failed: {e}")
    
    def init_sensors(self):
        """初始化传感器"""
        try:
            # 首先列出所有设备，帮助调试
            print("Listing all available devices:")
            n = self.robot.getNumberOfDevices()
            for i in range(n):
                device = self.robot.getDeviceByIndex(i)
                print(f"   Device {i}: {device.getName()} - Type: {device.getNodeType()}")
            
            # 彩色相机 - 根据Astra PROTO文件，RGB相机设备名称应该是"camera color"
            self.camera_color = self.robot.getDevice('camera color')
            if not self.camera_color:
                # 尝试其他可能的名称
                possible_names = ['camera color', 'camera rgb', 'camera', 'rgb']
                for name in possible_names:
                    print(f"   Trying to get camera: {name}")
                    self.camera_color = self.robot.getDevice(name)
                    if self.camera_color:
                        print(f"Found color camera: {name}")
                        break
            
            if self.camera_color:
                self.camera_color.enable(self.timestep)
                print(f"Color camera enabled")
                print(f"   Resolution: {self.camera_color.getWidth()}x{self.camera_color.getHeight()}")
                print(f"   Field of View: {self.camera_color.getFov()}")
                print(f"   Near plane: {self.camera_color.getNear()}")
            else:
                print("Error: Color camera not found")
                
            # 深度相机 - 根据Astra PROTO文件，深度相机设备名称应该是"camera depth"
            self.camera_depth = self.robot.getDevice('camera depth')
            if not self.camera_depth:
                # 尝试其他可能的名称
                possible_names = ['camera depth', 'camera range', 'depth']
                for name in possible_names:
                    print(f"   Trying to get depth camera: {name}")
                    self.camera_depth = self.robot.getDevice(name)
                    if self.camera_depth:
                        print(f"Found depth camera: {name}")
                        break
            
            if self.camera_depth:
                self.camera_depth.enable(self.timestep)
                print(f"Depth camera enabled")
                print(f"   Resolution: {self.camera_depth.getWidth()}x{self.camera_depth.getHeight()}")
                print(f"   Field of View: {self.camera_depth.getFov()}")
            else:
                print("Error: Depth camera not found")
                
            # 激光雷达 (RpLidar A2)
            self.lidar = self.robot.getDevice('lidar')
            if not self.lidar:
                # 尝试其他可能的名称
                possible_names = ['lidar', 'laser', 'rplidar', 'scanner']
                for name in possible_names:
                    print(f"   Trying to get lidar: {name}")
                    self.lidar = self.robot.getDevice(name)
                    if self.lidar:
                        print(f"Found lidar: {name}")
                        break
            
            if self.lidar:
                self.lidar.enable(self.timestep)
                self.lidar.enablePointCloud()
                print("Lidar enabled")
                print(f"   Horizontal Resolution: {self.lidar.getHorizontalResolution()}")
                print(f"   Field of View: {self.lidar.getFov()} radians")
            else:
                print("Error: Lidar not found")
                
            # GPS传感器
            self.gps = self.robot.getDevice('gps')
            if not self.gps:
                # 尝试其他可能的名称
                possible_names = ['gps', 'GPS']
                for name in possible_names:
                    print(f"   Trying to get GPS: {name}")
                    self.gps = self.robot.getDevice(name)
                    if self.gps:
                        print(f"Found GPS: {name}")
                        break
                        
            if self.gps:
                self.gps.enable(self.timestep)
                print("GPS enabled")
            else:
                print("Error: GPS not found")
                
            # IMU/指南针传感器
            self.imu = self.robot.getDevice('imu')
            if not self.imu:
                # 尝试其他可能的名称
                possible_names = ['imu', 'compass', 'inertial unit']
                for name in possible_names:
                    print(f"   Trying to get IMU/Compass: {name}")
                    self.imu = self.robot.getDevice(name)
                    if self.imu:
                        print(f"Found IMU/Compass: {name}")
                        break
                        
            if self.imu:
                self.imu.enable(self.timestep)
                print("IMU/Compass enabled")
            else:
                print("Error: IMU/Compass not found")
                
            # 触摸传感器（碰撞检测）
            self.touch_sensor = self.robot.getDevice('touch sensor')
            if not self.touch_sensor:
                # 尝试其他可能的名称
                possible_names = ['touch sensor', 'bumper', 'collision']
                for name in possible_names:
                    print(f"   Trying to get touch sensor: {name}")
                    self.touch_sensor = self.robot.getDevice(name)
                    if self.touch_sensor:
                        print(f"Found touch sensor: {name}")
                        break
                        
            if self.touch_sensor:
                self.touch_sensor.enable(self.timestep)
                print("Touch sensor enabled")
            else:
                print("Error: Touch sensor not found")
                
            # 距离传感器
            distance_sensor_names = ['fl_range', 'fr_range', 'rl_range', 'rr_range']
            self.distance_sensors = {}
            
            for name in distance_sensor_names:
                sensor = self.robot.getDevice(name)
                if sensor:
                    sensor.enable(self.timestep)
                    self.distance_sensors[name] = sensor
                    print(f"Distance sensor enabled: {name}")
            
            # 位置传感器
            position_sensor_names = {
                'front_left': 'front left wheel motor sensor',
                'rear_left': 'rear left wheel motor sensor',
                'front_right': 'front right wheel motor sensor',
                'rear_right': 'rear right wheel motor sensor'
            }
            
            self.position_sensors = {}
            for key, name in position_sensor_names.items():
                sensor = self.robot.getDevice(name)
                if sensor:
                    sensor.enable(self.timestep)
                    self.position_sensors[key] = sensor
                    print(f"Position sensor enabled: {name}")
                
        except Exception as e:
            print(f"Warning: Sensor initialization warning: {e}")
    
    def set_wheel_speeds(self, left_speed, right_speed):
        """设置轮子速度"""
        # 限制速度范围
        left_speed = max(min(left_speed, self.max_speed), -self.max_speed)
        right_speed = max(min(right_speed, self.max_speed), -self.max_speed)
        
        # 设置左侧轮子速度
        if 'front_left' in self.motors:
            self.motors['front_left'].setVelocity(left_speed)
        if 'rear_left' in self.motors:
            self.motors['rear_left'].setVelocity(left_speed)
            
        # 设置右侧轮子速度
        if 'front_right' in self.motors:
            self.motors['front_right'].setVelocity(right_speed)
        if 'rear_right' in self.motors:
            self.motors['rear_right'].setVelocity(right_speed)
            
        self.current_left_speed = left_speed
        self.current_right_speed = right_speed
    
    def stop_robot(self):
        """完全停止机器人"""
        self.set_wheel_speeds(0.0, 0.0)
        print("Robot stopped")
    
    def handle_keyboard(self):
        """处理键盘输入"""
        key = self.robot.getKeyboard().getKey()
        
        if key == ord('W') or key == ord('w'):  # 前进
            self.set_wheel_speeds(10.0, 10.0)  # 增加速度
            print("Moving forward")
        elif key == ord('S') or key == ord('s'):  # 后退
            self.set_wheel_speeds(-10.0, -10.0)  # 增加速度
            print("Moving backward")
        elif key == ord('A') or key == ord('a'):  # 左转
            self.set_wheel_speeds(-5.0, 5.0)  # 增加转弯速度
            print("Turning left")
        elif key == ord('D') or key == ord('d'):  # 右转
            self.set_wheel_speeds(5.0, -5.0)  # 增加转弯速度
            print("Turning right")
        elif key == ord(' '):  # 停止
            self.stop_robot()
        elif key == ord('C') or key == ord('c'):  # 识别图像
            self.send_image_for_recognition()
        elif key == ord('Q') or key == ord('q'):  # 退出
            return False
            
        return True
    
    def send_image_for_recognition(self, backend_url="http://127.0.0.1:8008/recognize"):
        """ captures an image, sends it to the backend, and prints the result."""
        if not hasattr(self, 'camera_color') or not self.camera_color:
            print("Error: Color camera not available.")
            return

        print("Capturing image...")
        image_data = self.camera_color.getImage()
        if not image_data:
            print("Warning: Could not get color camera image.")
            return

        width = self.camera_color.getWidth()
        height = self.camera_color.getHeight()

        # Convert to OpenCV format (BGRA to BGR)
        image = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Encode as JPEG for multipart upload
        _, buffer = cv2.imencode('.jpg', image_bgr)
        
        print(f"Sending image to {backend_url}...")
        try:
            # Send as multipart/form-data (file upload) instead of JSON
            files = {'file': ('image.jpg', buffer.tobytes(), 'image/jpeg')}
            response = requests.post(backend_url, files=files, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # Print the recognition result
            result = response.json()
            print(f"Recognition Result: {result}")
            
            # Print detailed OCR results if available
            if result.get('success') and result.get('texts'):
                print(f"✓ OCR recognition succeeded - Total {len(result['texts'])} text lines recognized:")
                for i, (text, conf) in enumerate(zip(result['texts'], result.get('confidences', []))):
                    print(f"  {i+1}. {text} (Confidence: {conf:.2f})")
                print(f"Full text: {result.get('total_text', '')}")
                print(f"Processing time: {result.get('processing_time', 0):.2f} seconds")
            elif result.get('success') == False:
                print(f"✗ OCR recognition failed: {result.get('message', 'Unknown error')}")

        except requests.exceptions.RequestException as e:
            print(f"Error sending image: {e}")

    def send_robot_status_env(self,action,status_backend_url="http://127.0.0.1:8009/status"):
        """发送机器人状态信息到监控后端"""
        try:
            # 构建状态数据
            status_data = {
                # "linear_velocity": {
                #     "x": (self.current_left_speed + self.current_right_speed) / 2.0 / self.max_speed,  # 归一化线速度
                #     "y": 0.0,
                #     "z": 0.0
                # },
                # "angular_velocity": {
                #     "x": 0.0,
                #     "y": 0.0,
                #     "z": (self.current_right_speed - self.current_left_speed) / (2.0 * self.max_speed)  # 归一化角速度
                # },
                "robot_id": "rosbot_01",
                "status_message": f"running - left:{action[0]:.1f}, right:{action[1]:.1f}"
            }
            
           
            # try:
            #     gps_values = env._get_sup_position()
            #     if gps_values and len(gps_values) >= 3:
            #         status_data["position"] = {
            #             "x": gps_values[0],
            #             "y": gps_values[1], 
            #             "z": gps_values[2]
            #         }
            #         "orientation": {
            #                 "x": 0.0,
            #                 "y": 0.0,
            #                 "z": 0.0,
            #                 "w": 1.0
            #             }
            # except:
            #     pass
            
            # 如果有IMU数据，添加姿态信息
            # if hasattr(self, 'imu') and self.imu:
            #     try:
            #         rpy_values = self.imu.getRollPitchYaw()
            #         if rpy_values and len(rpy_values) >= 3:
            #             status_data["orientation"] = {
            #                 "x": rpy_values[0],  # roll
            #                 "y": rpy_values[1],  # pitch
            #                 "z": rpy_values[2],  # yaw
            #                 "w": 1.0  # 简化的四元数w分量
            #             }
            #     except:
            #         pass
            
            # 发送状态数据
            response = requests.post(status_backend_url, json=status_data, timeout=2)
            
            if response.status_code == 200:
                result = response.json()
                if hasattr(self, 'status_send_success_count'):
                    self.status_send_success_count += 1
                else:
                    self.status_send_success_count = 1
                    
                # 每10次成功发送打印一次确认
                if self.status_send_success_count % 100 == 0:
                    print(f"send status {self.status_send_success_count},{result.get('message', '')}")
            else:
                print(f"send status failed: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            # 只在第一次失败或每50次失败时打印错误，避免刷屏
            if not hasattr(self, 'status_send_error_count'):
                self.status_send_error_count = 0
            self.status_send_error_count += 1
            
            if self.status_send_error_count == 1 or self.status_send_error_count % 50 == 0:
                print(f"send status error (#{self.status_send_error_count}): {e}")
        except Exception as e:
            print(f"send status error: {e}")

    def send_robot_status(self, status_backend_url="http://127.0.0.1:8009/status"):
        """发送机器人状态信息到监控后端"""
        try:
            # 构建状态数据
            status_data = {
                "linear_velocity": {
                    "x": (self.current_left_speed + self.current_right_speed) / 2.0 / self.max_speed,  # 归一化线速度
                    "y": 0.0,
                    "z": 0.0
                },
                "angular_velocity": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": (self.current_right_speed - self.current_left_speed) / (2.0 * self.max_speed)  # 归一化角速度
                },
                "robot_id": "rosbot_01",
                "status_message": f"running - left:{self.current_left_speed:.1f}, right:{self.current_right_speed:.1f}"
            }
            
            # 如果有GPS数据，添加位置信息
            if hasattr(self, 'gps') and self.gps:
                try:
                    gps_values = self.gps.getValues()
                    if gps_values and len(gps_values) >= 3:
                        status_data["position"] = {
                            "x": gps_values[0],
                            "y": gps_values[1], 
                            "z": gps_values[2]
                        }
                except:
                    pass
            
            # 如果有IMU数据，添加姿态信息
            if hasattr(self, 'imu') and self.imu:
                try:
                    rpy_values = self.imu.getRollPitchYaw()
                    if rpy_values and len(rpy_values) >= 3:
                        status_data["orientation"] = {
                            "x": rpy_values[0],  # roll
                            "y": rpy_values[1],  # pitch
                            "z": rpy_values[2],  # yaw
                            "w": 1.0  # 简化的四元数w分量
                        }
                except:
                    pass
            
            # 发送状态数据
            response = requests.post(status_backend_url, json=status_data, timeout=2)
            
            if response.status_code == 200:
                result = response.json()
                if hasattr(self, 'status_send_success_count'):
                    self.status_send_success_count += 1
                else:
                    self.status_send_success_count = 1
                    
                # 每10次成功发送打印一次确认
                if self.status_send_success_count % 100 == 0:
                    print(f"send status {self.status_send_success_count},{result.get('message', '')}")
            else:
                print(f"send status failed: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            # 只在第一次失败或每50次失败时打印错误，避免刷屏
            if not hasattr(self, 'status_send_error_count'):
                self.status_send_error_count = 0
            self.status_send_error_count += 1
            
            if self.status_send_error_count == 1 or self.status_send_error_count % 50 == 0:
                print(f"send status error (#{self.status_send_error_count}): {e}")
        except Exception as e:
            print(f"send status error: {e}")

    def print_status(self):
        """打印状态信息（每秒一次）"""
        if not hasattr(self, 'status_counter'):
            self.status_counter = 0
            
        self.status_counter += 1
        if self.status_counter >= 1000 // self.timestep:  # 每秒
            self.status_counter = 0
            
            # 打印IMU数据
            if hasattr(self, 'imu') and self.imu:
                try:
                    values = self.imu.getRollPitchYaw()
                    if values:
                        print(f"Attitude: Roll={values[0]:.2f}, Pitch={values[1]:.2f}, Yaw={values[2]:.2f}")
                except:
                    pass
                
            # 打印轮子位置传感器数据
            if hasattr(self, 'position_sensors') and len(self.position_sensors) > 0:
                try:
                    fl_pos = self.position_sensors.get('front_left', None)
                    if fl_pos:
                        print(f"Front left wheel position: {fl_pos.getValue():.2f}")
                except:
                    pass
                
            # 打印速度
            print(f"Speed: Left={self.current_left_speed:.2f}, Right={self.current_right_speed:.2f}")
            
            # 打印距离传感器数据
            if hasattr(self, 'distance_sensors') and len(self.distance_sensors) > 0:
                try:
                    distances = []
                    for name, sensor in self.distance_sensors.items():
                        if sensor:
                            distances.append(f"{name[-2:]}:{sensor.getValue():.2f}")
                    if distances:
                        print(f"Distances: {', '.join(distances)}")
                except:
                    pass
    
    def display_camera_images(self):
        """显示摄像头图像"""
        try:
            # 显示彩色相机图像
            if hasattr(self, 'camera_color') and self.camera_color:
                # 强制获取图像，即使不处理也会触发Webots显示
                image = self.camera_color.getImage()
                if image:
                    # 图像已成功获取，Webots会自动显示
                    #print("Color camera image acquired")
                    # 检查图像是否全黑
                    import numpy as np
                    width = self.camera_color.getWidth()
                    height = self.camera_color.getHeight()
                    np_image = np.frombuffer(image, np.uint8).reshape((height, width, 4))
                    if np.mean(np_image) < 5:  # 如果平均亮度很低，可能是全黑图像
                        print("Warning: Color camera image might be all black, please check camera position and orientation")
                else:
                    # If acquisition fails, re-enable the camera
                    print("Warning: Could not get color camera image, trying to re-enable")
                    self.camera_color.enable(self.timestep)
                    
            # 显示深度相机图像
            if hasattr(self, 'camera_depth') and self.camera_depth:
                # 强制获取深度图像，即使不处理也会触发Webots显示
                depth_image = self.camera_depth.getRangeImage()
                if depth_image:
                    pass
                    # 深度图像已成功获取，Webots会自动显示
                    #print("Depth camera image acquired")
                else:
                    # If acquisition fails, re-enable the camera
                    print("Warning: Could not get depth camera image, trying to re-enable")
                    self.camera_depth.enable(self.timestep)
                    
        except Exception as e:
            print(f"Warning: Error while displaying camera images: {e}")
            # 尝试重新初始化相机
            try:
                # 尝试重新获取相机设备
                if not hasattr(self, 'camera_color') or not self.camera_color:
                    # 尝试其他可能的名称
                    for name in ['camera color', 'camera rgb', 'camera', 'rgb']:
                        self.camera_color = self.robot.getDevice(name)
                        if self.camera_color:
                            self.camera_color.enable(self.timestep)
                            print(f"Color camera re-initialized: {name}")
                            break
                
                if not hasattr(self, 'camera_depth') or not self.camera_depth:
                    # 尝试不同的深度相机名称
                    for name in ['camera depth', 'camera range', 'depth']:
                        self.camera_depth = self.robot.getDevice(name)
                        if self.camera_depth:
                            self.camera_depth.enable(self.timestep)
                            print(f"Depth camera re-initialized: {name}")
                            break
            except Exception as e2:
                print(f"Warning: Error during camera re-initialization: {e2}")
    
    def run(self):
        """主循环"""
        # 启用键盘
        keyboard = self.robot.getKeyboard()
        keyboard.enable(self.timestep)
        
        # 确保开始时机器人停止
        self.stop_robot()
        
        print("\nController started!")
        print("Robot is stationary, use the keyboard to move.")
        print("Camera images should be displayed in the Webots window.")
        
        # 主控制循环
        while self.robot.step(self.timestep) != -1:
            # 处理键盘输入
            if not self.handle_keyboard():
                break
                
            # 显示摄像头图像
            self.display_camera_images()
                
            # 打印状态
            self.print_status()
            
            # 发送机器人状态到监控后端（每个时间步都发送）
            self.send_robot_status()
        
        # 退出前停止机器人
        self.stop_robot()
        print("\nController has exited")

# 主程序
if __name__ == "__main__":
    controller = RosbotController()
    controller.run()