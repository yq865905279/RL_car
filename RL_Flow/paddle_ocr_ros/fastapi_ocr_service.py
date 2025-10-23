#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from paddleocr import TextRecognition
from PIL import Image
import io
import time
import logging
from typing import List, Dict, Any
import uvicorn

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
app = FastAPI(title="PaddleOCR Service", description="OCR文字识别服务", version="1.0.0")

class OCRService:
    def __init__(self):
        """初始化OCR服务"""
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        
        logger.info("正在初始化模型...")
        self.ocr = TextRecognition(model_name="PP-OCRv5_mobile_rec")
        logger.info("模型初始化完成")

    def process_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """处理图像并返回OCR识别结果"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            logger.info(f"[请求#{self.request_count}] 收到OCR识别请求")
            
            # 将字节数据转换为PIL图像
            image = Image.open(io.BytesIO(image_bytes))
            
            # 转换为RGB格式（如果需要）
            if image.mode != 'RGB':
                image = image.convert('RGB')

            logger.info(f"图像尺寸: {image.size[0]}x{image.size[1]}")
            
            # 将PIL图像转换为numpy数组（PaddleOCR TextRecognition需要numpy数组）
            import numpy as np
            image_array = np.array(image)
            # 使用PaddleOCR进行文字识别
            logger.info("正在进行文字识别...")
            results = self.ocr.predict(image_array, batch_size=1)
            
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
                logger.warning("Unrecognized TextRecognition result structure")

            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 更新统计信息
            self.success_count += 1
            
            # 创建响应
            response = {
                "success": True,
                "message": f"recognized_lines={len(recognized_texts)}, time={processing_time:.2f}s",
                "texts": recognized_texts,
                "confidences": confidences,
                "total_text": ' '.join(recognized_texts),
                "processing_time": processing_time,
                "request_id": self.request_count
            }
            
            # 打印识别结果到控制台
            if recognized_texts:
                logger.info(f"识别成功 - 共 {len(recognized_texts)} 行文字:")
                for i, (text, conf) in enumerate(zip(recognized_texts, confidences)):
                    logger.info(f"{i+1}.{text} (置信度: {conf:.2f})")
                logger.info(f"完整文本: {response['total_text']}")
            else:
                logger.info("未识别到任何文字")
                response["message"] = "未识别到任何文字"
            
            logger.info(f"处理完成，耗时: {processing_time:.2f}秒")
            
            return response
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"✗ OCR服务处理错误: {str(e)}")
            
            # 更新错误统计
            self.error_count += 1
            
            # 返回错误响应
            response = {
                "success": False,
                "message": f"ocr_failed: {str(e)}",
                "texts": [],
                "confidences": [],
                "total_text": "",
                "processing_time": processing_time,
                "request_id": self.request_count
            }
            
            logger.error(f"处理失败，耗时: {processing_time:.2f}秒")
            
            return response

    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        return {
            "total_requests": self.request_count,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "success_rate": self.success_count / max(self.request_count, 1) * 100
        }

# 创建全局OCR服务实例
ocr_service = OCRService()

@app.post("/recognize")
async def recognize_text(file: UploadFile = File(...)):
    """
    OCR文字识别接口
    
    接受图像文件并返回识别的文字内容
    """
    # 检查文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="只支持图像文件")
    
    try:
        # 读取文件内容
        image_bytes = await file.read()
        # 处理图像
        result = ocr_service.process_image(image_bytes)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.get("/stats")
async def get_service_stats():
    """
    获取服务统计信息
    """
    return JSONResponse(content=ocr_service.get_stats())

@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {"status": "healthy", "service": "PaddleOCR FastAPI Service"}

@app.get("/")
async def root():
    """
    根路径，返回服务信息
    """
    return {
        "service": "PaddleOCR FastAPI Service",
        "version": "1.0.0",
        "endpoints": {
            "recognize": "POST /recognize - 上传图像进行OCR识别",
            "stats": "GET /stats - 获取服务统计信息",
            "health": "GET /health - 健康检查"
        }
    }

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("PaddleOCR FastAPI服务启动中...")
    logger.info("服务地址: http://127.0.0.1:8008")
    logger.info("识别接口: POST http://127.0.0.1:8008/recognize")
    logger.info("="*60)
    
    uvicorn.run(app, host="127.0.0.1", port=8008)
