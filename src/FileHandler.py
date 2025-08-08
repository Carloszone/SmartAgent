from .tool import timing_decorator
import os
import json
import time
import re

# 文本操作
from .ConfigManager import config
import hashlib
from .ContentHandler import file_loader
from typing import List, Optional, Union
from langchain_core.documents import Document
from weaviate.util import generate_uuid5

# 图像相关操作
import pytesseract
from PIL import Image
import cv2
import numpy as np


class FileProcessor:
    def __init__(self, file_loader=file_loader,
                 hash_algorithm: str='sha256',
                 buffer_size: int=65536,
                 encoding: str = 'utf-8'):
        self.file_loader = file_loader
        self.hash_algorithm = hash_algorithm
        self.buffer_size = buffer_size
        self.encoding = encoding
        self.file_type_setting = config.get_setting("files")["types"]  # 支持的文件类型信息
    
    def file_type_identifier(self, file_path) -> str:
        """
        对输入的文件进行类型区分的函数
        """
        # 根据传入的path类型决定处理方式
        if isinstance(file_path, dict):
            return "quesion-answer"  # 如果传入dict，视为问答对
        else:
            # 获取文件后缀
            file_extension = os.path.splitext(file_path)[1].lower()

            # 匹配预设的文件类型，并返回
            if file_extension in ["text_file_extension"]:
                return "text"
            elif file_extension in self.file_type_setting["document_file_extension"]:
                return "document"
            elif file_extension in self.file_type_setting["audio_file_extension"]:
                return "audio"
            elif file_extension in self.file_type_setting["table_file_extension"]:
                return "table"
            elif file_extension in self.file_type_setting["image_file_extension"]:
                return "image"
            else:
                return "other"

    def calculate_file_hash(self, file_path):
        """
        计算文件的hash值
        """
        try:
            hasher = hashlib.new(self.hash_algorithm)
        except ValueError as e:
            raise ValueError(f"无效的哈希算法: {self.hash_algorithm}。")
        
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(self.buffer_size):
                    hasher.update(chunk)
        except FileNotFoundError:
            raise FileNotFoundError(f"错误：文件未找到，路径为 '{file_path}'")
        except Exception as e:
            # 捕获其他可能的读取错误
            raise IOError(f"读取文件时出错: {e}")
        return hasher.hexdigest()

    def calculate_chunk_hash(self, chunk_string):
        """
        计算内容块的hash值
        """
        # 字符串转字节序列
        encoded_string = chunk_string.encode(self.encoding)

        # 新建hash对象
        try:
            hasher = hashlib.new(self.hash_algorithm)
        except ValueError as e:
            raise ValueError(f"无效的哈希算法: {self.hash_algorithm}")
        
        # 更新hash对象
        hasher.update(encoded_string)

        return hasher.hexdigest()

    @timing_decorator
    def load_file(self, user_id, file_id, file_name, file_path, output_dir):
        # 计算文件hash
        file_hash = self.calculate_file_hash(file_path)

        # 识别文件类型
        file_type = self.file_type_identifier(file_path)

        # 构建文本元数据
        file_metadata = {
            "user_id": user_id,
            "file_hash": file_hash,
            "file_id": file_id,
            "file_name": file_name,
            "file_path": file_path,
            "file_type": file_type
        }

        # 读取文件
        file_info = self.file_loader(file_path, output_dir)

        return {"file_info": file_info, "file_metadata": file_metadata}
    
    def clean_docs(self, 
                   docs: List[Document],
                   do_clear_space: bool = True,
                   do_clear_enter: bool = True,
                   do_clear_hyphen: bool = True,
                   do_clear_space_ch: bool = True,
                   do_sentence_fix: bool = True,
                   do_clear_page_tag: bool = True,
                   do_clear_noise: bool = True,
                   do_clear_unknown_letter: bool = True,
                   do_clear_spend_in_head_tail: bool = True
                   ) -> List[Document]:
        """
        文本清洗函数
        """
        new_docs = []
        for doc in docs:
            # 文档复制
            content = doc.page_content 
            # print(f'文档清洗前的原文： {content}')

            # (英文内容)替换多个空格为一个
            if do_clear_space:
                # print('合并空格')
                # print(f"输入内容：{content}")
                content = re.sub(r'[ \t]+', ' ', content)
            # print(f'合并空格后的原文： {content}')

            # (英文内容)替换多个换行符为一个
            if do_clear_enter:
                # print('合并换行符')
                # print(f"输入内容：{content}")
                content = re.sub(r'\n{3,}', '\n\n', content)
            # print(f'合并换行符后的原文： {content}')

            # (英文内容)去除文档中的连字符
            if do_clear_hyphen:
                # print('去除连字符')
                # print(f"输入内容：{content}")
                content = re.sub(r'-\s*\n', '', content)
            # print(f'去除连字符后的原文： {content}')

            # (中文内容)移除中文文本之间的多余空格
            if do_clear_space_ch:
                # print('删除中文文本之间的空格')
                # print(f"输入内容：{content}")
                content = re.sub(r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])', r'\1\2', content)
            # print(f'删除中文文本之间的空格后的原文： {content}')

            # 合并被错误切分的段落或句子
            if do_sentence_fix:
                # print('修复被错误切分的段落或句子')
                # print(f"输入内容：{content}")
                content = re.sub(r'(?<![。！？\n])\n', '', content)
                content = re.sub(r'\n+', '\n', content)

            # 去除页眉页脚和页码等噪音信息
            if do_clear_page_tag:
                # print('删除页眉页脚等信息')
                # print(f"输入内容：{content}")
                content = re.sub(r'(?i)第\s*\d+\s*页', '', content)
                content = re.sub(r'(?i)page\s*\d+', '', content)

            # 移除重复性的乱码噪声
            if do_clear_noise:
                # print("删除重复性乱码")
                # print(f"输入内容：{content}")
                noise_pattern = r"([\s`\\/’'\'V丶、()]+){5,}"
                content = re.sub(noise_pattern, ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()

            # 去除无法识别的乱码与字符
            if do_clear_unknown_letter:
                # print("统一文字与符号")
                # print(f"输入内容：{content}")
                allowed_chars = re.compile(
                    r'[^\u4e00-\u9fa5'  # 中日韩统一表意文字
                    r'a-zA-Z0-9'       # 字母和数字
                    r'\s'              # 空白符 (包括空格, \n, \t)
                    r'，。！？：；（）《》【】｛｝“”’、' # 中文标点
                    r',.?!:;()\[\]{}<>"\'~`@#$%^&*-_=+|\\/' # 英文标点和符号
                    r']'
                )
                content = allowed_chars.sub('', content)

            # 去除开头和结尾处的空白字符
            if do_clear_spend_in_head_tail:
                # print("删除开头和结尾的空白符")
                # print(f"输入内容：{content}")
                content = content.strip()

            # 构建一个新的Document对象
            new_doc = Document(page_content=content, metadata=doc.metadata)
            new_docs.append(new_doc)
        return new_docs

    def get_prompt_template(self):
        pass

    def calculate_uuid(self, docs: List[Document]) -> List[Document]:
        new_docs = []

        for doc in docs:
            doc_content = doc.page_content
            doc_text = doc_content.strip().lower()
            time_stamp = str(int(time.time()))
            generated_id = generate_uuid5(identifier=doc_text, namespace=time_stamp)
            doc.metadata["uuid"] = generated_id
            new_docs.append(doc)
        return new_docs
    
    def json_extractor(self, response: str):

        """
        用于从大模型返回的内容中提取json格式的函数
        :param content: 大模型返回的字符串
        :return:
        """
        # 解析和清洗模型返回
        raw_content = response['message']['content']
        content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

        # 转义\检查
        pattern = re.compile(r'\\(?![\\"\/bfnrtu])')
        content = pattern.sub(r'\\\\', content)

        # 第一次尝试：大模型返回了纯净的json字符串
        try: 
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 第二次尝试，大模型返回的是markdown代码块
        try:
            match = re.search(r'```(json)?\s*(\{.*\}|\[.*\])\s*```', content, re.DOTALL)
            if match:
                json_str = match.group(2)
                return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            pass

        # 第三次尝试，寻找第一个{和最后一个}
        try:
            # 尝试寻找JSON对象
            start_brace = content.find('{')
            end_brace = content.rfind('}')
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                json_str = content[start_brace : end_brace + 1]
                # print('3th :',json_str)
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # 第四次尝试，寻找第一个[和最后一个]
        try:
            start_bracket = content.find('[')
            end_bracket = content.rfind(']')
            if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
                json_str = content[start_bracket : end_bracket + 1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass 

        print(f"无法从输入中提取JSON数据")
        return content


class ImageObject:
    def __init__(self, image_path: str):
        self.image_path =  image_path
        self.image = Image.open(image_path)
    def preprocess_image_for_ocr(self, image, target_dpi=300) -> Image.Image:
        """
        放大图片，以满足ORC识别需要
        """
        if image is None:
            image = self.image

        # 获取当前DPI，如果不存在则默认为70
        current_dpi = image.info.get('dpi', (70, 70))[0]
        
        # 如果当前DPI已经很高，则无需处理
        if current_dpi >= target_dpi:
            return image
            
        # 计算放大比例
        scale_factor = target_dpi / current_dpi
        
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        
        # 使用高质量的重采样算法放大图片
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        return resized_image      

    def get_image_rotation(self, crop_num: int) -> int:
        """
        使用Tesseract OSD来检测图片中文字的方向，并返回使其摆正所需逆时针旋转的角度。
        """
        try:
            # 使用 Pillow 打开图片
            width, height = self.image.size

            # 处理图片
            new_image = self.image.crop((0, height // crop_num, width, height * (crop_num - 1) // crop_num))
            new_image =  self.preprocess_image_for_ocr(new_image)
            
            # 调用Tesseract的方向和文字脚本检测功能
            osd_data = pytesseract.image_to_osd(new_image)
            
            # 使用正则表达式查找 'Rotate' 后面的数字
            rotation_angle_match = re.search(r'(?<=Rotate: )\d+', osd_data)
            
            # 调整角度：Tesseract返回的是“需要顺时针旋转多少度”，后续旋转默认是逆时针旋转
            if rotation_angle_match:
                rotation_angle = int(rotation_angle_match.group(0))
                output_angle = (360 - rotation_angle) % 360
                return output_angle
            else:
                print('没有找到Rotate信息，可能图片为空或无法识别，默认不旋转')
                return 0
                
        except Exception as e:
            print(f"处理图片时出错: {e}")
            print(f'图片地址为:{self.image_path}')
            return 0

    def correct_image_rotation_and_save(self, rotation_angle: int, output_path: str):
        """
        根据给定的逆时针旋转角度，校正图片方向并保存。

        参数:
        - image_path (str): 原始图片的路径。
        - rotation_angle (int): 需要逆时针旋转的角度 (必须是 0, 90, 180, 270 之一)。
        - output_path (str): 校正后图片的保存路径。
        """
        if rotation_angle not in [0, 90, 180, 270]:
            print(f"错误：无效的旋转角度 {rotation_angle}。角度必须是 0, 90, 180, 270 之一。")
            return None

        try:
            # 如果角度为0，则无需旋转，可以直接保存或不进行任何操作
            if rotation_angle == 0:
                print("图片方向正常，无需旋转。")
                # 如果您希望无论如何都生成一个输出文件，可以取消下面这行注释
                # image.save(output_path)
                return None

            print(f"正在将图片逆时针旋转 {rotation_angle} 度...")
            corrected_image = self.image.rotate(rotation_angle, expand=True)

            # 保存校正后的图片
            corrected_image.save(output_path)
            print(f"校正后的图片已成功保存到: {output_path}")

        except Exception as e:
            print(f"处理图片时发生错误: {e}")




    

