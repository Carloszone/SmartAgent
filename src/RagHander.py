# 目前没有合适的中文图文embedding模型，因此对图片的处理模式为：描述图片->描述文本embedding 所以不支持以图搜图

from .ConfigManager import config
import os
import re
import numpy as np
import time
from typing import List, Union, Optional
from langchain_core.documents import Document
from .ContentHandler import get_system_message, json_extractor, file_loader, html_to_json, split_text_by_paragraphs
from .DBHandler import VectorDatabase
from src.Models import RagModels
from weaviate.classes.query import MetadataQuery
from weaviate.util import generate_uuid5
import pandas as pd
import hashlib
import ollama
from funasr import AutoModel
import markdown
from bs4 import BeautifulSoup
import shutil
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from pathlib import Path
import warnings
import json
import gc
from .tool import standard_competition_ranking, get_chunk_text, timing_decorator
from tqdm import tqdm


class RAGTool:
    """
    执行RAG相关操作的类
    """
    def __init__(self, db_instance, model_instance):
        # 传入参数
        self.db_instance = db_instance
        self.model_instance = model_instance

        # 文档相关
        self.long_text_threshold = config.get_setting("files")["long_text_threshold"]  # 长文本的最低字符限制
        self.default_parag_per_chunk = config.get_setting("files")["default_parag_per_chunk"]  # 文本的默认分块段落数
        self.chunk_seq_threshold = config.get_setting("files")["chunk_seq_threshold"]  # 不分块的最大段落数
        self.file_types_category = config.get_setting("files")["types"]  # 当前支持的文档类型字典

        # 输出相关
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        project_root = os.path.dirname(script_dir)
        self.output_dir = os.path.join(project_root, config.get_setting("files")["output_dir"])

        # 其他参数：
        self.model_retry_num = 3  # 遇到模型报错时的重试次数
        self.concurrency_limit = 4  # 并发控制最大数

    ########################################################  模型操作  ########################################################
    def model_check(self, is_remote: bool) -> bool:
        """
        加载模型或模型服务客户端的函数
        """

        # 检查ollama模型是否存在
        print(f'开始检查ollama模型是否存在')
        model_list = [model["model"] for model in self.client.list()['models']]
        if not is_remote:
            model_names = self.model_settings.get("local_ollama_models")
        else:
            model_names = self.model_settings.get("remote_ollama_models")
            
        for key, value in model_names.items():
            if key[-5:] == 'model':
                if value in model_list:
                    print(f'{key}模型({value})存在')
                else:
                    print(f'{key}模型({value})不存在')
                    return False
        print(f'ollama模型检查完毕')

        # 加载音频识别模型
        print(f'开始加载音频模型')
        audio_model_name = self.model_settings["audio_model_address"]
        audio_vad_mode_address = self.model_settings["audio_vad_model_address"]
        try:
            self.audio_model = AutoModel(
                model=audio_model_name,
                vad_model=audio_vad_mode_address,
                vad_kwargs={"max_single_segment_time": 30000},
                device="cuda:0",
                disable_update = True,
            )
        except Exception as e:
            print(f'音频模型加载失败')
            return False
        print(f'音频模型加载成功')
        return True

    def unload_model(self, model_name: str):
        """
        在模型使用完毕后，卸载模型以释放资源
        """
        try:
            print(f"尝试手动卸载模型：{model_name}")
            self.client.generate(
                model=model_name,
                prompt='.',
                options={'keep_alive': 0}
            )
            print(f"✅ 成功发送卸载请求给模型: {model_name}")
        except Exception as e:
            # 如果模型本身就没加载，可能会收到一个错误，可以忽略
            print(f"⚠️ 在卸载模型 {model_name} 时出现异常: {e}")
            print("   (这通常是正常的，如果模型本来就没有被加载)")

    ######################################################  数据库操作函数  #####################################################
    def create_collection(self, collection_name: str, collection_type:str="knowledge_base", is_rebuild: bool=False):
        """
        创建数据表的函数
        """
        if is_rebuild:
            print(f'删除数据表{collection_name},并重建')
            delete_collection(db_client=self.db_client, collection_name=collection_name)
        
        create_collection(db_client=self.db_client, 
                          collection_name=collection_name, 
                          embedding_model_name=self.ollama_model_settings["text_embedding_model"],
                          api_endpoint=self.ollama_model_settings["weaviate_api_endpoint"],
                          collection_type=collection_type)
    
    def close_db(self):
        try:
            close_weavier_connection(self.db_client)
            print("成功端开和向量数据库的连接")
        except Exception as e:
            print(f"关闭向量数据库连接失败: {e}")
            raise
    
        
    ########################################################  文件处理函数  ########################################################
    def file_type_identifier(self, file_path: Union[str, dict]) -> str:
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
            if file_extension in self.file_types_category["text_file_extension"]:
                return "text"
            elif file_extension in self.file_types_category["document_file_extension"]:
                return "document"
            elif file_extension in self.file_types_category["audio_file_extension"]:
                return "audio"
            elif file_extension in self.file_types_category["table_file_extension"]:
                return "table"
            elif file_extension in self.file_types_category["image_file_extension"]:
                return "image"
            else:
                return "other"

    def text_clear_formatting(self, docs:List[Document]) -> List[Document]:
        """
        清除文档中的格式化信息,并生成Document对象
        """
        new_docs = []  # 储存清洗后的文档
        for doc in docs:
            meta = doc.metadata
            # 转为纯文本
            doc_text = doc.page_content
            html = markdown.markdown(doc_text)
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text()

            # 转为Document
            text_doc = Document(page_content=text, metadata=meta)

            # 保存对象
            new_docs.append(text_doc)

        return new_docs

    def clean_text(self, 
                   doc: List[Document],
                   do_clear_space: bool = True,
                   do_clear_enter: bool = True,
                   do_clear_hyphen: bool = True,
                   do_clear_space_ch: bool = True,
                   do_sentence_fix: bool = True,
                   do_clear_page_tag: bool = True,
                   do_clear_noise: bool = True,
                   do_clear_unknown_letter: bool = True,
                   do_clear_emoji: bool = False,
                   do_clear_spend_in_head_tail: bool = True
                   ) -> List[Document]:
        """
        文本清洗函数
        """
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
            content = re.sub(r'\n+', '\n', content)
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

        # 去除表情
        if do_clear_emoji:
            # print("删除表情符号")
            # print(f"输入内容：{content}")
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+",
                flags=re.UNICODE,
            )
            content = emoji_pattern.sub(r'', content)

        # 去除开头和结尾处的空白字符
        if do_clear_spend_in_head_tail:
            # print("删除开头和结尾的空白符")
            # print(f"输入内容：{content}")
            content = content.strip()

        # 构建一个新的Document对象
        new_doc = Document(page_content=content, metadata=doc.metadata)
        return new_doc

    def text_window_retrieval(self, docs: List[Document]) -> List[dict]:
        """
        利用滑动窗口技术进行内容拼接
        """
        page_lens = len(docs)
        window_docs = []
        for index in range(page_lens):
            if index == 0:
                if page_lens >= 2:
                    window_docs.append(
                        {
                            'target_content': docs[index],
                            'below_content': docs[index + 1]
                        }
                    )
                else:
                    window_docs.append(
                        {
                            'target_content': docs[index]
                        }
                    )
            elif index == page_lens - 1:
                window_docs.append(
                    {
                        'above_content': docs[index - 1],
                        'target_content': docs[index]
                    }
                )
            else:
                window_docs.append(
                    {
                        'above_content': docs[index - 1],
                        'target_content': docs[index],
                        'below_content': docs[index + 1]
                    }
                )
        return window_docs
    
    def add_info(self, docs: List[Document]) -> List[Document]:
        """
        对文本内容进行信息补充
        """
        output_docs = []
        for doc in docs:
            # 提取信息
            content = doc.page_content
            meta = doc.metadata
            title_info = meta.get('text_title', False)
            chapter_relation = meta.get("chapter_relation", False)

            # 补充信息
            if title_info:
                if chapter_relation:
                    content = "文本主题：" + title_info + "\n" + "章节关系：" + chapter_relation + "\n" + "文本内容：" + content
                else:
                    content = "文本主题：" + title_info + "\n" + "文本内容：" + content
            else:
                if chapter_relation:
                    content = "章节关系：" + chapter_relation + "\n" + "文本内容：" + content
            
            # 生成新的doc
            new_doc = Document(page_content=content, metadata=meta)
            output_docs.append(new_doc)
        return output_docs

    def add_uuid(self, contents: List[Document]) -> List[Document]:
        """
        """
        # 输入解析
        output_list = []
        for content in contents:
            input_content = content.page_content

            # 生成分块的哈希和uuid
            orignal_text = input_content.strip().lower()
            orignal_hash = hashlib.sha256(orignal_text.encode('utf-8')).hexdigest()
            generated_id = generate_uuid5(identifier=orignal_text, namespace=orignal_hash)
            content.metadata['hash'] = orignal_hash
            content.metadata['uuid'] = generated_id
            output_list.append(content)
        return output_list

    def text_chunk_splitter(self, content: Document) -> List[Document]:
        """
        文本分块函数(同步)
        """
        # 输入解析
        input_content = content.page_content
        chunk_metadata = content.metadata


        # message构建
        system_message = get_system_message('chunker')
        message = [
            {
                "role": "system",
                "content": system_message
             },
             {
                "role": "user",
                "content": f"请对以下文本进行分块：{input_content}" 
             }
        ]

        # 模型交互,针对503错误重试
        last_exception = None
        for attempt in range(self.model_retry_num):
            try:
                response = self.client.chat(model=self.ollama_model_settings["generative_model"], 
                                                        messages=message,
                                                        options=self.ollama_model_option)
            except ollama.ResponseError as e:
                if e.status_code == 503:
                    last_exception = e
                    wait_time = 2 * attempt
                    print(f'遇到503错误，等待{wait_time}秒后重试...')
                    time.sleep(wait_time)
                else:
                    return e
            except Exception as e:
                return e
 
        # 提取输出json
        raw_content = response['message']['content']
        clean_output = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        # print(f'文本分块原文: {input_content}')
        # print(f"***clean content: {clean_output}")
        try:
            output_json = json_extractor(clean_output)
            # print(f"***提取出的json对象为:{output_json}")

            # 返回格式化后的文档内容
            formatted_chunkers = []
            if "chunks" in output_json:
                output_list = output_json["chunks"]
            else:
                output_list = split_text_by_paragraphs(clean_output, self.default_parag_per_chunk)

        except Exception as e:
            print(f"文档分块的json输出提取失败, 将模型返回直接作为对象输出,按照默认段落值分块")
            output_list = split_text_by_paragraphs(clean_output, self.default_parag_per_chunk)

        for index, output_content in enumerate(output_list):
                    chunk_metadata['chunk_seq_id'] = index
                    chunk_document = Document(page_content=output_content, metadata=chunk_metadata)
                    formatted_chunkers.append(chunk_document)
        return formatted_chunkers

    def text_summary(self, contents: List[Document]) -> List[Document]:
        # 输入解析
        new_docs = []
        for content in contents:
            input_content = content.page_content

            # 构建message
            if len(input_content) >= 20:
                # print('构建message')
                system_message = get_system_message('text_summary')
                message = [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": f"请对以下文本进行概括总结：{input_content}" 
                    }
                ]

                # 模型交互,针对503错误重试
                for attempt in range(self.model_retry_num):
                    try:
                        response = self.client.chat(model=self.ollama_model_settings["generative_model"], 
                                                                messages=message,
                                                                options=self.ollama_model_option)
                    except ollama.ResponseError as e:
                        print('遭遇503错误：', e)
                        if e.status_code == 503:
                            wait_time = 2 * attempt
                            print(f'遇到503错误，等待{wait_time}秒后重试...')
                            time.sleep(wait_time)
                        else:
                            return e
                    except Exception as e:
                        print('遇到错误:', e)
                        return e

                # 提取输出json
                # print('提取json', response)
                raw_content = response['message']['content']
                clean_output = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
                # print(f'原文：{input_content}')
                # print(f"概括内容：{clean_output}")
                try:
                    summary_json = json_extractor(clean_output)
                except Exception as e:
                    summary_json = {"summary": clean_output}

                if "summary" not in summary_json:
                    summary_json = {"summary": input_content}

            else:
                summary_json = {"summary": input_content}

            # 构建doc
            new_doc = Document(page_content=summary_json["summary"], metadata=content.metadata)
            new_docs.append(new_doc)
        return new_docs

    def image_caption(self, image_path:str, sys_message:str=None, user_content: str='请尽可能详尽地描述传入的图片'):
        # 构建图片描述信息
        # print('开始图片描述')
        if sys_message:
            system_message = sys_message
        else:
            system_message = get_system_message('image_caption')
        message = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_content,
                "images": [Path(image_path)]
            }
        ]
        # print(message)

        # 模型交互,针对503错误重试
        for attempt in range(self.model_retry_num):
            try:
                response = self.client.chat(model=self.ollama_model_settings["image_caption_model"],
                                            messages=message,
                                            options=self.ollama_model_option)
            except ollama.ResponseError as e:
                if e.status_code == 503:
                    wait_time = 2 * attempt
                    print(f'遇到503错误，等待{wait_time}秒后重试...')
                    time.sleep(wait_time)
                else:
                    print(f'图片描述模型失败 {e}')
                    return ""
            except Exception as e:
                print(f'图片描述模型失败 {e}')
                return ""
 
        # 提取输出json
        # print('解析大模型输出')
        raw_content = response['message']['content']
        # print(raw_content)
        clean_output = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        # print('clean output', clean_output)
        try:
            # print(f'开始提取json')
            output_content = json_extractor(clean_output)
            # print(f'***图像描述：{output_content}')
            return output_content
        except Exception as e:
            print("json提取失败")
            return {"image_caption": clean_output}

    def search_extract_keywords(self, request: str):
        """
        提取request中的关键词的函数
        """
        # 创建message
        system_message = get_system_message('keywords')
        message = [
            {
                "role": "system",
                "content": system_message
             },
             {
                "role": "user",
                "content": f"请求信息的内容如下:{request}" 
             }
        ]

        # 模型交互
        response = self.client.chat(model=self.ollama_model_settings["generative_model"], 
                                                messages=message,
                                                options=self.ollama_model_option)
        
        # 提取输出json
        raw_content = response['message']['content']
        clean_output = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        output_keywords = json_extractor(clean_output)
        keyswords = ''
        for keyword in output_keywords["keywords"]:
            keyswords = keyswords + keyword + ' '

        if len(keyswords) == 0:
            warnings.warn(message=f"请求{request}提取关键词失败，关键词检索的结果可能不准确")
        return keyswords

    def save_knowledge_data_generator(self, contents:List[dict]):
        """
        生成用于知识库向量数据库的dict对象
        """
        data_dict = {}
        for content in contents:
            # print('知识库元素:', content)
            # 文本内容
            content_text = content["content"].page_content
            summary_text = content["summary"].page_content
            fusion_text = content['content_for_rerank'].page_content

            # 元数据：来自文件
            metadata = content["content"].metadata
            uuid = metadata.get("uuid")
            source = metadata.get("source")
            file_type = metadata.get("file_type", "")
            page_number = metadata.get("page_number", "-1")
            chunk_seq_id = metadata.get("chunk_seq_id", -1)
            access_level = metadata.get("access_level", 0)
            chunk_type = metadata.get("chunk_type", "")
            content_id = metadata.get("content_id", -1)
            chapter_info = metadata.get("chapter_info", "")

            # 组合字段
            if uuid not in data_dict.keys():
                data_dict[uuid] = {"properties": {
                                       "content": content_text, 
                                       "summary": summary_text,
                                       "content_for_rerank": fusion_text,
                                       "source": source,
                                       "file_type": file_type,
                                       "chunk_type": chunk_type,
                                       "access_level": access_level,
                                    #    "page_number": page_number,
                                       "chunk_seq_id": chunk_seq_id,
                                       "content_id": content_id,
                                       "chapter_info": chapter_info
                                   }
                                }
        return data_dict

    def image_description_fusion(self, image_description:str, image_content: str, image_text: str=""):
        """
        基于图片的描述信息，文字信息和上下文，概括图片的内容
        """
        # 构建message
        system_message = get_system_message('image_description')
        message = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": f"请结合以下信息，综合概括图片描述的内容。模型识别出的图片内容{image_description}, 图像的上下文的文本内容{image_content}" 
            }
        ]

        # 模型交互,针对503错误重试
        last_exception = None
        for attempt in range(self.model_retry_num):
            try:
                response = self.client.chat(model=self.ollama_model_settings["generative_model"], 
                                                        messages=message,
                                                        options=self.ollama_model_option)
            except ollama.ResponseError as e:
                if e.status_code == 503:
                    last_exception = e
                    wait_time = 2 * attempt
                    print(f'遇到503错误，等待{wait_time}秒后重试...')
                    time.sleep(wait_time)
                else:
                    return e
            except Exception as e:
                return e
 
        # 提取输出json
        try:
            output_content = json_extractor(response['message']['content'])
            return output_content['description']
        except Exception as e:
            return e

    def keyword_generator(self, request: str):
        """
        提取request中的关键词的函数
        """
        # 创建message
        system_message = get_system_message('keywords')
        message = [
            {
                "role": "system",
                "content": system_message
             },
             {
                "role": "user",
                "content": f"请求信息的内容如下:{request}" 
             }
        ]

        # 模型交互
        response = self.client.chat(model=self.ollama_model_settings["generative_model"], 
                                    messages=message,
                                    options=self.ollama_model_option)
        
        # 提取输出json
        try:
            raw_content = response['message']['content']
            clean_output = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            output_keywords = json_extractor(clean_output)
            return output_keywords
        except Exception as e:
            print('提取关键字失败，返回空列表')
            return []

    def hyde_answer_generator(self,request: str):
        """
        用于生成hyde模型的输入的函数
        """
        # 构建message
        system_message = get_system_message('hyde')
        message = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": f"用户请求信息：{request}" 
            }
        ]

        # 模型交互,针对503错误重试
        for attempt in range(self.model_retry_num):
            try:
                response = self.client.chat(model=self.ollama_model_settings["generative_model"], 
                                                        messages=message,
                                                        options=self.ollama_model_option)
            except ollama.ResponseError as e:
                if e.status_code == 503:
                    wait_time = 2 * attempt
                    print(f'遇到503错误，等待{wait_time}秒后重试...')
                    time.sleep(wait_time)
                else:
                    return e
            except Exception as e:
                return e
            
        # 提取输出json
        try:
            output_content = json_extractor(response['message']['content'])
            return output_content['answer']
        except Exception as e:
            print('提取hyde回答失败，返回原始文本')
            return request

    ###################################################### 整合函数  ######################################################
    def document_content_parser_fusion(self, parse_result: dict, metadata: dict):
        """
        将document对象的解析结果进行处理和融合的函数
        以解析得到的一级标题作为首选docs标准。对于没有一级标题的对象，以页码+滑动段落的方式进行分页
        """
        # 提取解析信息
        content_json_path = parse_result.get("content_json_path")
        temp_dir_path = parse_result.get("temp_dir_path")

        # 获取内容json
        with open(content_json_path, 'r', encoding='utf-8') as f:
                content_json = json.load(f)

        # 遍历json，提取table和image内容， 并判断docs模式
        print('解析文档结构...')
        image_chunks = []
        table_chunks = []
        page_cut_mode = 0  # docs模式。0：基于页码滑动docs； 其他：基于标题docs
        for chunk_idx, chunk in enumerate(content_json):
            chunk_type = chunk["type"]
            chunk_text_level = chunk.get("text_level", 0) 
            page_cut_mode = max(page_cut_mode, chunk_text_level)
            if chunk_type == 'image':
                img_caption = ""
                img_footnote = ""
                if len(chunk["img_caption"]):
                    for caption in chunk["img_caption"]:
                        img_caption = img_caption + caption + '\n'
                if len(chunk["img_footnote"]):
                    for footnote in chunk["img_footnote"]:
                        img_footnote = img_footnote + footnote + '\n'
                
                image_path = os.path.join(temp_dir_path, chunk["img_path"])
                if self.file_type_identifier(image_path) == 'image':
                    image_chunks.append({
                        "element_path": image_path,
                        "img_caption": img_caption,
                        "img_footnote": img_footnote,
                        "chunk_id": chunk_idx
                    })
            if chunk_type == 'table':
                table_caption = ""
                table_footnote = ""
                if len(chunk["table_caption"]):
                    for caption in chunk["table_caption"]:
                        table_caption = table_caption + caption + '\n'
                if len(chunk["table_footnote"]):
                    for footnote in chunk["table_footnote"]:
                        table_footnote = table_footnote + footnote + '\n'

                image_path = os.path.join(temp_dir_path, chunk["img_path"])
                if self.file_type_identifier(image_path) != 'image':
                    image_path = None

                table_chunks.append({
                    "element_content": get_chunk_text(chunk=chunk, chunk_type=chunk_type),
                    "image_path": image_path,
                    "table_caption": table_caption,
                    "table_footnote": table_footnote,
                    "chunk_id": chunk_idx
                })
        
        # 提取得到的表格和图片信息
        print('提取图片信息')
        image_docs = self.image_pipeline(image_chunks)
        print('图片信息提取完毕')

        print('提取表格信息')
        table_docs = self.table_pipeline(table_chunks)
        print('表格信息提取完毕')

        # 更新文档结构信息
        for chunk_info, doc in zip(image_chunks, image_docs):
            chunk_id = chunk_info["chunk_id"]
            content_json[chunk_id]["type"] = 'text'
            content_json[chunk_id]["text"] = doc.page_content
            content_json[chunk_id]["chunk_type"] = "image"

        for chunk_info, doc in zip(table_chunks, table_docs):
            chunk_id = chunk_info["chunk_id"]
            content_json[chunk_id]["type"] = 'text'
            content_json[chunk_id]["text"] = doc.page_content
            content_json[chunk_id]["chunk_type"] = "table"

        # 遍历content_json并Document化
        print('将解析后的文档元素转化为docs')
        level_text_manager = {}
        record_page_num = None
        page_content = ''  # 文档的正文内容
        output_docs = []
        start_text_id = -1  # 页面第一段文本的chunk id
        text_pages = []
        chapter_relation = ''  # 初始化标题关系
        chunk_seq_id = 1
        for chunk_idx, chunk in enumerate(content_json):
            # 提取每一个文本块的信息
            print("chunk:", chunk)
            chunk_type = chunk["type"]  # 文本块的类型
            chunk_page_num = chunk["page_idx"]  # 文本块所属的页码
            chunk_text_level = chunk.get("text_level", -1)  # 文本块的级别，正文为-1, 类正文为0, 多级标题为级别数
            chunk_chunk_type = chunk.get("chunk_type", 'text')
            chunk_text = chunk.get("text")
            text_pages.append(chunk_page_num)

            if chunk_text_level == -1:  # 对于正文文本，进行拼接
                page_content = page_content + chunk_text + '\n'
            else:  # 对于标题信息
                # 首先判断是否需要生成doc
                if len(page_content) > 0 or chunk_idx == len(content_json) -1:  # 已经有正文内容，生成doc
                    # 准备元数据
                    if chunk_seq_id > self.chunk_seq_threshold:
                        need_chunk_flag = True
                    else:
                        need_chunk_flag = False

                    doc_metadata = {
                        "file_type": "document",
                        "page_number": str(list(set(text_pages))),
                        "chunk_type": chunk_chunk_type,
                        "text_title": level_text_manager.get("text_title", ""),
                        "chapter_relation": chapter_relation,
                        "need_chunk": need_chunk_flag,
                        "chunk_seq_id": chunk_seq_id,
                        "content_id": chunk_idx
                        } | metadata

                    # 生成doc
                    new_doc = Document(page_content=page_content, metadata=doc_metadata)
                    output_docs.append(new_doc)
                    chunk_seq_id += 1

                    # 重置page_content
                    page_content = ""

                    # 记录标题章节信息
                    if 'text_title' not in level_text_manager:
                        level_text_manager['text_title'] = chunk_text
                    else:
                        level_text_manager[str(chunk_text_level)] = chunk_text
                        # 当上级章节更新时，重置下级章节的内容
                        if chunk_text_level < page_cut_mode:
                            for i in range(chunk_text_level+1, page_cut_mode+1):
                                level_text_manager[str(i)] == ''
                else:  
                    # 记录和更新标题信息
                    if 'text_title' not in level_text_manager:
                        level_text_manager['text_title'] = chunk_text
                    else: # 拼接同一个级别的内容
                        if str(chunk_text_level) not in level_text_manager:
                            level_text_manager[str(chunk_text_level)] = ''
                        else:
                            level_text_manager[str(chunk_text_level)] =  level_text_manager[str(chunk_text_level)] + '-' + chunk_text
                        # 当上级章节更新时，重置下级章节的内容
                        if chunk_text_level < page_cut_mode:
                            for i in range(chunk_text_level+1, page_cut_mode+1):
                                level_text_manager[str(i)] == ''
                
                # 更新当前的标题关系信息
                chapter_relation = ''
                for i in range(1, page_cut_mode+1):
                    if str(i) in level_text_manager:
                        chapter_relation = chapter_relation + level_text_manager[str(i)] + '-'
                    else:
                        chapter_relation = chapter_relation + '-'
                chapter_relation = chapter_relation[:len(chapter_relation)-1] + '\n'

        return output_docs
    
    def docs_pipeline(self,
                      docs: Optional[Document] = None, 
                      do_text_clean: bool=True, 
                      do_text_summary: bool=True) -> List[dict]:
        """
        文本处理流水线函数(同步)
        """
        # 文本去格式化(markdown转text)
        print(f'开始对文本去格式化')
        processed_contents = self.text_clear_formatting(docs)
        # processed_contents = docs

        output_list = []
        for processed_content in processed_contents:
            # print(f'doc的主题：{processed_content.metadata.get("text_title", "")}')
            # print(f'doc的章节信息：{processed_content.metadata.get("chapter_relation", "")}')
            # print(f'doc的内容：{processed_content.page_content}')
            do_text_chunk = processed_content.metadata.get("need_chunk", True)

            # 文本清洗
            if do_text_clean:
                print(f'执行文本清洗')
                # print(f'输入的变量信息：{processed_contents}')
                processed_content = self.clean_text(processed_content)

            # 文本分块
            if do_text_chunk:
                print(f'执行文本分块**')
                # print(f'文本分块 输入的变量信息：{processed_contents}')
                processed_contents = self.text_chunk_splitter(processed_content)
                print(f'文本分块后的待处理文本对象数量:{len(processed_contents)}')
                # print(f'文本分块 输出的变量信息：{processed_contents}')
                gc.collect()
            else:
                processed_contents = [processed_content]

            # 为分块处理后的doc添加uuid
            print("添加uuid")
            # print(f'输入的变量信息：{processed_contents}')
            processed_contents = self.add_uuid(processed_contents)

            # 内容补完
            print('文本补完')
            fusion_contents = self.add_info(processed_contents)
            summary_contents = Document(page_content='', metadata={})

            # 文本概括
            if do_text_summary:
                print(f'执行文本概括')
                # print(f'输入的变量信息：{processed_contents}')
                summary_contents = self.text_summary(fusion_contents)
                gc.collect()

            # 拼接输出字典
            for processed_content, summary_content, fusion_content in zip(processed_contents, summary_contents, fusion_contents):
                output_list.append(
                    {
                        "content": processed_content,
                        "summary": summary_content,
                        "content_for_rerank": fusion_content
                    }
                )

            #返回结果
        return output_list
            
    def table_pipeline(self, file_info: List[dict]) -> List[Document]:
        """
        表格文件/对象的处理管道
        """
        table_docs = []
        for f_info in tqdm(file_info):
            # print('f_info', f_info)
            table_path = f_info.get("element_path", None)
            if table_path:
                table_text = file_loader(table_path)
            else:
                table_text = f_info["element_content"]

            # 表格内容拼接
            table_caption = f_info.get("table_tile", "")
            table_footnote = f_info.get("table_footnote", "")
            if len(table_caption):
                table_text = "表格标题：" + "\n" + table_caption + '\n' + "表格内容:" + '\n' + table_text + '\n'
            if len(table_footnote):
                table_text = table_text + '\n' + "表格尾注信息：" + "\n" + table_footnote + '\n'

            # 读取表格图片（如有）
            image_path = f_info.get("image_path", None)
            print(image_path)
            if image_path:
                print('识别到图表图片，尝试利用图片解析图表')
                # 提取模型交互的sytem_message
                system_message = get_system_message('table_description_1')
                time_point_3 = time.time()
                table_description = self.image_caption(image_path=image_path)
                # table_description = self.image_caption(image_path=image_path, 
                #                                        sys_message=system_message,
                #                                        user_content=table_text)
                time_point_4 = time.time()
                print(f'表格图片解析的耗时为：{time_point_4 - time_point_3}s')
                table_content = table_description["image_caption"]
                print('表格解析结果：', table_description)
            else:
                print('直接解析图表')
                # 提取模型交互的sytem_message
                system_message = get_system_message('table_description_2')
                # table_description = self.image_caption(image_path=image_path, 
                #                                        sys_message=system_message,
                #                                        user_content=table_text)

            
            # 更新元数据信息
            table_doc = Document(page_content=table_content, metadata={})
            table_doc.metadata["source"] = f_info.get("source", "")
            table_doc.metadata["file_type"] = f_info.get("file_type", "")
            table_doc.metadata["access_level"] = f_info.get("access_level")
            table_doc.metadata['page_number'] = f_info.get("page_num", "-1")
            table_doc.metadata["chunk_type"] = 'table'
            table_docs.append(table_doc)
        return table_docs

    def audio_pipeline(self, file_info: List[dict]) -> List[Document]:
        """
        音频文件的处理管道
        """
        audio_docs = []
        for f_info in file_info:
            audio_path = file_info['element_path']
            audio_response = self.audio_model.generate(
                input= audio_path,
                cache={},
                language="auto",  
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,  
                merge_length_s=15,
            )
            audio_text = rich_transcription_postprocess(audio_response[0]["text"])
            print(f'音频文本:{audio_text}')

            # 生成doc
            audio_doc = Document(page_content=audio_text, metadata={})

            # 更新元数据信息
            audio_doc.metadata["source"] = f_info.get("source", "")
            audio_doc.metadata["file_type"] = f_info.get("file_type", "")
            audio_doc.metadata["access_level"] = f_info.get("access_level")
            audio_doc.metadata["chunk_type"] = 'audio'
            audio_doc.metadata['page_number'] = f_info.get("page_num", "-1")
            audio_docs.append(audio_doc)
        
        return audio_docs

    def image_pipeline(self, file_info:List[dict]) -> List[Document]:
        """
        图像处理管道函数
        """
        image_docs = []
        for f_file in tqdm(file_info):
            # print('图像信息:', f_file)
            # 信息提取
            image_path = f_file.get("element_path", None)
            image_caption = f_file.get("img_caption", "")
            image_footnote = f_file.get("img_footnote", "")

            # 生成图片描述
            image_description = self.image_caption(image_path=image_path)  # 图像模型调用耗费资源，暂时不异步执行
            print(f'图像描述内容: {image_description["image_caption"]}')

            # 综合生成图片描述
            image_description_fusion = '图片描述：' + image_description["image_caption"]
            if len(image_caption):
                image_description_fusion = f'图片标题：{image_caption}\n' + image_description_fusion
            if len(image_footnote):
                image_description_fusion = image_description_fusion + '\n' + '图片尾注：\n' + image_footnote

            # 生成文本doc
            image_doc = Document(page_content=image_description_fusion, metadata={})

            # 更新元数据信息
            # print("开始更新和同步图片的元数据信息")
            image_doc.metadata["source"] = f_file.get("source", "")
            image_doc.metadata["file_type"] = f_file.get("file_type", "")
            image_doc.metadata["access_level"] = f_file.get("access_level")
            image_doc.metadata["chunk_type"] = "image"
            image_doc.metadata['page_number'] = f_file.get("page_num", "-1")
            image_docs.append(image_doc)

        return image_docs

    ###################################################### 流程函数  ######################################################
    def document_parse(self,
                       file_path: str,
                       metadata_dict: dict):
        """
        负责单个文件的ETF流程,同步操作。
        """
        temp_dir_path = None  # 初始化临时保存地址
        try:
            # 1 识别文件类型
            print('识别文件类型')
            file_type = self.file_type_identifier(file_path)
            print(f'当前文件的类型是：{file_type}')

            # 2 基于文件类型，提取其中的元素信息
            if file_type == 'document':
                # 文件解析
                print('开始解析文档文件')
                parse_result = file_loader(file_path, self.output_dir)
                docs = self.document_content_parser_fusion(parse_result=parse_result, metadata=metadata_dict)
                # for doc in docs:
                #     print(f'doc的主题：{doc.metadata.get("text_title", "")}')
                #     print(f'doc的章节信息：{doc.metadata.get("chapter_relation", "")}')
                #     print(f'doc的内容：{doc.page_content}')
                #     print(f'是否需要再分块：{doc.metadata.get("need_chunk", "")}')
                # raise ValueError('人工暂停')
                temp_dir_path = parse_result['temp_file_dir']  # 更新temp_dir_path
            elif file_type == 'image':
                image_info = [
                    {
                        "element_path": file_path,
                        "file_type": 'image',
                        "need_chunk": False,
                        "source": file_path
                    } | metadata_dict
                ]
                docs = self.image_pipeline(file_info=image_info)
            elif file_type == 'audio':
                audio_info = [
                    {
                        "element_path": file_path,
                        "file_type": 'audio',
                        "need_chunk": False,
                        "source": file_path
                    } | metadata_dict
                ]
                docs = self.audio_pipeline(file_info=audio_info)
            elif file_type == 'table':
                table_info = [
                    {
                        "element_path": file_path,
                        "file_type": 'table',
                        "source": file_path,
                        "need_chunk": False
                    } | metadata_dict
                ]
                docs = self.table_pipeline(file_info=table_info)
            elif file_type == 'text':
                text_info = [
                    {
                        "element_path": file_path,
                        "file_type": 'text',
                        "source": file_path
                    } | metadata_dict
                ]
                docs = self.text_pipeline(file_info=text_info)
            else:
                raise ValueError("不支持的文件类型，无法解析")
                
            # 3 处理docs
            print(f'docs数量: {len(docs)}')
            if file_type in ['document', 'text']:
                processed_contents = self.docs_pipeline(docs=docs)
            else:
                processed_contents = self.docs_pipeline(docs=docs)

            # 4 入库信息生成
            print('入库信息生成')
            # print(f'输入内容：{processed_contents}')
            records_to_save = self.save_knowledge_data_generator(processed_contents)
            # print("入库信息如下：")
            # print(records_to_save)

            # 5 删除临时文件和文件夹
            print()
            if temp_dir_path:
                shutil.rmtree(temp_dir_path)

            print(f"文件解析成功: {file_path}")
            return records_to_save
        except Exception as e:
            print(f"文件解析失败: {file_path}, 错误: {e}")
            return {"state": False}    
       
    def file_to_knowledge_database_handler(self, file_paths: List[str], file_ids: List[dict] = None, collection_name:str=None):
        """
        文件处理函数,将传入的文件处理后存入向量数据库
        """
        # 元数据处理
        if file_ids is None:
            file_ids = ["0"] * len(file_paths)

        if len(file_ids) != len(file_paths):
            print(f'文件路径和文件id数量不一致')
            return {"status": "05", "msg": "处理失败"}

        for file_path, file_id in zip(file_paths, file_ids):
            # 拼接元数据信息
            meta_info = {"source": file_path,"access_level": file_id} 

            # 解析文件，生成保存用的字典
            print(f'开始解析文件:{file_path}')
            dict_to_save = self.document_parse(file_path=file_path, metadata_dict=meta_info)

            if dict_to_save.get("state", True):
                print(f'文档解析成功')
            else:
                print(f'文档解析失败')
                return {"status": "05", "msg": "处理失败"}
            
            # 信息存入向量数据库
            try:
                res = save_to_vector_database(db_client=self.db_client,
                                        data_dict=dict_to_save, 
                                        collection_name=collection_name)
                if not res:
                    print('写入向量数据库发生错误')
                    return {"status": "05", "msg": "处理失败"}
            except Exception as e:
                print(f'写入向量数据库时发生错误，写入失败。错误信息：{e}')
                return {"status": "05", "msg": "处理失败"}

        return {"result": "00", "msg": "处理成功"}

    def search_knowledge_database(self, user_request: str, 
                                        collection_names: str,
                                        lite_mode:bool=True,
                                        Sorted_by_seq_id:bool=False):
        """
        请求向量数据库,从知识库中搜索相关信息
        """
        print(f'检索请求为:{user_request}')
        if lite_mode:
            print('使用快速检索模式')
            # 开始检索
            query_results = []
            query_consuming_times = []
            for collection_name in collection_names:
                query_response = db_search(db_client=self.vector_database_client,
                                           collection_name=collection_name,
                                           user_request=user_request,
                                           search_num=self.search_max_num,
                                           output_num=self.search_rerank_num)
                
                query_results.append(query_response["response"])
                query_consuming_times.append(query_response["search_time"])       
        else:
            print('使用强力检索模式')
            # 生成HyDE回答
            print('用户请求信息为:', user_request)
            hyde_response = self.hyde_answer_generator(request=user_request)
            print('生成的虚拟回答：', hyde_response)

            # 生成原始请求的关键词
            request_keywords = self.keyword_generator(request=user_request)
            print('基于原始请求的关键词为：', request_keywords)

            # 生成虚拟回答的关键词
            hyde_keywords = self.keyword_generator(request=hyde_response)
            print('虚拟回答的关键词为：', hyde_keywords)

            # 拼接关键词
            keywords_string = ''
            for keyword in request_keywords['keywords'] + hyde_keywords["keywords"]:
                keywords_string += (keyword + ' ')

            # 形成融合请求
            fusion_request = user_request + '\n\n' + keywords_string + '\n\n' + hyde_response

            # 开始检索
            query_results = []
            query_consuming_times = []
            for collection_name in collection_names:
                query_response = db_search(db_client=self.vector_database_client,
                                           collection_name=collection_name,
                                           user_request=user_request,
                                           search_num=self.search_max_num,
                                           output_num=self.search_rerank_num,
                                           fusion_request=fusion_request)

                query_results.append(query_response["response"])
                query_consuming_times.append(query_response["search_time"])

        # 将distance转化为分数（softmax）并返回
        search_results = []
        distances = [query_object.metadata.distance for query_result in query_results for query_object in query_result.objects]
        scores = rerank_distances_with_softmax(distances)
        score_index = 0
        for query_result in query_results:
            new_queries = []
            for return_object in query_result.objects:
                new_result = {
                    "properties" : return_object.properties,
                    "metadata": {
                        "distances": return_object.metadata.distance,
                        "softmax_score": scores[score_index]
                    }
                }
                new_queries.append(new_result)
                score_index += 1
            search_results.append(new_queries)
            

        #计算检索平均耗时
        print(f'检索的总耗时为:{np.sum(query_consuming_times)}s' )
        print(f'单次检索的平均耗时为:{np.mean(query_consuming_times)}s')

        # 统计结果
        search_info = {}
        for search_result in search_results:
            for object_ in search_result:
                object_path = object_["properties"]["source"]
                object_content = object_["properties"]["content"]
                object_seq_id = object_["properties"]["chunk_seq_id"]
                object_score = object_["metadata"]["softmax_score"]
                if object_path in search_info:
                    search_info[object_path]["content_text"].append(object_content)
                    search_info[object_path]["content_score"].append(object_score)
                    search_info[object_path]["content_seq_id"].append(object_seq_id)
                else:
                    search_info[object_path] = {
                        'content_text': [object_content],
                        "content_score": [object_score],
                        "content_seq_id": [object_seq_id]
                    }
        
        # 计算每个文件的检索信息
        file_outputs = []
        for key, value in search_info.items():
            # seq_ids = value["content_seq_id"]
            chunk_contents = value["content_text"]
            chunk_scores = value["content_score"]
            sum_score = np.sum(chunk_scores)
            file_outputs.append(
                {
                    "file_path": key,
                    "contents": chunk_contents,
                    "content_scores": chunk_scores,
                    "search_total_score": sum_score,
                }
            )

        # 对文件按分数进行rank
        output_rank = standard_competition_ranking([file_output["search_total_score"] for file_output in file_outputs], reverse=True)
        search_outputs = [file_outputs[ind] for ind, rank_value in enumerate(output_rank) if rank_value <= 3]

        # 结果返回
        outputs = {
            "search_consuming_times": np.sum(query_consuming_times),
            "search_detail": search_outputs
        }
        return outputs
    
    
if __name__ == '__main__':
    # /home/carlos/Projects/SmartAgent/data/Documents/2025数据分析Agent实践与案例研究报告.pdf
    # /home/carlos/Projects/SmartAgent/data/Documents/室内空气质量检测报告.pdf
    file_paths = [
        '/home/carlos/Projects/SmartAgent/data/Documents/奥 特 迅：2024年年度报告.pdf',
        # '/home/carlos/Projects/SmartAgent/data/Documents/室内空气质量检测报告.pdf',
        # '/home/carlos/Projects/SmartAgent/data/Documents/dokumen.pub_3-9787559427427.pdf',
        # '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海 】秦洋 5年.pdf',
        # '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海 】申靳超 6年.pdf',
        # '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海】曾柏栋 9年.pdf',
        # '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海】卫驰 7年.pdf',
        # '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海】张书琪 6年.pdf',
        # '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师(J10001)_上海 】张昊明 4年.pdf',
        # '/home/carlos/Projects/SmartAgent/data/Documents/【算法_上海】吴金亮 6年.pdf',
        # '/home/carlos/Projects/SmartAgent/data/Documents/C00015642-王明谦-数据分析工程师.pdf',
        # '/home/carlos/Projects/SmartAgent/data/Documents/C00016488-李明-数据分析工程师.pdf',
        # '/home/carlos/Projects/SmartAgent/data/Documents/C00016934-于畅-数据分析工程师.pdf'
        ]
    
    # 创建向量数据库实例
    print('创建向量数据库实例')
    db_client_params = config.get_setting("vector_database")["local"]
    weaviate_db_instance = VectorDatabase(db_params=db_client_params)

    # 创建模型实例
    print('创建模型实例')
    rag_model_instance = RagModels()

    # 创建RAG实例
    print('创建RAG实例')
    test_tool = RAGTool(db_client=db_client, ollama_client=ollama_client, async_ollama_client=ollama_async_client, is_remote=remote_flag)
    collection_name = 'report_collection'

    # 检查和加载模型
    print('模型检查')
    test_tool.model_check(is_remote=remote_flag)


    # # 生成多个表
    # c_names = []
    # for ind, file_path in enumerate(file_paths):
    #     # 生成元数据信息
    #     # 元数据处理
    #     metadata = {"access_level": 0, "source": file_path} 
    #     response = test_tool.file_to_knowledge_database(file_path=file_path, metadata_dict=metadata, collection_name=collection_name)
    #     c_names.append(collection_name)
    # print(c_names)

    # 删除与创建数据表
    collection_name = 'test_collection'
    test_tool.create_collection(collection_name=collection_name, is_rebuild=True)

    # pdf处理和保存测试
    response = test_tool.file_to_knowledge_database_handler(file_paths, collection_name=collection_name)
    print(f'文档处理结果:{response}')

    # # 数据库写入测试
    # test_data = {'08e6fc42-2293-5d63-ab18-21862cad7d6c': {'properties': {'content': '18521569056 丨qinyang4396@163.com 丨上海图片描述：图片展示了一位穿着正式的男性，他穿着深色西装，搭配白色衬衫和深色领带。背景是纯白色的，使得人物成为画面的焦点。他的头发是黑色的，整齐地向后梳，给人一种专业和整洁的印象。整体风格简洁，适合用于证件照或正式场合的肖像。', 'summary': '秦洋（联系方式：18521569056丨qinyang4396@163.com，地点：上海）的图片展示其正式形象：黑色头发整齐向后梳，深色西装搭配白衬衫与深色领带，纯白背景突显人物，整体风格简洁适用于证件照或正式场合肖像。', 'content_for_rerank': '文本主题：秦洋\n章节关系：\n\n文本内容：18521569056 丨qinyang4396@163.com 丨上海图片描述：图片展示了一位穿着正式的男性，他穿着深色西装，搭配白色衬衫和深色领带。背景是纯白色的，使得人物成为画面的焦点。他的头发是黑色的，整齐地向后梳，给人一种专业和整洁的印象。整体风格简洁，适合用于证件照或正式场合的肖像。', 'source': '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海 】秦洋 5年.pdf', 'file_type': 'document', 'chunk_type': 'text', 'access_level': '0', 'chunk_seq_id': 1, 'content_id': 3, 'chapter_info': ''}}, 'bf60ff5e-9f40-5a2d-a14e-5a0ef353ccb2': {'properties': {'content': '2023年02月 - 2024年09月悉尼大学数据科学硕士江苏大学车辆工程本科2012年09月 - 2016年06月', 'summary': '秦洋教育经历包含2023年2月-2024年9月悉尼大学数据科学硕士及2012年9月-2016年6月江苏大学车辆工程本科', 'content_for_rerank': '文本主题：秦洋\n章节关系：教育经历\n\n文本内容：2023年02月 - 2024年09月悉尼大学数据科学硕士江苏大学车辆工程本科2012年09月 - 2016年06月', 'source': '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海 】秦洋 5年.pdf', 'file_type': 'document', 'chunk_type': 'text', 'access_level': '0', 'chunk_seq_id': 2, 'content_id': 7, 'chapter_info': ''}}, 'de37cb08-164b-52c8-9a6b-8c1a3d28c663': {'properties': {'content': '2018年12月 - 2021年08月数据分析高级工程师', 'summary': '秦洋于2018年12月至2021年8月任职于上海蔚来汽车有限公司，担任数据分析高级工程师。', 'content_for_rerank': '文本主题：秦洋\n章节关系：工作经历-上海蔚来汽车有限公司\n\n文本内容：2018年12月 - 2021年08月数据分析高级工程师', 'source': '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海 】秦洋 5年.pdf', 'file_type': 'document', 'chunk_type': 'text', 'access_level': '0', 'chunk_seq_id': 3, 'content_id': 11, 'chapter_info': ''}}, 'c55e698f-4367-51bf-a7ba-ff08cdd47829': {'properties': {'content': '电池安全指标监控体系： 基于SQL提取上百个电池包历史数据，通过纵向（快充/慢充/静置工况）与横向（电压/电流/温度/绝缘值等特征）多维分析，识别自放电、电解液泄漏、内短路、螺栓松动等高危场景特征，科学设定三级报警阈值。算法优化与效果验证： 开发云端实时报警框架，通过回收电池包数据回测优化特征组合策略（如电压上升速率 $^ +$ 电流波动），推动召回准确率达 $9 5 \\%$ ，累计召回高危电池包 $^ { 1 0 0 + }$ 个。联合电池设计、运营团队制定分级响应SOP，建立周度异动分析闭环流程，并输出报告$^ { 4 0 + }$ 份，高危场景处置时效缩短 $30 \\%$ 。', 'summary': '秦洋构建电池安全指标监控体系，基于SQL提取电池包历史数据，通过纵向（快充/慢充/静置）与横向（电压/电流/温度/绝缘值）多维分析，识别自放电、电解液泄漏等高危场景并设定三级报警阈值；开发云端实时报警框架优化特征组合策略（如电压上升速率+电流波动），实现95%召回准确率及100+高危电池包召回，联合团队制定分级响应SOP，建立周度异动分析闭环流程，输出40+报告并使高危场景处置时效缩短30%。', 'content_for_rerank': '文本主题：秦洋\n章节关系：1. 电池安全异动归因\n\n文本内容：电池安全指标监控体系： 基于SQL提取上百个电池包历史数据，通过纵向（快充/慢充/静置工况）与横向（电压/电流/温度/绝缘值等特征）多维分析，识别自放电、电解液泄漏、内短路、螺栓松动等高危场景特征，科学设定三级报警阈值。算法优化与效果验证： 开发云端实时报警框架，通过回收电池包数据回测优化特征组合策略（如电压上升速率 $^ +$ 电流波动），推动召回准确率达 $9 5 \\%$ ，累计召回高危电池包 $^ { 1 0 0 + }$ 个。联合电池设计、运营团队制定分级响应SOP，建立周度异动分析闭环流程，并输出报告$^ { 4 0 + }$ 份，高危场景处置时效缩短 $30 \\%$ 。', 'source': '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海 】秦洋 5年.pdf', 'file_type': 'document', 'chunk_type': 'text', 'access_level': '0', 'chunk_seq_id': 4, 'content_id': 13, 'chapter_info': ''}}, '7606f5c3-628d-5b2d-a872-83ab6f6266ef': {'properties': {'content': '全生命周期建模： 基于1万 $^ +$ 电池包数据构建随机森林模型，提取行驶里程、快慢充频次、SOH等12维特征，预测2万+未知SOH电池包的SOH，将RMSE控制在5%以内，支撑资产估值决策。通过Pearson相关系数分析，发现急减速次数与SOH衰减强相关，经与电池设计团队沟通，确认动能回收控制系统存在设计缺陷。', 'summary': '秦洋在电池健康度预测与归因分析章节中，基于1万+电池包数据构建随机森林模型，提取行驶里程、快慢充频次、SOH等12维特征，实现2万+未知SOH电池包的SOH预测（RMSE≤5%），通过Pearson相关分析发现急减速次数与SOH衰减强相关，经与设计团队确认动能回收控制系统存在设计缺陷。', 'content_for_rerank': '文本主题：秦洋\n章节关系：2. 电池健康度预测与归因分析\n\n文本内容：全生命周期建模： 基于1万 $^ +$ 电池包数据构建随机森林模型，提取行驶里程、快慢充频次、SOH等12维特征，预测2万+未知SOH电池包的SOH，将RMSE控制在5%以内，支撑资产估值决策。通过Pearson相关系数分析，发现急减速次数与SOH衰减强相关，经与电池设计团队沟通，确认动能回收控制系统存在设计缺陷。', 'source': '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海 】秦洋 5年.pdf', 'file_type': 'document', 'chunk_type': 'text', 'access_level': '0', 'chunk_seq_id': 5, 'content_id': 15, 'chapter_info': ''}}, '64c5abfb-e1b8-52a7-b30e-ae1af3d10182': {'properties': {'content': '为电池系统、整车部门提供全面的电池数据查询支持。 负责整理和归纳平台报警的异常值及其产生原因，定期形成周报，为团队提供数据洞察和决策参考。', 'summary': '秦洋负责为电池系统及整车部门提供电池数据查询支持，通过整理平台报警异常值及其成因并形成周报，为团队提供数据洞察与决策参考。', 'content_for_rerank': '文本主题：秦洋\n章节关系：3. 日常数据查询与分析\n\n文本内容：为电池系统、整车部门提供全面的电池数据查询支持。 负责整理和归纳平台报警的异常值及其产生原因，定期形成周报，为团队提供数据洞察和决策参考。', 'source': '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海 】秦洋 5年.pdf', 'file_type': 'document', 'chunk_type': 'text', 'access_level': '0', 'chunk_seq_id': 6, 'content_id': 17, 'chapter_info': ''}}, 'c8ae1359-a80a-5a32-93ed-6e03f334556c': {'properties': {'content': '2016年07月 - 2018年12月电池开发工程师电池性能参数优化专项：为实现续航里程 $3 2 5 \\mathsf { k m }$ 目标，设计能量密度（52.5kwh）与快充效率（45min充 $8 0 \\%$ ）的电池PACK，平衡安全性与性能指标；建立轻量化验证体系，分析 $^ { 1 0 0 + }$ 组结构强度测试数据，量化重量降低 $10 \\%$ 对电池安全及续航里程的正向影响。 $\\bullet$ 供应链管理及成本归因分析：构建电池包BOM成本监控图，拆解材料/模具/加工等二级成本结构，定位模具费用异常波动点（ $\\pm 3 \\%$ ）；输出供应商质量指标体系，制定尺寸公差、材料强度等8项SOR标准，沉淀10份可量化验收文档；建立供应商交付数据日报体系，通过过程指标异常预警推动交付延迟率下降 $10 \\%$ 。', 'summary': '秦洋在上汽商用车技术中心（2016.7-2018.12）主导电池开发工程：针对325km续航目标设计52.5kWh能量密度/45min充80%快充效率的电池PACK，通过100+组结构强度测试验证重量降低10%对安全性和续航的正向影响；构建电池包BOM成本监控体系，定位模具费用±3%异常波动，制定8项SOR标准并沉淀10份验收文档，建立供应商交付数据日报体系使交付延迟率下降10%。', 'content_for_rerank': '文本主题：秦洋\n章节关系：上汽商用车技术中心\n\n文本内容：2016年07月 - 2018年12月电池开发工程师电池性能参数优化专项：为实现续航里程 $3 2 5 \\mathsf { k m }$ 目标，设计能量密度（52.5kwh）与快充效率（45min充 $8 0 \\%$ ）的电池PACK，平衡安全性与性能指标；建立轻量化验证体系，分析 $^ { 1 0 0 + }$ 组结构强度测试数据，量化重量降低 $10 \\%$ 对电池安全及续航里程的正向影响。 $\\bullet$ 供应链管理及成本归因分析：构建电池包BOM成本监控图，拆解材料/模具/加工等二级成本结构，定位模具费用异常波动点（ $\\pm 3 \\%$ ）；输出供应商质量指标体系，制定尺寸公差、材料强度等8项SOR标准，沉淀10份可量化验收文档；建立供应商交付数据日报体系，通过过程指标异常预警推动交付延迟率下降 $10 \\%$ 。', 'source': '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海 】秦洋 5年.pdf', 'file_type': 'document', 'chunk_type': 'text', 'access_level': '0', 'chunk_seq_id': 7, 'content_id': 21, 'chapter_info': ''}}, '9317aa01-99f4-514b-a10f-bf1dbcdb0017': {'properties': {'content': '2024年02月 - 2024年06月$\\bullet$ 项目背景：针对工程安全评估中防火性能检测高成本痛点，构建预测模型替代部分实验检测，提升评估效率。 . 多维特征分析与建模：构建17维工程指标评估体系（含材料属性/荷载参数/结构特征） ，通过分布分析发现 $30 \\%$ 数据存在量纲不统一问题，设计标准化处理流程 。运用SHAP值解析特征贡献度 ，识别荷载比 、FRP包裹率为核心决策因子（累计贡献度 $6 2 \\%$ ），指导后续特征组合优化。 $\\bullet$ 策略优化与效果验证：基于业务场景设计三分类预测框架（高危/中危/安全），通过多模型对比验证， 采用特征交叉策略提升模型对非线性关系的捕捉能力。通过分层交叉验证确保方案鲁棒性，最终模型预测准确率较传统评估方法提升 $2 8 \\%$ ，F1值达0.88。', 'summary': '秦洋于2024年2月至6月主导RC剪力墙失效模式分析项目，针对防火性能检测高成本问题构建预测模型。通过构建17维工程指标体系（含材料属性/荷载参数/结构特征），标准化处理30%量纲不统一数据，运用SHAP值识别荷载比与FRP包裹率（累计贡献度62%）为核心因子。基于三分类预测框架（高危/中危/安全）设计特征交叉策略，经分层交叉验证后模型准确率较传统方法提升28%（F1值达0.88）。', 'content_for_rerank': '文本主题：秦洋\n章节关系：项目经历-RC剪力墙的失效模式分析\n\n文本内容：2024年02月 - 2024年06月$\\bullet$ 项目背景：针对工程安全评估中防火性能检测高成本痛点，构建预测模型替代部分实验检测，提升评估效率。 . 多维特征分析与建模：构建17维工程指标评估体系（含材料属性/荷载参数/结构特征） ，通过分布分析发现 $30 \\%$ 数据存在量纲不统一问题，设计标准化处理流程 。运用SHAP值解析特征贡献度 ，识别荷载比 、FRP包裹率为核心决策因子（累计贡献度 $6 2 \\%$ ），指导后续特征组合优化。 $\\bullet$ 策略优化与效果验证：基于业务场景设计三分类预测框架（高危/中危/安全），通过多模型对比验证， 采用特征交叉策略提升模型对非线性关系的捕捉能力。通过分层交叉验证确保方案鲁棒性，最终模型预测准确率较传统评估方法提升 $2 8 \\%$ ，F1值达0.88。', 'source': '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海 】秦洋 5年.pdf', 'file_type': 'document', 'chunk_type': 'text', 'access_level': '0', 'chunk_seq_id': 8, 'content_id': 25, 'chapter_info': ''}}, 'db8de45d-f49a-5630-b723-7ce8ced47d08': {'properties': {'content': '2023年07月 - 2023年11月$\\bullet$ 项目目标：验证不同机器学习方案在复杂图像分类场景中的适用性 ，为算法选型提供决策依据。 $\\bullet$ 数据质量治理：分析发现 $20 \\%$ 图像存在分辨率不一致问题 ，设计标准化预处理流程（尺寸归一化 $^ +$ 数据增强）提升数据可用性。 $\\bullet$ 特征有效性验证：基于灰度直方图分布分析，识别3类高区分度特征，构建特征组合策略提升传统模型效果。 $\\bullet$ 核心成果：建立双维度评估标准（准确率/计算效率），针对不同业务场景提出解决方案建议（CNN用于高精度场景 ，SVM/RF适配轻量化需求）。通过网格搜索优化模型参数，CNN模型F1值达0.92。', 'summary': '秦洋于2023年7月至11月主导血细胞图片分类项目，通过验证机器学习方案适用性构建算法选型决策体系。针对20%分辨率不一致图像实施尺寸归一化与数据增强预处理，基于灰度直方图识别3类高区分度特征并构建组合策略。建立准确率/计算效率双维度评估标准，提出CNN（F1=0.92）适配高精度场景、SVM/RF满足轻量化需求的解决方案，通过网格搜索完成模型参数优化。', 'content_for_rerank': '文本主题：秦洋\n章节关系：血细胞图片分类\n\n文本内容：2023年07月 - 2023年11月$\\bullet$ 项目目标：验证不同机器学习方案在复杂图像分类场景中的适用性 ，为算法选型提供决策依据。 $\\bullet$ 数据质量治理：分析发现 $20 \\%$ 图像存在分辨率不一致问题 ，设计标准化预处理流程（尺寸归一化 $^ +$ 数据增强）提升数据可用性。 $\\bullet$ 特征有效性验证：基于灰度直方图分布分析，识别3类高区分度特征，构建特征组合策略提升传统模型效果。 $\\bullet$ 核心成果：建立双维度评估标准（准确率/计算效率），针对不同业务场景提出解决方案建议（CNN用于高精度场景 ，SVM/RF适配轻量化需求）。通过网格搜索优化模型参数，CNN模型F1值达0.92。', 'source': '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海 】秦洋 5年.pdf', 'file_type': 'document', 'chunk_type': 'text', 'access_level': '0', 'chunk_seq_id': 9, 'content_id': 28, 'chapter_info': ''}}}

    # flag = save_to_vector_database(db_client=db_client,
    #                                     data_dict=test_data, 
    #                                     collection_name=collection_name)

    # 遍历向量数据库
    collection = test_tool.db_client.collections.get(collection_name)
    count = 1
    for obj in collection.iterator(
        return_properties=True, # 获取所有自定义字段
        return_metadata=MetadataQuery(
            creation_time=True,
            last_update_time=True,
            # 如果您也需要向量本身，可以取消下面这行的注释
            # vector=True 
        )
    ):
        print(f"***第{count}条数据记录***")
        print(f"数据记录uuid:{str(obj.uuid)}")
        print(f"数据记录的类型：{obj.properties['chunk_type']}")
        print(f"数据记录的来源：{obj.properties['source']}")
        # print(f"文本所在的页码：{obj.properties['page_number']}")
        # print(f"数据记录的内容：{obj.properties['content']}")
        print(f"数据记录的摘要：{obj.properties['summary']}")
        count += 1


    # # 向量数据库检索
    # test_collection = test_tool.vector_database_client.collections.get(collection_name)
    # request_info = [
    #     "请告诉我brent的相关信息",  #1
    #     # "请推荐有汽车相关行业工作经历的候选人简历",  #2
    #     # "请推荐有博士学位的候选人",  #3
    #     # "请总结卫驰的工作经历",  #4
    #     # "请简述张书琪的教育经历",  #5
    #     # "哪些求职者可以熟练使用Python",  #6
    #     # "请告诉我Lili Tong的个人信息",  #7
    #     # "哪些求职者有5年以上的工作经验？",  #8
    #     # "请推荐有市场分析经验的候选人",  #9
    #     # "请推荐有复旦背景的候选人",  #10
    #     # "在金融领域，数据分析agent主要应用与哪些任务中？",  #11
    #     # "企业在引入数据分析Agent前，需要如何进行准备？",  #12
    #     # "目前，国内有哪些数据分析Agent的解决方案？",  #13
    #     # "SwiftAgent的产品架构一共有几层？",  #14
    #     # "wiftAgent的核心优势有哪些？",  #15
    #     # "请提供一些数据分析Agent的应用案例",  #16
    #     # "反问机制是如何解决人员模糊提问问题的？",  #17
    #     # "AI报告生成项目的落地步骤是怎样的？",  #18
    #     # "数据分析agent的定义是什么？",  #19
    #     # "请简述数据分析agent的应用现状"  #20
    # ]
    # for ind, request in enumerate(request_info):
    #     search_result = test_tool.search_knowledge_database(user_request=request,
    #                                                              collection_name=collection_name)
    #     print(search_result)

    # 关闭数据库连接
    close_weavier_connection(test_tool.db_client)