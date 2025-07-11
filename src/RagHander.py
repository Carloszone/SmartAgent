# 目前没有合适的中文图文embedding模型，因此对图片的处理模式为：描述图片->描述文本embedding 所以不支持以图搜图

from ConfigManager import config
import os
import re
import time
from typing import List, Union, Optional
from langchain_core.documents import Document
from ContentHandler import get_system_message, json_extractor, apply_rrf, file_loader, fusion_list_str, html_to_json, split_text_by_paragraphs
import weaviate
import asyncio
from weaviate.classes.query import Filter, MetadataQuery, BM25Operator
from weaviate.util import generate_uuid5
from weaviate.classes.config import Configure, Property, DataType, Vectorizers, Tokenization
import pandas as pd
import hashlib
import ollama
from funasr import AutoModel
import markdown
from bs4 import BeautifulSoup
import shutil
from FileLoaders import image_text_reader
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from pathlib import Path
import warnings
import json
import copy
import gc


class RAGTool:
    """
    执行RAG相关操作的类
    """
    def __init__(self):
        # 模型相关
        self.ollama_client = None  # ollama客户端
        self.async_ollama_client = None  # 异步ollama客户断
        self.audio_model = None  # 音频模型1
        self.generative_model_name = config.get_setting("models")["generative_model_name"]  # 生成模型名称
        self.text_embedding_model_name = config.get_setting("models")["text_embedding_model_name"]  # 文本embedding模型
        self.audio_model_address = config.get_setting("models")["audio_model_address_1"]  # 音频模型的地址
        self.audio_vad_model_address = config.get_setting("models")["audio_vad_model_address"]  # 音频ad模型地址
        self.image_caption_model_name = config.get_setting("models")["image_caption_model_name"]  #  图像描述模型
        self.ollama_model_option = config.get_setting("models")["ollama_model_option"]  # ollama模型额外参数


        # 数据库相关
        self.vector_database_client = None  # 向量数据库的客户端
        self.vector_database_host = config.get_setting("vector_database")["host"]  # 向量数据库的地址
        self.vector_database_port = config.get_setting("vector_database")["port"]  # 向量数据库的端口
        self.vector_database_grpc_port = config.get_setting("vector_database")["grpc_port"]  # 向量数据库的grpc端口
        self.knewledge_base_collection_name = config.get_setting("vector_database")["knewledge_base_collection_name"]  # 知识库collection的名称
        self.chat_collection_name = config.get_setting("vector_database")["chat_collection_name"]  # 聊天记录collection的名称
        

        # 文档相关
        self.long_text_threshold = config.get_setting("files")["long_text_threshold"]  # 长文本的最低字符限制
        self.default_parag_per_chunk = config.get_setting("files")["default_parag_per_chunk"]  # 文本的默认分块段落数
        self.document_types = config.get_setting("files")["types"]["text_file_extension"]  # 文档后缀
        self.table_types = config.get_setting("files")["types"]["table_file_extension"]  # 表格后缀
        self.audio_types = config.get_setting("files")["types"]["audio_file_extension"]  # 音频后缀
        self.image_types = config.get_setting("files")["types"]["image_file_extension"]  # 图像后缀
        self.default_file_access_level = config.get_setting("files")["default_file_access_level"]  # 文档的默认访问级别


        # 检索相关
        self.search_max_num = config.get_setting("search")["max_mun"]  # 每次独立搜索返回的结果数量
        self.search_output_num = config.get_setting("search")["output_mun"]  # rrf综合后的结果数量


        # 输出相关
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        project_root = os.path.dirname(script_dir)
        self.output_dir = os.path.join(project_root, config.get_setting("files")["output_dir"])

        # 其他参数：
        self.model_retry_num = 3  # 遇到模型报错时的重试次数
        self.concurrency_limit = 4  # 并发控制最大数

    ###################################################### 数据库操作函数  ######################################################
    def save_to_vector_database(self, data_dict: dict = None, collection_name: str=None):
        """
        将数据存入数据库
        """
        # print('待入库信息：', data_dict)
        if data_dict is None:
            print(f'没有数据需要保存')
            return
        else:
            # print(f'需要保存的数据:{data_dict}')
            print(f'检查数据表{collection_name}是否存在')
            # 提取配置信息
            if collection_name is None:
                vector_database_collection_name = self.knewledge_base_collection_name
            else:
                vector_database_collection_name = collection_name

            # 检查数据表是否存在
            if not self.vector_database_client.collections.exists(vector_database_collection_name):
                raise Exception(f'向量数据库中不存在{vector_database_collection_name}数据表')
            else:
                # 验证数据表是否存在
                if not self.vector_database_client.collections.exists(vector_database_collection_name):
                    print(f'数据表{vector_database_collection_name}不存在')
                    raise
                else:
                    document_collection = self.vector_database_client.collections.get(vector_database_collection_name)
                    aggregation_result = document_collection.aggregate.over_all(total_count=True)
                    total = aggregation_result.total_count
                    print(f"\n 数据表 '{vector_database_collection_name}' 的总记录数为: {total}")
                # 尝试保存数据
                try:
                    print(f"\n准备插入 {len(data_dict)} 条数据...")
                    successful_mapping = []
                    with document_collection.batch.dynamic() as batch:
                        for uuid, item in data_dict.items():
                            batch.add_object(
                                uuid=uuid,
                                properties=item.get("properties")
                            )

                    # 检查批量操作中是否有错误
                    failed_objects = document_collection.batch.failed_objects
                    if failed_objects:
                        print(f"插入数据时发生错误数量: {len(document_collection.batch.failed_objects)}")
                    else:
                        print(f'插入数据成功')
                    return successful_mapping
                            

                except Exception as e:
                    print(f"插入数据时发生错误: {e}")
                    raise
        return []
  
    def craete_collection(self, collection_name):
        if self.vector_database_client.collections.exists(collection_name):
            print(f'该数据表已经存在')
        else:
            properties_to_create = [
                Property(
                    name="content",
                    description="原始文本",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.GSE,
                    index_searchable=True
                ),
                Property(
                    name="summary",
                    description="原始文本的概括",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.GSE,
                    index_searchable=True
                ),
                Property(
                    name="source",
                    description="来源文件路径或URL",
                    data_type=DataType.TEXT,
                    index_filterable=True
                ),
                Property(
                    name="chapter_info",
                    description="来源文件路径或URL",
                    data_type=DataType.TEXT,
                    index_filterable=True
                ),
                Property(
                    name="file_type",
                    description="原始文件的类型 (e.g., pdf, docx, md)",
                    data_type=DataType.TEXT,
                    index_filterable=True
                ),
                Property(
                    name="chunk_type",
                    description="内容块的类型 (e.g., pdf, docx, md)",
                    data_type=DataType.TEXT,
                    index_filterable=True
                ),
                Property(
                    name="access_level",
                    description="访问控制级别",
                    data_type=DataType.INT,
                    index_filterable=True
                ),
                Property(
                    name="page_number",
                    description="在原始文件中的页码",
                    data_type=DataType.TEXT,
                    index_filterable=True
                ),
                Property(
                    name="chunk_seq_id",
                    description="内容块的序列ID",
                    data_type=DataType.INT,
                    index_filterable=True
                ),
            ]

            self.vector_database_client.collections.create(
                name=collection_name,
                vectorizer_config=[
                    Configure.NamedVectors.text2vec_ollama(
                        name="content_vector",
                        source_properties=["content"],
                        api_endpoint="http://ollama-host:11434",
                        model=self.text_embedding_model_name
                    ),
                    Configure.NamedVectors.text2vec_ollama(
                        name="summary_vector",
                        source_properties=["summary"],
                        api_endpoint="http://ollama-host:11434",
                        model=self.text_embedding_model_name
                    )
                ],
                properties=properties_to_create
            )
            print(f'向量数据表{collection_name}创建成功')

    def delete_collection(self, collection_name: str=None, delete_all: bool=False):
        if delete_all:
            if self.vector_database_client.is_ready(): # is_ready() 检查连接
                self.vector_database_client.collections.delete_all()
                print(f'已删除全部数据表')
                return None

        if collection_name:
            if self.vector_database_client.collections.exists(collection_name): # is_ready() 检查连接
                self.vector_database_client.collections.delete(collection_name)
                print(f'数据表{collection_name}删除成功')
        else:
            raise ValueError(f'参数collection_name不能为空')

    ########################################################  功能函数  ########################################################
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
            if file_extension in self.document_types:
                return "document"
            elif file_extension in self.audio_types:
                return "audio"
            elif file_extension in self.table_types:
                return "table"
            elif file_extension in self.image_types:
                return "image"
            else:
                raise ValueError(f'不支持的文件后缀:{file_extension}')

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
                response = self.client.chat(model=self.generative_model_name, 
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
            if len(input_content) >= 50:
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
                last_exception = None
                for attempt in range(self.model_retry_num):
                    try:
                        response = self.client.chat(model=self.generative_model_name, 
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
                print(f'原文：{input_content}')
                print(f"概括内容：{clean_output}")
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
    
    def table_content_description(self, table_content: str):
        """
        对JSON化的表格内容进行描述同步
        """
        # 构建message
        system_message = get_system_message('table_description')
        message = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": f"请对以下JSON格式的表格内容进行描述：{table_content}" 
            }
        ]

        # 模型交互,针对503错误重试
        last_exception = None
        for attempt in range(self.model_retry_num):
            try:
                response = self.client.chat(model=self.generative_model_name, 
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
        try:
            output_content = json_extractor(clean_output)
            return output_content['content']
        except Exception as e:
            return clean_output

    def image_caption(self, image_path):
        # 构建图片描述信息
        # print('开始图片描述')
        system_message = get_system_message('image_caption')
        message = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": "请尽可能详尽地描述传入的图片",
                "images": [Path(image_path)]
            }
        ]

        # 模型交互,针对503错误重试
        last_exception = None
        for attempt in range(self.model_retry_num):
            try:
                response = self.client.chat(model=self.image_caption_model_name,
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
                print(f'图片描述模型失败 {e}')
                return e
 
        # 提取输出json
        # print(f'开始提取json')
        raw_content = response['message']['content']
        clean_output = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        try:
            output_content = json_extractor(clean_output)
            # print(f'***图像描述：{output_content}')
            return output_content["image_caption"]
        except Exception as e:
            print("json提取失败")
            return clean_output

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
        response = self.client.chat(model=self.generative_model_name, 
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

            # 元数据：来自文件
            metadata = content["content"].metadata
            uuid = metadata.get("uuid")
            source = metadata.get("source")
            file_type = metadata.get("file_type", "")
            page_number = metadata.get("page_number", "-1")
            chunk_seq_id = metadata.get("chunk_seq_id", -1)
            access_level = metadata.get("access_level", 0)
            chunk_type = metadata.get("chunk_type", "")
            chapter_info = metadata.get("chapter_info", "")

            # 组合字段
            if uuid not in data_dict.keys():
                data_dict[uuid] = {"properties": {
                                       "content": content_text, 
                                       "summary": summary_text,
                                       "source": source,
                                       "file_type": file_type,
                                       "chunk_type": chunk_type,
                                       "access_level": access_level,
                                       "page_number": page_number,
                                       "chunk_seq_id": chunk_seq_id,
                                       "chapter_info": chapter_info
                                   }
                                }
        return data_dict

    async def text_chunk_splitter_async(self, content: Document) -> List[Document]:
        """
        基于markdown格式的分本分块函数
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
                response = await self.async_client.chat(model=self.generative_model_name, 
                                                        messages=message,
                                                        options=self.ollama_model_option)
            except ollama.ResponseError as e:
                if e.status_code == 503:
                    last_exception = e
                    wait_time = 2 * attempt
                    print(f'遇到503错误，等待{wait_time}秒后重试...')
                    await asyncio.sleep(wait_time)
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

    async def text_chunk_splitter_handler_async(self, contents: List[Document]) -> List[Document]:
        """
        基于markdown格式的文本分块函数(批量)
        """
        # print(f'文档分块输入内容：', contents)
        tasks = [self.text_chunk_splitter_async(content) for content in contents if isinstance(content, Document)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # print(f'文档分块输出内容：', results)

        # 展评list
        output_results = []
        for result in results:
            if isinstance(result, list):
                for res in result:
                    if isinstance(res, Document):
                        output_results.append(res)
        return output_results
    
    async def text_summary_async(self, content: Document) -> Document:
        # 输入解析
        input_content = content.page_content

        # 构建message
        if len(input_content) >= 50:
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
            last_exception = None
            for attempt in range(self.model_retry_num):
                try:
                    response = await self.async_client.chat(model=self.generative_model_name, 
                                                            messages=message,
                                                            options=self.ollama_model_option)
                except ollama.ResponseError as e:
                    if e.status_code == 503:
                        last_exception = e
                        wait_time = 2 * attempt
                        print(f'遇到503错误，等待{wait_time}秒后重试...')
                        await asyncio.sleep(wait_time)
                    else:
                        return e
                except Exception as e:
                    return e

            # 提取输出json
            summary_json = json_extractor(response['message']['content'])

            if "summary" not in summary_json:
                summary_json = {"summary": input_content}

        else:
            summary_json = {"summary": input_content}

        # 构建doc
        new_doc = Document(page_content=summary_json["summary"], metadata=content.metadata)
        return new_doc
    
    async def text_summary_handler_async(self, contents: List[Document]) -> List[Document]:
        """
        """
        tasks = [self.text_summary_async(content) for content in contents if isinstance(content, Document)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 结果过滤
        successful_results = [res for res in results if isinstance(res, Document)]
        return successful_results
    
    async def text_question_derive_async(self, content: Document) -> Document:
        # 输入解析
        input_content = content.page_content

        # 构建message
        system_message = get_system_message('questions')
        message = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": f"请针对以下内容设计提问：{input_content}" 
            }
        ]

        # 模型交互,针对503错误重试
        last_exception = None
        for attempt in range(self.model_retry_num):
            try:
                response = await self.async_client.chat(model=self.generative_model_name, 
                                                        messages=message,
                                                        options=self.ollama_model_option)
            except ollama.ResponseError as e:
                if e.status_code == 503:
                    last_exception = e
                    wait_time = 2 * attempt
                    print(f'遇到503错误，等待{wait_time}秒后重试...')
                    await asyncio.sleep(wait_time)
                else:
                    return e
            except Exception as e:
                return e
        
        # 提取输出json
        question_json = json_extractor(response['message']['content'])

        if "questions" not in question_json:
            question_json = {"questions": ""}

        # 构建新的doc
        new_doc = Document(page_content=question_json["questions"], metadata=content.metadata)
        return new_doc

    async def text_question_derive_handler_async(self, contents: List[Document]) -> List[Document]:
        """
        """
        tasks = [self.text_question_derive_async(content) for content in contents if isinstance(content, Document)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 结果过滤
        successful_results = [res for res in results if isinstance(res, Document)]
        return successful_results

    async def table_content_description_async(self, table_content: dict):
        """
        对JSON化的表格内容进行描述
        """
        # 构建message
        system_message = get_system_message('table_description')
        message = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": f"请对以下文本进行概括总结：{table_content}" 
            }
        ]

        # 模型交互,针对503错误重试
        last_exception = None
        for attempt in range(self.model_retry_num):
            try:
                response = await self.async_client.chat(model=self.generative_model_name, 
                                                        messages=message,
                                                        options=self.ollama_model_option)
            except ollama.ResponseError as e:
                if e.status_code == 503:
                    last_exception = e
                    wait_time = 2 * attempt
                    print(f'遇到503错误，等待{wait_time}秒后重试...')
                    await asyncio.sleep(wait_time)
                else:
                    return e
            except Exception as e:
                return e
 
        # 提取输出json
        raw_content = response['message']['content']
        clean_output = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        try:
            output_content = json_extractor(clean_output)
            return output_content['content']
        except Exception as e:
            return clean_output

    async def table_summary_async(self, content: Document) -> Document:
        """
        用于提取表格内容的函数
        """
        table_json = content.page_content

        # 表格内容提炼:当行数大于20时,视为非文本表格
        if len(table_json) >= 20:
            raise ValueError(f'暂不支持大表格处理')
        else:
            # 概括信息
            table_description = await self.table_content_description_async(table_json)

        # 构建doc
        new_doc = Document(page_content=table_description, metadata=content.metadata)
        return new_doc

    async def table_summary_handler_async(self, contents: List[Document]) -> List[Document]:
        """
        批量提取表格内容概括的函数
        """
        tasks = [self.text_summary_async(content) for content in contents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # print(f'表格概括结果为:{results}')
        # 结果过滤
        successful_results = [res for res in results if isinstance(res, Document)]
        return successful_results

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
                response = self.client.chat(model=self.generative_model_name, 
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

    async def search_extract_keywords_async(self, request: str):
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
        response = await self.async_client.chat(model=self.generative_model_name, 
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

    ###################################################### 整合函数  ######################################################
    def document_content_parser_fusion(self, parse_result: dict, metadata: dict):
        """
        将document对象的解析结果进行处理和融合的函数
        以解析得到的一级标题作为首选docs标准。对于没有一级标题的对象，以页码+滑动段落的方式进行分页
        """
        def get_chunk_text(chunk, chunk_type):
            # 处理文本内容
            if chunk_type == "text":
                chunk_text = fusion_list_str(chunk["text"])  # 文本块的文本内容

            # 处理表格内容块
            if chunk_type == 'table':
                # 表格信息提取
                table_caption = fusion_list_str(chunk.get("table_caption", ""))
                table_footnote = fusion_list_str(chunk.get("table_footnote", ""))
                table_context = f"表格名称：: {table_caption}\n 表格尾注: {table_footnote}\n"
                table_content = fusion_list_str(chunk.get("table_body", ""))
                chunk_text = "表格名称：" + "\n" + table_caption + '\n' + "表格正文:" + "\n" + table_content + '\n' + "表格尾注信息：" + "\n" + table_footnote + '\n'

            # 处理图片内容块
            if chunk_type == 'image':
                # 图片信息提取
                img_caption = fusion_list_str(chunk.get("img_caption", ""))
                img_footnote = fusion_list_str(chunk.get("img_footnote", ""))
                chunk_text = "图片的名称信息:" + "\n" + img_caption + '\n' + "图片的尾注信息:" + img_footnote
            return chunk_text

        # 提取解析信息
        content_json_path = parse_result.get("content_json_path")
        temp_dir_path = parse_result.get("temp_dir_path")

        # 获取内容json
        with open(content_json_path, 'r', encoding='utf-8') as f:
                content_json = json.load(f)

        # 遍历json，提取table和image内容， 并判断docs模式
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

                image_chunks.append({
                    "element_path": os.path.join(temp_dir_path, chunk["img_path"]),
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

                table_chunks.append({
                    "element_content": get_chunk_text(chunk=chunk, chunk_type=chunk_type),
                    "table_caption": table_caption,
                    "table_footnote": table_footnote,
                    "chunk_id": chunk_idx
                })
        
        # 提取得到的表格和图片信息
        image_docs = self.image_pipeline(image_chunks)
        table_docs = self.table_pipeline(table_chunks)

        # 利用提取到的表格和图片信息，更新content_json
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
        level_text_manager = {}
        record_page_num = None
        page_content = ''  # 文档的正文内容
        output_docs = []
        start_text_id = -1  # 页面第一段文本的chunk id
        text_pages = []
        chapter_relation = ''  # 初始化标题关系
        for chunk_idx, chunk in enumerate(content_json):
            # 提取每一个文本块的信息
            # print("chunk:", chunk)
            chunk_type = chunk["type"]  # 文本块的类型
            chunk_page_num = chunk["page_idx"]  # 文本块所属的页码
            chunk_text_level = chunk.get("text_level", 0)  # 文本块的级别，正文为0
            chunk_chunk_type = chunk.get("chunk_type", 'text')
            chunk_text = chunk.get("text")
            text_pages.append(chunk_page_num)

            if page_cut_mode:  # 标题docs模式
                if chunk_text_level == 0:  # 对于正文文本，进行拼接
                    page_content = page_content + chunk_text + '\n'
                else:  # 对于标题信息
                    # 首先判断是否需要生成doc
                    if len(page_content) > 0 or chunk_idx == len(content_json) -1:  # 已经有正文内容，生成doc
                        # 准备元数据
                        doc_metadata = {
                            "file_type": "document",
                            "page_number": str(list(set(text_pages))),
                            "chunk_type": chunk_chunk_type,
                            "text_title": level_text_manager.get("text_title", ""),
                            "chapter_relation": chapter_relation,
                            "need_chunk": False
                            } | metadata

                        # 生成doc
                        new_doc = Document(page_content=page_content, metadata=doc_metadata)
                        output_docs.append(new_doc)

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
            else:  # 页码docs模式
                if len(chunk_text) == 0:
                    continue
                if record_page_num is None:
                    record_page_num = chunk_page_num

                if record_page_num != chunk_page_num or chunk_idx == len(content_json) - 1:  # 如果进入新的一页， 将旧页内容拼接上下文后保存为Document,同时更新record page信息以及重置page content
                    # print(f'保存当前页的内容:{page_content}')
                    end_text_id = chunk_idx
                    doc_metadata = {"file_type": "document",
                                    "page_number": str(record_page_num),
                                    "chunk_type": chunk_chunk_type,
                                    "text_title": level_text_manager.get("text_title", "")
                                    } | metadata
                    
                    # 需要拼接的上文内容
                    before_content = ""
                    before_chunks = content_json[max(start_text_id - self.default_parag_per_chunk, 0): start_text_id]
                    for before_chunk in before_chunks:
                        before_chunk_type = before_chunk["type"]
                        if before_chunk_type == "text":
                            before_chunk_text = get_chunk_text(before_chunk, before_chunk_type)
                        else:
                            before_chunk_text = ""
                        before_content = before_content + before_chunk_text + "\n"

                    after_content = ""
                    after_chunks = content_json[end_text_id: end_text_id + self.default_parag_per_chunk]
                    for after_chunk in after_chunks:
                        after_chunk_type = after_chunk["type"]
                        if after_chunk_type == "text":
                            after_chunk_text = get_chunk_text(after_chunk, after_chunk_type)
                        else:
                            after_chunk_text = ""
                        after_content = after_content + after_chunk_text + "\n"

                    # 拼接上下文
                    all_content = "*正文内容*:" + '\n' + page_content
                    if len(before_content):
                        all_content = '*上文内容*:' + '\n' + before_content + "\n" + all_content
                    if len(after_content):
                        all_content = all_content + '\n' + "*下文内容*:" + '\n' + after_content
                    new_doc = Document(page_content=all_content, metadata=doc_metadata)
                    # print(f'第{record_page_num}页的内容：\n{all_content}')
                    output_docs.append(new_doc)

                    # 变量更新与重置
                    record_page_num = chunk_page_num
                    start_text_id = chunk_idx
                    page_content = chunk_text + '\n'
                else:
                    if start_text_id == -1:
                        start_text_id = chunk_idx
                    page_content = page_content + chunk_text + '\n'

        return output_docs
    
    def docs_pipeline(self,
                      docs: Optional[Document] = None, 
                      do_text_clean: bool=True, 
                      do_text_chunk: bool=True,
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

            # 内容补完
            print('文本补完')
            processed_contents = self.add_info(processed_contents)

            # 为分块处理后的doc添加uuid
            print("添加uuid")
            # print(f'输入的变量信息：{processed_contents}')
            processed_contents = self.add_uuid(processed_contents)
            summary_contents = Document(page_content='', metadata={})

            # 文本概括
            if do_text_summary:
                print(f'执行文本概括')
                # print(f'输入的变量信息：{processed_contents}')
                summary_contents = self.text_summary(processed_contents)
                gc.collect()

            # 拼接输出字典
            for processed_content, summary_content in zip(processed_contents, summary_contents):
                output_list.append(
                    {
                        "content": processed_content,
                        "summary": summary_content
                    }
                )

            #返回结果
        return output_list
            
    def table_pipeline(self, file_info: List[dict]) -> List[Document]:
        """
        表格文件/对象的处理管道
        """
        table_docs = []
        for f_info in file_info:
            table_path = f_info.get("element_path", None)
            if table_path:
                table_text = file_loader(table_path)
            else:
                table_text = f_info["element_content"]

            # 表格内容转json
            table_json = html_to_json(table_text)

            # 表格内容拼接
            table_content = '表格内容JSON：' + "\n" + str(table_json)
            table_caption = f_info.get("table_tile", "")
            table_footnote = f_info.get("table_footnote", "")
            if len(table_caption):
                table_content = "表格标题：" + "\n" + table_caption + '\n' + table_content
            if len(table_footnote):
                table_content = table_content + '\n' + "表格尾注信息：" + "\n" + table_footnote + '\n'
            # print(f'拼接得到的表格内容为:{table_content}')

            # 提取表格描述
            table_content = self.table_content_description(table_content)
            
            # 更新元数据信息
            table_doc = Document(page_content=table_content, metadata={})
            table_doc.metadata["source"] = f_info.get("source", "")
            table_doc.metadata["file_type"] = f_info.get("file_type", "")
            table_doc.metadata["access_level"] = f_info.get("access_level", self.default_file_access_level)
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
            audio_doc.metadata["access_level"] = f_info.get("access_level", self.default_file_access_level)
            audio_doc.metadata["chunk_type"] = 'audio'
            audio_doc.metadata['page_number'] = f_info.get("page_num", "-1")
            audio_docs.append(audio_doc)
        
        return audio_docs

    def image_pipeline(self, file_info:List[dict]) -> List[Document]:
        """
        图像处理管道函数
        """
        image_docs = []
        for f_file in file_info:
            # print('图像信息:', f_file)
            # 信息提取
            image_path = f_file.get("element_path", None)
            image_caption = f_file.get("img_caption", "")
            image_footnote = f_file.get("img_footnote", "")

            # 生成图片描述
            image_description = self.image_caption(image_path=image_path)  # 图像模型调用耗费资源，暂时不异步执行
            # print(f'图像描述内容: {image_description}')

            # 综合生成图片描述
            image_description_fusion = '图片描述：' + 'image_description'
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
            image_doc.metadata["access_level"] = f_file.get("access_level", self.default_file_access_level)
            image_doc.metadata["chunk_type"] = "image"
            image_doc.metadata['page_number'] = f_file.get("page_num", "-1")
            image_docs.append(image_doc)

        return image_docs

    async def text_pipeline_async(self, 
                                  file_info: List[dict] = None,
                                  text_docs: Optional[Document] = None, 
                                  do_text_clean: bool=True, 
                                  do_text_chunk: bool=True,
                                  do_text_summary: bool=True,
                                  do_table_summary: bool=False,
                                  do_question_derive: bool=True,
                                  need_load: bool=True
                                  ) -> List[dict]:
        """
        文本处理流水线函数
        """
        if file_info:
            text_docs = []
            for f_info in file_info:
                if need_load:
                    print("f_info:", type(f_info), f_info)
                    # 读取文件，生成document对象
                    text_path = f_info.get("element_path", None)
                    text_doc = file_loader(text_path)[0]
                    # print(f"文本信息：{text_doc}")

                    # 更新元数据信息
                    text_doc.metadata["source"] = f_info.get("source", "")
                    text_doc.metadata["file_type"] = f_info.get("file_type", "")
                    text_doc.metadata["access_level"] = f_info.get("access_level", self.default_file_access_level)
                    text_doc.metadata["chunk_type"] = "text"
                    text_docs.append(text_doc)
                else:
                    text_docs = file_info
                    break
        elif text_docs:
            pass
        else:
            raise ValueError(f'file_info 和 text_docs 不能同时为空')

        # 文本去格式化(markdown转text)
        # print(f'开始对文本去格式化')
        # processed_contents = self.text_clear_formatting(text_docs)
        processed_contents = text_docs

        # 文本清洗
        if do_text_clean:
            print(f'执行文本清洗')
            # print(f'输入的变量信息：{processed_contents}')
            processed_contents = self.clean_text(processed_contents)

        # 文本分块
        if do_text_chunk:
            print(f'执行文本分块')
            # print(f'文本分块 输入的变量信息：{processed_contents}')
            processed_contents = await self.text_chunk_splitter_handler_async(processed_contents)
            # print(f'文本分块 输出的变量信息：{processed_contents}')

        # 为分块处理后的doc添加uuid
        print("添加uuid")
        # print(f'输入的变量信息：{processed_contents}')
        processed_contents = self.add_uuid(processed_contents)
        summary_contents = [Document(page_content='', metadata={})] * len(processed_contents)
        question_contents = [Document(page_content='', metadata={})] * len(processed_contents)

        # 文本概括
        if do_text_summary:
            print(f'执行文本概括')
            # print(f'输入的变量信息：{processed_contents}')
            summary_contents = await self.text_summary_handler_async(processed_contents)

        # 表格概括
        if do_table_summary:
            print(f'执行表格内容概括')
            # print(f'输入的变量信息：{processed_contents}')
            summary_contents = await self.table_summary_handler_async(processed_contents)
            # print(f'输出的变量信息：{summary_contents}')

        # 问题派生
        # if do_question_derive:
        #     print(f'执行派生提问')
        #     # print(f'输入的变量信息：{processed_contents}')
        #     question_contents = await self.text_question_derive_handler_async(processed_contents)
        #     # print(f'输出的变量信息：{question_contents}')

        # 拼接输出字典
        output_list = []
        for content_doc, summary_doc, questions_doc in zip(processed_contents, summary_contents, processed_contents):
            output_list.append(
                {
                    "content": content_doc,
                    "summary": summary_doc,
                    "questions": questions_doc
                }
            )


        return output_list

    async def table_pipeline_async(self, file_info: dict) -> List[dict]:
        """
        表格文件/对象的处理管道
        """
        table_docs = []
        for f_info in file_info:
            table_path = f_info.get("element_path", None)
            if table_path:
                table_text = file_loader(table_path)
            else:
                table_text = f_info["element_content"]

            # 表格内容转json
            table_json = html_to_json(table_text)

            # 表格内容拼接
            table_caption = f_info.get("table_tile", "")
            table_footnote = f_info.get("table_footnote", "")
            table_content = table_caption + '\n' + "表格内容：" + "\n" + str(table_json) + "\n" + table_footnote
            # print(f'拼接得到的表格内容为:{table_content}')
            
            # 更新元数据信息
            # print(f'表格json信息：{table_json}')
            table_doc = Document(page_content=table_content, metadata={})
            table_doc.metadata["source"] = f_info.get("source", "")
            table_doc.metadata["file_type"] = f_info.get("file_type", "")
            table_doc.metadata["access_level"] = f_info.get("access_level", self.default_file_access_level)
            table_doc.metadata['page_number'] = f_info.get("page_num", "-1")
            table_doc.metadata["chunk_type"] = 'table'
            table_docs.append(table_doc)

        # 表格元素处理
        if table_docs:
            processed_contents = await self.text_pipeline_async(text_docs=table_docs, do_text_chunk=False, do_text_summary=False, do_table_summary=True)
            return processed_contents
        else:
            table_docs

    async def audio_pipeline_async(self, file_info: dict) -> List[dict]:
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
            audio_doc.metadata["access_level"] = f_info.get("access_level", self.default_file_access_level)
            audio_doc.metadata["chunk_type"] = 'audio'
            audio_doc.metadata['page_number'] = f_info.get("page_num", "-1")
            audio_docs.append(audio_doc)
        
        # 音频文本处理
        if audio_docs:
            processed_contents = await self.text_pipeline_async(text_docs=audio_docs)
            return processed_contents
        else:
            return audio_docs

    async def image_pipeline_async(self, file_info:dict):
        """
        图像处理管道函数
        """
        image_docs = []
        for f_file in file_info:
            # 信息提取
            image_path = f_file.get("element_path", None)
            image_content = f_file.get("element_content", "")
            print(f'图像上下文内容: {image_content}')

            # 生成图片描述
            image_description = self.image_caption_async(image_path=image_path)  # 图像模型调用耗费资源，暂时不异步执行
            print(f'图像描述内容: {image_description}')

            # 识别图片的文字内容
            # image_text = image_text_reader(image_path)
            # print(f'图像文字内容: {image_text}')

            # 综合生成图片描述
            image_description_fusion = self.image_description_fusion_async(image_description=image_description,
                                                                           image_content=image_content
                                                                           )
            print(f"图像的综合描述结果：{image_description_fusion}")

            # 生成文本doc
            image_doc = Document(page_content=image_description_fusion, metadata={})

            # 更新元数据信息
            print("开始更新和同步图片的元数据信息")
            image_doc.metadata["source"] = f_file.get("source", "")
            image_doc.metadata["file_type"] = f_file.get("file_type", "")
            image_doc.metadata["access_level"] = f_file.get("access_level", self.default_file_access_level)
            image_doc.metadata["chunk_type"] = "image"
            image_doc.metadata['page_number'] = f_file.get("page_num", "-1")
            image_docs.append(image_doc)

        # 图像描述文本处理
        print("开始对图像文本进行处理")
        if image_docs:
            processed_contents = await self.text_pipeline_async(text_docs=image_docs, do_text_chunk=False)
            return processed_contents
        else:
            return image_docs

    
    ###################################################### 工具/流程函数  ######################################################
    def load_models(self):
        """
        加载模型或模型服务客户端的函数
        """
        # 加载ollam客户端
        self.async_client = ollama.AsyncClient()
        self.client = ollama.Client()

        # 检查ollama模型是否存在本地
        local_models = [model["model"] for model in ollama.list()['models']]

        if self.generative_model_name not in local_models:
            print(f'未找到ollama模型{self.generative_model_name},请检查')
        if self.text_embedding_model_name not in local_models:
            print(f'未找到ollama模型{self.text_embedding_model_name},请检查')
        if self.image_caption_model_name not in local_models:
            print(f'未找到ollama模型{self.image_caption_model_name},请检查')

        # 加载音频识别函数
        audio_model_name = self.audio_model_address
        audio_vad_mode_address = self.audio_vad_model_address
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

    def connect_to_weative_database(self):
        """
        创建一个到weative数据库的连接
        """
        # 参数提取
        vector_database_host = self.vector_database_host
        vector_database_port = self.vector_database_port
        vector_database_grpc_port = self.vector_database_grpc_port

        # 构建连接
        try:
            self.vector_database_client = weaviate.connect_to_custom(
                http_host=vector_database_host,
                http_port=int(vector_database_port),
                http_secure=False,
                grpc_host='localhost',
                grpc_port=int(vector_database_grpc_port),
                grpc_secure=False,
            )

            # 检查连接是否成功
            self.vector_database_client.is_ready() # 检查连接是否成功
            print(f'连接到weative数据库成功')
            return self.vector_database_client
        except Exception as e:
            print(f"Failed to connect to Weaviate: {e}")
            raise

    def close_weavier_connection(self):
        """
        一个专用的关闭方法，用于在应用退出前释放资源。
        """
        print("断开和Weavier数据库的连接")
        if self.vector_database_client:
            self.vector_database_client.close()

    def file_to_knowledge_database(self,
                                   file_path: str,
                                   metadata_dict: dict,
                                   collection_name: str):
        """
        负责单个文件的ETF流程,同步操作。
        """
        print(f'正在处理文件:{file_path}')
        if "source" not in metadata_dict:
            metadata_dict["source"] = file_path

        temp_dir_path = None  # 初始化临时保存地址
        try:
            # 1 识别文件类型
            print('识别文件类型')
            file_type = self.file_type_identifier(file_path)

            # 2 基于文件类型，提取其中的元素信息
            print(f'当前文件的类型是：{file_type}')
            if file_type == 'document':
                # 文件解析
                parse_result = file_loader(file_path, self.output_dir)
                docs = self.document_content_parser_fusion(parse_result=parse_result, metadata=metadata_dict)
                # for doc in docs:
                #     print(f'doc的主题：{doc.metadata.get("text_title", "")}')
                #     print(f'doc的章节信息：{doc.metadata.get("chapter_relation", "")}')
                #     print(f'doc的内容：{doc.page_content}')
                #     print(f'是否需要再分块：{doc.metadata.get("need_chunk", "")}')
                # raise ValueError('人工暂停')
            elif file_type == 'image':
                image_info = [
                    {
                        "element_path": file_path,
                        "file_type": 'image',
                        "source": file_path
                    } | metadata_dict
                ]
                docs = self.image_pipeline(file_info=image_info)
            elif file_type == 'audio':
                audio_info = [
                    {
                        "element_path": file_path,
                        "file_type": 'audio',
                        "source": file_path
                    } | metadata_dict
                ]
                docs = self.audio_pipeline(file_info=audio_info)
            elif file_type == 'table':
                table_info = [
                    {
                        "element_path": file_path,
                        "file_type": 'table',
                        "source": file_path
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
            print(f'doc数量: {len(docs)}')
            if file_type in ['document', 'text']:
                processed_contents = self.docs_pipeline(docs=docs)
            else:
                processed_contents = self.docs_pipeline(docs=docs, do_text_chunk=False)

            # 4 入库信息生成
            print('入库信息生成')
            # print(f'输入内容：{processed_contents}')
            records_to_save = self.save_knowledge_data_generator(processed_contents)
            # print("入库信息如下：")
            # print(records_to_save)

            # 5 信息入库
            print('信息入库')
            successful_insert_info = self.save_to_vector_database(records_to_save, collection_name)

            # 6 删除临时文件和文件夹
            if temp_dir_path:
                shutil.rmtree(temp_dir_path)

            print(f"文件处理成功: {file_path}")
            return {"status": "00", "msg": '处理成功'}
        except Exception as e:
            print(f"文件处理失败: {file_path}, 错误: {e}")
            return {"status": "05", "msg": f"处理失败。错误信息为:{e}"}      
       
    def file_to_knowledge_database_handler(self, file_paths: List[str], metadata: List[dict] = None, collection_name:str=None):
        """
        文件处理函数,将传入的文件处理后存入向量数据库
        """
        # 元数据处理
        if metadata is None:
            metadata = [{"access_level": 0}] * len(file_paths)

        for file_path, meta_info in zip(file_paths, metadata):
            response = self.file_to_knowledge_database(file_path=file_path, metadata_dict=meta_info, collection_name=collection_name)

        return response

    async def file_to_knowledge_database_async(self, 
                                               file_path: str, 
                                               metadata_dict: dict,  # 至少包含原始文件地址（source）和访问等级（access_level）两个信息
                                               semaphore: asyncio.Semaphore):
        """
        负责单个文件的ETF流程。这个函数内部包含了所有的异步和批量操作
        """
        if "source" not in metadata_dict:
            metadata_dict["source"] = file_path

        async with semaphore:
            temp_dir_path = None  # 初始化临时保存地址
            try:
                # 1. 识别文件类型
                print('识别文件类型')
                file_type = self.file_type_identifier(file_path)

            
                # 2 基于文件类型，提取其中的元素信息
                print(f'当前文件的类型是：{file_type}')
                if file_type == 'document':
                    # 文件解析
                    element_list = []
                    print("开始解析文件")
                    parse_result = file_loader(file_path, self.output_dir)
                    print("开始聚合解析结果：")
                    print(metadata_dict)
                    docs, images, tables = self.document_content_parser_fusion(parse_result=parse_result, metadata=metadata_dict)
                    print(f"文本页数:{len(docs)}")
                    for idx, doc in enumerate(docs):
                        print("当前页数", doc.metadata["page_number"])
                        print("页面字数", len(doc.page_content))
                    print("开始拼接输出")
                    element_list = [
                        {
                            "element_type": 'text',
                            "need_load": False,
                            "element_info": docs
                        },
                        {
                            "element_type": 'table',
                            "need_load": True,
                            "element_info": tables
                        },
                        {
                            "element_type": 'image',
                            "need_load": True,
                            "element_info": images
                        }
                    ]
                    # element_list = [{"text_docs": docs, "element_type": "text", "source": file_path, "file_type": file_type}] + images + tables
                else:
                    # 读取文件
                    element_list = [
                        {
                            "element_type": file_type,
                            "need_load": True,
                            "element_info": [
                                {
                                "element_path": file_path, 
                                "element_type": file_type, 
                                "source": file_path, 
                                "file_type": file_type
                                } | metadata_dict
                            ]
                        } 
                    ]
                # 3 基于元素类型进行分流处理
                for element in element_list:
                    element_type = element["element_type"]
                    element_info = element["element_info"]
                    need_load = element["need_load"]
                    print(f'当前文档元素的类型为:{element_type}')
                    if element_type == "text":  # 3.1 文本处理
                        # print('暂不处理文本内容')
                        # continue
                        print('开始处理文本内容')
                        processed_contents = await self.text_pipeline_async(file_info=element_info, need_load=need_load)
                    elif element_type == "table":  # 3.2 表格文件处理
                        # print('暂不处理表格内容')
                        # continue
                        print('开始处理表格内容')
                        processed_contents = await self.table_pipeline_async(file_info=element_info)
                    elif element_type == "audio": # 3.3 音频文件处理
                        # print('暂不处理表音频内容')
                        # continue
                        print('开始处理音频内容')
                        processed_contents = await self.audio_pipeline_async(file_info=element_info)
                    elif element_type == "image":  # 3.4 图像文件处理
                        # print('暂不处理图像内容')
                        # continue
                        print('开始处理图像内容')
                        processed_contents = await self.image_pipeline_async(file_info=element_info)
                    else:
                        raise ValueError(f'不支持的文件类型：{file_type}')

                    # 入库信息生成
                    print('入库信息生成')
                    # print(f'输入内容：{processed_contents}')
                    records_to_save = self.save_knowledge_data_generator(processed_contents)
                    # print("入库信息如下：")
                    # print(records_to_save)

                    # 6. 信息入库
                    print('信息入库')
                    successful_insert_info = await asyncio.to_thread(self.save_to_vector_database, records_to_save)

                # 删除临时文件和文件夹
                if temp_dir_path:
                    shutil.rmtree(temp_dir_path)

                print(f"文件处理成功: {file_path}")
                return {"status": "00", "msg": '处理成功'}
            except Exception as e:
                print(f"文件处理失败: {file_path}, 错误: {e}")
                return {"status": "05", "msg": f"处理失败。错误信息为:{e}"}      
        
    async def file_to_knowledge_database_handler_async(self, file_paths: List[str], metadata: List[dict] = None):
        """
        文件处理函数,将传入的文件处理后存入向量数据库
        """
        # 元数据处理
        if metadata is None:
            metadata = [{"access_level": 0}] * len(file_paths)

        # 限制并发数量
        semaphore = asyncio.Semaphore(self.concurrency_limit)

        # 创建协程任务
        tasks = [asyncio.create_task(self.file_to_knowledge_database_async(path, meta, semaphore)) for path, meta in zip(file_paths, metadata)]

        # 执行任务
        results = await asyncio.gather(*tasks)
        return results

    async def search_knowledge_database(self, request: str, 
                                        enable_request_serach: bool=True, 
                                        enable_summary_search: bool=True, 
                                        enable_keyword_search: bool=True):
        """
        请求向量数据库,从知识库中搜索相关信息
        """
        # 连接数据表
        collection = self.vector_database_client.collections.get("knowledge_base_collection")
        
        # 检索数据库
        print(f"请求为:{request}")
        search_response_collections = []
        if enable_request_serach:  # 基于请求信息的检索
            question_query_response = collection.query.near_text(
                query=request,
                limit=self.search_max_num,
                target_vector="content_vector",
                return_metadata=MetadataQuery(distance=True)           
            )
            search_response_collections.append(question_query_response)
            print(f"基于请求的检索结果为:")
            for i, o in enumerate(question_query_response.objects):
                print(f'返回次序{i+1}')
                # print(f'当前内容对应的请求/问题为:{o.properties["questions"]}')
                print(o.properties["content"])
                print('***')
            print("***")

        if enable_summary_search:  # 基于回答概述的检索
            # 生成请求对应的答案的概括内容
            # answer_summary = await self.search_answer_summary_async(request=request)
            summary_query_response = collection.query.near_text(
                query=request,
                # query=request,
                limit=self.search_max_num,
                target_vector="summary_vector",
                return_metadata=MetadataQuery(distance=True)
            )
            search_response_collections.append(summary_query_response)
            print(f"基于概括的检索结果为:")
            for i, o in enumerate(summary_query_response.objects):
                print(f'返回次序{i+1}')
                print(o.properties["content"])
                print('***')
            print("***")

        if enable_keyword_search:  # 关键字/词检索
            keyswords = await self.search_extract_keywords_async(request=request)
            keywords_response = collection.query.bm25(
                query=keyswords,
                query_properties=["content"],
                limit=self.search_max_num,
                operator=BM25Operator.or_(minimum_match=1)
            )
            search_response_collections.append(keywords_response)
            print(f"问题的关键字为:{keyswords}")
            print(f"基于关键词的回答为:")
            for i, o in enumerate(keywords_response.objects):
                print(f'返回次序{i+1}')
                print(o.properties["content"])
                print('***')
            print("***")

        # rrf算法输出结果
        # print("rrf算法rerank")
        rrf_response = apply_rrf(search_response_collections)
        # print("rerank结果", rrf_response)

        # 提取top匹配结果
        uuid_lst = rrf_response
        id_filter = Filter.by_id().contains_any(uuid_lst)
        response =  collection.query.fetch_objects(filters=id_filter)

        # 返回结果
        return response

if __name__ == '__main__':
    # /home/carlos/Projects/SmartAgent/data/Documents/2025数据分析Agent实践与案例研究报告.pdf
    # /home/carlos/Projects/SmartAgent/data/Documents/室内空气质量检测报告.pdf
    file_paths = [
        '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海 】秦洋 5年.pdf',
        '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海 】申靳超 6年.pdf',
        '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海】曾柏栋 9年.pdf',
        '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海】卫驰 7年.pdf',
        '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师_上海】张书琪 6年.pdf',
        '/home/carlos/Projects/SmartAgent/data/Documents/【数据分析工程师(J10001)_上海 】张昊明 4年.pdf',
        '/home/carlos/Projects/SmartAgent/data/Documents/【算法_上海】吴金亮 6年.pdf',
        '/home/carlos/Projects/SmartAgent/data/Documents/C00015642-王明谦-数据分析工程师.pdf',
        '/home/carlos/Projects/SmartAgent/data/Documents/C00016488-李明-数据分析工程师.pdf',
        '/home/carlos/Projects/SmartAgent/data/Documents/C00016934-于畅-数据分析工程师.pdf',
        '/home/carlos/Projects/SmartAgent/data/Documents/2025数据分析Agent实践与案例研究报告.pdf'
        ]
    test_tool = RAGTool()
    collection_name = 'text_collection'

    # 检查和加载模型
    test_tool.load_models()

    # 获取数据库连接
    test_tool.connect_to_weative_database()

    # 删除数据表
    test_tool.delete_collection(collection_name=collection_name)

    # 创建数据表
    test_tool.craete_collection(collection_name=collection_name)

    # pdf处理和保存测试
    response = test_tool.file_to_knowledge_database_handler(file_paths, collection_name=collection_name)
    print(f'文档处理结果:{response}')

    # # 遍历向量数据库
    # collection = test_tool.vector_database_client.collections.get(collection_name)
    # count = 1
    # for obj in collection.iterator(
    #     return_properties=True, # 获取所有自定义字段
    #     return_metadata=MetadataQuery(
    #         creation_time=True,
    #         last_update_time=True,
    #         # 如果您也需要向量本身，可以取消下面这行的注释
    #         # vector=True 
    #     )
    # ):
    #     print(f"***第{count}条数据记录***")
    #     print(f"数据记录uuid:{str(obj.uuid)}")
    #     print(f"数据记录的类型：{obj.properties['chunk_type']}")
    #     print(f"数据记录的来源：{obj.properties['source']}")
    #     print(f"文本所在的页码：{obj.properties['page_number']}")
    #     print(f"数据记录的内容：{obj.properties['content']}")
    #     print(f"数据记录的摘要：{obj.properties['summary']}")
    #     count += 1


    # # 向量数据库检索
    # test_collection = test_tool.vector_database_client.collections.get("knowledge_base_collection")
    # request_info = [
    #     "本次空气检测的委托单位是哪一家？",  #1
    #     "本次检测使用了哪些检测设备？",  #2
    #     "本次检测的主要检测点都有哪里？",  #3
    #     "检测的主要污染物有哪些?",  #4
    #     "本次检测的结果如何？",  #5
    #     "数据分析agent研究报告一共多少章节？",  #6
    #     "数据分析的发展经过了哪几个阶段？",  #7
    #     "企业AI Agent有哪些应用方向？",  #8
    #     "数据分析Agent的关键能力体现在哪里？",  #9
    #     "请问“多Agent”这个概念的定义",  #10
    #     "在金融领域，数据分析agent主要应用与哪些任务中？",  #11
    #     "企业在引入数据分析Agent前，需要如何进行准备？",  #12
    #     "目前，国内有哪些数据分析Agent的解决方案？",  #13
    #     "SwiftAgent的产品架构一共有几层？",  #14
    #     "wiftAgent的核心优势有哪些？",  #15
    #     "请提供一些数据分析Agent的应用案例",  #16
    #     "反问机制是如何解决人员模糊提问问题的？",  #17
    #     "AI报告生成项目的落地步骤是怎样的？",  #18
    #     "数据分析agent的定义是什么？",  #19
    #     "请简述数据分析agent的应用现状"  #20
    # ]
    # for ind, request in enumerate(request_info):
    #     seach_results = asyncio.run(test_tool.search_knowledge_database(request=request))
    #     print('*****************************')
    #     print(f'第{ind+1}个问题:{request}')
    #     for idx, result in enumerate(seach_results.objects):
    #         print(f'第{idx+1}顺位的关联文本：')
    #         print(result.properties["content"])
    #         print('**\n ** \n')
    #     print("************")

    # 关闭数据库连接
    test_tool.close_weavier_connection()