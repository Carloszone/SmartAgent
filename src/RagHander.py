# 目前没有合适的中文图文embedding模型，因此对图片的处理模式为：描述图片->描述文本embedding 所以不支持以图搜图

from ConfigManager import config
import os
import re
from typing import List, Union, Optional
from langchain_core.documents import Document
from ContentHandler import get_system_message, json_extractor, apply_rrf, find_html_tables_in_markdown, find_files_in_directory, get_context_around_image, file_loader
from ContentHandler import html_to_json, file_loader
import weaviate
import asyncio
from weaviate.classes.query import Filter, MetadataQuery, BM25Operator
from weaviate.util import generate_uuid5
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
        self.long_text_threshold = config.get_setting("file")["long_text_threshold"]  # 长文本的最低字符限制
        self.document_types = config.get_setting("files")["types"]["text_file_extension"]  # 文档后缀
        self.table_types = config.get_setting("files")["types"]["table_file_extension"]  # 表格后缀
        self.audio_types = config.get_setting("files")["types"]["audio_file_extension"]  # 音频后缀
        self.image_types = config.get_setting("files")["types"]["image_file_extension"]  # 图像后缀
        self.default_file_access_level = config.get_setting("files")["default_file_access_level"]  # 文档的默认访问级别

        # 输出相关
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        project_root = os.path.dirname(script_dir)
        self.output_dir = os.path.join(project_root, config.get_setting("files")["output_dir"])

        # 其他参数：
        self.model_retry_num = 3  # 遇到模型报错时的重试次数
        self.concurrency_limit = 4  # 并发控制最大数

    ###################################################### 数据库操作函数  ######################################################
    def save_to_vector_database(self, data_dict: dict = None, mode="knowledge_base"):
        """
        将数据存入数据库
        """
        # print('待入库信息：', data_dict)
        if data_dict is None:
            print(f'没有数据需要保存')
            return
        else:
            # 提取配置信息
            if mode == "knowledge_base":
                vector_database_collection_name = self.knewledge_base_collection_name
            elif mode == "chat_history":
                vector_database_collection_name = self.chat_collection_name
            else:
                raise ValueError(f'参数mode错误，当前参数为{mode}')

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
                    if mode == "knowledge_base":
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
  
    def get_weavier_collection_info(self):
        """
        获取指定collection的字段信息
        """
        pass

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
                raise ValueError(f'不支持的文件格式:{file_extension}')

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
                   docs: List[Document],
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
        new_docs = []  # 储存清洗后的文档

        for doc in docs:
            # 文档复制
            content = doc.page_content 

            # (英文内容)替换多个空格为一个
            if do_clear_space:
                print('合并空格')
                print(f"输入内容：{content}")
                content = re.sub(r'\s+', ' ', content)

            # (英文内容)替换多个换行符为一个
            if do_clear_enter:
                print('合并换行符')
                print(f"输入内容：{content}")
                content = re.sub(r'\n+', '\n', content)

            # (英文内容)去除文档中的连字符
            if do_clear_hyphen:
                print('去除连字符')
                print(f"输入内容：{content}")
                content = re.sub(r'-\s*\n', '', content)

            # (中文内容)移除中文文本之间的多余空格
            if do_clear_space_ch:
                print('删除中文文本之间的空格')
                print(f"输入内容：{content}")
                content = re.sub(r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])', r'\1\2', content)

            # 合并被错误切分的段落或句子
            if do_sentence_fix:
                print('修复被错误切分的段落或句子')
                print(f"输入内容：{content}")
                content = re.sub(r'(?<![。！？\n])\n', '', content)
                content = re.sub(r'\n+', '\n', content)

            # 去除页眉页脚和页码等噪音信息
            if do_clear_page_tag:
                print('删除页眉页脚等信息')
                print(f"输入内容：{content}")
                content = re.sub(r'(?i)第\s*\d+\s*页', '', content)
                content = re.sub(r'(?i)page\s*\d+', '', content)

            # 移除重复性的乱码噪声
            if do_clear_noise:
                print("删除重复性乱码")
                print(f"输入内容：{content}")
                noise_pattern = r"([\s`\\/’'\'V丶、()]+){5,}"
                content = re.sub(noise_pattern, ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()

            # 去除无法识别的乱码与字符
            if do_clear_unknown_letter:
                print("统一文字与符号")
                print(f"输入内容：{content}")
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
                print("删除表情符号")
                print(f"输入内容：{content}")
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
                print("删除开头和结尾的空白符")
                print(f"输入内容：{content}")
                content = content.strip()

            # 构建一个新的Document对象
            new_doc = Document(page_content=content, metadata=doc.metadata)
            new_docs.append(new_doc)
        return new_docs

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
    
    async def text_fusion_async(self, content: dict) -> Document:
        """"
        基于上下文对当前页面的内容进行补完
        """
        # 输入解析
        above_document = content.get('above_content', '')
        if isinstance(above_document, Document):
            above_content = above_document.page_content
        else:
            above_content = above_document

        target_document = content.get('target_content')
        target_content = target_document.page_content

        below_document = content.get('below_content', '')
        if isinstance(below_document, Document):
            below_content = below_document.page_content
        else:
            below_content = below_document

        # message构建
        system_message = get_system_message('content_pitch')
        message = [
            {
                "role": "system",
                "content": system_message
             },
             {
                "role": "user",
                "content": f"请基于以下信息，对目标页内容进行拼接：上文页内容: {above_content}, 目标页内容：{target_content}, 下文页内容：{below_content}" 
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
        try:
            output_json = json_extractor(response['message']['content'])

            # 格式化输出
            if "target_content" in output_json:
                output_content = output_json["target_content"]
                output_object = Document(page_content=output_content, metadata=target_document.metadata)
                return output_object
            else:
                return last_exception
        except ValueError as e:
            return e

    async def text_fusion_handler_async(self, contents: List[dict]) -> List[Document]:
        """
        对输入的文档内容进行内容拼接的函数(异步处理)
        """
        # print('拼接输入:', contents)
        tasks = [self.text_fusion_async(content) for content in contents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print('拼接返回：')
        for i, result in enumerate(results):
            print(i)
            print(result)

        # 过滤掉处理失败的结果
        successful_results = [res for res in results if isinstance(res, Document)]

        # 返回警告
        if len(successful_results) < len(results):
            print(f'发现{len(results) - len(successful_results)}个文档拼接失败，跳过处理')
        return successful_results

    async def text_chunk_splitter_async(self, content: Document) -> List[Document]:
        """
        基于markdown格式的分本分块函数
        """
        # 输入解析
        input_content = content.page_content

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
        print(f"***clean content: {clean_output}")
        try:
            output_json = json_extractor(clean_output)
            print(f"***提取出的json对象为:{output_json}")

            # 返回格式化后的文档内容
            formatted_chunkers = []
            if "chunks" in output_json:
                output_list = output_json["chunks"]
                for index, output_content in enumerate(output_list):
                    chunk_metadata = content.metadata
                    chunk_metadata['chunk_seq_id'] = index
                    chunk_document = Document(page_content=output_content, metadata=chunk_metadata)
                    formatted_chunkers.append(chunk_document)
                return formatted_chunkers
            else:
                return last_exception
        except Exception as e:
            print(f"文档分块失败：{e}")
            return e

    async def text_chunk_splitter_handler_async(self, contents: List[Document]) -> List[Document]:
        """
        基于markdown格式的文本分块函数(批量)
        """
        # print(f'文档分块输入内容：', contents)
        tasks = [self.text_chunk_splitter_async(content) for content in contents if isinstance(content, Document)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print(f'文档分块输出内容：', results)

        # 展评list
        output_results = []
        for result in results:
            for res in result:
                if isinstance(res, Document):
                    output_results.append(res)
        return output_results
    
    def add_uuid(self, contents: List[Document]) -> List[Document]:
        """
        """
        # 输入解析
        new_contents = []
        for content in contents:
            input_content = content.page_content

            # 生成分块的哈希和uuid
            orignal_text = input_content.strip().lower()
            orignal_hash = hashlib.sha256(orignal_text.encode('utf-8')).hexdigest()
            generated_id = generate_uuid5(identifier=orignal_text, namespace=orignal_hash)
            content.metadata['hash'] = orignal_hash
            content.metadata['uuid'] = generated_id
            new_contents.append(content)
        return new_contents

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
        try:
            output_content = json_extractor(response['message']['content'])
            return output_content['content']
        except Exception as e:
            return e

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
        print(f'表格概括结果为:{results}')
        # 结果过滤
        successful_results = [res for res in results if isinstance(res, Document)]
        return successful_results
    
    async def image_caption_async(self, image_path):
        # 构建图片描述信息
        print('开始图片描述')
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
                response = await self.async_client.chat(model=self.image_caption_model_name, 
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
                print(f'图片描述模型失败 {e}')
                return e
 
        # 提取输出json
        print(f'开始提取json')
        try:
            output_content = json_extractor(response['message']['content'])
            print(f'***图像描述：{output_content}')
            return output_content["image_caption"]
        except Exception as e:
            print("json提取失败")
            return e

    async def image_description_fusion_async(self, image_description:str, image_content: str, image_text: str):
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
                "content": f"请结合以下信息，综合概括图片描述的内容。模型识别出的图片内容{image_description}, 图像的上下文的文本内容{image_content}, 图像里的文字内容{image_text}" 
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
        try:
            output_content = json_extractor(response['message']['content'])
            return output_content['description']
        except Exception as e:
            return e

    async def extract_keywords(self, request: str):
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

    async def text_model_processor_async(self, content, mode: str):
        """
        基于模式生成大模型message用于各种任务
        """
        # 解析content
        if isinstance(content, Document):
            if mode == "text_fusion":
                above_document = content.get('above_content', '')
                if isinstance(above_document, Document):
                    above_content = above_document.page_content
                else:
                    above_content = above_document

                target_document = content.get('target_content')
                target_content = target_document.page_content

                below_document = content.get('below_content', '')
                if isinstance(below_document, Document):
                    below_content = below_document.page_content
                else:
                    below_content = below_document
            else:
                message_info = Document.page_content
        elif isinstance(content, str):
            message_info = content
        else:
            raise ValueError(f'传入大模型处理的参数不正确，当前参数为 content = {content}, mode = {mode}')
        
        # 基于mode抽取system message,并构建与大模型交互的message
        message = None
        system_message = get_system_message(mode)
        if mode == "text_fusion":  # 基于页面上下文对当前页面内容进行补完
            message = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": f"请基于以下信息，对目标页内容进行拼接：上文页内容: {above_content}, 目标页内容：{target_content}, 下文页内容：{below_content}" 
                }
            ]
        elif mode == "chunks":  # 对传入的文本进行语义分块
            message = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": f"请对以下文本进行分块：{message_info}" 
                }
            ]
        elif mode == "text_summary":  # 对传入的语义块进行内容概括
            if len(message_info) >= self.long_text_threshold:
                message = [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": f"请对以下文本进行概括总结：{message_info}" 
                    }
                ]
            else:
                summary_json = {"summary": message_info}
                message = None
        elif mode == "questions":  # 基于传入的内容，生成可能的提问
            message = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": f"请针对以下内容设计提问：{message_info}" 
                }
            ]
        elif mode == "table_description":  # 对传入的表格进行内容概括
            if len(message_info) > 20:
                print('暂时不支持大型表格处理')
                pass
            else:
                message = [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": f"请对以下文本进行概括总结：{message_info}" 
                    }
                ]
        elif mode == "image_caption":  # 对传入的图像进行描述
            message = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": "请尽可能详尽地描述传入的图片",
                    "images": [Path(message_info)]
                }
            ]
        elif mode == "keywords":  #  对传入的请求信息进行关键词提取
            message = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": f"请提取以下请求信息中的关键词:{message_info}" 
                }
            ]
        else:
            raise ValueError(f'传入的mode参数不正确，当年参数值为{mode}')
        
        # 大模型交互
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
            
        # 删除模型的思考内容
        raw_response = response['message']['content']
        clean_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()

        # 提取返回的json信息
        output_json = json_extractor(clean_response)

        # 基于mode对模型进行针对性处理
        if mode == "text_fusion":
            if "target_content" in output_json:
                new_doc = Document(page_content=output_json["target_content"], metadata=content.metadata)
                return new_doc
            else:
                raise ValueError(f'没有在模型的返回中找到目标字段，当前返回内容为{output_json}')
        if mode == "chunks":
            formatted_chunkers = []
            if "chunks" in output_json:
                output_list = output_json["chunks"]
                for index, output_content in enumerate(output_list):
                    chunk_metadata = content.metadata
                    chunk_metadata['chunk_seq_id'] = index
                    chunk_document = Document(page_content=output_content, metadata=chunk_metadata)
                    formatted_chunkers.append(chunk_document)
                return formatted_chunkers
            else:
                return ValueError(f'没有在模型的返回中找到目标字段，当前返回内容为{output_json}')
        if mode == "text_summary":
            if "summary" not in output_json:
                warnings.warm(f'没有在模型的返回中找到目标字段，当前返回内容为，返回默认值信息')
                output_json = {"summary": message_info}
            new_doc = Document(page_content=output_json["summary"], metadata=content.metadata)
            return new_doc
        if mode == "questions":
            if "questions" not in output_json:
                warnings.warm(f'没有在模型的返回中找到目标字段，当前返回内容为，返回默认值信息')
                output_json = {"questions": ""}
            new_doc = Document(page_content=output_json["questions"], metadata=content.metadata)
        if mode == "table_description":
            if "content" in output_json:
                return output_json["content"]
            else:
                warnings.warm(f'没有在模型的返回中找到目标字段，当前返回内容为，返回默认值信息')
        if mode == "image_caption":
            if "image_caption" in output_json:
                return output_json["image_caption"]
            else:
                warnings.warm(f'没有在模型的返回中找到目标字段，当前返回内容为，返回默认值信息')
        if mode == "keywords":
            keyswords = ''
            if "keywords" in output_json:
                for keyword in output_json["keywords"]:
                    keyswords = keyswords + keyword + ' '

                if len(keyswords) == 0:
                    warnings.warn(message=f"请求{request}提取关键词失败，关键词检索的结果可能不准确")
                return keyswords
            else:
                warnings.warm(f'没有在模型的返回中找到目标字段，当前返回内容为，返回默认值信息')
        return last_exception

    async def text_model_processor_handler_async(self, contents: List[Document], mode: str) -> List[Document]:
        """
        批量调用大模型并进行处理的函数
        """
        modes = [mode] * len(contents)
        tasks = [self.text_model_processor_async(content, mode) for content, mode in zip(contents, modes)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print(f'表格概括结果为:{results}')
        # 结果过滤
        successful_results = [res for res in results if (isinstance(res, Document) or isinstance(res, dict) or isinstance(res, str))]
        return successful_results


    def save_knowledge_data_generator(self, contents:List[Document]):
        """
        生成用于知识库向量数据库的dict对象
        """
        data_dict = {}
        for content in contents:
            # 文本内容
            content_text = content["content"].page_content
            summary_text = content["summary"].page_content
            questions_text = content["questions"].page_content

            # 元数据：来自文件
            metadata = content["content"].metadata
            uuid = metadata.get("uuid")
            source = metadata.get("source")
            file_type = metadata.get("file_type", "")
            # page_number = metadata.get("page_number", -1)
            chunk_seq_id = metadata.get("chunk_seq_id", -1)
            access_level = metadata.get("access_level", 0)
            chunk_type = metadata.get("element_type", "")

            # 组合字段
            if uuid not in data_dict.keys():
                data_dict[uuid] = {"properties": {
                                       "content": content_text, 
                                       "summary": summary_text,
                                       "questions": questions_text,
                                       "source": source,
                                       "file_type": file_type,
                                       "chunk_type": chunk_type,
                                       "access_level": access_level,
                                    #    "page_number": page_number,
                                       "chunk_seq_id": chunk_seq_id
                                   }
                                }
        return data_dict


    ###################################################### 整合函数  ######################################################
    async def text_pipeline_async(self, 
                                  file_info: Optional[dict] = None,
                                  text_doc: Optional[Document] = None, 
                                  do_text_clean: bool=True, 
                                  do_text_fusion: bool=False,
                                  do_text_chunk: bool=True,
                                  do_text_summary: bool=True,
                                  do_table_summary: bool=False,
                                  do_question_derive: bool=True
                                  ) -> List[dict]:
        """
        文本处理流水线函数
        """
        if file_info:
            # 读取文件，生成document对象 
            text_path = file_info['element_path']
            text_doc = file_loader(text_path)[0]
            print(f"文本信息：{text_doc}")

            # 更新元数据信息
            text_doc.metadata["source"] = file_info.get("source", "")
            text_doc.metadata["file_type"] = file_info.get("file_type", "")
            text_doc.metadata["access_level"] = file_info.get("access_level", self.default_file_access_level)

        # 文本去格式化(markdown转text)
        print(f'开始对文本去格式化')
        processed_contents = self.text_clear_formatting([text_doc])

        # 文本清洗
        if do_text_clean:
            print(f'执行文本清洗')
            print(f'输入的变量信息：{processed_contents}')
            processed_contents = self.clean_text(processed_contents)

        # 文本拼接
        if do_text_fusion:
            print(f'执行文本内容拼接')
            print(f'输入的变量信息：{processed_contents}')
            processed_contents = await self.text_fusion_handler_async(processed_contents)

        # 文本分块
        if do_text_chunk:
            print(f'执行文本分块')
            print(f'输入的变量信息：{processed_contents}')
            processed_contents = await self.text_chunk_splitter_handler_async(processed_contents)

        # 为分块处理后的doc添加uuid
        print("添加uuid")
        print(f'输入的变量信息：{processed_contents}')
        processed_contents = self.add_uuid(processed_contents)
        summary_contents = [Document(page_content='', metadata={})] * len(processed_contents)
        question_contents = [Document(page_content='', metadata={})] * len(processed_contents)

        # 文本概括
        if do_text_summary:
            print(f'执行文本概括')
            print(f'输入的变量信息：{processed_contents}')
            summary_contents = await self.text_summary_handler_async(processed_contents)

        # 表格概括
        if do_table_summary:
            print(f'执行表格内容概括')
            print(f'输入的变量信息：{processed_contents}')
            summary_contents = await self.table_summary_handler_async(processed_contents)
            print(f'输出的变量信息：{summary_contents}')

        # 问题派生
        if do_question_derive:
            print(f'执行派生提问')
            print(f'输入的变量信息：{processed_contents}')
            question_contents = await self.text_question_derive_handler_async(processed_contents)
            print(f'输出的变量信息：{question_contents}')

        # 拼接输出字典
        output_list = []
        for content_doc, summary_doc, questions_doc in zip(processed_contents, summary_contents, question_contents):
            output_list.append(
                {
                    "content": content_doc,
                    "summary": summary_doc,
                    "questions": questions_doc
                }
            )

        #返回结果
        return output_list

    async def table_pipeline_async(self, file_info: dict) -> List[dict]:
        """
        表格文件/对象的处理管道
        """
        table_path = file_info.get("element_path", None)
        if table_path:
            table_text = file_loader(table_path)
        else:
            table_text = file_info["element_content"]

        # 表格内容转json
        table_json = html_to_json(table_text)
        
        # 更新元数据信息
        print(f'表格json信息：{table_json}')
        table_doc = Document(page_content=str(table_json), metadata={})
        table_doc.metadata["source"] = file_info.get("source", "")
        table_doc.metadata["file_type"] = file_info.get("file_type", "")
        table_doc.metadata["access_level"] = file_info.get("access_level", self.default_file_access_level)

        # 表格元素处理
        processed_contents = await self.text_pipeline_async(text_doc=table_doc, do_text_chunk=False, do_text_summary=False, do_table_summary=True)
        return processed_contents

    async def audio_pipeline_async(self, file_info: dict) -> List[dict]:
        """
        音频文件的处理管道
        """
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
        audio_doc.metadata["source"] = file_info.get("source", "")
        audio_doc.metadata["file_type"] = file_info.get("file_type", "")
        audio_doc.metadata["access_level"] = file_info.get("access_level", self.default_file_access_level)
        
        # 音频文本处理
        processed_contents = await self.text_pipeline_async(text_doc=audio_doc)
        return processed_contents

    async def image_pipeline_async(self, file_info:dict):
        """
        图像处理管道函数
        """
        # 信息提取
        image_path = file_info.get("element_path", None)
        image_content = file_info.get("element_content", "")
        print(f'图像上下文内容: {image_content}')

        # 生成图片描述
        image_description = await self.image_caption_async(image_path=image_path)
        print(f'图像描述内容: {image_description}')

        # 识别图片的文字内容
        image_text = image_text_reader(image_path)
        print(f'图像文字内容: {image_text}')

        # 综合生成图片描述
        image_description_fusion = await self.image_description_fusion_async(image_description=image_description,
                                                                             image_content=image_content,
                                                                             image_text=image_text)
        print(f"图像的综合描述结果：{image_description_fusion}")

        # 生成文本doc
        image_doc = Document(page_content=image_description_fusion, metadata={})

        # 更新元数据信息
        print("开始更新和同步图片的元数据信息")
        image_doc.metadata["source"] = file_info.get("source", "")
        image_doc.metadata["file_type"] = file_info.get("file_type", "")
        image_doc.metadata["access_level"] = file_info.get("access_level", self.default_file_access_level)

        # 图像描述文本处理
        print("开始对图像文本进行处理")
        processed_contents = await self.text_pipeline_async(text_doc=image_doc, do_text_chunk=False)
        return processed_contents

    
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

    async def file_to_knowledge_database_async(self, 
                                               file_path: str, 
                                               metadata_dict: dict, 
                                               semaphore: asyncio.Semaphore):
        """
        负责单个文件的ETF流程。这个函数内部包含了所有的异步和批量操作
        """
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
                    parse_result = file_loader(file_path, self.output_dir)
                    formatted_text_path = parse_result.get("text_path")
                    image_dir = parse_result.get("image_path")
                    temp_dir_path = parse_result.get("temp_dir_path")
                    element_list.append({"element_path": formatted_text_path, "element_type": "text", "source": file_path, "file_type": file_type} | metadata_dict)  # 组织元素信息并添加元数据信息

                    # 表格提取
                    table_list = find_html_tables_in_markdown(file_path=formatted_text_path)
                    for table_content in table_list:
                        element_list.append({"element_type": "table", "element_content": table_content, "source": file_path, "file_type": file_type} | metadata_dict)

                    # 图像内容提取
                    full_image_paths = find_files_in_directory(image_dir, target_extension="jpg")  # 遍历寻找所有jpg图像文件
                    for full_image_path in full_image_paths:
                        # 提取图像文件名
                        file_name_format = r'[^/\\]+$'
                        image_name = re.search(file_name_format, full_image_path).group(0)
                        image_content = get_context_around_image(file_path=formatted_text_path, image_name=image_name)
                        element_list.append({
                            "element_path": full_image_path,
                            "element_type": "image",
                            "element_content": image_content,
                            "source": file_path, "file_type": file_type, "chunk_type": 'image'} | metadata_dict)
                else:
                    # 读取文件
                    element_list = [{"element_path": file_path, "element_type": file_type, "source": file_path, "file_type": file_type} | metadata_dict]
                # 3 基于元素类型进行分流处理
                for element_info in element_list:
                    element_type = element_info["element_type"]
                    print(f'当前文档元素的类型为:{element_type}')
                    if element_type == "text":  # 3.1 文本处理
                        print('开始处理文本内容')
                        processed_contents = await self.text_pipeline_async(file_info=element_info)
                    elif element_type == "table":  # 3.2 表格文件处理
                        print('开始处理表格内容')
                        processed_contents = await self.table_pipeline_async(file_info=element_info)
                    elif element_type == "audio": # 3.3 音频文件处理
                        print('开始处理音频内容')
                        processed_contents = await self.audio_pipeline_async(file_info=element_info)
                    elif element_type == "image":  # 3.4 图像文件处理
                        print('开始处理图像内容')
                        processed_contents = await self.image_pipeline_async(file_info=element_info)
                    else:
                        raise ValueError(f'不支持的文件类型：{file_type}')

                    # 入库信息生成
                    print('入库信息生成')
                    # print(f'输入内容：{processed_contents}')
                    records_to_save = self.save_knowledge_data_generator(processed_contents)
                    print(records_to_save)

                    # 6. 信息入库
                    print('信息入库')
                    successful_insert_info = await asyncio.to_thread(self.save_to_vector_database, records_to_save)

                # 删除临时文件和文件夹
                if temp_dir_path:
                    shutil.rmtree(temp_dir_path)

                print(f"文件处理成功: {file_path}")
                return {"status": "success", "file_path": file_path}
            except Exception as e:
                print(f"文件处理失败: {file_path}, 错误: {e}")
                return {"status": "failed", "file_path": file_path, "error": str(e)}      
        
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

    # async def chat_to_vector_database_handler_async(self, user_request: str, answer: str, metadata_dict: dict):
    #     """
    #     负责单个请求-问答对的数据库入库流程。这个函数内部包含了所有的异步和批量操作(异步处理)
    #     """
    #     pass       

    # def chat_to_vector_database_handler(self, user_requests, answers,  metadata: List[dict] = None):
    #     """
    #     将聊天记录存入向量数据库
    #     """
    #     pass

    async def search_knowledge_database(self, request: str, limit_search_num: int=50,  limit_output_nun: int = 5, return_metadata: List[str]=None):
        """
        请求向量数据库,从知识库中搜索相关信息
        """
        # 连接数据表
        collection = self.vector_database_client.collections.get("knowledge_base_collection")

        # 搜索请求提炼
        system_message = get_system_message('answer_summary')
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
        answer_summary = json_extractor(clean_output)
        print('答案概括', answer_summary)
        
        # 针对问题的检索
        print("针对问题的检索")
        question_query_response = collection.query.near_text(
            query=request,
            limit=limit_search_num,
            target_vector="question_vector",
            return_metadata=MetadataQuery(distance=True)           
        )

        # 针对概述的检索
        print("针对概述的检索")
        summary_query_response = collection.query.near_text(
            query=answer_summary["answer_summary"],
            # query=request,
            limit=limit_search_num,
            target_vector="summary_vector",
            return_metadata=MetadataQuery(distance=True)
        )

        # 关键字/词检索
        response = collection.query.bm25(
            query=keyswords,
            query_properties=["content"],
            operator=BM25Operator.or_(minimum_match=1),
            limit=5
        )

        print(f"问题为:{request}")
        print(f"问题的关键字为:{keyswords}")
        print(f"关键字回答为:")
        for o in response.objects:
            print(o.properties)
            print('***')
        print("***")


        # rrf算法输出结果
        print("rrf算法rerank")
        rrf_response = apply_rrf([question_query_response, summary_query_response])
        # print("rerank结果", rrf_response)

        # 提取top匹配结果
        uuid_lst = rrf_response
        id_filter = Filter.by_id().contains_any(uuid_lst)
        response =  collection.query.fetch_objects(filters=id_filter)

        # 返回结果
        return response

if __name__ == '__main__':
    file_paths = ['data/Documents/室内空气质量检测报告.pdf']
    test_tool = RAGTool()

    # 检查和加载模型
    test_tool.load_models()

    # 获取数据库连接
    test_tool.connect_to_weative_database()

    # # pdf处理和保存测试
    # try:
    #     asyncio.run(test_tool.file_to_knowledge_database_handler_async(file_paths=file_paths))
    # except Exception as e:
    #     print(f"测试过程中发生错误: {e}")
    # asyncio.run(test_tool.file_to_knowledge_database_handler_async(file_paths=file_paths))


    # 向量数据库检索
    test_collection = test_tool.vector_database_client.collections.get("knowledge_base_collection")
    request_info = [
        "负责这次检测的单位是？",
        "主要使用的设备有什么？",
        "请简述本次的主要依据",
        "目标污染物有哪些?",
        "测点位置在哪里?？"
    ]
    for ind, request in enumerate(request_info):
        seach_results = asyncio.run(test_tool.search_knowledge_database(request=request))
        # print(f'seach results for qeustion {ind+1}:', seach_results)
        # print("************")

    # # 获取uuid
    # uuids = []
    # for ind, result in enumerate(seach_results.objects):
    #     print(f'###第{ind+1}个返回结果###:')
    #     print(f'***查询结果的分数(原始)***:', result.metadata.score)
    #     print('***返回的文本内容***:')
    #     print(result.properties["text"])
    #     print(" ")

    # 关闭数据库连接
    test_tool.close_weavier_connection()