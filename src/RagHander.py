from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredPDFLoader
from ConfigManager import config
import os
import re
from typing import List, Union
from langchain_core.documents import Document
from ContentHandler import get_system_message, json_extractor, apply_rrf
import weaviate
import weaviate.classes as wvc
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.util import generate_uuid5
import pandas as pd
import hashlib
import ollama
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import tqdm
import markdown
from bs4 import BeautifulSoup



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
        self.image_embedding_model_name = config.get_setting("models")["image_embedding_model_name"]  #  图像embedding模型
        self.ollama_model_option = config.get_setting("models")["ollama_model_option"]  # ollama模型额外参数


        # 数据库相关
        self.vector_database_client = None  # 向量数据库的客户端
        self.vector_database_host = config.get_setting("vector_database")["host"]  # 向量数据库的地址
        self.vector_database_port = config.get_setting("vector_database")["port"]  # 向量数据库的端口
        self.vector_database_grpc_port = config.get_setting("vector_database")["grpc_port"]  # 向量数据库的grpc端口
        self.knewledge_base_collection_name = config.get_setting("vector_database")["knewledge_base_collection_name"]  # 知识库collection的名称
        self.chat_collection_name = config.get_setting("vector_database")["chat_collection_name"]  # 聊天记录collection的名称
        

        # 文档类型相关
        self.document_types = config.get_setting("file_types")["text_file_extension"]  # 文档后缀
        self.table_types = config.get_setting("file_types")["table_file_extension"]  # 表格后缀
        self.audio_types = config.get_setting("file_types")["audio_file_extension"]  # 音频后缀
        self.image_types = config.get_setting("file_types")["image_file_extension"]  # 图像后缀

        self.file_loader_mapping = {
            ".pdf": UnstructuredPDFLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".txt": TextLoader
        }

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
                                if "content_vector" in item:
                                    batch.add_object(
                                        uuid=uuid,
                                        properties=item.get("properties"),
                                        vector={"content_vector": item.get("content_vector")},
                                    )
                                else:
                                    batch.add_object(
                                        uuid=uuid,
                                        properties=item.get("properties")
                                    )

                    # 检查批量操作中是否有错误
                    failed_objects = document_collection.batch.failed_objects
                    if failed_objects:
                        print(f"插入数据时发生错误数量: {len(document_collection.batch.failed_objects)}")

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
                return "text"
            elif file_extension in self.audio_types:
                return "audio"
            elif file_extension in self.table_types:
                return "table"
            elif file_extension in self.image_types:
                return "image"
            else:
                raise ValueError(f'不支持的文件格式:{file_extension}')

    def file_reader(self, file_path) -> List:
        """
        用于读取导入的文档的函数
        """
        # 获取文件后缀
        file_extension = os.path.splitext(file_path)[1].lower()

        # 进行后缀匹配
        if file_extension in self.file_loader_mapping:
            document_loader = self.file_loader_mapping.get(file_extension)

            # 基于后缀，创建loader实例
            if file_extension in self.file_loader_mapping.keys():
                loader = document_loader(file_path)
            else:
                loader = document_loader(file_path, encoding='utf-8')

            # 加载并返回文档内容
            return loader.load()
        else:
            print(f"警告：不支持的文件类型 {file_extension}，跳过文件 {file_path}")
            loader
            return []

    def text_clear_formatting(self, docs:List[Document], metadata_dicts: List[dict]) -> List[Document]:
        """
        清除文档中的格式化信息,并生成Document对象
        """
        new_docs = []  # 储存清洗后的文档
        for doc, meta in zip(docs, metadata_dicts):
            # 转为纯文本
            doc_text = doc.page_content
            html = markdown.markdown(doc_text)
            soup = BeautifulSoup(html, "html.paerser")
            text = soup.get_text()

            # 转为Document
            text_doc = Document(page_content=text, metadata=meta)

            # 保存对象
            new_docs.append(text_doc)

        return new_docs

    def clean_text(self, docs: List[Document]) -> List[Document]:
        """
        文本清洗函数
        """
        new_docs = []  # 储存清洗后的文档

        for doc in docs:
            # 文档复制
            content = doc.page_content 

            # (英文内容)替换多个空格为一个
            content = re.sub(r'\s+', ' ', content)

            # (英文内容)替换多个换行符为一个
            content = re.sub(r'\n+', '\n', content)

            # (英文内容)去除文档中的连字符
            content = re.sub(r'-\s*\n', '', content)

            # (中文内容)移除中文文本之间的多余空格
            ontent = re.sub(r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])', r'\1\2', content)

            # 合并被错误切分的段落或句子
            content = re.sub(r'(?<![。！？\n])\n', '', content)
            content = re.sub(r'\n+', '\n', content)

            # 去除页眉页脚和页码等噪音信息
            content = re.sub(r'(?i)第\s*\d+\s*页', '', content)
            content = re.sub(r'(?i)page\s*\d+', '', content)

            # 移除重复性的乱码噪声
            noise_pattern = r"([\s`\\/’'\'V丶、()]+){5,}"
            content = re.sub(noise_pattern, ' ', content)
            content = re.sub(r'\s+', ' ', content).strip()

            # 去除无法识别的乱码与字符
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
        tasks = [self.text_pitch(content) for content in contents]
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
        try:
            output_json = json_extractor(response['message']['content'])

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
            return e

    async def text_chunk_splitter_handler_async(self, contents: List[Document]) -> List[Document]:
        """
        基于markdown格式的文本分块函数(批量)
        """
        # print(f'文档分块输入内容：', contents)
        tasks = [self.text_chunk_splitter(content) for content in contents if isinstance(content, Document)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # print(f'文档分块输出内容：', len(results))

        # 展评list
        output_results = []
        for result in results:
            # print('result', result)
            for res in result:
                # print("res", res)
                if isinstance(res, Document):
                    output_results.append(res)
        return output_results
    
    async def text_processor_async(self, content: Document) -> dict:
        """
        对文档块内容进行处理的函数（概括与问题派生）,以及生成哈希和uuid
        """
        # 输入解析
        input_content = content.page_content

        # 生成分块的哈希和uuid
        orignal_text = input_content.strip().lower()
        orignal_hash = hashlib.sha256(orignal_text.encode('utf-8')).hexdigest()
        generated_id = generate_uuid5(identifier=orignal_text, namespace=orignal_hash)
        content.metadata['hash'] = orignal_hash
        content.metadata['uuid'] = generated_id

        # 内容概括部分
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

        # 问题派生部分
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

        # 构建输出对象
        output_obejct = {"content": Document(page_content=input_content, metadata=content.metadata),
                         "summary": Document(page_content=summary_json["summary"], metadata=content.metadata),
                         "questions": Document(page_content=question_json["questions"], metadata=content.metadata)}


        return output_obejct

    async def text_processor_handler_async(self, contents) -> List[dict]:
        """
        对文档内容进行精简和概括的函数(批量)
        """
        # for content in contents[:15]:
        #     print(f'文档概括输入内容：', type(content), content.page_content)
        tasks = [self.text_processor(content) for content in contents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # for content in results[:5]:
        #     print(content)

        # 结果过滤
        successful_results = [res for res in results if isinstance(res, dict)]
        return successful_results   
    
    def save_knowledge_data_generator(self, contents:List[Document], metadata_dict:dict, image_embeddings: list):
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
            document_type = metadata.get("info_type", "")
            page_number = metadata.get("page_number", -1)
            chunk_seq_id = metadata.get("chunk_seq_id", -1)

            # 元数据：外部传入
            if metadata_dict:
                access_level = metadata_dict.get("access_level", 0)
            else:
                access_level = 0

            # 组合字段
            if uuid not in data_dict.keys():
                data_dict[uuid] = {"properties": {
                                       "content": content_text, 
                                       "summary": summary_text,
                                       "questions": questions_text,
                                       "source": source,
                                       "document_type": document_type,
                                       "access_level": access_level,
                                       "page_number": page_number,
                                       "chunk_seq_id": chunk_seq_id
                                   }
                                }
            # 添加图像向量
            if image_embeddings:
                data_dict[uuid]["content_vector"] = image_embeddings
        return data_dict

    async def audio_to_text_async(self, audio_path: str, meta: dict) -> Document:
            """
            对音频文件进行语音识别的函数
            """
            try:
                res = self.audio_model_1.generate(
                    input=audio_path, 
                    cache={},
                    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True, 
                    merge_length_s=15,
                )
                text = rich_transcription_postprocess(res[0]["text"])
            except:
                text = ""

            # 结果输出
            content = Document(page_content=text, metadata=meta)
            return {'content': content}
    
    async def audio_to_text_handler_async(self, audio_paths: List[str], metas: List[dict]) -> List[Document]:
        """
        异步音频读取函数
        """
        # 任务生成与执行
        tasks = [self.audio_to_text_async(path, meta) for path, meta in zip(audio_paths, metas)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 结果筛选
        output_results = [result for result in results if isinstance(result, Document)]
        return output_results

    async def text_correction_async(self, text_dict: dict) -> Document:
        """
        对音频识别的结果进行纠偏处理
        """
        # 提取音频识别结果
        text_1 = text_dict.get('text_1', "")
        text_2 = text_dict.get('text_2', "")

        # message构建
        system_message = get_system_message('text_correction')
        message = [
            {
                "role": "system",
                "content": system_message
             },
             {
                "role": "user",
                "content": f"请结合两个版本的音频识别结果，对比校正的出一个纠正文本错误后的新版本。ARS识别结果1：{text_1} \nARS识别结果2:{text_2}" 
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
 
        # 提取输出json并处理
        try:
            output_json = json_extractor(response['message']['content'])
            if "corrected_text" in output_json:
                corrected_text = output_json["corrected_text"]
                return Document(page_content=corrected_text, metadata={"page_number": 0})
            else:
                return last_exception
        except ValueError as e:
            return e

    async def text_correction_handler_async(self, text_dict_list: List[dict]) -> List[Document]:
        """
        异步对音频识别文档结果进行纠错
        """
        # print(f'文档纠错输入内容：', contents)
        tasks = [self.text_correction_async(text_dict) for text_dict in text_dict_list if isinstance(text_dict, dict)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # print(f'文档纠错输出内容：', len(results))

        # 结果筛选
        output_results = [result for result in results if isinstance(result, Document)]
        return output_results
    

    ###################################################### 整合函数  ######################################################
    async def text_pipeline_async(self, 
                                  text_contents: List, 
                                  do_text_clean: bool=True, 
                                  do_text_correction: bool=False,
                                  do_text_fusion: bool=False,
                                  do_text_chunk: bool=True,
                                  do_text_processor: bool=True) -> List[dict]:
        """
        文本处理流水线函数
        """
        # 文本去格式化(markdown转text)
        processed_contents = self.text_clear_formatting(text_contents)

        # 文本清洗
        if do_text_clean:
            processed_contents = self.clean_text(processed_contents)

        # 文本纠正
        if do_text_correction:
            processed_contents = await self.text_correction_handler_async(processed_contents)

        # 文本拼接
        if do_text_fusion:
            processed_contents = await self.text_fusion_handler_async(processed_contents)

        # 文本分块
        if do_text_chunk:
            processed_contents = await self.text_chunk_splitter_handler_async(processed_contents)

        # 文本处理
        if do_text_processor:
            processed_contents = await self.text_processor_handler_async(processed_contents)

        #返回结果
        return processed_contents


    async def audio_pipeline(self, audio_paths: List[str], meta_dicts=List[dict]) -> List[dict]:
        """
        音频处理管道函数
        """
        # 音频识别
        text_contents = await self.audio_to_text_handler_async(audio_paths=audio_paths, metas=meta_dicts)

        # 文本处理
        processed_content = await self.text_pipeline(text_contents=text_contents)

        return processed_content

    async def image_pipeline(self, image_paths: List, meta_dict: list[dict]) -> List[dict]:
        """
        图像处理管道函数
        """
        # 图像文本识别

        # 图像描述

        # 文本信息整合

        # 文本处理

        # 图像嵌入向量生成

        # 整合结果

        # 返回结果
        pass




    
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
        if self.image_embedding_model_name not in local_models:
            print(f'未找到ollama模型{self.image_embedding_model_name},请检查')

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

    async def file_to_knowledge_database_async(self, file_path: str, metadata_dict: dict, semaphore: asyncio.Semaphore):
        """
        负责单个文件的ETF流程。这个函数内部包含了所有的异步和批量操作
        """
        async with semaphore:
            try:
                # 1. 识别文件类型
                print('识别文件类型')
                file_type = self.file_type_identifier(file_path)

                # 2 依据文件类型进行处理
                print(f'当前文件的类型是：{file_type}')
                image_embeddings = None
                if file_type == "text":
                    # 3.1 文本文件处理
                    # 读取文件
                    print('读取文件')
                    raw_docs = self.file_reader(file_path)
                    print(f'### 读取文件{len(raw_docs)}个')

                    # 文本清洗
                    print('文本清洗')
                    cleaned_docs = self.clean_text(raw_docs)
                    print(f'### 文本清洗{len(cleaned_docs)}个')

                    # 窗口拼接
                    print('窗口拼接')
                    window_contexts = self.text_window_retrieval(cleaned_docs)
                    print(f'### 窗口拼接{len(window_contexts)}个')

                    # 文本拼接
                    print('文本拼接')
                    text_contexts = await self.text_pitch_async(window_contexts)
                    print(f'### 文本拼接{len(text_contexts)}个')

                    # 文本分块
                    print('文本分块')
                    new_contents = await self.text_chunk_splitter_async(text_contexts)
                    print(f'### 文本分块完成，共{len(new_contents)}个')

                elif file_type == "table":
                    # 3.2 表格文件处理

                    # 表格分块
                    pass
                elif file_type == "audio":
                    # 3.3 音频文件处理
                    auido_to_text = self.audio_to_text(file_path)
                    
                    # 文本纠偏
                    correction_contents = await self.text_correction_handler_async([auido_to_text])

                    # 文本分块
                    new_contents = await self.text_chunk_splitter_async(correction_contents)
                    pass
                elif file_type == "image":
                    # 3.4 图像文件处理

                    # 图像向量化

                    # 图像描述
                    pass
                elif file_type == 'question-answer':
                    pass
                else:
                    raise ValueError(f'不支持的文件类型：{file_type}')

                # 4. 文本处理
                print('文本处理')
                processed_contents = await self.text_processor_async(new_contents)

                # 5. 入库信息生成
                print('入库信息生成')
                records_to_save = self.save_knowledge_data_generator(processed_contents, metadata_dict, image_embeddings)
                print(records_to_save)

                # 6. 信息入库
                print('信息入库')
                # successful_insert_info = await asyncio.to_thread(self.save_to_vector_database, records_to_save)

                print(f"文件处理成功: {file_path}")
                return {"status": "success", "file_path": file_path}
            except Exception as e:
                print(f"文件处理失败: {file_path}, 错误: {e}")
                return {"status": "failed", "file_path": file_path, "error": str(e)}      
        
    async def file_to_knowledge_database_handler_async(self, file_paths: List[str], metadata: List[dict] = None):
        """
        文档处理函数,将传入的文档格式化，分块，提炼并embeding化存入向量数据库
        """
        # 元数据处理
        if metadata is None:
            metadata = [None] * len(file_paths)

        # 限制并发数量
        semaphore = asyncio.Semaphore(self.concurrency_limit)

        # 创建协程任务
        tasks = [asyncio.create_task(self.file_to_knowledge_database_async(path, meta, semaphore)) for path, meta in zip(file_paths, metadata)]

        # 执行任务
        results = await asyncio.gather(*tasks)
        return results

    async def chat_to_vector_database_handler_async(self, user_request: str, answer: str, metadata_dict: dict):
        """
        负责单个请求-问答对的数据库入库流程。这个函数内部包含了所有的异步和批量操作(异步处理)
        """
        pass       

    def chat_to_vector_database_handler(self, user_requests, answers,  metadata: List[dict] = None):
        """
        将聊天记录存入向量数据库
        """
        pass

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
        answer_summary = json_extractor(response['message']['content'])
        print('答案概括', answer_summary)

        # 获取搜索参数
        filter_info = None
        
        # 针对问题的检索
        question_query_response = collection.query.near_text(
            query=request,
            limit=limit_search_num,
            return_metadata=MetadataQuery(distance=True)           
        )

        # 针对概述的检索
        summary_query_response = collection.query.near_text(
            query=answer_summary["answer_summary"],
            limit=limit_search_num,
            return_metadata=MetadataQuery(distance=True)
        )

        # 关键字/词检索


        # rrf算法输出结果
        rrf_response = apply_rrf([question_query_response, summary_query_response])
        print(rrf_response)

        # 提取top匹配结果
        uuid_lst = rrf_response
        id_filter = Filter.by_id().contains_any(uuid_lst)
        response =  collection.query.fetch_objects(filters=id_filter)

        # 返回结果
        return response

if __name__ == '__main__':
    file_paths = ['data/Audios/vad_example.wav']
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
    asyncio.run(test_tool.file_to_knowledge_database_handler_async(file_paths=file_paths))
    # # 向量数据库检索
    # test_tool.connect_to_weative_database()
    # test_collection = test_tool.vector_database_client.collections.get("knowledge_base_collection")
    # request_info = "为什么温度设置为0时，大模型依然无法保证输出的稳定性？"
    # seach_results = test_tool.search_knowledge_database(request=request_info)
    # test_tool.close_weavier_connection()
    # print('return search result:', seach_results)

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