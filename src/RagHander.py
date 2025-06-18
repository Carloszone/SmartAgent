from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from google import genai
import redis.client
from ConfigManager import config
import os
import re
from typing import List
from langchain_core.documents import Document
from ContentHandler import ContentHandler
import weaviate
import weaviate.classes as wvc
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from google.genai.types import EmbedContentConfig
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from google.genai import types
from weaviate.util import generate_uuid5
import redis


class RAGTool:
    """
    执行RAG相关操作的类
    """
    def __init__(self, agent_client = None, embeding_model = None, redis_client = None, thread_num: int = 8, batch_size: int = 32, session_ttl_s: int = 86400):
        # 加载文本内容处理器
        self.content_handler = ContentHandler()

        # 模型加载
        # self.model_client = self.content_handler.client
        self.model_client = genai.Client(api_key=config.get_setting('model_api_key'))   # 模型客户端
        self.generative_model_name = config.get_setting('generative_model_name')  # 使用的生成模型名称
        self.embeding_model_name = config.get_setting('embeding_model_name')  # 使用的嵌入向量生成模型名称

        # 客户端信息
        self.vector_database_client = None  # 向量数据库的客户端
        self.redis_database_client = redis_client  # redis数据库的客户端
        self.session_ttl_s = session_ttl_s  # redis过期时间

        # 处理参数
        self.thread_num = thread_num   # 线程数
        self.batch_size = batch_size   # 批次规模
        
        # 文档类型-解析工具映射表
        self.documente_loader_mapping = {
            ".pdf": PyMuPDFLoader,
            ".docx": UnstructuredWordDocumentLoader
        }

        # 默认文本文件的解析工具
        self.default_text_reader = TextLoader

        # 自描述
        self.tool_description = self.self_description()

    ###################################################### 数据库操作函数  ######################################################
    def connect_to_weative_database(self):
        """
        创建一个到weative数据库的连接
        """
        # 参数提取
        vector_database_config = config.get_setting('vector_database_config')
        vector_database_host = vector_database_config.get('host')
        vector_database_port = vector_database_config.get('port')
        vector_database_grpc_port = vector_database_config.get('grpc_port')

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

    def save_to_vector_database(self, data_list: List[dict] = None, mode="knowledge_base"):
        """
        将数据存入数据库
        """
        if data_list is None:
            print(f'没有数据需要保存')
            return
        else:
            # 提取配置信息
            if mode == "knowledge_base":
                vector_database_collection_name = config.get_setting('vector_database_config').get('knewledge_base_collection_name')
            elif mode == "chat_history":
                vector_database_collection_name = config.get_setting('vector_database_config').get('chat_collection_name')
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
                    print(f"\n准备插入 {len(data_list)} 条数据...")
                    successful_mapping = []
                    with document_collection.batch.dynamic() as batch:
                        for item in data_list:
                            generated_uuid = generate_uuid5(item)
                            text = item.get("text", '')
                            successful_mapping.append({"original_text": text, "uuid": generated_uuid})
                            batch.add_object(properties=item.get("properties"),
                                             vector=item.get("vector"),
                                             uuid=generated_uuid)

                    # 检查批量操作中是否有错误
                    failed_objects = document_collection.batch.failed_objects
                    if failed_objects:
                        print(f"插入数据时发生错误数量: {len(document_collection.batch.failed_objects)}")

                    return successful_mapping
                            

                except Exception as e:
                    print(f"插入数据时发生错误: {e}")
                    raise
        return []
  
    def close_weavier_connection(self):
        """
        一个专用的关闭方法，用于在应用退出前释放资源。
        """
        print("断开和Weavier数据库的连接")
        if self.vector_database_client:
            self.vector_database_client.close()

    def get_weavier_collection_info(self):
        """
        获取指定collection的字段信息
        """
        pass

    def connect_to_database(self):
        """
        连接到传统数据库的函数
        """
        pass

    def save_to_database(self):
        """
        存入信息到传统数据库的函数
        """
        pass
    
    def close_database(self):
        """
        关闭传统数据库的函数
        """
        pass
    
    def extract_topic_vector_from_vector_database(self, chat_id):
        """
        基于chat_id在向量数据库检索对应的topic_vector
        """
        # 查询向量数据库
        collection = self.vector_database_client.collections.get(config.get_setting('vector_database_config').get('chat_history_collection_name'))
        response = collection.query.fetch_objects(
            filters=Filter.by_property("chat_id").equal(chat_id),
            limit=1,
            return_vectors=["topic_vector"] 
        )
        if response.objects:
            return response.objects[0].vectors["topic_vector"]
        else:
            return None

    async def get_topic_embeddings(self, chat_id, docs):
        """
         异步获取或创建会话的主题向量
         """
        # 定义redis中储存的键名
        redis_key = f"session:topic_vector:{chat_id}"

        try:
            # 检查主题信息是否在缓存中
            cached_vector_str = self.redis_client.get(redis_key)

            if cached_vector_str:
                print(f'从redis缓存中查找到会话{chat_id}对应的主题向量')
                # 刷新过期时间
                self.redis_client.expire(redis_key, self.session_ttl_s)
                return json.loads(cached_vector_str)
            
            # 如果缓存中不存在，查询向量数据库
            print(f'未在缓存中查找到会话{chat_id}对应的主题向量，开始从向量数据库中查找')
            topic_vector = self.extract_topic_vector_from_vector_database(chat_id)
            if topic_vector:
                print(f"从向量数据库中提取到会话{chat_id}对应的主题向量")
            else:
                # 如果向量数据中没有，提炼主题并生成主题嵌入向量
                print(f"未在向量数据库中提取到会话{chat_id}对应的主题向量，开始提炼主题并生成主题嵌入向量")
                topic, topic_vector = await self.chat_topic_embeddings(docs)

                # topic 以后可以存入数据库，供前端显示

            # 存入redis缓存中
            topic_vector_str = json.dumps(topic_vector)
            self.redis_database_client.set(redis_key, topic_vector_str, ex=self.session_ttl_s)
            return topic_vector
        except redis.RedisError as e:
            print(f"警告：与Redis交互时发生错误: {e}。本次操作将不使用缓存。")
            return None
        
    ########################################################  功能函数  ########################################################
    def self_description(self):
        """
        用于生成自身说明书的函数，解释自身功能，供agent使用
        """
        pass
    
    def document_reader(self, file_path) -> List:
        """
        用于读取导入的文档的函数
        """
        # 获取文件后缀
        file_extension = os.path.splitext(file_path)[1].lower()

        # 进行后缀匹配
        if file_extension in self.documente_loader_mapping:
            document_loader = self.documente_loader_mapping.get(file_extension, self.default_text_reader)

            # 基于后缀，创建loader实例
            if file_extension in self.documente_loader_mapping.keys():
                loader = document_loader(file_path)
            else:
                loader = document_loader(file_path, encoding='utf-8')

            # 加载并返回文档内容
            return loader.load()
        else:
            print(f"警告：不支持的文件类型 {file_extension}，跳过文件 {file_path}")
            return []

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

            # 去除开头和结尾处的空白字符
            content = content.strip()

            # 构建一个新的Document对象
            new_doc = Document(page_content=content, metadata=doc.metadata)
            new_docs.append(new_doc)
        return new_docs

    def text_window_retrieval(self, docs):
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

    def markdown_formatter_single_flow(self, contents):
        """
        对输入的文档内容进行markdown格式化
        """
        # 输入解析
        output_contents = []
        for content in contents:
            # 输入解析
            text = {key: value.page_content for key, value in content.items() if isinstance(value, Document)}

            # prompt生成（文本markdown格式化）
            output_prompt = self.content_handler.prompt_generator(content=text, mode="content_markdown")

            # 执行prompt
            response = self.content_handler.client.models.generate_content(model=self.generative_model_name, contents=output_prompt)

            # 提取输出json
            output_json = self.content_handler.json_extractor(response.text)

            # 返回格式化后的文档内容
            output_content = {key: Document(page_content=value, metadata=content[key].metadata) for key, value in output_json.items()if len(value) > 0}

            output_contents.append(output_content)
        return output_contents
    async def markdown_formatter(self, content: dict):
        """
        对输入的文档内容进行markdown格式化
        """
        # 输入解析
        text = {key: value.page_content for key, value in content.items() if isinstance(value, Document)}

        # prompt生成（文本markdown格式化）
        output_prompt = self.content_handler.prompt_generator(content=text, mode="content_markdown")

        # 执行prompt
        response = await self.content_handler.client.aio.models.generate_content(model=self.generative_model_name, 
                                                                                 contents=output_prompt,
                                                                                 config=types.GenerateContentConfig(temperature=0.0))

        # 提取输出json
        output_json = self.content_handler.json_extractor(response.text)

        # 返回格式化后的文档内容
        output_content = {key: Document(page_content=value, metadata=content[key].metadata) for key, value in output_json.items()if len(value) > 0}
        return output_content

    async def markdown_formatter_async(self, contents: List[dict]) -> List[dict]:
        """
        对输入的文档内容进行markdown格式化(异步处理)
        """
        # print(f'开始进行文档格式化的输入：', contents)
        tasks = [self.markdown_formatter(content) for content in contents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # print(f'文档格式化的输出：', results)

        # 过滤掉处理失败的结果
        # successful_results = [res for res in results if isinstance(res, Document)]
        return results

    async def text_chunk_splitter(self, content: dict):
        """
        基于markdown格式的分本分块函数
        """
        # 输入解析
        text = {key: value.page_content for key, value in content.items()}

        # prompt生成（文本markdown格式化）
        output_prompt = self.content_handler.prompt_generator(content=text, mode="chunker")

        # 执行prompt
        response = await self.content_handler.client.aio.models.generate_content(model=self.generative_model_name, 
                                                                                 contents=output_prompt,
                                                                                 config=types.GenerateContentConfig(temperature=0.0))
 
        # 提取输出json
        output_json = self.content_handler.json_extractor(response.text)

        # 返回格式化后的文档内容
        formatted_chunkers = []
        for index, output_content in enumerate(output_json['chunks']):
            chunk_metadata = content['target_content'].metadata
            chunk_metadata['chunk_seq_id'] = index
            chunk_document = Document(page_content=output_content, metadata=chunk_metadata)
            formatted_chunkers.append(chunk_document)
        return formatted_chunkers

    async def text_chunk_splitter_async(self, contents: List[dict]) -> List[List[Document]]:
        """
        基于markdown格式的文本分块函数(批量)
        """
        # print(f'文档分块输入内容：', contents)
        tasks = [self.text_chunk_splitter(content) for content in contents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # print(f'文档分块输出内容：', results)
        
        # 过滤掉处理失败的结果
        # successful_results = [res for res in results if isinstance(res, Document)]
        return results

    def token_calculator(self, content):
        """
        计算输入的token使用量的函数,tokenizer必须和embeding模型以及生成模型配套
        """
        response = self.model_client.models.count_tokens(model=self.generative_model_name,
                                                         contents=content)
        return response.total_tokens
    
    async def text_summary(self, content):
        """
        对文档块内容进行精简和概括的函数
        """
        # 输入解析
        if isinstance(content, Document):
            text = {"summary": content.page_content}
        elif isinstance(content, str):
            text = {"summary": content}
        else:
            raise

        if self.token_calculator(text["summary"]) > 200:
            # prompt生成（文本markdown格式化）
            output_prompt = self.content_handler.prompt_generator(content=text, mode="text_summary")

            # 执行prompt
            response = await self.content_handler.client.aio.models.generate_content(model=self.generative_model_name, 
                                                                                     contents=output_prompt,
                                                                                     config=types.GenerateContentConfig(temperature=0.0))

            # 提取输出json
            output_json = self.content_handler.json_extractor(response.text)
        else:
            output_json = text

        # 返回格式化后的文档内容
        output_content = Document(page_content=output_json['summary'], metadata=content.metadata)

        return output_content

    async def text_summary_async(self, contents):
        """
        对文档内容进行精简和概括的函数(批量)
        """
        # for content in contents:
        #     for chunks in content:
        #         print(f'文档概括输入内容：', type(chunks), chunks.page_content)
        tasks = [self.text_summary(chunk_content) for content in contents for chunk_content in content]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # for res in results:
        #     print(f'文档概括输出类型：', type(res))
        #     print(f'文档概括输出内容：', res.page_content)

        # 过滤掉处理失败的结果
        # print('概括输出结果数量', len(results))
        successful_results = [res for res in results if isinstance(res, Document)]
        return successful_results 

    async def text_embeding_async(self, contents):
        """
        接受一个文本列表，并批量生成embeidngs
        """
        if not contents:
            return []

        try:
            # 输入解析
            texts_to_embed = [content.page_content for content in contents]
            print('文档分块数目：', len(texts_to_embed))

            # 获取embeding
            response = await self.model_client.aio.models.embed_content(model=self.embeding_model_name,
                                                                    contents=texts_to_embed,
                                                                    config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                                                                )
            print('embedding数量', len(response.embeddings), type(response.embeddings))
            return response
        except Exception as e:
            print(f"Embedding过程中发生错误: {e}")
            # 返回一个与输入长度相匹配的空向量列表或进行其他错误处理
            return [[] for _ in contents]

    def text_embeding(self, contents):
        """
        接受一个文本列表,并批量生成embeidngs
        """
        # print(f'embeding输入的内容为：', contents)
        if not contents:
            return []

        try:
            # 输入解析
            texts_to_embed = [content.page_content for content in contents]
            print('文档分块数目：', len(texts_to_embed))

            # 获取embeding
            response = self.model_client.models.embed_content(model=self.embeding_model_name,
                                                                    contents=texts_to_embed,
                                                                    config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                                                                )
            # print('embedding数量', len(response.embeddings), type(response.embeddings))
            return response.embeddings
        except Exception as e:
            print(f"Embedding过程中发生错误: {e}")
            # 返回一个与输入长度相匹配的空向量列表或进行其他错误处理
            return [[] for _ in contents]
    
    async def chat_topic_embeddings(self, docs: List[Document]):
        """
        基于问答对生成主题和对应embeddings的函数
        """
        # 参数构建
        request = {"user_request": docs[0].page_content,
                   "ai_answer": docs[1].page_content}

        # 生成主题提炼prompt
        output_prompt = self.content_handler.prompt_generator(content=request, mode="topic_extraction")

        # 执行prompt并获取主题
        chat_topic = self.model_client.models.generate_content(model=self.generative_model_name, 
                                                             contents=output_prompt,
                                                             config=types.GenerateContentConfig(temperature=0.0)
        )
        topic = Document(page_content=chat_topic)

        # 生成主题embeddings
        topic_embeding = await self.text_embeding_async(topic)

        # 返回结果
        return (topic, topic_embeding)
    
    def save_knowledge_data_generator(self, contents, embeddings):
        """
        生成用于保存到是指向量数据库的dict对象
        """
        data_list = []
        for content, embedding in zip(contents, embeddings.embeddings):
            # print(f'传入的内容变量类型：', type(content))
            # print(f'传入的内容变量信息：', type(embedding))
            # print(f'传入的向量变量类型：', type(content))
            # print(f'传入的向量变量信息：', type(embedding))
            text = content.page_content
            metadata = content.metadata

            # 构建储存字典
            dict_for_storage = {
                "vector": embedding.values,
                'text': text,
                "properties":{
                    "source": metadata.get("source", ''),

                    # 过滤和检索信息
                    "document_type": metadata.get("document_type", "pdf"),
                    "access_level": metadata.get("access_level", 0),

                    # 内容与结构信息
                    "page_number": metadata.get("page_number", -1),
                    "chunk_seq_id": metadata.get("chunk_seq_id", -1)
                }
            }

            data_list.append(dict_for_storage)
        return data_list

    def save_chat_data_generator(self, contents, embeddings):
        """
        """
        data_list = []
        for content, embedding in zip(contents, embeddings.embeddings):
            text = content.page_content
            metadata = content.metadata

            # 提取topic_embeddings
            chat_id = metadata.get("chat_id", '')
            topic_embeddings = self.get_topic_embeddings(chat_id)

            # 构建储存字典
            dict_for_storage = {
                "chunk_vector": embedding.values,
                "topic_vector": topic_embeddings,
                'text': text,
                "properties":{
                    # 过滤和检索信息
                    "chat_id": chat_id,
                    "user_id": metadata.get("user_id", "NoName"),
                    "record_type": metadata.get("record_type", 0),

                    # 内容与结构信息
                    "access_level": metadata.get("access_level", -1),
                    "chunk_seq_id": metadata.get("chunk_seq_id", -1)
                }
            }

    def output_rerank(self, user_quest, reference_chunks):
        """
        对返回的内容进行重要性重排序
        """
        # 参数构建
        request = {"user_request": user_quest, "reference_chunks": reference_chunks}

        # 生成prompt
        output_prompt = self.content_handler.prompt_generator(content=request, mode="re-rank")

        # 执行prompt
        response = self.content_handler.client.models.generate_content(model=self.generative_model_name, contents=output_prompt)

        # 提取json结果
        output_json = self.content_handler.json_extractor(response.text)

        # 结果解析

        pass
    
    ###################################################### 工具/流程函数  ######################################################
    async def document_to_vector_database_async(self, file_path: str, metadata_dict: dict):
        """
        负责单个文件的ETF流程。这个函数内部包含了所有的异步和批量操作
        """
        try:
            # 1. 准备工作：读取文件，文本清洗和分块
            print('***开始读取文件')
            raw_docs = self.document_reader(file_path)
            print(f'###读取文件完成，共{len(raw_docs)}个文档')

            print('***添加元数据到doc')
            if metadata_dict:
                for key, value in metadata_dict.items():
                    raw_docs.metadata[key] = value

            print('***开始文本清洗')
            cleaned_docs = self.clean_text(raw_docs)
            print(f'###文本清洗完成，共{len(cleaned_docs)}个文档')

            print('***开始文本滑动窗口拼接')
            if len(cleaned_docs) > 1:
                window_contexts = self.text_window_retrieval(cleaned_docs)
            else:
                window_contexts = cleaned_docs
            print(f'###文本滑动窗口拼接完成，共{len(window_contexts)}个文档')

            # 2. 并发Markdown格式化
            print('***开始markdown格式化')
            markdown_contents = await self.markdown_formatter_async(window_contexts)
            # markdown_contents = self.markdown_formatter_single_flow(window_contexts)
            print(f'#### markdown格式化完成，共{len(markdown_contents)}个文档')

            # 3. 并发文本分块
            print('***开始文本分块')
            chunked_contents = await self.text_chunk_splitter_async(markdown_contents)
            print(f'### 文本分块完成，共{len(chunked_contents)}个文档')

            # 4. 并发文本概括
            print('***开始文本概括')
            summary_contents = await self.text_summary_async(chunked_contents)
            print(f'### 文本概括完成，共{len(summary_contents)}个文档')

            # 5. 并发文本embeding
            print('***开始文本embeding')
            embeded_contents = await self.text_embeding_async(summary_contents)
            # embeded_contents = self.text_embeding(summary_contents)
            print(f'###文本embeding完成，共{len(embeded_contents.embeddings)}个嵌入向量')

            # 6. 生成保存用的数据
            print(f'***数据格式化')
            records_to_save = self.save_knowledge_data_generator(summary_contents, embeded_contents)
            print(f'###生成保存用的数据完成，共{len(records_to_save)}个文档')

            # 7. 并发保存数据
            print(f'***开始保存数据到向量数据库数据库')
            successful_insert_info = await asyncio.to_thread(self.save_to_vector_database, records_to_save)

            # 8. 将原文和向量数据库的UUID储存到传统数据库中
            print('successful_insert_info:')
            print(successful_insert_info)

            print(f"文件处理成功: {file_path}")
            return {"status": "success", "file_path": file_path}
        except Exception as e:
            print(f"文件处理失败: {file_path}, 错误: {e}")
            return {"status": "failed", "file_path": file_path, "error": str(e)}        

    def document_to_vector_database_handler(self, file_paths: List[str], metadata: List[dict] = None):
        """
        文档处理函数,将传入的文档格式化，分块，提炼并embeding化存入向量数据库
        """
        # 获取数据库连接
        self.connect_to_weative_database()

        # 多线程处理
        if metadata is None:
            metadata = [None] * len(file_paths)

        with ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            futures = [executor.submit(asyncio.run, self.document_to_vector_database_async(path, metadata_dict)) for path, metadata_dict in zip(file_paths, metadata)]

            # 等待所有任务完成并收集结果
            for future in futures:
                result = future.result()
                print(f"任务完成: {result}")

        # 关闭数据库连接
        self.close_weavier_connection()

    async def chat_to_vector_database_handler_async(self, user_request: str, answer: str, metadata_dict: dict):
        """
        负责单个请求-问答对的数据库入库流程。这个函数内部包含了所有的异步和批量操作(异步处理)
        """
        try:
            # 构建Document对象
            request_document = Document(page_content=user_request, metadata=metadata_dict)
            answer_document = Document(page_content=answer, metadata=metadata_dict)

            # 补充元数据信息
            request_document.metadata["record_type"] = "request"
            answer_document.metadata["record_type"] = "answer"
            docs = [request_document, answer_document]

            # 文本markdown处理
            markdown_contents = await self.markdown_formatter_async(docs)

            # 文本分块
            print('***开始文本分块')
            chunked_contents = await self.text_chunk_splitter_async(markdown_contents)
            print(f'### 文本分块完成，共{len(chunked_contents)}个')

            # 文本概括
            print('***开始文本概括')
            summary_contents = await self.text_summary_async(chunked_contents)
            print(f'### 文本概括完成，共{len(summary_contents)}个')

            # 文本embeding
            print('***开始文本embeding')
            embeded_contents = await self.text_embeding_async(summary_contents)
            # embeded_contents = self.text_embeding(summary_contents)
            print(f'###文本embeding完成，共{len(embeded_contents.embeddings)}个嵌入向量')

            # 生成保存用的数据
            print(f'***数据格式化')
            records_to_save = self.save_chat_data_generator(summary_contents, embeded_contents)
            print(f'###生成保存用的数据完成，共{len(records_to_save)}个')

            # 并发保存数据
            print(f'***开始保存数据到向量数据库数据库')
            successful_insert_info = await asyncio.to_thread(self.save_to_vector_database, records_to_save)

            # 将文本快的原文和向量数据库的UUID储存到传统数据库中
            print('successful_insert_info:')
            print(successful_insert_info)

            print(f"文件处理成功: {file_path}")
            return {"status": "success", "file_path": file_path}
        except Exception as e:
            print(f"文件处理失败: {file_path}, 错误: {e}")
            return {"status": "failed", "file_path": file_path, "error": str(e)}        

    def chat_to_vector_database_handler(self, user_requests, answers,  metadata: List[dict] = None):
        """
        将聊天记录存入向量数据库
        """
        # 获取数据库连接
        self.connect_to_weative_database()

        # 多线程处理
        if metadata is None:
            metadata = [None] * len(metadata)

        with ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            futures = [executor.submit(asyncio.run, self.chat_to_vector_database_handler_async(user_request, answer, metadata_dict)) 
                       for user_request, answer , metadata_dict in zip(user_requests, answers, metadata)]

            # 等待所有任务完成并收集结果
            for future in futures:
                result = future.result()
                print(f"任务完成: {result}")

        # 关闭数据库连接
        self.close_weavier_connection()

    def search_knowledge_base(self, request):
        """请求向量数据库,从知识库中搜索相关信息"""
        # 搜索请求提炼

        # 获取搜索参数

        # 近似搜索

        # 输出结果重排序

        # 返回结果
        pass

    def serach_chat_history(self, request):
        """请求向量数据库,从聊天记录中搜索相关信息"""
        pass

if __name__ == '__main__':
    file_path = 'data/Documents/室内空气质量检测报告.pdf'
    test_tool = RAGTool()

    test_tool.document_to_vector_database_handler(file_paths=[file_path])

