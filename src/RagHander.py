from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from google import genai
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
import math


class RAGTool:
    """
    执行RAG相关操作的类
    """
    def __init__(self, agent_client=None, embeding_model=None, tokenizer=None, thread_num=8, batch_size=32):
        # 加载文本内容处理器
        self.content_handler = ContentHandler()

        # 模型客户端
        # self.model_client = self.content_handler.client
        self.model_client = genai.Client(api_key=config.get_setting('model_api_key'))

        # 模型信息
        self.generative_model_name = config.get_setting('generative_model_name')
        self.embeding_model_name = config.get_setting('embeding_model_name')

        # 向量数据库客户端
        self.vector_database_client = None

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

    def save_to_vector_database(self, data_list: List[dict] = None):
        """
        将数据存入数据库
        """
        if data_list is None:
            print(f'没有数据需要保存')
            return
        else:
            # 提取配置信息
            vector_database_collection_name = config.get_setting('vector_database_config').get('knewledge_base_collection_name')

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
                    print(f"\n📊 数据表 '{vector_database_collection_name}' 的总记录数为: {total}")
                # 尝试保存数据
                try:
                    print(f"\n准备插入 {len(data_list)} 条数据...")

                    with document_collection.batch.dynamic() as batch:
                        for item in data_list:
                            my_vector=item.get("vector")

                            # 内部检查
                            if my_vector is None or any(v is None for v in my_vector):
                                print("错误：向量为 None 或内部包含 None 值，跳过此条数据。")
                                continue

                            # 检查 NaN
                            if any(math.isnan(v) for v in my_vector):
                                print("错误：向量中包含 NaN 值，跳过此条数据。")

                            # 检查无穷大
                            if any(math.isinf(v) for v in my_vector):
                                print("错误：向量中包含无穷大值，跳过此条数据。")

                            try:
                                clean_vector = [float(v) for v in my_vector]
                            except (ValueError, TypeError):
                                print("错误：向量中的某个值无法被转换为浮点数。")

                            batch.add_object(properties=item.get("properties"),
                                             vector=item.get("vector"))

                    # 检查批量操作中是否有错误
                    if document_collection.batch.failed_objects:
                        print(f"插入数据时发生错误: {document_collection.batch.failed_objects}")
                except Exception as e:
                    print(f"插入数据时发生错误: {e}")
                    raise

    def close_weavier_connection(self):
        """
        一个专用的关闭方法，用于在应用退出前释放资源。
        """
        print("断开和Weavier数据库的连接")
        if self.vector_database_client:
            self.vector_database_client.close()

    def get_weavier_collection_info(self):
        pass

    def self_description(self):
        """
        用于生成自身说明书的函数，解释自身功能，供agent使用
        """
    
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
    
    async def text_summary_asyn(self, content):
        """
        对文档块内容进行精简和概括的函数
        """
        # 输入解析
        text = {"summary": content.page_content}

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
        tasks = [self.text_summary_asyn(chunk_content) for content in contents for chunk_content in content]
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
        接受一个文本列表，并批量生成embeidngs
        """
        print(f'embeding输入的内容为：', contents)
        if not contents:
            return []

        try:
            # 输入解析
            texts_to_embed = [content.page_content for content in contents] * 3
            print('文档分块数目：', len(texts_to_embed))

            # 获取embeding
            response = self.model_client.models.embed_content(model=self.embeding_model_name,
                                                                    contents=texts_to_embed,
                                                                    config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                                                                )
            print('embedding数量', len(response.embeddings), type(response.embeddings))
            return response.embeddings
        except Exception as e:
            print(f"Embedding过程中发生错误: {e}")
            # 返回一个与输入长度相匹配的空向量列表或进行其他错误处理
            return [[] for _ in contents]

    def save_data_generator(self, contents, embeddings):
        """
        生成用于保存到向量数据库的dict对象
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
                    "metadata": {
                        # 来源与溯源信息
                        # "file_id": '001', 文件id在储存时自动生成
                        "source": metadata.get("source", ''),

                        # 过滤和检索信息
                        "document_type": metadata.get("document_type", "pdf"),
                        "access_level": metadata.get("access_level", 0),

                        # 内容与结构信息
                        "page_number": metadata.get("page_number", -1),
                        "chunk_seq_id": metadata.get("chunk_seq_id", -1)

                        # 维护与版本信息
                        # "file_hash": "",
                        # "file_short_hash": "",
                        # "version": "v2.2"
                    }

                }
            }

            data_list.append(dict_for_storage)
        return data_list
    
    def output_rerank(self):
        """
        对返回的内容进行重要性重排序
        """
        pass
    
    async def document_to_vector_database_handler_async(self, file_path: str, metadata_dict: dict):
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
            window_contexts = self.text_window_retrieval(cleaned_docs)
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
            print(f'***开始保存数据')
            records_to_save = self.save_data_generator(summary_contents, embeded_contents)
            print(f'###生成保存用的数据完成，共{len(records_to_save)}个文档')

            # 7. 并发保存数据
            print(f'***开始保存数据到数据库')
            await asyncio.to_thread(self.save_to_vector_database, records_to_save)

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
            futures = [executor.submit(asyncio.run, self.document_to_vector_database_handler_async(path, metadata_dict)) for path, metadata_dict in zip(file_paths, metadata)]

            # 等待所有任务完成并收集结果
            for future in futures:
                result = future.result()
                print(f"任务完成: {result}")

        # 关闭数据库连接
        self.close_weavier_connection()

    def chat_to_vector_database_handler(self, file_path: str):
        pass

    def search_knowledge_base(self, request):
        """请求向量数据库,从知识库中搜索相关信息"""
        pass

    def serach_chat_history(self, request):
        """请求向量数据库,从聊天记录中搜索相关信息"""
        pass

if __name__ == '__main__':
    file_path = 'data/Documents/室内空气质量检测报告.pdf'
    test_tool = RAGTool()

    test_tool.document_to_vector_database_handler(file_paths=[file_path])

