# 和weaviate向量数据库相关的操作方法
from .ConfigManager import config
import weaviate
from weaviate.classes.query import Filter, MetadataQuery, Rerank
from weaviate.classes.config import Configure, Property, DataType, Vectorizers, Tokenization
import time
from .tool import timing_decorator
from weaviate.classes.init import AdditionalConfig, Timeout


# knowledge_base settings
kb_properties = [
    Property(
        name="content",
        description="原始文本",
        data_type=DataType.TEXT,
        tokenization=Tokenization.GSE,
        index_searchable=True
    ),
    Property(
        name="fusion_content",
        description="由原始文本和文本元数据组合而成的融合文本",
        data_type=DataType.TEXT,
        tokenization=Tokenization.GSE,
        index_searchable=True
    ),
    Property(
        name="user_id",
        description="文件上传者的id",
        data_type=DataType.TEXT,
        index_filterable=True
    ),
    Property(
        name="file_hash",
        description="所属文件的hash",
        data_type=DataType.TEXT,
        index_filterable=True
    ),
    Property(
        name="file_id",
        description="所属文件的id(雪花id，用于鉴权)",
        data_type=DataType.TEXT,
        index_filterable=True
    ),
    Property(
        name="file_name",
        description="所属文件的文件名",
        data_type=DataType.TEXT,
        index_filterable=True
    ),
    Property(
        name="file_path",
        description="所属文件所在S3路径或URL",
        data_type=DataType.TEXT,
        index_filterable=True
    ),
    Property(
        name="file_type",
        description="所属文件的类型 (e.g., pdf, docx, md)",
        data_type=DataType.TEXT,
        index_filterable=True
    ),
    Property(
        name="chunk_hash",
        description="内容块的hash",
        data_type=DataType.TEXT,
        index_filterable=True
    ),
    Property(
        name="chunk_chapter",
        description="文本所属的章节信息，仅限格式化文档/文本对象",
        data_type=DataType.TEXT,
        index_filterable=True
    ),
    Property(
        name="chunk_type",
        description="内容块的原始类型（text, image, table等）",
        data_type=DataType.TEXT_ARRAY,
        index_filterable=True
    ),
    Property(
        name="chunk_seq_id",
        description="内容块在原文中的序列ID",
        data_type=DataType.INT,
        index_filterable=True
    ),
    Property(
        name="chunk_page_number",
        description="内容块所在页码（仅限pdf类型）",
        data_type=DataType.INT_ARRAY,
        index_filterable=True
    )
]



class VectorDatabase:
    def __init__(self, db_params: dict):
        self.db_params = db_params
        try:
            self.db_client = weaviate.connect_to_custom(**self.db_params,     
                                                        additional_config=AdditionalConfig(
                                                            timeout=Timeout(init=60, query=240, insert=120)
    ))
        except Exception as e:
            print(f"连接向量数据库失败，错误信息: {e}")

        if self.db_client.is_ready():
            print(f'连接到向量数据库成功')
        else:
            raise ValueError(f'向量数据库连接失败，请检查数据库参数：{self.db_params}')

    def close_weavier_connection(self):
        """
        一个专用的关闭方法，用于在应用退出前释放资源。
        """
        print("断开和Weavier数据库的连接")
        if self.db_client:
            self.close()

    def create_collection(self, 
                          collection_name,
                          collection_type,
                          api_endpoint,
                          embedding_model_name
                          ):
        # 检查数据表是否存在
        if self.db_client.collections.exists(collection_name):
            print(f'该数据表已经存在,无需创建')
        else:
            if collection_type == 'knowledge_base':
                vectorizer_settings = [
                    Configure.NamedVectors.text2vec_ollama(
                        name="content_vector",
                        source_properties=["content"],
                        api_endpoint=api_endpoint,
                        model=embedding_model_name
                    ),
                    Configure.NamedVectors.text2vec_ollama(
                        name="fusion_content_vector",
                        source_properties=["fusion_content"],
                        api_endpoint=api_endpoint,
                        model=embedding_model_name
                    )
                ]
                self.db_client.collections.create(
                    name=collection_name,
                    reranker_config=Configure.Reranker.transformers(),
                    vectorizer_config=vectorizer_settings,
                    properties=kb_properties
                        )
                print(f'{collection_type}类的向量数据表{collection_name}创建成功')
            elif collection_type == "chat_history":
                pass
            else:
                raise ValueError(f'不支持的数据表类型，当前类型为{collection_type}')
    
    def delete_collection(self, collection_name: str=None, delete_all: bool=False):
        if delete_all:
            if self.db_client.is_ready(): # is_ready() 检查连接
                self.db_client.collections.delete_all()
                print(f'已删除全部数据表')

        if collection_name:
            if self.db_client.collections.exists(collection_name): # is_ready() 检查连接
                self.db_client.collections.delete(collection_name)
                print(f'数据表{collection_name}删除成功')
            else:
                print(f'数据表{collection_name}，不存在，不执行删除操作')
        else:
            raise ValueError(f'参数collection_name不能为空')

    @timing_decorator
    def save_to_db(self, data_dict: dict = None, collection_name: str=None):
        """
        将数据存入数据库
        """
        # print('待入库信息：', data_dict)
        if data_dict is None:
            print(f'没有数据需要保存')
            return False
        else:
            # print(f'需要保存的数据:{data_dict}')
   

            # 检查数据表是否存在
            print(f'检查数据表{collection_name}是否存在')
            if not self.db_client.collections.exists(collection_name):
                print(f'向量数据库中不存在{collection_name}数据表')
                return False
            else:
                document_collection = self.db_client.collections.get(collection_name)
                aggregation_result = document_collection.aggregate.over_all(total_count=True)
                total = aggregation_result.total_count
                print(f"\n 数据表 '{collection_name}' 的总记录数为: {total}")
                # 尝试保存数据
                try:
                    print(f"\n准备插入 {len(data_dict)} 条数据...")
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
                        print(f'传入数据的格式信息：{data_dict.keys()}')
                        return False
                    else:
                        print(f'插入数据成功')
                    return True
                except Exception as e:
                    print(f"插入数据时发生错误: {e}")
                    return False
                
    @timing_decorator
    def search_collection(self,
                          collection_name: str, 
                          user_request: str,
                          fusion_request: str=None):
        """
        请求向量数据库,从知识库中搜索相关信息
        """
        # 连接数据表
        if self.db_client.collections.exists(collection_name):
            collection = self.db_client.collections.get(collection_name)
        else:
            raise ValueError(f'数据表{collection_name}不存在')

        # # 统计表格规模
        # aggregation_result = collection.aggregate.over_all(total_count=True)
        # print(f'检索表为:{collection_name},表格规模为{aggregation_result.total_count}条')
        
        if fusion_request:
            print('开始强力检索')
            # 利用fusion request进行初筛
            hybrid_query_response = collection.query.hybrid(
                query=fusion_request,
                query_properties=["fusion_content"],
                target_vector="fusion_content_vector",
                limit=config.get_setting("search")["first_output"]
            )

            # 检索结果的uuid提取
            print('提取uuid')
            content_uuids = [result.uuid for result in hybrid_query_response.objects]
            unique_uuids = list(set(content_uuids))
            id_filter = Filter.by_id().contains_any(unique_uuids)

            # 使用原始请求精筛
            print('执行rerank')
            rerank_query_response = collection.query.near_text(
                query=user_request,
                filters=id_filter,
                target_vector="fusion_content_vector",
                rerank=Rerank(prop="fusion_content",
                              query=user_request),
                return_metadata=MetadataQuery(distance=True, certainty=True),
                limit=config.get_setting("search")["second_output"]
            )

        else:
            print('开始快速检索')
            rerank_query_response = collection.query.near_text(
                query=user_request,
                limit=config.get("search")["second_output"],
                target_vector="fusion_content_vector",
                return_metadata=MetadataQuery(distance=True, certainty=True)           
            )

        return rerank_query_response



