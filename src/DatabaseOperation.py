# 和weaviate向量数据库相关的操作方法

import weaviate
from weaviate.classes.query import Filter, MetadataQuery, BM25Operator, Rerank
from weaviate.util import generate_uuid5
from weaviate.classes.config import Configure, Property, DataType, Vectorizers, Tokenization
import weaviate.classes as wvc
import numpy as np
import time
from .tool import timing_decorator

def connect_to_weative_database(db_params: dict):
    """
    创建一个到weative数据库的连接
    """
    # 构建连接
    try:
        db_client = weaviate.connect_to_custom(**db_params)

        # 检查连接是否成功
        db_client.is_ready() # 检查连接是否成功
        print(f'连接到weative数据库成功')
        return db_client
    except Exception as e:
        print(f"Failed to connect to Weaviate: {e}")
        raise


def close_weavier_connection(db_client):
    """
    一个专用的关闭方法，用于在应用退出前释放资源。
    """
    print("断开和Weavier数据库的连接")
    if db_client:
        db_client.close()


def create_collection(db_client, collection_name, embedding_model_name: str, api_endpoint: str, collection_type='knowledge_base'):
    # 检查数据表是否存在
    if db_client.collections.exists(collection_name):
        print(f'该数据表已经存在,无需创建')
    else:
        # 基于不同的collection_type，生成不同的向量和属性配置
        if collection_type == 'knowledge_base':
            properties_to_create = [
                Property(
                    name="content",
                    description="原始文本",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.GSE,
                    index_searchable=True
                ),
                Property(
                    name="content_for_rerank",
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
                    description="文本所属的章节信息",
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
                    data_type=DataType.TEXT,
                    index_filterable=True
                ),
                Property(
                    name="content_id",
                    description="内容块在原文中的次序ID",
                    data_type=DataType.INT,
                    index_filterable=True
                ),
                Property(
                    name="chunk_seq_id",
                    description="内容块的序列ID",
                    data_type=DataType.INT,
                    index_filterable=True
                ),
            ]

            vectorizer_settings = [
                Configure.NamedVectors.text2vec_ollama(
                    name="content_vector",
                    source_properties=["content"],
                    api_endpoint=api_endpoint,
                    model=embedding_model_name
                ),
                Configure.NamedVectors.text2vec_ollama(
                    name="content_for_rerank_vector",
                    source_properties=["content_for_rerank"],
                    api_endpoint=api_endpoint,
                    model=embedding_model_name
                ),
                Configure.NamedVectors.text2vec_ollama(
                    name="summary_vector",
                    source_properties=["summary"],
                    api_endpoint=api_endpoint,
                    model=embedding_model_name
                )
            ]
        else:
            raise ValueError(f'不支持的collection_type参数: {collection_type}')

        db_client.collections.create(
            name=collection_name,
            reranker_config=Configure.Reranker.transformers(),
            vectorizer_config=vectorizer_settings,
            properties=properties_to_create
        )
        print(f'向量数据表{collection_name}创建成功')


def delete_collection(db_client, collection_name: str=None, delete_all: bool=False):
    if delete_all:
        if db_client.is_ready(): # is_ready() 检查连接
            db_client.collections.delete_all()
            print(f'已删除全部数据表')
            return None

    if collection_name:
        if db_client.collections.exists(collection_name): # is_ready() 检查连接
            db_client.collections.delete(collection_name)
            print(f'数据表{collection_name}删除成功')
        else:
            print(f'数据表{collection_name}，不存在，不执行删除操作')
    else:
        raise ValueError(f'参数collection_name不能为空')

@timing_decorator
def save_to_db(db_client, data_dict: dict = None, collection_name: str=None):
    """
    将数据存入数据库
    """
    # print('待入库信息：', data_dict)
    if data_dict is None:
        print(f'没有数据需要保存')
        return False
    else:
        # print(f'需要保存的数据:{data_dict}')
        print(f'检查数据表{collection_name}是否存在')

        # 检查数据表是否存在
        if not db_client.collections.exists(collection_name):
            print(f'向量数据库中不存在{collection_name}数据表')
            return False
        else:
            # 验证数据表是否存在
            if not db_client.collections.exists(collection_name):
                print(f'数据表{collection_name}不存在')
            else:
                document_collection = db_client.collections.get(collection_name)
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
                    return False
                else:
                    print(f'插入数据成功')
                return True
                        
            except Exception as e:
                print(f"插入数据时发生错误: {e}")
                return False



def rerank_distances_with_softmax(distances: list[float]) -> list[float]:
    """
    使用Softmax将一组距离转换为概率分布分数。
    返回的分数列表总和为1。
    """
    # 距离越小越好，所以我们对负距离应用softmax
    # numpy可以很好地处理数值稳定性
    neg_distances = -np.array(distances)
    
    # 计算Softmax
    e_x = np.exp(neg_distances - np.max(neg_distances)) # 减去最大值防止溢出
    probabilities = e_x / e_x.sum()
    
    return probabilities.tolist()


def db_search(db_client,
              collection_name: str,
              user_request: str,
              search_num: int=100,
              output_num: int=10,
              fusion_request: str=None):
    """
    请求向量数据库,从知识库中搜索相关信息
    """
    # 连接数据表
    collection = db_client.collections.get(collection_name)

    # # 统计表格规模
    # aggregation_result = collection.aggregate.over_all(total_count=True)
    # print(f'检索表为:{collection_name},表格规模为{aggregation_result.total_count}条')
    
    if fusion_request:
        print('开始强力检索')
        start_time = time.time()
        # 利用fusion request进行初筛
        hybrid_query_response = collection.query.hybrid(
            query=fusion_request,
            query_properties=["content_for_rerank"],
            target_vector="content_for_rerank_vector",
            limit=search_num
        )
        temp_time = time.time()
        print(f'初筛耗时为:{temp_time - start_time}s')

        # 检索结果的uuid提取
        content_uuids = [result.uuid for result in hybrid_query_response.objects]
        unique_uuids = list(set(content_uuids))
        id_filter = Filter.by_id().contains_any(unique_uuids)

        # 使用原始请求精筛
        print('执行rerank')
        rerank_query_response = collection.query.near_text(
            query=user_request,
            filters=id_filter,
            target_vector="content_for_rerank_vector",
            rerank=Rerank(prop="content_for_rerank",
                          query=user_request),
            return_metadata=MetadataQuery(distance=True),
            limit=output_num
        )
        end_time = time.time()
        search_time = end_time - start_time
        print(f'精筛耗时为：{end_time - temp_time}')
        print(f'强力检索模式下的单次检索耗时为:{search_time}s')

    else:
        print('开始快速检索')
        start_time = time.time()
        rerank_query_response = collection.query.near_text(
            query=user_request,
            limit=output_num,
            target_vector="summary_vector",
            return_metadata=MetadataQuery(distance=True)           
        )
        end_time = time.time()
        search_time = end_time - start_time
        print(f'快速检索模式下的单次检索耗时为:{search_time}s')

    return {"response":rerank_query_response, "search_time": search_time}


# if __name__ == '__main__':
#     client = weaviate.connect_to_local()

#     try:
#         meta_info = client.get_meta()
#         print(meta_info)

#     finally:
#         client.close()
