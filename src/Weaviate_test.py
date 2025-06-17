import weaviate
# v4 版本中，配置相关的类（如 Property, DataType）需要从 weaviate.classes.config 导入
from weaviate.classes.config import Configure, Property, DataType, Vectorizers
# v4 版本中，查询相关的类（如 Filter）需要从 weaviate.classes.query 导入
from weaviate.classes.query import Filter
import json

# --- 第一步：连接到您的 Weaviate 服务 ---
try:
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051
    )
    print("成功连接到 Weaviate!")

#     # --- 第二步：删除全部collections ---
#     if client.is_ready(): # is_ready() 检查连接
#         client.collections.delete_all()

#     # --- 第三步：定义新 Collection 的名称和字段结构 ---
#     collection_name = "knowledge_base_collection"
    
#     properties_to_create = [
#     Property(
#         name="metadata",
#         data_type=DataType.OBJECT,
#         # 在 nested_properties 中定义所有元数据子字段
#         nested_properties=[
#             Property(
#                 name="source",
#                 description="来源文件路径或URL",
#                 data_type=DataType.TEXT,
#                 index_filterable=True
#             ),
#             Property(
#                 name="document_type",
#                 description="文档类型 (e.g., PDF, DOCX, MD)",
#                 data_type=DataType.TEXT,
#                 index_filterable=True
#             ),
#             Property(
#                 name="access_level",
#                 description="访问控制级别",
#                 data_type=DataType.INT,
#                 index_filterable=True
#             ),
#             Property(
#                 name="page_number",
#                 description="在原始文件中的页码",
#                 data_type=DataType.INT,
#                 index_filterable=True
#             ),
#             Property(
#                 name="chunk_seq_id",
#                 description="文本块在其所属文本中的序列ID",
#                 data_type=DataType.INT,
#                 index_filterable=True
#             ),
#         ]
#     )

#         # 注意：我们不需要定义 embedding 字段，因为它是每个对象的固有部分
#     ]


#     # --- 第四步：创建新的 Collection ---
#     print(f"\n## 正在创建新的 Collection: '{collection_name}'...")
#     client.collections.create(
#         name=collection_name,
#         properties=properties_to_create,
#         # 因为您要直接传入 embedding，所以必须将 vectorizer 设置为 none
#         vectorizer_config=Configure.Vectorizer.none()
#     )
#     print(f"✅ Collection '{collection_name}' 创建成功！")


#     # --- 重复第三步和第四步，构建第二张表 ---
#     collection_name = "chat_histroy_collection"
    
#     properties_to_create = [
#         Property(
#             name="metadata",
#             data_type=DataType.OBJECT,
#             # 在 nested_properties 中定义所有元数据子字段
#             nested_properties=[
#                 Property(
#                     name="chat_id",
#                     description="聊天会话id",
#                     data_type=DataType.TEXT,
#                     index_filterable=True
#                 ),
#                 Property(
#                     name="user_id",
#                     description="发起聊天的用户的id",
#                     data_type=DataType.TEXT,
#                     index_filterable=True
#                 ),
#                 Property(
#                     name="record_type",
#                     description="对应文本所属类型,可以是问题(question)或回答(answer)",
#                     data_type=DataType.TEXT,
#                     index_filterable=True
#                 ),
#                 Property(
#                     name="access_level",
#                     description="访问控制级别",
#                     data_type=DataType.INT,
#                     index_filterable=True
#                 ),
#                 Property(
#                     name="chunk_seq_id",
#                     description="文本块在其所属文本中的序列ID",
#                     data_type=DataType.INT,
#                     index_filterable=True
#                 ),
#             ]
#         )
#     ]


#     print(f"\n## 正在创建新的 Collection: '{collection_name}'...")
#     client.collections.create(
#         name=collection_name,
#         properties=properties_to_create,
#         vectorizer_config=[
#             Configure.NamedVectors.none(name='chunk_vector'),
#             Configure.NamedVectors.none(name='topic_vector')
#         ]
#     )
#     print(f"✅ Collection '{collection_name}' 创建成功！")


# except Exception as e:
#     print(f"\n❌ 操作过程中发生错误: {e}")

    collection_name = "knowledge_base_collection"
    document_collection = client.collections.get(collection_name)
    aggregation_result = document_collection.aggregate.over_all(total_count=True)
    total = aggregation_result.total_count
    print(f"\n📊 数据表 '{collection_name}' 的总记录数为: {total}")

finally:
    # --- 最后一步：关闭连接 ---
    if 'client' in locals() and client.is_connected():
        client.close()
        print("\n连接已关闭。")