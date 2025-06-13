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

    # --- 第二步：定义并创建“数据表” (Class) ---
    collection_name = "TestCollection"

    # 定义元数据字段结构
    metadata_properties = [
        Property(name="file_source", data_type=DataType.TEXT, description="文件来源路径或URL"),
        Property(name="document_type", data_type=DataType.TEXT, description="文档类型, e.g., 'PDF', 'Markdown'"),
        Property(name="access_level", data_type=DataType.TEXT, description="访问级别, e.g., 'public', 'private'"),
        Property(name="page_number", data_type=DataType.INT, description="在原始文件中的页码"),
        Property(name="chunk_seq_id", data_type=DataType.INT, description="文件内文本块的序列号"),
    ]

    # 然后，定义顶层的字段结构
    collection_properties = [
        # 用于存储原始文本，可以用于混合搜索或直接查看
        Property(
            name="summary_text",
            data_type=DataType.TEXT,
            description="文本内容的摘要",
            index_searchable=True, # 允许对摘要进行关键词搜索
            index_filterable=True
        ),
        
        # 定义 metadata 字段，其类型为 OBJECT，并引用上面定义的嵌套结构
        Property(
            name="metadata",
            data_type=DataType.OBJECT,
            description="包含所有元信息的对象",
            nested_properties=metadata_properties
        )
    ]

    # --- 2. 连接 Weaviate 并执行创建 ---

    try:
        with weaviate.connect_to_local() as client:
            print("✅ 成功连接到 Weaviate！")

            if client.collections.exists(collection_name):
                print(f"🟡 数据表 '{collection_name}' 已存在，无需创建。")
            else:
                print(f"## 正在创建数据表 '{collection_name}'...")
                client.collections.create(
                    name=collection_name,
                    properties=collection_properties,
                    vectorizer_config=Configure.Vectorizer.none()
                )
                print(f"✅ 数据表 '{collection_name}' 创建成功！")

    except Exception as e:
        print(f"❌ 操作失败: {e}")

    print('读取字段信息')
    my_collection = client.collections.get(collection_name)
    print("提取表配置信息")
    updated_config = my_collection.config.get()
    print("打印表属性")
    for prop in updated_config.properties:
        print(f"- 现有字段: {prop.name}")
finally:
    # --- 最后一步：关闭连接 ---
    if 'client' in locals() and client.is_connected():
        client.close()
        print("\n连接已关闭。")