# 模型相关函数

import ollama


def connect_to_ollama(ollama_address: str=None, quick_timeout:int=60, normal_timeout:int = 600) -> ollama.Client:
    remote_flag = True
    if ollama_address:
        try:
            client = ollama.Client(host=ollama_address, trust_env=False, timeout=normal_timeout)
            quick_client = ollama.Client(host=ollama_address, trust_env=False, timeout=quick_timeout)
            print(f'成功连接到远端({ollama_address})ollama服务器')
        except Exception as e:
            print('无法连接到ollama，采用本地ollama服务')
            client = ollama.Client(timeout=normal_timeout)
            quick_client = ollama.Client(timeout=quick_timeout)
            remote_flag = False
    else:
        async_client = ollama.AsyncClient()
        client = ollama.Client()
        remote_flag = False

    return client, quick_client, remote_flag


def unload_ollama_model(client, model_name: str):
    """
    在模型使用完毕后，卸载模型以释放资源
    """
    try:
        print(f"尝试手动卸载模型：{model_name}")
        client.client.generate(
            model=model_name,
            prompt='.',
            options={'keep_alive': 0}
        )
        print(f"✅ 成功发送卸载请求给模型: {model_name}")
    except Exception as e:
        # 如果模型本身就没加载，可能会收到一个错误，可以忽略
        print(f"⚠️ 在卸载模型 {model_name} 时出现异常: {e}")
        print("   (这通常是正常的，如果模型本来就没有被加载)")