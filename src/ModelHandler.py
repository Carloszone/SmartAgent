# 模型相关函数
import ollama
from .ConfigManager import config
from funasr import AutoModel
from .tool import timing_decorator


class RagModels:
    def __init__(self):
        # ollama model
        self.model_settings = config.get_setting("models")
        self.ollama_model_settings = None
        self.ollama_quick_timeout = None
        self.ollama_normal_timeout = None
        self.ollama_client = None
        self.ollama_quick_client = None
        self.remote_flag = False
        self.audio_model_client = None

    def connect_to_ollama(self, ollama_address: str=None):  # 后期添加timeout控制参数
        if ollama_address:
            print('尝试连远端ollama服务器')
            self.ollama_client = ollama.Client(host=ollama_address, trust_env=False)
            self.ollama_flash_client = ollama.Client(host=ollama_address, trust_env=False, timeout=1)
            self.remote_flag = True
            self.ollama_model_settings = self.model_settings["remote_ollama_models"]
            self.ollama_model_check()
            self.load_audio_model()
            print('远端服务器连接成功')
        else:
            print('无法连接到远端ollama服务器，尝试连接本地ollama服务')
            self.ollama_client = ollama.Client(trust_env=False)
            self.ollama_flash_client = ollama.Client(trust_env=False, timeout=1)
            self.remote_flag = False
            self.ollama_model_settings = self.model_settings["local_ollama_models"]
            self.ollama_model_check()
            self.load_audio_model()
    
    def ollama_model_check(self, model_info: list=None):
        if model_info is None:
            if self.remote_flag:
                model_info = self.model_settings.get("remote_ollama_models")
            else:
                model_info = self.model_settings.get("local_ollama_models")

        model_list = [model["model"] for model in self.ollama_client.list()['models']]

        for key, value in model_info.items():
            if key[-5:] == 'model':
                if value in model_list:
                    print(f'{key}模型({value})存在')
                else:
                    print(f'{key}模型({value})不存在')
                    raise ValueError(f'{key}模型({value})不存在')
                
        # 预先加载图像和文本处理
        # self.warm_up_model(model_name=self.ollama_model_settings['image_caption_model'])
        # self.warm_up_model(model_name=self.ollama_model_settings['text_embedding_model'])

    def load_audio_model(self, audio_model_address: str=None, 
                         audio_vad_mode_address:str=None):
        if audio_model_address is None:
            audio_model_address = config.get_setting('models')['audio_model_address']
        if audio_vad_mode_address is None:
            audio_vad_mode_address = config.get_setting('models')["audio_vad_model_address"]

        try:
            self.audio_model_client = AutoModel(
                model=audio_model_address,
                vad_model=audio_vad_mode_address,
                vad_kwargs={"max_single_segment_time": 30000},
                device="cuda:0",
                disable_update = True)
            print(f'音频模型加载成功')
        except Exception as e:
            print(f'音频模型加载失败,错误信息：{e}')
            return False
        
    @timing_decorator
    def call_model(self, model_params: dict, model_client=None) -> dict:
        """
        调用模型的函数
        """
        if model_client is None:
            model_client = self.ollama_client

        if isinstance(model_client, ollama.Client):
            response = self.ollama_client.chat(**model_params)
        elif isinstance(model_client, AutoModel):
            response = self.audio_model_client.generate(**model_params)
        else:
            raise ValueError(f'无法识别的模型客户端类型。当前类型为:{type(model_client)}')
        return response

    def warm_up_model(self, model_name: str):
        """
        预加载模型，应对模型冷启动问题
        """
        try:
            self.ollama_client.chat(
                model=model_name,
                messages=[{"role": "user", "content": ""}],
                options={"keep_alive": -1, "think": False}
            )
            print(f'预加载模型：{model_name}成功')
        except Exception as e:
            print(f'预加载模型失败，错误信息:{e}')


    def unload_ollama_model(self, model_name: str):
        """
        在模型使用完毕后，卸载模型以释放资源
        """
        try:
            self.ollama_client.chat(
                model=model_name,
                messages=[{"role": "user", "content": ""}],
                options={'keep_alive': 0, "think": False}
            )
            print(f"成功发送卸载请求给模型: {model_name}")
        except Exception as e:
            print(f"在卸载模型 {model_name} 时出现异常: {e}")