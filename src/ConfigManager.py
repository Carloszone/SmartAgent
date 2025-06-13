import json
import os
import logging


class ConfigManager:
    _instance = None
    _config_data = {}
    _config_loaded = False

    def __new__(cls):
        # 确保只有一个实例
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()  # 首次创建时加载配置
        return cls._instance

    def _load_config(self):
        # 实际加载配置的逻辑
        if self._config_loaded:
            return

        # 从JSON文件加载
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)
        config_file_path = os.path.join(project_root, 'data/config.json')
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                self._config_data = json.load(f)
            self._config_loaded = True
            logging.info(f"成功读取配置信息。配置文件地址：{config_file_path}")
        except FileNotFoundError:
            logging.info(f"无法找到配置文件。配置文件地址：{config_file_path}")
        except json.JSONDecodeError:
            logging.info(f"无法解码配置文件。配置文件地址：{config_file_path}")

    def get_setting(self, key_path, default=None):
        """
        根据键路径获取配置值。
        """
        keys = key_path.split('.')
        current_level = self._config_data
        for key in keys:
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            else:
                return default
        return current_level

    def get_all_config(self):
        """返回所有配置数据"""
        return self._config_data


# 创建一个全局可用的配置实例
config = ConfigManager()
