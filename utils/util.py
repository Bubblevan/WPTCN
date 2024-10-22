import yaml

def load_config(config_file):
    """加载配置文件"""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)