import yaml
import argparse
from typing import Any, Dict, Optional, Union

class Config:
    def __init__(self, config_file: str, overrides: Dict[str, Any], key_config):
        self.config = self.load_config(config_file, key_config)
        if overrides is not None:
            self.apply_overrides(overrides)

    def load_config(self, config_file: str, key_config: Optional[Union[str, None]] = None) -> Dict[str, Any]:
        with open(config_file, 'r') as file:
            if key_config:
                return yaml.safe_load(file)[key_config]
            return yaml.safe_load(file)

    def apply_overrides(self, overrides: Dict[str, Any]):
        for key, value in overrides.items():
            if value is not None:
                keys = key.split('__')
                self._set_value(self.config, keys, value)

    def _set_value(self, config_dict, keys, value):
        for key in keys[:-1]:
            config_dict = config_dict.setdefault(key, {})
        config_dict[keys[-1]] = value

def add_arguments(parser, config, parent_key=''):
    """ Рекурсивно добавляет аргументы в парсер на основе конфигурации """
    for key, value in config.items():
        full_key = f"{parent_key}__{key}" if parent_key else key
        if isinstance(value, dict):
            add_arguments(parser, value, full_key)
        else:
            parser.add_argument(f'--{full_key}', type=type(value), default=value)

def parse_args(config_file):
    try:
        config = Config.load_config(None, config_file, key_config='experiment')
        parser = argparse.ArgumentParser(description='Override config parameters.')
        parser.add_argument(f'--config', type=str, default=None)
        parser.add_argument(f'--sweep_config', type=str, default=None)
        add_arguments(parser, config)
        args = vars(parser.parse_known_args()[0])
    except Exception as e:
        print(f"Error with experiment config parsing -> {e}")
        return dict()
    return args

def parse_config(config_path, use_args=False):
    config_file = config_path
    if use_args:
        args = parse_args(config_file)
    else:
        args = None
    config = Config(config_file, args, key_config='experiment')
    print(config.config)
    return config.config

def update_dict(dest, source):
    for key, value in source.items():
        # Разбиваем ключ на части по точке
        keys = key.split('.')
        # Получаем последний словарь для ключа
        d = dest
        for sub_key in keys[:-1]:
            # Создаем новый словарь, если нужно
            if sub_key not in d:
                d[sub_key] = {}
            d = d[sub_key]
        # Обновляем значение последнего ключа
        d[keys[-1]] = value
    return dest

# if __name__ == '__main__':
#     main()
