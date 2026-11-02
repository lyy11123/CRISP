
import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))


import configparser


def cfg_to_dict(cfg_path):
    config = configparser.ConfigParser()
    config.optionxform = str

    config.read(cfg_path)

    cfg_dict = {}

    for section in config.sections():
        cfg_dict[section] = {}
        for key, value in config.items(section):
            try:
                if value.startswith('[') and value.endswith(']'):
                    value = eval(value)
                else:
                    value = int(value) if value.isdigit() else float(value) if '.' in value else value
            except Exception:
                pass
            cfg_dict[section][key] = value
    return cfg_dict


if __name__ == "__main__":
    template_cfg = cfg_to_dict(os.path.join(file_path, '../src/platform/Dojo/cfg/template.cfg'))
    print(template_cfg)
