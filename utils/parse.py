import json
import ruamel_yaml

root = r'root/workspace/gene/full_lung_code/code'

def parse_yaml(file='/root/workspace/gene/full_lung_code/config/EGFR/default.yaml'):
    with open(file) as f:
        return ruamel_yaml.load(f, Loader=ruamel_yaml.Loader)


def format_config(config, indent=2):
    return json.dumps(config, indent=indent)