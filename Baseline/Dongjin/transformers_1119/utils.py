import os
import json
import re

def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def save_json(dicts, file_path):
    with open(file_path, "w") as f:
        json.dump(dicts, f, indent=4)

def load_conf(work_dir_path='', rel_default_path='conf/default.json', rel_exp_path=None):
    default_path = os.path.join(work_dir_path, rel_default_path)
    conf = read_json(default_path)

    if rel_exp_path is not None:
        exp_path = os.path.join(work_dir_path, rel_exp_path)
        new_conf = read_json(exp_path)

        for key, value in new_conf.items():
            conf[key] = value

    return conf


def renew_if_path_exist(path, num_try=100000):
    if not(os.path.exists(path)):
        return path
    else:
        for i in range(num_try):
            temp_path = path + f'_{i}'
            if not(os.path.exists(temp_path)):
                return temp_path            
                
        raise Exception("path 지정을 실패했습니다.")


def combine_paths(base_path, rel_paths):
    # 여러 상대 경로를 절대 경로로 변경하기
    paths = [os.path.join(base_path, rel_path) for rel_path in rel_paths]
    return paths