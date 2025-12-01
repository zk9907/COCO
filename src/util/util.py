import json
import os
from types import SimpleNamespace
import pickle
from config.GenConfig import GenConfig
import numpy as np

def load_schema_json(dataset):
    schema_path = os.path.join(GenConfig.schema_dir, f'{dataset}.json')
    assert os.path.exists(schema_path), f"Could not find schema.json ({schema_path})"
    return load_json(schema_path)

def load_meta_data(dataset):
    all_table_info = pickle.load(open(GenConfig.table_feature_path, 'rb'))
    all_column_info = pickle.load(open(GenConfig.column_feature_path, 'rb'))
    index_info = pickle.load(open(GenConfig.assindex_path, 'rb'))
    table_info = {}
    column_info = {}
    for table_name in index_info['table2idx'].keys():
        if table_name.split('.')[0] == dataset:
            idx = index_info['table2idx'][table_name]
            table_info[table_name.split('.')[1]] = {'features':{key:all_table_info[key][idx] for key in all_table_info.keys()}, 'include_column':[]}
    for column_name in index_info['column2idx'].keys():
        if column_name.split('.')[0] == dataset:
            idx = index_info['column2idx'][column_name]
            column_info['.'.join(column_name.split('.')[1:])] = {key:all_column_info[key][idx] for key in all_column_info.keys()}
            table_info[column_name.split('.')[1]]['include_column'].append('.'.join(column_name.split('.')[1:]))
    meta_info = {'table':table_info, 'column':column_info}
    return meta_info
def load_index_info():
    index_info = pickle.load(open(GenConfig.assindex_path, 'rb'))

    return index_info

def load_column_statistics(dataset, namespace=True):
    path = os.path.join(GenConfig.column_stats_dir, f'{dataset}_column_statistics.json')
    assert os.path.exists(path), f"Could not find file ({path})"
    return load_json(path, namespace=namespace)


def load_json(path, namespace=True):
    with open(path) as json_file:
        if namespace:
            json_obj = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
        else:
            json_obj = json.load(json_file)
    return json_obj


def rand_choice(randstate, l, no_elements=None, replace=False):
    if no_elements is None:
        idx = randstate.randint(0, len(l)) - 1
        return l[idx]
    else:
        idxs = randstate.choice(range(len(l)), no_elements, replace=replace)
        return [l[i] for i in idxs]
def swap_dict_items(data, key1, key2):
    if key1 not in data or key2 not in data:
        raise KeyError("Index Error")
    items = list(data.items())
    index1 = next(i for i, (k, v) in enumerate(items) if k == key1)
    index2 = next(i for i, (k, v) in enumerate(items) if k == key2)
    items[index1], items[index2] = items[index2], items[index1]
    new_data = dict(items)
    return new_data

def minmax_transform(x, min_max):
    min_val, max_val = min_max
    if min_val == max_val:
        return 0.0
    return (x - min_val) / (max_val - min_val)
def normalize_narray(scores: np.ndarray) -> np.ndarray:
    min_score = scores.min()
    max_score = scores.max()
    return (scores - min_score) / (max_score - min_score)

def planByGeqo():
    hints = [
        "SET GEQO TO ON;",
        "SET geqo_threshold TO 2;"
    ]
    return hints