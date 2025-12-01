import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LCQO.pghelper import PGHelper
import pickle
import json
from config.GenConfig import GenConfig
from util.util import load_schema_json
import collections
def get_meta_info(database_names,accuracy_mode=True):
    # Database configurations
    if os.path.exists(GenConfig.meta_info_path):
        old_metaInfo = pickle.load(open(GenConfig.meta_info_path, 'rb'))
    else:
        old_metaInfo = {}
    pg_helper = PGHelper(GenConfig.db_config)
    # Initialize metaInfo dictionary
    metaInfo = {db_name: {} for db_name in database_names}
    
    # Get statistics and column types for each database
    for db_name in database_names:
        print(db_name,flush=True)
        schema = load_schema_json(db_name)
        pg_helper.reconnect(db_name)
        metaInfo[db_name] = pg_helper.get_statistics(old_metaInfo[db_name] if db_name in old_metaInfo else None, accuracy_mode=accuracy_mode)
        columnType = pg_helper.get_column_type()
        # Convert column types to indices
        for colName, columnType in columnType.items():
            # if columnType not in columnType2Idx:
            #     columnType2Idx[columnType] = len(columnType2Idx) + 1
            if columnType in ['smallint','integer','bigint','smallserial','serial','bigserial']:
                metaInfo[db_name]['colAttr'][colName]['colDtype'] = 1
            elif columnType in ['decimal','numeric','real','double precision']:
                metaInfo[db_name]['colAttr'][colName]['colDtype'] = 2
            else:
                metaInfo[db_name]['colAttr'][colName]['colDtype'] = 3

        possible_joins = collections.defaultdict(list)
        for table_l, column_l, table_r, column_r in schema.relationships:
            if isinstance(column_l, list) and isinstance(column_r, list):
                for i in range(len(column_l)):
                    possible_joins[table_l + '.' + column_l[i]].append(table_r + '.' + column_r[i])
                    possible_joins[table_r + '.' + column_r[i]].append(table_l + '.' + column_l[i])
            else:
                possible_joins[table_l + '.' + column_l].append(table_r + '.' + column_r)
                possible_joins[table_r + '.' + column_r].append(table_l + '.' + column_l)
        for col_name in metaInfo[db_name]['colAttr']:
            if col_name not in possible_joins:
                metaInfo[db_name]['colAttr'][col_name]['join_num'] = 0
            else:
                metaInfo[db_name]['colAttr'][col_name]['join_num'] = len(possible_joins[col_name])

    # post processing
    max_avg_width = 0
    # max_correlation = 0
    max_table_size = 0
    max_num_columns = 0
    # max_num_references = 0
    max_num_indexes = 0
    max_join_num_per_col = 0
    for db_name in database_names:
        for col_name,col_info in metaInfo[db_name]['colAttr'].items():
            if math.log2(col_info['avg_width'] + 1) > max_avg_width:
                max_avg_width = math.log2(col_info['avg_width'] + 1)
            max_join_num_per_col = max(max_join_num_per_col, col_info['join_num'])
        for table_name,table_info in metaInfo[db_name]['tableAttr'].items():
            max_table_size = max(max_table_size,table_info['table_size_ori'])
            max_num_columns = max(max_num_columns,table_info['num_columns'])
            max_num_indexes = max(max_num_indexes,table_info['num_indexes'])

    for db_name in database_names:
        for col_name,col_info in metaInfo[db_name]['colAttr'].items():
            norm_avg_width = math.log2(col_info['avg_width'] + 1) / max_avg_width
            if norm_avg_width > 0.5:
                col_info['avg_width'] = 3
            elif norm_avg_width < 0.25:
                col_info['avg_width'] = 1
            else:
                col_info['avg_width'] = 2
            col_info['join_num'] = col_info['join_num'] / max_join_num_per_col
        for table_name,table_info in metaInfo[db_name]['tableAttr'].items():
            norm_table_size = math.log10(table_info['table_size_ori'] + 1) / math.log10(max_table_size + 1)
            norm_num_columns = table_info['num_columns'] / max_num_columns
            norm_num_indexes = table_info['num_indexes'] / max_num_indexes
            metaInfo[db_name]['tableAttr'][table_name]['table_size'] = norm_table_size
            metaInfo[db_name]['tableAttr'][table_name]['num_columns'] = norm_num_columns
            metaInfo[db_name]['tableAttr'][table_name]['num_indexes'] = norm_num_indexes

    pg_helper.close()
    return metaInfo
os.makedirs(os.path.dirname('./features/meta/'), exist_ok=True)
metaInfo = get_meta_info(GenConfig.databases,accuracy_mode=True)  # accuracy_mode=True for accuracy mode, False for speed mode
with open(GenConfig.meta_info_path, 'wb') as f:
    pickle.dump(metaInfo, f)