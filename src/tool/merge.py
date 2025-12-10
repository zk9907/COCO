from copy import deepcopy
import pickle
import json 
import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.GenConfig import GenConfig
import pandas as pd
import matplotlib.pyplot as plt
def classify_distribution(db_names):
    all_distribution = []
    for db_name in db_names:
        distribution_path = f'{GenConfig.distribution_dir}/{db_name}.json'
        if not os.path.exists(distribution_path):
            continue
        distribution = json.load(open(distribution_path, 'r'))
        if 'features' in distribution:
            for dist_feature in distribution['features'].values():
                all_distribution.append(dist_feature)

    if not all_distribution:
        print("No distribution data found.")
        return

    # Flatten the distribution data
    flattened_data = np.array(all_distribution).reshape(len(all_distribution), -1)
    
    # Find the optimal number of clusters using the elbow method
    # inertias = []
    # K = range(2, 60)
    # for k in K:
    #     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    #     kmeans.fit(flattened_data)
    #     inertias.append(kmeans.inertia_)
    
    # # Simple elbow detection
    # diffs = np.diff(inertias, 2)
    # optimal_k = np.argmax(diffs) + 2 # add 2 to offset the diff index
    # print(diffs)
    # print(f"Optimal number of clusters: {optimal_k}")

    # # Perform clustering with the optimal k
    kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
    kmeans.fit(flattened_data)
    # print(kmeans.labels_)
    # Save the clustering model
    # with open(GenConfig.distribution_cluster_path, 'wb') as f:
    #     pickle.dump(kmeans, f)
    # print(f"Clustering model saved to {GenConfig.distribution_cluster_path}")
    return kmeans
def one_hot_transform(i, max_class):
    one_hot_data = np.zeros(max_class, dtype=np.int8)
    one_hot_data[i-1] = 1
    return one_hot_data
def get_additional_info(metaInfo):
    distributionDir = GenConfig.distribution_dir
    # attrEmbedDir = GenConfig.attrEmbed_dir
    kmeans = classify_distribution(GenConfig.databases)
    # target data
    # table meta : table_size, num_columns, num_references, num_indexes
    # column meta : n_distinct, null_frac, is_primary_key, is_foreign_key, has_index
    # max_column = 60
    # table = {'tableMeta':[[0,0,0]],'tableEmbed':[[0.0] * 16],'columnIdx':[[0] * max_column]}
    table = {'tableMeta':[[0,0,0]]}
    # column = {'semantic':[[0.0] * 16],'colDtype':[0],'meta':[],'distribution':[0]}
    column = {'colDtype':[0],'meta':[],'distribution':[0]}
    # database = [[0.0] * 50]
    assindex = {'column2idx':{'NA':0},'idx2column':{'0':'NA'},'table2idx':{'NA':0},'idx2table':{'0':'NA'},'database2idx':{'NA':0},'idx2database':{'0':'NA'}}
    for dbName in metaInfo.keys():
        # if dbName == 'baseball': continue
        # if os.path.exists(f'{attrEmbedDir}/{dbName}.json'):
        #     attrEmbed = json.load(open(f'{attrEmbedDir}/{dbName}.json', 'r'))
        # else:
        #     attrEmbed = {'database':[0.0] * 50,'Table':{},'Column':{}}
        if os.path.exists(f'{distributionDir}/{dbName}.json'):
            distribution = json.load(open(f'{distributionDir}/{dbName}.json', 'r'))
        else:
            print(f'{distributionDir}/{dbName}.json not found')
            distribution = {'features':{}}
        idx = len(assindex['database2idx'])
        # database.append(attrEmbed['database'])
        assindex['database2idx'][dbName] = idx
        assindex['idx2database'][idx] = dbName
        for tableName in metaInfo[dbName]['tableAttr'].keys():
            idx = len(table['tableMeta'])
            table['tableMeta'].append([metaInfo[dbName]['tableAttr'][tableName]['table_size'],
                                      metaInfo[dbName]['tableAttr'][tableName]['num_columns'],
                                    #   metaInfo[dbName]['tableAttr'][tableName]['num_references'],
                                      metaInfo[dbName]['tableAttr'][tableName]['num_indexes']])
            # if tableName in attrEmbed['Table'].keys():
            #     table['tableEmbed'].append(attrEmbed['Table'][tableName])
            # else:
            #     table['tableEmbed'].append([0.0] * 16)
            # table['columnIdx'].append([0] * max_column)
            assindex['table2idx'][dbName+ '.' + tableName] = idx
            assindex['idx2table'][idx] = dbName+ '.' + tableName
        for table_col in metaInfo[dbName]['colAttr'].keys():
            tableName, colName = table_col.split('.')
            # table_idx = assindex['table2idx'][dbName+ '.' + tableName]
            # column_idx = 0
            # for colName, _ in colEmbedDict.items():
            idx = len(column['colDtype'])
            # if tableName + '.' + colName in colEmbed:
            #     column['semantic'].append(colEmbed[tableName + '.' + colName])
            # else:
            #     column['semantic'].append([0.0] * 16)
            column['colDtype'].append(metaInfo[dbName]['colAttr'][tableName + '.' + colName]['colDtype'])
            meta_data = [
            [metaInfo[dbName]['tableAttr'][tableName]['table_size'],
            metaInfo[dbName]['tableAttr'][tableName]['num_columns'],
            # metaInfo[dbName]['tableAttr'][tableName]['num_references'],
            metaInfo[dbName]['tableAttr'][tableName]['num_indexes']],
            [metaInfo[dbName]['colAttr'][tableName + '.' + colName]['join_num']],
            one_hot_transform(metaInfo[dbName]['colAttr'][tableName + '.' + colName]['correlation'], 3),
            one_hot_transform(metaInfo[dbName]['colAttr'][tableName + '.' + colName]['avg_width'], 3),
            one_hot_transform(metaInfo[dbName]['colAttr'][tableName + '.' + colName]['n_distinct'], 3),
            one_hot_transform(metaInfo[dbName]['colAttr'][tableName + '.' + colName]['null_frac'], 3),
            one_hot_transform(metaInfo[dbName]['colAttr'][tableName + '.' + colName]['is_primary_key'], 2),
            one_hot_transform(metaInfo[dbName]['colAttr'][tableName + '.' + colName]['is_foreign_key'], 2),
            one_hot_transform(metaInfo[dbName]['colAttr'][tableName + '.' + colName]['has_index'], 2)]
            meta_data = np.concatenate(meta_data)#.reshape(1, -1)
            column['meta'].append(meta_data)
            if dbName + '.' + tableName + '.' + colName in distribution['features'].keys():
                column['distribution'].append(kmeans.predict(np.array(distribution['features'][dbName + '.' + tableName + '.' + colName]).reshape(1, -1))[0] + 1)
            else:
                column['distribution'].append(0)#([[0.0] * 6 for _ in range(10)])
            assindex['column2idx'][dbName + '.' + tableName + '.' + colName] = idx
            assindex['idx2column'][idx] = dbName + '.' + tableName + '.' + colName
            # if column_idx < max_column:
            #     table['columnIdx'][table_idx][column_idx] = idx
            # column_idx += 1
    column['meta'] = np.array([np.zeros(len(column['meta'][0]), dtype=np.int8)] + column['meta'])
    # print(column['meta'][-1])
    # print(len(column['meta']))
    # print(column['meta'][1])
    # meta_data = np.array(column['meta'][1:], dtype=np.float64)
    # meta_data = np.nan_to_num(meta_data, nan=0.0)
    # print(meta_data[:,1])
    # pd.DataFrame(meta_data[:,2]).hist()
    # plt.savefig('correlation.png')
    # if len(column['meta']) > 1:
    #     meta_data = np.array(column['meta'][1:], dtype=np.float64)
    #     # take log2 of the first column(table_size), add 1 to avoid log(0)
    #     # meta_data[:, 0] = np.log2(meta_data[:, 0] + 1)
    #     meta_data = np.nan_to_num(meta_data, nan=0.0)
    #     min_vals = np.min(meta_data, axis=0)
    #     max_vals = np.max(meta_data, axis=0)
    #     range_vals = max_vals - min_vals
    #     # Avoid division by zero
    #     range_vals[range_vals == 0] = 1.0
    #     normalized_meta = (meta_data - min_vals) / range_vals
    #     # normalized_meta = np.nan_to_num(normalized_meta, nan=0.0)
    #     column['meta'] = [[0.0] * 7] + normalized_meta.tolist()
    # print(len(column['meta']))
    # semantic_column = deepcopy(column['semantic'])
    # semantic_table = deepcopy(table['tableEmbed'])
    # del column['semantic']
    # del table['tableEmbed']
    pickle.dump(assindex, open(GenConfig.assindex_path, 'wb'))
    pickle.dump(column, open(GenConfig.column_feature_path, 'wb'))
    pickle.dump(table, open(GenConfig.table_feature_path, 'wb'))
    # pickle.dump(database, open(GenConfig.database_path, 'wb'))
    # pickle.dump(semantic_column, open(GenConfig.semantic_column_path, 'wb'))
    # pickle.dump(semantic_table, open(GenConfig.semantic_table_path, 'wb'))
metaInfo = pickle.load(open(GenConfig.meta_info_path, 'rb'))
# print(metaInfo)
get_additional_info(metaInfo)
# classify_distribution(GenConfig.databases)
