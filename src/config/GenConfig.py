import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
class GenConfig:
    db_config = {
            'database': '',
            'user': '',
            'password': '',
            'port': 5431,
            'host': ''
        }
    base_db_config = db_config
    remote_db_config_list = [
        base_db_config.copy(),
    ]
    databases = ['basketball', 'walmart','financial', 'movielens',
                'carcinogenesis','accidents', 'tournament',
                'employee', 'geneea', 'genome','seznam', 'fhnk', 'consumer',
                'ssb', 'hepatitis', 'credit','chembl','ergastf1','grants',
                'legalacts','sap','talkingdata','baseball', 'tpch','stats','imdb', 'tpcds']
    test_databases = ['stats','imdb','tpcds']
    
    N_bins = 10
    random_source_workloads_path = './TestWorkload/random_source_workload.pkl'
    warmup_workloads_path = './TestWorkload/warmup_workload.pkl'
    checkpoint_dir = './ckpt'
    column_stats_dir = './features/dataset/' 
    meta_info_path = './features/meta/metaInfo.pkl'
    assindex_path = './features/meta/assindex.pkl'
    column_feature_path = './features/meta/column.pkl'
    table_feature_path = './features/meta/table.pkl'
    sample_dir = './features/sample'
    distribution_dir = './features/distribution'
    schema_dir = './features/schema'