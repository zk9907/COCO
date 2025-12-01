import os
import sys
import pickle
import json
import ray
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.SQLBuilder import SQLBuilderActor
from config.LCQOConfig import Config
from util.SQLBuffer import SQL

def load_queries(path):
    sql_statements = {}
    if path.endswith('.json'):
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                sql_statements[data[0]] = data[1]
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            sql_statements = pickle.load(f)
    return sql_statements

def run_sql(workload, dbName):
    # Initialize Ray
    ray.init()
    if os.path.exists(f'./TestWorkload/test_{workload}_sql.pkl'):
        with open(f'./TestWorkload/test_{workload}_sql.pkl', 'rb') as f:
            test_sql = pickle.load(f)
        exists_qid = []
        tm_test_sql = []
        for sql in test_sql:
            # if not sql.id.startswith('q11'):
            print(f'add {sql.id}')
            exists_qid.append(sql.id)
            tm_test_sql.append(sql)
        test_sql = tm_test_sql
    else:
        test_sql = []
        exists_qid = []
    # Load queries
    queries = load_queries(f'./TestWorkload/{workload}.json')
    items = list(queries.items())
    total = len(items)
    print(f'Total queries: {total}')
    idx = 0

    # Create builder actors, one per DB config
    lqo_config = Config()
    actors = []
    for dbConfig in lqo_config.remote_db_config_list:
        actor_config = Config()
        actor_config.db_config = dbConfig
        actors.append(SQLBuilderActor.remote(actor_config))
        print(f'actor created',flush=True)

    # Map of pending futures to their actors
    pending = {}
    
    # Launch initial batch of tasks
    fut = None
    for actor in actors:
        qid, qtxt = items[idx]
        if qid not in exists_qid:
            fut = actor.build.remote(dbName, qtxt, qid)
            pending[fut] = actor
        else:
            print(f'{qid} exists', flush = True)
            fut = None
        idx += 1

    # Dynamically assign new tasks as actors become free
    while pending or idx < total:
            
        done, _ = ray.wait(list(pending.keys()), num_returns=1)
        if len(done) == 0:
            continue
        fut = done[0]
        actor = pending.pop(fut)
        sql:SQL = ray.get(fut)
        if sql:
            test_sql.append(sql)
            print(sql.id, sql.base_latency, sql.min_latency, len(sql.plans), flush=True)
            with open(f'./TestWorkload/test_{workload}_sql.pkl', 'wb') as f:
                pickle.dump(test_sql, f)
        if idx < total:
            qid, qtxt = items[idx]
            new_fut = actor.build.remote(dbName, qtxt, qid)
            pending[new_fut] = actor
            idx += 1

    print(f'Built {len(test_sql)} SQLs successfully',flush=True)
    ray.get([actor.shutdown.remote() for actor in actors])
    ray.shutdown()

if __name__ == '__main__':
    workloads = {'job':'imdb','jobext':'imdb','stats':'stats','tpcds':'tpcds'}
    for w,dbName in workloads.items():
        run_sql(w,dbName = dbName + '_' + w)
