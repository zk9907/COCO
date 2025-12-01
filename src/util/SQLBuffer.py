import numpy as np
from typing import List
import os,sys
import torch
from config.LCQOConfig import Config as LCQOConfig
import LCQO,util
sys.modules['LQO'] = util
config  = LCQOConfig()

class Node:
    def __init__(self, env_state, plan_state, action=None, parent=None):
        self.env_state = env_state
        self.plan_state = plan_state
        self.action = action  # Action that led to this state
        self.parent = parent
        self.children = {}  # action -> Node
        self.visited = False
        self.code = 0  # Track the current hint code

class Plan:
    def __init__(self, hint_code, plan_str):
        self.hint_code = hint_code
        self.plan_str = plan_str

    def __eq__(self, other):
        return self.plan_str == other.plan_str
    
    def __repr__(self):
        return self.plan_str
    
    def update(self, plan_json, latency, istimeout, feature_dict = None, hint_dict = None):
        self.plan_json = plan_json
        self.latency = latency
        self.istimeout = istimeout
        self.feature_dict = feature_dict
        self.hint_dict = hint_dict

class SQL:
    def __init__(self, dbName: str, sql_statement: str, plans: List[Plan], has_result = None ,sql_infos = None):
        self.dbName = dbName
        self.sql_statement = sql_statement
        # self.pos_in_sqlgen_buffer = None
        self.sqlgen_reward = None
        self.plans = {}
        self.has_result = has_result
        self.q_min_latency = None
        self.sql_infos = sql_infos
        if plans:
            self.min_latency = plans[0].latency
            self.max_latency = plans[0].latency
            for plan in plans:
                self.plans[plan.hint_code] = plan
                self.min_latency = min(self.min_latency, plan.latency)
                self.max_latency = max(self.max_latency, plan.latency)
            self.base_latency = self.plans[0].latency
        else:
            self.min_latency = None
            self.base_latency = None
        
    def get_plan_by_hint_code(self, hint_code: int):
        return self.plans[hint_code]
    
    def update_reward(self, sqlgen_reward: float):
        self.sqlgen_reward = sqlgen_reward

    def update_pos_in_sqlgen_buffer(self, pos: int):
        self.pos_in_sqlgen_buffer = pos

    def update_idx(self, idx: str):
        self.id = idx

class SQLBuffer:
    def __init__(self, buffer_state_dim: int = config.hidden_dim):
        self.db_sql_dict = {}
        self.buffer = []
        self.buffer_state_vec = np.zeros(buffer_state_dim, dtype=np.float32)
        self.latest_idx = []
        self.num_sql = 0
        self.current_all_vectors = None
        
    def push(self, sql: SQL):
        self.buffer.append(sql)
        if sql.dbName not in self.db_sql_dict:
            self.db_sql_dict[sql.dbName] = 0
        self.db_sql_dict[sql.dbName] += 1
        sql_id = sql.dbName + '_' + str(self.db_sql_dict[sql.dbName])
        sql.update_idx(sql_id)
        self.num_sql += 1
        self.latest_idx.append(len(self.buffer) - 1)  # the new added sql will be labeled and update its rewards after updating the LQO Model
        return sql_id
    
    def update_state_vector(self, all_vectors):
        self.buffer_state_vec = np.mean(all_vectors, axis=0)
        self.current_all_vectors = all_vectors

    def get_sql_buffer(self,volume = 'all'):
        if volume == 'all':
            return self.buffer
        elif volume == 'latest':
            latest_sql = [self.buffer[idx] for idx in self.latest_idx]
            self.latest_idx = []
            return latest_sql
        else:
            raise ValueError(f"Invalid volume: {volume}")
    
    # def get_latest_idx(self):
    #     return self.latest_idx
    
    def get_buffer_size(self):
        return len(self.buffer)
    
    def update_sql_reward(self, idx: int, sqlgen_reward: float):
        self.buffer[idx].update_reward(sqlgen_reward)

    def save_state(self, checkpoint_dir):
        if checkpoint_dir.endswith('.pkl'):
            buffer_path = checkpoint_dir
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
            buffer_path = os.path.join(checkpoint_dir, "sql_buffer.pkl")
        buffer_state = {
            'buffer': self.buffer,
            'buffer_state_vec': self.buffer_state_vec
        }
        torch.save(buffer_state, buffer_path)
        return buffer_path

    def load_state(self, checkpoint_dir, sample_size = None, include_test = False):
        if checkpoint_dir.endswith('.pkl'):
            buffer_path = checkpoint_dir
        else:
            buffer_path = os.path.join(checkpoint_dir, "sql_buffer.pkl")
        if not os.path.exists(buffer_path):
            print(f"Buffer state not found at {buffer_path}")
            return False
        print(f"Loading buffer state from {buffer_path}")
        buffer_state = torch.load(buffer_path)
        if not include_test:
            new_buffer = []
            for sql in buffer_state['buffer']:
                if sql.dbName not in config.test_databases:
                    new_buffer.append(sql)
            buffer_state['buffer'] = new_buffer
        if sample_size is not None:
            # random_indices = np.random.choice(len(buffer_state['buffer']), sample_size, replace=False)
            db_sql = {}
            for idx, sql in enumerate(buffer_state['buffer']):
                if sql.dbName not in db_sql:
                    db_sql[sql.dbName] = []
                db_sql[sql.dbName].append(idx)
            random_indices = []
            for db in db_sql:
                random_indices.extend(np.random.choice(db_sql[db], sample_size, replace=False))
            buffer_state['buffer'] = [buffer_state['buffer'][i] for i in random_indices]
        self.buffer = buffer_state['buffer']
        self.num_sql = len(self.buffer)
        self.latest_idx = list(range(self.num_sql))
        self.db_sql_dict = {}
        for sql in self.buffer:
            if sql.dbName not in self.db_sql_dict:
                self.db_sql_dict[sql.dbName] = 0
            self.db_sql_dict[sql.dbName] += 1
            sql.update_idx(sql.dbName + '_' + str(self.db_sql_dict[sql.dbName]))
        if 'buffer_state_vec' in buffer_state:
            self.buffer_state_vec = buffer_state['buffer_state_vec']
        return True
