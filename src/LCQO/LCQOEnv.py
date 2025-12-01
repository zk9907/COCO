import numpy as np
from LCQO.planhelper import PlanHelper
from LCQO.Constant import *
from util.SQLBuffer import SQL,Plan

def get_step_reward(curr_latency, prev_latency): # curr_latecncy: UnKnown_latency; prev_latency: Known_latency
    # return 0.0
    if abs(curr_latency - prev_latency) / max(curr_latency, prev_latency) < 0.05:
        return 0.0
    else:
        reward = (prev_latency  - curr_latency) / max(prev_latency, curr_latency)
    return reward

class LCQOEnvTest:
    def __init__(self, env_config):
        self.action_space_size = len(HINT2POS)
        self.config = env_config
        self.planHelper = PlanHelper(self.config, build_pghelper=False)

    def reset(self, options=None):
        self.current_code = 0
        self.current_hintsno = set()
        self.action_code = np.zeros(self.action_space_size)
        self.sql :SQL = options['sql']
        plan: Plan = self.sql.get_plan_by_hint_code(self.current_code)
        self.base_latency = plan.latency
        obs, hint_dict, _ = self.planHelper.get_feature_from_planJson(plan.plan_json, self.sql.dbName)
        self.action_mask = self.get_action_mask(hint_dict)
        # self.action_mask[-1] = 1
        obs['action_mask'] = self.action_mask.copy()
        obs['action_code'] = self.action_code.copy()
        return obs, {'Planning Time':plan.plan_json['Planning Time']}
    
    def step(self, action):
        self.action_code[action] = 1
        self.current_code += pow(2, action)
        self.current_hintsno.add(action)
        plan: Plan = self.sql.get_plan_by_hint_code(self.current_code)
        obs, hint_dict, _  = self.planHelper.get_feature_from_planJson(plan.plan_json, self.sql.dbName)
        # obs, hint_dict = copy.deepcopy(plan.feature_dict), plan.hint_dict
        self.action_mask = self.get_action_mask(hint_dict)
        terminated = False
        obs['action_mask'] = self.action_mask.copy()
        obs['action_code'] = self.action_code.copy()
        reward = plan.latency
        return obs, reward, terminated, False, {'Planning Time':plan.plan_json['Planning Time']}
    
    def get_action_mask(self, hint_dict):
        join_type = list(set(hint_dict['join operator']))
        scan_type = list(set(hint_dict['scan operator']))
        action_mask = np.zeros(self.action_space_size)
        # action_mask[-1] = 1
        for k in join_type + scan_type:
            if k in HINT2POS and HINT2POS[k] not in self.current_hintsno:
                action_mask[HINT2POS[k]] = 1
        return action_mask
    
    def save_env_state(self):
        return {
            'current_code': self.current_code,
            'current_hintsno': self.current_hintsno.copy(),
            'action_code': self.action_code.copy(),
            'action_mask': self.action_mask.copy()
        }

    def load_env_state(self, state):
        self.current_code = state['current_code']
        self.current_hintsno = state['current_hintsno'].copy()
        self.action_code = state['action_code'].copy()
        self.action_mask = state['action_mask'].copy()

class LCQOEnvCollect:
    def __init__(self, planHelper : PlanHelper):
        self.action_space_size = len(HINT2POS)
        self.planHelper = planHelper

    def reset(self, options=None):
        self.current_code = 0
        self.current_hintsno = set()
        self.action_code = np.zeros(self.action_space_size)
        self.query, self.dbName = options['query'], options['dbName']
        obs, hint_dict, _, plan_json = self.planHelper.get_feature(self.current_code, self.query, self.dbName)
        self.action_mask = self.get_action_mask(hint_dict)
        # self.action_mask[-1] = 1
        obs['action_mask'] = self.action_mask.copy()
        obs['action_code'] = self.action_code.copy()
        return obs, {'hint_dict': hint_dict, 'plan_json': plan_json}
    
    def step(self, action):
        self.action_code[action] = 1
        self.current_code += pow(2, action)
        self.current_hintsno.add(action)
        obs, hint_dict, _, plan_json = self.planHelper.get_feature(self.current_code, self.query, self.dbName)
        self.action_mask = self.get_action_mask(hint_dict)
        obs['action_mask'] = self.action_mask.copy()
        obs['action_code'] = self.action_code.copy()
        if np.all(self.action_mask == 0):
            terminated = True
        else:
            terminated = False
        return obs, 0, terminated, False, {'hint_dict': hint_dict, 'plan_json': plan_json}
    
    def get_action_mask(self, hint_dict):
        join_type = list(set(hint_dict['join operator']))
        scan_type = list(set(hint_dict['scan operator']))
        action_mask = np.zeros(self.action_space_size)
        # action_mask[-1] = 1
        for k in join_type + scan_type:
            if k in HINT2POS and HINT2POS[k] not in self.current_hintsno:
                action_mask[HINT2POS[k]] = 1
        return action_mask
    
    def save_env_state(self):
        return {
            'current_code': self.current_code,
            'current_hintsno': self.current_hintsno.copy(),
            'action_code': self.action_code.copy()
        }

    def load_env_state(self, state):
        self.current_code = state['current_code']
        self.current_hintsno = state['current_hintsno'].copy()
        self.action_code = state['action_code'].copy()

class LCQOEnvExp:
    def __init__(self, lqo_config):
        self.action_space_size = len(HINT2POS)
        self.planHelper = PlanHelper(lqo_config, build_pghelper=False)
    def reset(self, options=None):
        self.current_code = 0
        self.current_hintsno = set()
        self.action_code = np.zeros(self.action_space_size)
        self.sql: SQL = options['sql']
        plan: Plan = self.sql.get_plan_by_hint_code(self.current_code)
        
        obs, hint_dict, _ = self.planHelper.get_feature_from_planJson(plan.plan_json, self.sql.dbName)
        # obs, hint_dict = copy.deepcopy(plan.feature_dict), plan.hint_dict
        self.base_latency = plan.latency
        self.min_latency = self.sql.min_latency
        self.prev_latency = self.base_latency 
        self.action_mask = self.get_action_mask(hint_dict)
        # self.action_mask[-1] = 1
        obs['action_mask'] = self.action_mask.copy()
        obs['action_code'] = self.action_code.copy()
        return obs, self.base_latency 
    
    def step(self, action):
        self.action_code[action] = 1
        self.current_code += pow(2, action)
        self.current_hintsno.add(action)
        plan: Plan = self.sql.get_plan_by_hint_code(self.current_code)
        obs, hint_dict, _  = self.planHelper.get_feature_from_planJson(plan.plan_json, self.sql.dbName)
        # obs, hint_dict = copy.deepcopy(plan.feature_dict), plan.hint_dict
        self.action_mask = self.get_action_mask(hint_dict)
        obs['action_mask'] = self.action_mask.copy()
        obs['action_code'] = self.action_code.copy()
        reward = get_step_reward(plan.latency, self.prev_latency)
        self.prev_latency = plan.latency
        if np.all(self.action_mask == 0):
            terminated = True
        else:
            terminated = False
        return obs, reward, terminated, False, plan.latency
    
    def get_action_mask(self, hint_dict):
        join_type = list(set(hint_dict['join operator']))
        scan_type = list(set(hint_dict['scan operator']))
        action_mask = np.zeros(self.action_space_size)
        # action_mask[-1] = 1
        for k in join_type + scan_type:
            if k in HINT2POS and HINT2POS[k] not in self.current_hintsno:
                action_mask[HINT2POS[k]] = 1
        return action_mask
    
    def save_env_state(self):
        return {
            'current_code': self.current_code,
            'current_hintsno': self.current_hintsno.copy(),
            'action_code': self.action_code.copy(),
            'prev_latency': self.prev_latency
        }

    def load_env_state(self, state):
        self.current_code = state['current_code']
        self.current_hintsno = state['current_hintsno'].copy()
        self.action_code = state['action_code'].copy()
        self.prev_latency = state['prev_latency']

        