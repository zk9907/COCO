import numpy as np
from LCQO.planhelper import PlanHelper
from LCQO.LCQOEnv import LCQOEnvCollect
from util.SQLBuffer import SQL, Plan, Node
import ray
class SQLBuilder:
    def __init__(self, plan_helper: PlanHelper):
        self.plan_helper = plan_helper
        self.collect_env = LCQOEnvCollect(plan_helper)

    def test_build_sql(self, query: str, dbName: str, only_sql_statement = False, sql_infos = None):
        try:
            if not self.plan_helper.test_sql_validity(query, dbName):   # the query can not be executed
                return None
            re_result, _ = self.plan_helper.test_sql_meaningful(query, dbName)  #
            if re_result == None:   #  timeout
                return None
            elif re_result == False: 
                featureDict, _, _, _ = self.plan_helper.get_feature(0, query, dbName)
                return (featureDict, sql_infos)
            # else: 
            exectime, istimeout, planJson= self.plan_helper.get_latency_analyze(0, query, dbName) # repeat twice to avoid cache effect
            if exectime > 50.0:   
                if only_sql_statement:
                    return (query, dbName)
                else:
                    sql = self.build_SQL(query, dbName, re_result, sql_infos)
                    return sql
            else:
                return None
        except Exception as e:
            print(query, dbName, e)
            return None
            
    def get_unique_code(self, query: str, dbName: str):
        initial_state, plan_state = self.collect_env.reset(options = {'query': query, 'dbName': dbName})
        root = Node(initial_state, plan_state)
        unique_code = {}
        visited_nodes = set()
        def expand_node(node: Node):
            if node.code in visited_nodes:
                return
            visited_nodes.add(node.code)
            valid_actions = np.where(node.env_state['action_mask'] == 1)[0].tolist()
            original_env_state = self.collect_env.save_env_state()
            if node.code not in unique_code:
                unique_code[node.code] = node.plan_state
            for action in valid_actions:
                self.collect_env.load_env_state(original_env_state)
                next_state, _, done, _, plan_state = self.collect_env.step(action)
                child = Node(next_state, plan_state, action, node)
                child.code = self.collect_env.current_code
                node.children[action] = child
                # if not done:
                expand_node(child)

            self.collect_env.load_env_state(original_env_state)
        
        expand_node(root)
        return unique_code
    @classmethod
    def hint2str(cls, hint,name2str):
        re_str = []
        for k in hint:
            if k != 'structure':
                for i in range(len(hint[k])):
                    if hint[k][i] not in name2str[k]:
                        name2str[k][hint[k][i]] = len(name2str[k])
                    re_str.append(str(name2str[k][hint[k][i]]))
            else:
                for i in range(len(hint[k])):
                    re_str.append(str(hint[k][i]))
        return '_'.join(re_str)
    
    def build_SQL(self, query: str, dbName: str, has_result, sql_infos):
        unique_code = self.get_unique_code(query, dbName)
        name2str = {'join order':{},'join operator':{},'scan operator':{},'structure':{}}
        plan_str = SQLBuilder.hint2str(unique_code[0]['hint_dict'], name2str)
        plans = [Plan(0, plan_str)]
        baseline_latency, istimeout, planJson = self.plan_helper.get_latency_analyze(0, query, dbName)
        plans[0].update(planJson, baseline_latency, istimeout)
        executed_plans = [plans[0]]  
        for hint_code, plan_state in unique_code.items():
            if hint_code == 0:
                continue
            plan_str = SQLBuilder.hint2str(plan_state['hint_dict'], name2str)
            plan = Plan(hint_code, plan_str)
            plans.append(plan)
            if plan not in executed_plans:
                executed_plans.append(plan)
        for plan in executed_plans:
            if plan.hint_code == 0:
                latency, istimeout, planJson = plan.latency, plan.istimeout, plan.plan_json
                # feature_dict, hint_dict = plan.feature_dict, plan.hint_dict
            else:
                latency, istimeout, planJson = self.plan_helper.get_latency_analyze(plan.hint_code, query, dbName)
                # feature_dict, hint_dict, _ = self.plan_helper.get_feature_from_planJson(planJson, dbName)
            for p in plans:
                if p == plan:
                    # p.update(planJson, latency, istimeout, feature_dict, hint_dict)
                    p.update(planJson, latency, istimeout)
        sql = SQL(dbName, query, plans, has_result, sql_infos)
        return sql

@ray.remote
class SQLBuilderActor:
    def __init__(self, config):
        self.plan_helper = PlanHelper(config, build_pghelper=True)
        self.builder = SQLBuilder(self.plan_helper)

    def test_build_sql(self, dbName, sql_statements, query_id = None, sql_infos = None):
        sql = self.builder.test_build_sql(sql_statements, dbName, sql_infos = sql_infos)
        if sql and query_id:
            sql.update_idx(query_id)
        return sql
    
    def test_sql_meaningful(self, query, dbName, timeout = 3e5):
        return self.plan_helper.test_sql_meaningful(query, dbName, timeout = timeout)
    
    def build(self, dbName, sql_statements,query_id = None, sql_infos = None):
        sql = self.builder.build_SQL(sql_statements, dbName, None, sql_infos)
        if sql and query_id:
            sql.update_idx(query_id)
        return sql
    def get_feature(self,hint_code, query: str, dbName: str):
        return self.plan_helper.get_feature(hint_code, query, dbName)
    def shutdown(self):
        if hasattr(self.plan_helper, '_exit'):
            self.plan_helper._exit()
        return True