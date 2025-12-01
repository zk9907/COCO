import random, os, copy
import time, math, pickle
from config.LCQOConfig import Config as LCQOConfig
from config.SQLGenConfig import Config as SQLGenConfig
from config.GenConfig import GenConfig
from LCQO.LCQOAgent import LCQOAgent, QDataset
from SQLGen.SQLGenAgent import SQLGenAgent
from SQLGen.ValueModelAgent import ValueAgent
from model.TailNet import *
import numpy as np
from LCQO.planhelper import PlanHelper
from util.SQLBuffer import SQLBuffer, SQL
from util.SQLBuilder import SQLBuilderActor
from util.ActorManager import ActorManager, TaskType
from torch.utils.tensorboard.writer import SummaryWriter
import logging
import ray
import sys

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)
# File handler
file_handler = logging.FileHandler('./train.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class Trainer:
    def __init__(self, seed = 1408):
        set_seed(seed)
        self.seed = seed
        self.lcqo_config = LCQOConfig()
        self.sqlgen_config = SQLGenConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        os.makedirs('./runs', exist_ok=True)
        self.run_dir = f'./runs/training_run_{int(time.time())}'
        self.test_databases = GenConfig.test_databases
        self.train_databases = [db for db in GenConfig.databases if db not in GenConfig.test_databases]
        self.writer = SummaryWriter(self.run_dir)
        self.db_writers = {db: SummaryWriter(os.path.join(self.run_dir, db)) for db in self.train_databases}
        self.planhelper = PlanHelper(self.lcqo_config, build_pghelper = True)

        self.benefit_model = BenefitNetwork().to(self.lcqo_config.device)  #  Benefit Model
        self.value_model = ValueNetwork(self.benefit_model.planStateEncoder).to(self.lcqo_config.device)    #  Reward Model or SQL Value Estimation Network
        self.generation_model = GenerationNetwork().to(self.lcqo_config.device)  #  SQL Query Generation Model
        self.benefit_initial_state = copy.deepcopy(self.benefit_model.state_dict())  # fix the initial state
        self.value_out_head_initial_state = copy.deepcopy(self.value_model.out_head.state_dict())

        self.builder_actors = []
        for dbConfig in GenConfig.remote_db_config_list:
            lcqo_config = LCQOConfig()
            lcqo_config.db_config = dbConfig
            self.builder_actors.append(SQLBuilderActor.remote(lcqo_config))
    
        self.actor_manager = ActorManager(self.builder_actors)
        
        self.sqlgen_agent = SQLGenAgent(self.generation_model, self.value_model, self.benefit_model, self.actor_manager, **self.sqlgen_config.train_params) 
        self.lcqo_agent = LCQOAgent(self.lcqo_config, self.benefit_model)  
        self.value_agent = ValueAgent(self.lcqo_config.device, self.value_model, self.lcqo_agent)

        self.test_sql, self.test_sql_dataset, self.test_info = self.load_public(self.lcqo_config.test_sql_path)
        self.validation_sql, self.validation_dataset, self.validation_info = self.load_synthetic(self.lcqo_config.random_source_workloads_path)
        self.sql_buffer = SQLBuffer(self.lcqo_config.hidden_dim)

        self.training_samples = {}
        self.false_samples = []
        self.iteration_count = 0
        self.generated_sql_counts = {db: {} for db in self.train_databases}
        self.executed_sql_counts = {db: 0 for db in self.train_databases}

    def initialize_from_random_workload(self, path):
        if os.path.exists(path):
            self.sql_buffer.load_state(path, sample_size = 10)
            for sql in self.sql_buffer.buffer:
                sql:SQL
                self.training_samples[sql.id] = self.lcqo_agent._collect_experiences_for_query(sql)
            self.iterate_LCQO_training()
            self.iterate_Value_training()
            self.iterate_SQLGen_training()
        else:
            print('Not Found {}. Initialize randomly'.format(path))

    def iterate_one_loop(self, num_generated_sql: int):
        self.iteration_count += 1
        iteration_start_time = time.time()
        self.logger.info(f"Starting iteration {self.iteration_count}")
        sql_gen_time = self.generate_N_sql(num_generated_sql)
        self.sql_buffer.save_state(self.sqlgen_config.checkpoint_dir)
        lcqo_update_time = self.iterate_LCQO_training()
        value_update_time = self.iterate_Value_training()
        sqlgen_update_time = self.iterate_SQLGen_training()
        iteration_time = time.time() - iteration_start_time
        self.writer.add_scalars('Time/Overall', {
            'SQL_Execution': sql_gen_time,
            'SQLGen_Update': sqlgen_update_time,
            'Value_Update': value_update_time,
            'LCQO_update': lcqo_update_time,
            'Total': iteration_time
        }, self.iteration_count)

    def iterate_LCQO_training(self):
        time_start = time.time()
        all_experiences = []
        for experience in self.training_samples.values():
            all_experiences.extend(experience.values())
        dataset = QDataset(all_experiences)
        if hasattr(self, 'benefit_initial_state') and self.benefit_initial_state is not None:
            self.benefit_model.load_state_dict(self.benefit_initial_state)
        public_result, synthetic_result = self.lcqo_agent.update(dataset,(self.test_sql, self.test_sql_dataset), (self.validation_sql, self.validation_dataset), num_epochs = 50, batch_size = 1024)
        self.log_validate_result(synthetic_result)
        self.log_test_result(public_result)
        self.lcqo_agent.save_model()
        time_end = time.time()
        return time_end - time_start

    def iterate_Value_training(self):
        time_start = time.time()
        default_plan_features = {}
        if hasattr(self, 'value_out_head_initial_state') and self.value_out_head_initial_state is not None:
            self.value_model.out_head.load_state_dict(self.value_out_head_initial_state)
        for sql_id, experience in self.training_samples.items():
            default_plan_features[sql_id] = experience[0][0]
        self.value_agent.update(self.training_samples, default_plan_features, self.false_samples)
        time_end = time.time()
        return time_end - time_start

    def iterate_SQLGen_training(self):
        time_start = time.time()
        stats = self.sqlgen_agent.rollout_policy(self.train_databases, self.writer, num_iterations = min(max(len(self.sql_buffer.buffer) // 50, 1), self.sqlgen_config.max_update_times_per_iter))
        time_end = time.time()
        return time_end - time_start
    
    def generate_N_sql(self, num_generated_sql: int, similarity_threshold=0.90):
        if self.iteration_count < 5:
            similarity_threshold = 0.99
        time_start = time.time()
        one_iter_count = 0
        collected_rewards = [] 
        backlog = self.sqlgen_agent.choose_for_execution(return_vectors=True)
        successful_vectors = []
        pending_vectors = {}  
        deferred_queries = {} 
        idle_wait_rounds = 0
        self.logger.info(f"[generate_N_sql] iter={self.iteration_count} start, backlog_size={len(backlog)}, target={num_generated_sql}, similarity_threshold={similarity_threshold}")
            
        # 检查并处理已完成的main任务
        # self.actor_manager.check_and_cache_completed_tasks()
        available = self.actor_manager.get_available_actors(TaskType.MAIN)
        self.logger.info(f"[generate_N_sql] available_main_actors={len(available)}, busy_main={self.actor_manager.get_busy_count(TaskType.MAIN)} before dispatch")
        # 处理已缓存的main结果
        for future in list(self.actor_manager.main_results.keys()):
            result, task_data = self.actor_manager.get_main_result(future)
            dbName = task_data['dbName']
            sql_statement = task_data['sql_statement']
            sql_reward = task_data['sql_reward']
            sql_infos = task_data['sql_infos']
            query_vector = task_data.get('query_vector')
            
            reward = self._try_collect_sql_result(result, sql_reward, sql_infos)
            if reward is not None and reward > 0.0:
                collected_rewards.append(reward)
                if query_vector is not None:
                    successful_vectors.append(query_vector)
            else:
                if reward is not None:  # empty result
                    self.logger.info(f"Empty Result SQL for {dbName} with sql_statement: {sql_statement}")
                else:
                    self.logger.info(f"Timeout SQL for {dbName} with sql_statement: {sql_statement}")
                if future in deferred_queries:
                    deferred_count = len(deferred_queries[future])
                    backlog[:0] = deferred_queries[future]
                    self.logger.info(f"Re-adding {deferred_count} deferred queries to front of backlog (failed pending)")
            if future in deferred_queries:
                del deferred_queries[future]
        
        # 获取当前pending任务的向量信息（用于相似度检查）
        for future in self.actor_manager.get_all_pending_futures(TaskType.MAIN):
            task_info = self.actor_manager.get_task_info(future)
            if task_info:
                _, task_data = task_info
                query_vector = task_data.get('query_vector')
                if query_vector is not None:
                    pending_vectors[future] = query_vector
        def is_similar_to_successful(query_vector):
            similarity = 0.0
            if len(successful_vectors) == 0:
                return False, similarity
            query_norm = np.linalg.norm(query_vector)
            if query_norm < 1e-8:
                return False, similarity
            normalized_query = query_vector / query_norm
                
            for success_vec in successful_vectors:
                success_norm = np.linalg.norm(success_vec)
                if success_norm < 1e-8:
                    continue
                normalized_success = success_vec / success_norm
                similarity = np.dot(normalized_query, normalized_success)
                if similarity > similarity_threshold:
                    return True, similarity
            return False, similarity
            
        # 辅助函数：检查查询是否与pending查询相似
        def is_similar_to_pending(query_vector):
            similarity = 0.0
            if len(pending_vectors) == 0:
                return False, similarity, None
            query_norm = np.linalg.norm(query_vector)
            if query_norm < 1e-8:
                return False, similarity, None
            normalized_query = query_vector / query_norm
                
            for pending_fut, pending_vec in pending_vectors.items():
                pending_norm = np.linalg.norm(pending_vec)
                if pending_norm < 1e-8:
                    continue
                normalized_pending = pending_vec / pending_norm
                similarity = np.dot(normalized_query, normalized_pending)
                if similarity > similarity_threshold:
                    return True, similarity, pending_fut
            return False, similarity, None
            
        for actor in available:
            while backlog:
                item = backlog.pop(0)
                dbName, sql_statement, sql_reward, sql_infos, candiate_idx, query_vector = item
                if query_vector is not None:
                    is_similar, similarity = is_similar_to_successful(query_vector)
                    if is_similar:
                        self.logger.info(f"Skipped similar query for {dbName} (similarity: {similarity:.3f} with successful)")
                        continue
                        
                    is_sim_pending, sim_score, similar_fut = is_similar_to_pending(query_vector)
                    if is_sim_pending and similar_fut is not None:
                        if similar_fut not in deferred_queries:
                            deferred_queries[similar_fut] = []
                        deferred_queries[similar_fut].append(item)
                        self.logger.info(f"Deferred query for {dbName} (similarity: {sim_score:.3f} with pending)")
                        continue
                    
                if sql_infos['joins'] not in self.generated_sql_counts[dbName]:
                    self.generated_sql_counts[dbName][sql_infos['joins']] = 0
                self.generated_sql_counts[dbName][sql_infos['joins']] += 1
                self.sqlgen_agent.candidate_pool.remove(candiate_idx)
                new_fut = actor.test_build_sql.remote(dbName, sql_statement, sql_infos=sql_infos)
                task_data = {
                    'dbName': dbName,
                    'sql_statement': sql_statement,
                    'sql_reward': sql_reward,
                    'sql_infos': sql_infos,
                    'query_vector': query_vector
                }
                self.actor_manager.submit_main_task(actor, new_fut, task_data)
                if query_vector is not None:
                    pending_vectors[new_fut] = query_vector
                break

        while (backlog or self.actor_manager.get_busy_count(TaskType.MAIN) > (len(self.builder_actors) - 1)) and one_iter_count < num_generated_sql:
            busy_main = self.actor_manager.get_busy_count(TaskType.MAIN)
            if busy_main == 0:
                if backlog:
                    self.logger.warning(f"[generate_N_sql] busy_main=0 but backlog_size={len(backlog)}; pending={len(pending_vectors)}, deferred_keys={len(deferred_queries)}")
                break
            task_result = self.actor_manager.wait_any_task(TaskType.MAIN, timeout=1.0)
            if task_result is None:
                idle_wait_rounds += 1
                # if idle_wait_rounds % 10 == 0:
                    # self.logger.warning(f"[generate_N_sql] wait_any_task timeout x{idle_wait_rounds}, backlog={len(backlog)}, pending={len(pending_vectors)}, deferred_keys={len(deferred_queries)}, busy_main={busy_main}")
                continue
            idle_wait_rounds = 0
                
            fut, task_type, task_data, result, actor = task_result
            dbName = task_data['dbName']
            sql_statement = task_data['sql_statement']
            sql_reward = task_data['sql_reward']
            sql_infos = task_data['sql_infos']
            query_vector = task_data.get('query_vector')

            if fut in pending_vectors:
                del pending_vectors[fut]
                
            reward = self._try_collect_sql_result(result, sql_reward, sql_infos)
            if reward is not None and reward > 0.0:
                collected_rewards.append(reward)
                one_iter_count += 1
                if query_vector is not None:
                    successful_vectors.append(query_vector)
            else:
                if reward is not None: # empty result
                    self.logger.info(f"Empty Result SQL for {dbName} with sql_statement: {sql_statement}")
                else:
                    self.logger.info(f"Timeout SQL for {dbName} with sql_statement: {sql_statement}")
                if fut in deferred_queries:
                    deferred_count = len(deferred_queries[fut])
                    backlog[:0] = deferred_queries[fut]
                    self.logger.info(f"Re-adding {deferred_count} deferred queries to front of backlog (failed pending)")
                
            if fut in deferred_queries:
                del deferred_queries[fut]
                
            while backlog:
                item = backlog.pop(0)
                dbName, sql_statement, sql_reward, sql_infos, candiate_idx, query_vector = item
                if query_vector is not None:
                    is_similar, similarity = is_similar_to_successful(query_vector)
                    if is_similar:
                        self.logger.info(f"Skipped similar query for {dbName} (similarity: {similarity:.3f} with successful)")
                        continue
                    is_sim_pending, sim_score, similar_fut = is_similar_to_pending(query_vector)
                    if is_sim_pending and similar_fut is not None:
                        if similar_fut not in deferred_queries:
                            deferred_queries[similar_fut] = []
                        deferred_queries[similar_fut].append(item)
                        self.logger.info(f"Deferred query for {dbName} (similarity: {sim_score:.3f} with pending)")
                        continue
                    
                if sql_infos['joins'] not in self.generated_sql_counts[dbName]:
                    self.generated_sql_counts[dbName][sql_infos['joins']] = 0
                self.generated_sql_counts[dbName][sql_infos['joins']] += 1
                self.sqlgen_agent.candidate_pool.remove(candiate_idx)
                new_fut = actor.test_build_sql.remote(dbName, sql_statement, sql_infos=sql_infos)
            
                task_data = {
                    'dbName': dbName,
                    'sql_statement': sql_statement,
                    'sql_reward': sql_reward,
                    'sql_infos': sql_infos,
                    'query_vector': query_vector
                }
                self.actor_manager.submit_main_task(actor, new_fut, task_data)
                if query_vector is not None:
                    pending_vectors[new_fut] = query_vector
                break

        self.logger.info(f"[generate_N_sql] finished loop: generated={one_iter_count}, backlog_remaining={len(backlog)}, pending={len(pending_vectors)}, deferred_keys={len(deferred_queries)}")
        total_gene_num = sum(sum(self.generated_sql_counts[dbName].values()) for dbName in self.generated_sql_counts)
        generated_ratios = {}
        complexity_values = {}
        for dbName in self.generated_sql_counts:
            db_count = sum(self.generated_sql_counts[dbName].values())
            generated_ratios[dbName] = (db_count / total_gene_num) if total_gene_num > 0 else 0
            complexity_values[dbName] = (sum([joins * num for joins, num in self.generated_sql_counts[dbName].items()]) / db_count) if db_count > 0 else 0
        # Log per-DB to individual runs to ensure consistent colors/toggles across charts
        for dbName, ratio in generated_ratios.items():
            if dbName in self.db_writers:
                self.db_writers[dbName].add_scalar('Query Counts/Generated', ratio, self.iteration_count)
        overall_complexity = 0
        for dbName, comp in complexity_values.items():
            overall_complexity += generated_ratios[dbName] * comp
            if dbName in self.db_writers:
                self.db_writers[dbName].add_scalar('Query Counts/Complexity', comp, self.iteration_count)
        self.writer.add_scalar('Query Counts/Overall_Complexity', overall_complexity, self.iteration_count)

        # Record SQLGen average reward for this batch
        if collected_rewards:
            avg_reward = np.mean(collected_rewards)
            self.writer.add_scalar('SQLGen/Average_Reward', avg_reward, self.iteration_count)
            # self.writer.add_scalar('SQLGen/Reward_Count', len(collected_rewards), self.iteration_count)
            self.logger.info(f"SQLGen batch average reward: {avg_reward:.4f}")

        time_end = time.time()

        executed_ratios = {dbName: (self.executed_sql_counts[dbName] / sum(self.executed_sql_counts.values())) if sum(self.executed_sql_counts.values()) > 0 else 0 for dbName in self.executed_sql_counts}
        for dbName, ratio in executed_ratios.items():
            if dbName in self.db_writers:
                self.db_writers[dbName].add_scalar('Query Counts/Executed', ratio, self.iteration_count)
        return time_end - time_start

    def _try_collect_sql_result(self, sql_obj, sql_reward, sql_infos):
        if sql_obj is not None:
            if isinstance(sql_obj, SQL):
                try:
                    latency = self.lcqo_agent._evaluate_one_sql(sql_obj)
                    self.logger.info(f"dbName: {sql_obj.dbName} sql_statement: {sql_obj.sql_statement}")
                    self.logger.info(f"min_latency: {sql_obj.min_latency:.2f}, q_min_latency: {sql_obj.q_min_latency:.2f}, max_latency: {sql_obj.max_latency:.2f}, base_latency: {sql_obj.base_latency:.2f},lcqo_latency: {latency:.2f}, sqlgen_reward: {sql_reward:.2f}, has_result: {str(sql_obj.has_result)}, {sql_infos}")
                    sql_id = self.sql_buffer.push(sql_obj)
                    self.executed_sql_counts[sql_obj.dbName] += 1
                    experiences = self.lcqo_agent._collect_experiences_for_query(sql_obj)
                    self.training_samples[sql_id] = experiences
                    return sql_reward
                except Exception as exc:
                    self.logger.exception(f"Failed to collect result for db={sql_obj.dbName}: {exc}")
                    return None
            elif isinstance(sql_obj, tuple):
                # self.false_samples.append(sql_obj)
                # if len(self.false_samples) > 2000:
                #     self.false_samples.pop(0)
                return 0.0
            else:
                return None
        return None
    
    def load_public(self, test_sql_path):
        test_sqls = []
        test_dataset = {}
        test_info = {}
        for test_sql_path in test_sql_path:
            if os.path.exists(test_sql_path):   
                test_sql = pickle.load(open(test_sql_path, 'rb'))
                test_sqls.extend(test_sql)
        for sql in test_sqls:
            sql:SQL
            if not hasattr(sql, 'q_min_latency') or sql.q_min_latency is None:
                sql.q_min_latency = self.lcqo_agent._get_q_min_latency(sql)
            db_name = sql.dbName
            if db_name not in test_dataset:
                test_dataset[db_name] = []
                test_info[db_name] = {'min_latency': 0, 'base_latency': 0, 'q_min_latency': 0}
            test_dataset[db_name].extend(list(self.lcqo_agent._collect_experiences_for_query(sql).values()))    
            test_info[db_name]['min_latency'] += sql.min_latency
            test_info[db_name]['base_latency'] += sql.base_latency
            test_info[db_name]['q_min_latency'] += sql.q_min_latency
        return test_sqls, test_dataset, test_info
    
    def load_synthetic(self, synthetic_sql_path):
        if not os.path.exists(synthetic_sql_path):
            return None, None, None
        sqlbuffer = SQLBuffer()
        sqlbuffer.load_state(synthetic_sql_path)
        validation_dataset = {}
        validation_info = {}
        for sql in sqlbuffer.buffer:
            sql:SQL
            if not hasattr(sql, 'q_min_latency') or sql.q_min_latency is None:
                sql.q_min_latency = self.lcqo_agent._get_q_min_latency(sql)
            db_name = sql.dbName
            if db_name not in validation_dataset:
                validation_dataset[db_name] = []
                validation_info[db_name] = {'min_latency': 0, 'base_latency': 0, 'q_min_latency': 0}
            validation_dataset[db_name].extend(list(self.lcqo_agent._collect_experiences_for_query(sql).values()))
            validation_info[db_name]['min_latency'] += sql.min_latency
            validation_info[db_name]['base_latency'] += sql.base_latency
            validation_info[db_name]['q_min_latency'] += sql.q_min_latency
        return sqlbuffer.buffer, validation_dataset, validation_info
    
    def log_validate_result(self,result):
        if result is None:
            return
        total_lcqo_latency = 0
        for dbName in result[1]:
            total_lcqo_latency += result[1][dbName][1]
        total_q_min_latency = sum(self.validation_info[dbName]['q_min_latency'] for dbName in self.validation_info)
        total_base_latency = sum(self.validation_info[dbName]['base_latency'] for dbName in self.validation_info)
        total_min_latency = sum(self.validation_info[dbName]['min_latency'] for dbName in self.validation_info)
        self.writer.add_scalar(f'Validate/Ratio', (total_lcqo_latency - total_q_min_latency) / (total_base_latency - total_q_min_latency), self.iteration_count)
        self.writer.add_scalars(f'Validate/Latency', {
            'Min_Latency': total_min_latency / 1000,
            'Q_Min_Latency': total_q_min_latency / 1000,
            'Base_Latency': total_base_latency / 1000,
            'LCQO_Latency': total_lcqo_latency / 1000
        }, self.iteration_count)
        self.writer.add_scalar(f'Validate/Loss', result[0], self.iteration_count)
        self.logger.info(f"total_min_latency={total_min_latency / 1000:.2f}, total_q_min_latency={total_q_min_latency / 1000:.2f}, total_base_latency={total_base_latency / 1000:.2f}, total_lcqo_latency={total_lcqo_latency / 1000:.2f}")
        
    def log_test_result(self, result):
        # result:(avg_loss, {dbName: (avg_loss,lcqo_latency)})
        total_lcqo_latency = 0
        for dbName in result[1]:
            self.writer.add_scalar(f'Test/Loss/{dbName}', result[1][dbName][0], self.iteration_count)
            total_lcqo_latency += result[1][dbName][1]
            self.writer.add_scalars(f'Test/Latency/{dbName}', {
                'Min_Latency': self.test_info[dbName]['min_latency'] / 1000,
                'Q_Min_Latency': self.test_info[dbName]['q_min_latency'] / 1000,
                'Base_Latency': self.test_info[dbName]['base_latency'] / 1000,
                'LCQO_Latency': result[1][dbName][1] / 1000
            }, self.iteration_count)
            self.logger.info(f"dbName={dbName}, Min_Latency={self.test_info[dbName]['min_latency'] / 1000:.2f}, Q_Min_Latency={self.test_info[dbName]['q_min_latency'] / 1000:.2f}, Base_Latency={self.test_info[dbName]['base_latency'] / 1000:.2f}, LCQO_Latency={result[1][dbName][1] / 1000:.2f}")
        self.writer.add_scalars(f'Test/Latency/All', {
            'Min_Latency': sum(self.test_info[dbName]['min_latency'] for dbName in self.test_info) / 1000,
            'Q_Min_Latency': sum(self.test_info[dbName]['q_min_latency'] for dbName in self.test_info) / 1000,
            'Base_Latency': sum(self.test_info[dbName]['base_latency'] for dbName in self.test_info) / 1000,
            'LCQO_Latency': total_lcqo_latency / 1000
        }, self.iteration_count)
        self.writer.add_scalar(f'Test/Loss/All', result[0], self.iteration_count)
        self.logger.info(f"total_lcqo_latency={total_lcqo_latency / 1000:.2f}, total_base_latency={sum(self.test_info[dbName]['base_latency'] for dbName in self.test_info) / 1000:.2f}, total_min_latency={sum(self.test_info[dbName]['min_latency'] for dbName in self.test_info) / 1000:.2f}, total_q_min_latency={sum(self.test_info[dbName]['q_min_latency'] for dbName in self.test_info) / 1000:.2f}")
if __name__ == "__main__":
    trainer = Trainer(seed = 9999)
    trainer.initialize_from_random_workload('/home/zhongkai/pywork/SuperCOCO/TestWorkload/sql_buffer.pkl')
    while True:
        trainer.iterate_one_loop(25)