
import torch, random
import logging
import ray
from ray.exceptions import RayError
from model.TailNet import GenerationNetwork
from model.TailNet import ValueNetwork as RewardModel
from model.TailNet import BenefitNetwork
from SQLGen.SQLGenEnv import SQLGenEnv
from LCQO.planhelper import PlanHelper
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import OrderedDict
import time
from sklearn.cluster import KMeans
from util.ActorManager import TaskType
class PPODataset(Dataset):
    def __init__(self, states, actions, old_log_probs, action_masks, advantages):
        self.states = states
        self.actions = torch.LongTensor(actions)
        self.old_log_probs = torch.FloatTensor(old_log_probs)
        self.action_masks = action_masks
        self.advantages = torch.FloatTensor(advantages)
    
        self.state_tensors = []
        self.mask_tensors = []
        for state, mask in zip(states, action_masks):
            state_tensor = {
                'node_vector': torch.FloatTensor(state['node_vector']),
                'attn_bias': torch.FloatTensor(state['attn_bias']),
                'action_type': torch.LongTensor(state['action_type'])
            }
            self.state_tensors.append(state_tensor)
            self.mask_tensors.append(torch.FloatTensor(mask))
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return (
            self.state_tensors[idx],
            self.mask_tensors[idx],
            self.actions[idx],
            self.old_log_probs[idx],
            self.advantages[idx]
        )
    
class SQLPool:
    def __init__(self, max_size):
        self.max_size = max_size
        self.pool = OrderedDict()  # {key: [db_name, sql_statement, feature, advantage, sql_infos]}
        self.next_key = 0 

    def get_all_features(self):
        return [item[2] for item in self.pool.values()]
    
    def get_all_keys(self):
        return list(self.pool.keys())
    
    def update_advantage(self, advantages):
        for i, key in enumerate(self.pool.keys()):
            self.pool[key][3] = advantages[i]

    def add(self, sql):
        if len(self.pool) >= self.max_size:
            self.pool.popitem(last=False)
        self.pool[self.next_key] = sql
        self.next_key += 1
            
    def remove(self, key):
        if key in self.pool:
            del self.pool[key]

    def __len__(self):
        return len(self.pool)
    
    def __getitem__(self, key):
        return self.pool[key]
    
    def items(self):
        return self.pool.items()
    
    
    
class SQLGenAgent:
    def __init__(
        self, 
        policy_model, 
        reward_model,
        benefit_model,
        actor_manager,
        learning_rate=1e-4,
        group_size=8,  
        num_groups=8,  
        clip_epsilon=0.2,  
        entropy_coef=0.1,  
        max_grad_norm=0.5,  
        gamma=1.0,  
        update_epochs=15,  
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    ):
        self.policy_model:GenerationNetwork = policy_model
        self.reward_model:RewardModel = reward_model
        self.benefit_model:BenefitNetwork = benefit_model
        self.sqlgen_env = SQLGenEnv()
        self.device = device
        self.actor_manager = actor_manager

        self.candidate_pool = SQLPool(max_size = 2000)
        self.group_size = group_size
        self.num_groups = num_groups
        self.batch_size = group_size * num_groups
        
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.update_epochs = update_epochs
        
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        
        self.stats = {
            'policy_loss': [],
            'entropy': [],
            'rewards': [],
            'total_loss':[]
        }
        self.total_iteration_step = 0
        self.logger = logging.getLogger(__name__)
    
    def select_action(self, state, action_mask, deterministic = False):
        with torch.no_grad():
            obs_dict = {
                'node_vector': torch.FloatTensor(state['node_vector']).unsqueeze(0).to(self.device),
                'attn_bias': torch.FloatTensor(state['attn_bias']).unsqueeze(0).to(self.device),
                'action_type': torch.LongTensor(state['action_type']).unsqueeze(0).to(self.device),
            }
            action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
            action, probs = self.policy_model(
                obs_dict, 
                action_mask=action_mask_tensor,
                get_log_prob=True,
                deterministic=deterministic,
                add_noise = True
            )
            if deterministic:
                return action.cpu().item(), None
            else:
                return action.cpu().item(), probs.cpu().item()
        
    # def generation_for_execution(self, db_names):
    #     self.reward_model.eval()
    #     self.policy_model.eval()
    #     generation_sqls = []
    #     for db_name in db_names:
    #         for _ in range(self.group_size):
    #             trajectory = self.generate_one_sql(db_name, deterministic = False)
    #             generation_sqls.append((db_name,trajectory['sql'], trajectory['reward'], trajectory['sql_infos']))
        
    #     all_rewards = torch.tensor([sql[2] for sql in generation_sqls])
    #     advantages = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)

    #     filtered_generation_sqls = []
    #     for i, (db_name, sql, reward, sql_infos) in enumerate(generation_sqls):
    #         if advantages[i].item() > 0:
    #             filtered_generation_sqls.append((db_name, sql, reward, sql_infos))
        
    #     return filtered_generation_sqls

    def choose_for_execution(self, use_clustering=False, n_clusters=25, return_vectors=False):
        self.reward_model.eval()
        if len(self.candidate_pool) == 0:
            return []
        
        features = self.candidate_pool.get_all_features()

        batch_size = 256
        all_rewards = []
        all_vectors = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size]
                batch_tensors = {}
                for key in batch_features[0].keys():
                    batch_tensors[key] = torch.stack([
                        torch.FloatTensor(f[key]) for f in batch_features
                    ]).to(self.device)
                batch_vector = self.reward_model.only_vector(batch_tensors)
                batch_rewards = self.reward_model.only_value(batch_vector).cpu().numpy()
                # batch_rewards = self.reward_model(batch_tensors).cpu().numpy()
                all_rewards.extend(batch_rewards.flatten())
                all_vectors.extend(batch_vector.cpu().numpy())
        
        all_rewards = np.array(all_rewards)
        all_vectors = np.array(all_vectors)
        advantages = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)
        self.candidate_pool.update_advantage(advantages)
        
        positive_samples = []
        for idx, (pool_key, item) in enumerate(self.candidate_pool.items()):
            db_name, sql, _, _, sql_infos = item
            if advantages[idx] > 0:
                positive_samples.append({
                    'idx': idx,
                    'db_name': db_name,
                    'sql': sql,
                    'reward': all_rewards[idx],
                    'sql_infos': sql_infos,
                    'pool_key': pool_key,
                    'advantage': advantages[idx],
                    'vector': all_vectors[idx]
                })
        
        if not positive_samples:
            return []
        
        if use_clustering and len(positive_samples) > n_clusters:
            results = self._cluster_based_selection(positive_samples, n_clusters, return_vectors)
        else:
            positive_samples.sort(key=lambda x: x['advantage'], reverse=True)
            if return_vectors:
                results = [(s['db_name'], s['sql'], s['reward'], s['sql_infos'], s['pool_key'], s['vector']) 
                          for s in positive_samples]
            else:
                results = [(s['db_name'], s['sql'], s['reward'], s['sql_infos'], s['pool_key']) 
                          for s in positive_samples]
        
        return results
    
    def _cluster_based_selection(self, samples, n_clusters, return_vectors=False):
        vectors = np.array([s['vector'] for s in samples])
        
        actual_n_clusters = min(n_clusters, len(samples))
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(vectors)

        clusters = [[] for _ in range(actual_n_clusters)]
        for sample, label in zip(samples, cluster_labels):
            sample['cluster_label'] = label
            clusters[label].append(sample)
        
        for cluster in clusters:
            cluster.sort(key=lambda x: x['reward'], reverse=True)
        results = []
        max_cluster_size = max(len(cluster) for cluster in clusters)
        
        for round_idx in range(max_cluster_size):
            round_samples = []
            for cluster_idx in range(actual_n_clusters):
                if round_idx < len(clusters[cluster_idx]):
                    sample = clusters[cluster_idx][round_idx]
                    round_samples.append(sample)
            round_samples.sort(key=lambda x: x['reward'], reverse=True)
            for sample in round_samples:
                if return_vectors:
                    results.append((
                        sample['db_name'],
                        sample['sql'],
                        sample['reward'],
                        sample['sql_infos'],
                        sample['pool_key'],
                        sample['vector']
                    ))
                else:
                    results.append((
                        sample['db_name'],
                        sample['sql'],
                        sample['reward'],
                        sample['sql_infos'],
                        sample['pool_key']
                    ))
        
        return results
        
    def generate_one_sql(self, db_name, deterministic = False):
        state, _ = self.sqlgen_env.reset(options={'dbName': db_name})
        action_mask = state['action_mask']
        done = False
        states, actions, log_probs, action_masks = [], [], [], []
        while not done:
            states.append(state)
            action_masks.append(action_mask)
            action, log_prob = self.select_action(state, action_mask, deterministic)
            actions.append(action)
            log_probs.append(log_prob)
            next_state, _, done, _, _ = self.sqlgen_env.step(action)
            state, action_mask = next_state, next_state['action_mask']
        sql_statement, sql_infos = self.sqlgen_env.get_query()
        trajectory = {
                'states': states,
                'actions': actions,
                'log_probs': log_probs,
                'action_masks': action_masks,
                'reward': 0.0,
                'sql': sql_statement,
                'sql_infos': sql_infos
            }
        return trajectory
    
    def compute_sql_intrinsic_reward(self, sql_infos):
        return 0.0
        join_score = sql_infos['joins'] / self.sqlgen_env.max_no_joins
        filter_score = sql_infos['predicates'] / self.sqlgen_env.max_no_predicates
        sql_intric_score = 0.8 * join_score + 0.2 * filter_score
        return 0.1 * sql_intric_score

    def collect_trajectories(self, db_names):
        trajectories = []
        self.policy_model.train()
        selected_db_names = [random.choice(db_names) for _ in range(self.num_groups)]
        for db_name in selected_db_names:
            for _ in range(self.group_size):
                trajectory = self.generate_one_sql(db_name)
                trajectory['db_name'] = db_name
                trajectories.append(trajectory)
        
        # print(f"Generated {len(trajectories)} trajectories, starting test_sql_meaningful...")
        
        # ===== 第一阶段：并行执行test_sql_meaningful =====
        trajectory_futures = {}  # {future: trajectory_idx}
        pending_trajectories = list(range(len(trajectories)))  # 待处理的trajectory索引
        meaningful_results = {}  # {traj_idx: (re_result, _)}
        
        while pending_trajectories or trajectory_futures:
            available_actors = self.actor_manager.get_available_actors()
            for future in list(trajectory_futures.keys()):
                cached_result = self.actor_manager.get_sqlgen_result(future)
                if cached_result is not None:
                    result, task_data = cached_result
                    traj_idx = task_data['trajectory_idx']
                    meaningful_results[traj_idx] = result
                    del trajectory_futures[future]
            
            for actor in available_actors:
                if not pending_trajectories:
                    break
                
                traj_idx = pending_trajectories.pop(0)
                trajectory = trajectories[traj_idx]
                
                future = actor.test_sql_meaningful.remote(
                    trajectory['sql'],
                    trajectory['db_name'],
                    timeout = 1e4
                )
                
                task_data = {
                    'sql': trajectory['sql'],
                    'db_name': trajectory['db_name'],
                    'timeout': 1e4,
                    'trajectory_idx': traj_idx
                }
                
                success = self.actor_manager.submit_sqlgen_task(actor, future, task_data)
                if success:
                    trajectory_futures[future] = traj_idx
                else:
                    pending_trajectories.insert(0, traj_idx)
            
            if trajectory_futures:
                timeout = 1.0 if not pending_trajectories else 0.1
                task_result = self.actor_manager.wait_any_task(TaskType.SQLGEN, timeout=timeout)
                
                if task_result is not None:
                    future, task_type, task_data, result, actor = task_result
                    traj_idx = task_data['trajectory_idx']
                    meaningful_results[traj_idx] = result
                    
                    if future in trajectory_futures:
                        del trajectory_futures[future]
                    
                    # print(f"test_sql_meaningful completed for trajectory {traj_idx}: {len(meaningful_results)}/{len(trajectories)}")
        
        # 确保所有轨迹都已处理
        if len(meaningful_results) != len(trajectories):
            print(f"WARNING: Only {len(meaningful_results)}/{len(trajectories)} trajectories completed!")
            # 为未完成的轨迹设置默认结果
            for traj_idx in range(len(trajectories)):
                if traj_idx not in meaningful_results:
                    print(f"WARNING: Trajectory {traj_idx} missing result, setting as not meaningful")
                    meaningful_results[traj_idx] = (False, None)
        
        # print(f"All test_sql_meaningful completed. Starting feature computation...")
        
        # ===== 第二阶段：批量计算features和rewards =====
        meaningful_trajectories = []  # 需要计算feature的trajectory索引
        
        for traj_idx in range(len(trajectories)):
            trajectory = trajectories[traj_idx]
            re_result, _ = meaningful_results[traj_idx]
            
            if re_result is None or re_result is True:
                # SQL有意义，需要计算feature
                meaningful_trajectories.append(traj_idx)
            # else:
            #     # SQL无意义，只给内在奖励
                # trajectories[traj_idx]['reward'] = 0.0
        
        # print(f"Found {len(meaningful_trajectories)} meaningful SQLs, computing features...")
        
        # 批量计算features
        pending_features = list(range(len(trajectories))) #list(meaningful_trajectories) 
        feature_futures = {}  # {future: traj_idx}
        computed_features = {}  # {traj_idx: feature}
        
        while pending_features or feature_futures:
            # 获取空闲actors
            available_actors = self.actor_manager.get_available_actors()
            
            # 为空闲actors分配get_feature任务
            for actor in available_actors:
                if not pending_features:
                    break
                
                traj_idx = pending_features.pop(0)
                trajectory = trajectories[traj_idx]
                
                # 提交get_feature任务（这是一个快速操作，不通过ActorManager管理）
                future = actor.get_feature.remote(0, trajectory['sql'], trajectory['db_name'])
                feature_futures[future] = traj_idx
            
            # 等待任意feature计算完成
            if feature_futures:
                # 如果没有pending任务但有正在执行的任务，使用更长的超时
                timeout = 1.0 if not pending_features else 0.1
                done, _ = ray.wait(list(feature_futures.keys()), num_returns=1, timeout=timeout)
                
                if done:
                    future = done[0]
                    traj_idx = feature_futures.pop(future)
                    try:
                        feature, _, _, _ = ray.get(future)
                        computed_features[traj_idx] = feature
                    except RayError:
                        self.logger.exception(f"get_feature failed for trajectory {traj_idx}")
                        computed_features[traj_idx] = None

        # if len(computed_features) != len(meaningful_trajectories):
        #     print(f"WARNING: Only {len(computed_features)}/{len(meaningful_trajectories)} features computed!")
        
        batch_max_reward = 0.0
        for traj_idx in range(len(trajectories)):#meaningful_trajectories:
            trajectory = trajectories[traj_idx]
            feature = computed_features.get(traj_idx)
            if feature is None:
                self.logger.warning(f"Missing feature for trajectory {traj_idx}; assigning zero reward.")
                trajectories[traj_idx]['reward'] = 0.0
                continue
            
            # 使用reward model计算奖励
            with torch.no_grad():
                feature_tensor = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) for k, v in feature.items()}
                sql_reward = self.reward_model(feature_tensor).cpu().item()
            
            if traj_idx in meaningful_trajectories:
                self.candidate_pool.add([
                    trajectory['db_name'],
                    trajectory['sql'],
                    feature,
                    sql_reward,
                    trajectory['sql_infos']
                ])
            
            # 加上内在奖励
            # sql_reward += self.compute_sql_intrinsic_reward(trajectory['sql_infos'])
            trajectories[traj_idx]['reward'] = sql_reward
            batch_max_reward = max(batch_max_reward, sql_reward)
        # for traj_idx in range(len(trajectories)):
        #     trajectories[traj_idx]['reward'] += (self.compute_sql_intrinsic_reward(trajectories[traj_idx]['sql_infos']) * batch_max_reward)
        return trajectories
            
    def compute_group_advantages(self, trajectories):
        groups = [trajectories[i:i+self.group_size] 
                  for i in range(0, len(trajectories), self.group_size)]
        
        trajectories_with_advantages = []
        
        for group in groups:
            group_rewards = torch.tensor([traj['reward'] for traj in group])
            advantages = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)
            
            for i, traj in enumerate(group):
                traj['advantage'] = advantages[i].item() 
                trajectories_with_advantages.append(traj)
        
        return trajectories_with_advantages
    
    def _collate_fn(self, batch):
        states, masks, actions, old_log_probs, advantages = zip(*batch)
        
        batch_obs_dict = {
            'node_vector': torch.stack([s['node_vector'] for s in states]).to(self.device),
            'attn_bias': torch.stack([s['attn_bias'] for s in states]).to(self.device),
            'action_type': torch.stack([s['action_type'] for s in states]).to(self.device)
        }
        
        batch_masks = torch.stack(list(masks)).to(self.device)
        batch_actions = torch.stack(list(actions)).to(self.device)
        batch_old_log_probs = torch.stack(list(old_log_probs)).to(self.device)
        batch_advantages = torch.stack(list(advantages)).to(self.device)
        
        return batch_obs_dict, batch_masks, batch_actions, batch_old_log_probs, batch_advantages
    
    def update_policy(self, trajectories, mini_batch_size=256):
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_action_masks = []
        all_advantages = []
        
        for traj in trajectories:
            states = traj['states']
            actions = traj['actions']
            old_log_probs = traj['log_probs']
            action_masks = traj['action_masks']
            advantage = traj['advantage']
            
            for state, action, old_log_prob, mask in zip(states, actions, old_log_probs, action_masks):
                all_states.append(state)
                all_actions.append(action)
                all_old_log_probs.append(old_log_prob)
                all_action_masks.append(mask)
                all_advantages.append(advantage)
        
        dataset = PPODataset(all_states, all_actions, all_old_log_probs, all_action_masks, all_advantages)
        
        total_policy_loss = 0
        total_entropy = 0
        total_loss = 0
        num_updates = 0
        self.policy_model.train()
        for epoch in range(self.update_epochs):
            data_loader = DataLoader(
                dataset,
                batch_size=mini_batch_size,
                shuffle=True,
                collate_fn=self._collate_fn
            )
            
            for batch_obs_dict, batch_action_mask_tensor, batch_actions, batch_old_log_probs, batch_advantages in data_loader:

            
                probs, log_probs_all = self.policy_model(
                    batch_obs_dict,
                    action_mask=batch_action_mask_tensor,
                    get_all_probs_log_probs=True,
                    add_noise=False 
                )

                new_log_probs = log_probs_all.gather(1, batch_actions.unsqueeze(-1)).squeeze(-1)
       
                entropies = -(probs * log_probs_all).sum(dim=-1)
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                entropy_loss = -entropies.mean()
                

                loss = policy_loss + self.entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_entropy += entropies.mean().item()
                total_loss += loss.item()
                num_updates += 1
        
        avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0
        avg_entropy = total_entropy / num_updates if num_updates > 0 else 0
        avg_total_loss = total_loss / num_updates if num_updates > 0 else 0
        
        return avg_policy_loss, avg_entropy, avg_total_loss
    
    def rollout_policy(self, db_names, writer, num_iterations):
        self.reward_model.eval()
        for iteration in range(num_iterations):
            collection_time = time.time()
            trajectories = self.collect_trajectories(db_names)
            collection_time = time.time() - collection_time
            trajectories = self.compute_group_advantages(trajectories)
            update_time = time.time()
            avg_policy_loss, avg_entropy, avg_total_loss = self.update_policy(trajectories)
            update_time = time.time() - update_time
            avg_reward = sum(traj['reward'] for traj in trajectories) / len(trajectories)
            self.total_iteration_step += 1
            self.stats['policy_loss'].append(avg_policy_loss)
            self.stats['entropy'].append(avg_entropy)
            self.stats['rewards'].append(avg_reward)
            self.stats['total_loss'].append(avg_total_loss)
            writer.add_scalars('Time/SQLGen', {
                'Collection': collection_time,
                'Update': update_time
            }, self.total_iteration_step)
            writer.add_scalar('SQLGen/Policy_Loss', avg_policy_loss, self.total_iteration_step)
            writer.add_scalar('SQLGen/Entropy', avg_entropy, self.total_iteration_step)
            writer.add_scalar('SQLGen/Reward', avg_reward, self.total_iteration_step)
            writer.add_scalar('SQLGen/Total_Loss', avg_total_loss, self.total_iteration_step)
            print(f"Iteration {iteration} - Policy Loss: {avg_policy_loss:.4f}, Entropy: {avg_entropy:.4f}, Total Loss: {avg_total_loss:.4f}, Reward: {avg_reward:.4f}")
        return self.stats
    
    def save_checkpoint(self, path):
        torch.save({
            'policy_model_state_dict': self.policy_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats': self.stats
        }, path)
        print(f"Model saved to: {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.policy_model.load_state_dict(checkpoint['policy_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.stats = checkpoint['stats']
        print(f"Model loaded from: {path}")
