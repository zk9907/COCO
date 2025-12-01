import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.LCQOConfig import Config
from model.TailNet import BenefitNetwork
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from LCQO.LCQOEnv import *
from util.SQLBuffer import SQL, Node
import math
# from tqdm import tqdm
class QDataset(Dataset):
    def __init__(self, experiences):
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        state, rewards, weights = self.experiences[idx]
        
        state_tensors = {}
        for k, v in state.items():
            dtype = torch.float32
            if 'mask' in k:
                dtype = torch.bool if k != 'action_mask' else torch.float32

            if isinstance(v, np.ndarray):
                state_tensors[k] = torch.from_numpy(v).to(dtype)
            else:
                state_tensors[k] = torch.tensor(v, dtype=dtype)

        return state_tensors, torch.tensor(rewards, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32)
def huber_loss(x, delta=1.0):
    return torch.where(
        torch.abs(x) < delta,
        0.5 * x.pow(2),
        delta * (torch.abs(x) - 0.5 * delta)
    )

class LCQOAgent:
    def __init__(self, config:Config,  model : BenefitNetwork):
        self.config = config
        self.benefit_model = model
        self.lcqo_test_env = LCQOEnvTest(self.config)
        self.lcqo_collect_env = LCQOEnvExp(self.config)
        self.optimizer = optim.AdamW(self.benefit_model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.model_save_path = config.lqo_agent_path

    def update(self, train_dataset, test_data = None, validate_data = None, num_epochs=50, batch_size=1024):
        for param in self.benefit_model.parameters():
            param.requires_grad = True
        self.optimizer = optim.AdamW(self.benefit_model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-5)
        self.benefit_model.train()
        losses = []
        test_result = []
        validate_result = None
        for epoch in range(num_epochs):
            total_loss = 0
            data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for batch_data in data_loader:
                states, target_q_values, weights = batch_data
                
                states = {k: v.to(self.config.device) for k, v in states.items()}
                target_q_values = target_q_values.to(self.config.device)
                predicted_q_values = self.benefit_model(states)
                weights = weights.to(self.config.device)
                
                loss_mask = states['action_mask'].bool()
                squared_error = (predicted_q_values - target_q_values) ** 2
                masked_error = squared_error * loss_mask.float() * weights.float()
                
                if loss_mask.sum() > 0:
                    loss = masked_error.sum() / loss_mask.sum()
                else:
                    loss = torch.tensor(0.0, requires_grad=True).to(self.config.device)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()          
                total_loss += loss.item()
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
            losses.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                print(f"Benefit Model Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f} LR: {current_lr:.6f}",flush=True)
            if len(losses) > 15 and losses[-1] < 0.01:
                last_two = np.min(losses[-2:])
                if last_two > losses[-5] or (losses[-5] - last_two < 0.001):
                    print("Stopped training from convergence condition at epoch", epoch)
                    break
            if (epoch + 1) % 5 == 0 and test_data is not None:
                test_result.append(self._evaluate_sql_list(test_data[0], test_data[1]))
                self.benefit_model.train()
        if test_result:
            test_result = sorted(test_result)[0]
        if validate_data is not None and validate_data[0] is not None and validate_data[1] is not None:
            validate_result = self._evaluate_sql_list(validate_data[0], validate_data[1])
            self.benefit_model.train()
        return test_result, validate_result

    def _evaluate_one_sql(self, sql: SQL):
        if not hasattr(sql, 'q_min_latency') or sql.q_min_latency is None:
            sql.q_min_latency = self._get_q_min_latency(sql)
        obs, _ = self.lcqo_test_env.reset(options={'sql': sql})
        done = False
        latency = sql.base_latency
        while not done:
            obs['action_code'] = obs['action_code']
            q_values = self.predict(obs)
            action_mask = obs['action_mask']
            masked_q_values = np.where(action_mask == 1, q_values, -np.inf)
            if np.all(masked_q_values <= 0.0):
                done = True
            else:
                action = np.argmax(masked_q_values)
                obs, latency, done, _, _ = self.lcqo_test_env.step(action)
        return latency
    
    def _evaluate_sql_list(self, sql_list, sql_dataset):
        latency_info = {}
        for sql in sql_list:
            sql:SQL
            lcqo_latency = self._evaluate_one_sql(sql)
            db_name = sql.dbName
            if db_name not in latency_info:
                latency_info[db_name] = 0
            latency_info[db_name] += lcqo_latency
        result = {}
        all_loss = []
        for db, dataset in sql_dataset.items():
            q_dataset = QDataset(dataset)
            db_loss_values = self.obtain_loss_per_db(q_dataset)
            result[db] = (sum(db_loss_values) / len(db_loss_values), latency_info[db])
            all_loss.extend(db_loss_values)
        avg_loss = sum(all_loss) / len(all_loss)
        return (avg_loss, result)

    def _get_q_min_latency(self, sql: SQL):
        obs, _ = self.lcqo_test_env.reset(options={'sql': sql})
        done = False
        while not done:
            action_mask = obs['action_mask']
            max_idx = -1
            now_latency = sql.get_plan_by_hint_code(self.lcqo_test_env.current_code).latency
            for idx, value in enumerate(action_mask):
                if value == 1:
                    plan: Plan = sql.get_plan_by_hint_code(self.lcqo_test_env.current_code + pow(2, idx))
                    if plan.latency < now_latency:
                        max_idx = idx
                        now_latency = plan.latency
            if max_idx == -1:
                done = True
            else:
                obs, reward, done, _, latency = self.lcqo_test_env.step(max_idx)
        return now_latency

    def _collect_experiences_for_query(self, sql_instance: SQL):
        experiences = {}
        initial_state, current_latency = self.lcqo_collect_env.reset(options = {'sql': sql_instance})
        root = Node(initial_state, current_latency)
        visited_nodes = set()
        def expand_node(node: Node):
            if node.code in visited_nodes:
                return
            visited_nodes.add(node.code)
            original_env_state = self.lcqo_collect_env.save_env_state()
            valid_actions = np.where(node.env_state['action_mask'] == 1)[0].tolist()
            rewards = np.zeros(len(HINT2POS), dtype=np.float16)
            weights = np.zeros(len(HINT2POS), dtype=np.float16)
            for action in range(len(HINT2POS)):
                if action not in valid_actions:
                    rewards[action] = -1.0
                    continue
                self.lcqo_collect_env.load_env_state(original_env_state)
                next_state, reward, done, _, current_latency = self.lcqo_collect_env.step(action)
                rewards[action] = reward
                weights[action] = min(max(math.log2(abs(original_env_state['prev_latency']) + 1) / math.log2(PLANMAXTIMEOUT), 1e-2), 1)
                child = Node(next_state, current_latency, action, node)
                child.code = self.lcqo_collect_env.current_code
                node.children[action] = child
                if not done:
                    expand_node(child)
            if len(node.env_state['heights']) <= MAXNODE and node.env_state['action_mask'].sum() > 0:
                experiences[node.code] = (node.env_state, rewards, weights)
            self.lcqo_collect_env.load_env_state(original_env_state)
        expand_node(root)
        return experiences
    
    def obtain_all_vector(self, data_loader):
        self.benefit_model.eval()
        all_vectors = []
        with torch.no_grad():
            for batch_data in data_loader:
                states, _, _ = batch_data
                states = {k: v.to(self.config.device) for k, v in states.items()}
                state_vector = self.benefit_model.getStateVector(states)
                all_vectors.append(state_vector.cpu().numpy())
        if len(all_vectors) == 0:
            return None
        all_vectors = np.concatenate(all_vectors, axis=0)
        return all_vectors
    
    def obtain_loss_per_db(self, dataset: QDataset):
        data_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        self.benefit_model.eval()
        losses = []
        for batch_data in data_loader:
            states, target_q_values, weights = batch_data
            states = {k: v.to(self.config.device) for k, v in states.items()}
            target_q_values = target_q_values.to(self.config.device)
            weights = weights.to(self.config.device)
            with torch.no_grad():
                predicted_q_values = self.benefit_model(states)
            loss_mask = states['action_mask']
            squared_error = (predicted_q_values - target_q_values) ** 2
            masked_error = squared_error * loss_mask.float() * weights.float()
            loss_mask_sum = loss_mask.float().sum(dim=1)
            sample_losses = torch.zeros_like(loss_mask_sum)
            valid_mask = loss_mask_sum > 0
            sample_losses[valid_mask] = masked_error.sum(dim=1)[valid_mask] / loss_mask_sum[valid_mask]
            valid_mask = valid_mask.cpu().numpy()
            sample_losses = sample_losses.cpu().numpy()
            losses.extend(sample_losses[valid_mask].tolist())
        return losses
    
    def obtain_loss_vector_per_sql(self, dataset: QDataset):
        self.benefit_model.eval()
        state_lists = {}
        target_list = []
        weights_list = []
        for i in range(len(dataset)):
            states_i, target_q_values_i, weights_i = dataset[i]
            for k, v in states_i.items():
                if k not in state_lists:
                    state_lists[k] = []
                state_lists[k].append(v.unsqueeze(0))
            target_list.append(target_q_values_i.unsqueeze(0))
            weights_list.append(weights_i.unsqueeze(0))
        if len(target_list) == 0:
            raise ValueError("No data to obtain loss vector")
        states = {k: torch.cat(v_list, dim=0).to(self.config.device) for k, v_list in state_lists.items()}
        target_q_values = torch.cat(target_list, dim=0).to(self.config.device)
        weights = torch.cat(weights_list, dim=0).to(self.config.device)
        with torch.no_grad():
            predicted_q_values = self.benefit_model(states)
            state_vector = self.benefit_model.getStateVector(states)
        loss_mask = states['action_mask']
        squared_error = (predicted_q_values - target_q_values) ** 2
        masked_error = squared_error * loss_mask.float() * weights.float()
        loss_mask_sum = loss_mask.float().sum(dim=1)
        sample_losses = torch.zeros_like(loss_mask_sum)
        valid_mask = loss_mask_sum > 0
        sample_losses[valid_mask] = masked_error.sum(dim=1)[valid_mask] / loss_mask_sum[valid_mask]
        sample_losses = sample_losses.cpu().numpy()
        valid_mask = valid_mask.cpu().numpy()
        # avg_loss = masked_error.mean().item() if masked_error.numel() > 0 else 0.0
        state_vector = np.mean(state_vector.cpu().numpy(), axis=0)
        return sample_losses[valid_mask].tolist(), state_vector
    
    def save_model(self):
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.benefit_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.model_save_path, 'lqo_agent.pth'))
        print(f"Benefit Model saved to {self.model_save_path}")

    def load_model(self):
        if not os.path.exists(self.model_save_path):
            print(f"No model found at {self.model_save_path}. Starting from scratch.")
            return
        model_path = os.path.join(self.model_save_path, 'lqo_agent.pth')
        try:
            checkpoint = torch.load(model_path, map_location=self.config.device)
            self.benefit_model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            if k == 'step':
                                state[k] = v.cpu()
                            else:
                                state[k] = v.to(self.config.device)
            self.benefit_model.to(self.config.device)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}. Starting from scratch.")

    def predict(self,state):
        self.benefit_model.eval()
        model_input = {
            k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.config.device)
            for k, v in state.items() if k in ["x", "attn_bias", "heights", "action_code"]
        }
        with torch.no_grad():
            q_values = self.benefit_model(model_input)
        return q_values.cpu().numpy()
    