
from LCQO.LCQOAgent import LCQOAgent, QDataset
import numpy as np
# from LQO.constant import *
import math
import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr, spearmanr
from model.TailNet import ValueNetwork
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from util.util import normalize_narray

class ValueDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature_tensors = {}
        for key in ['x', 'attn_bias', 'heights']:
            if key in feature:
                if isinstance(feature[key], np.ndarray):
                    feature_tensors[key] = torch.from_numpy(feature[key]).to(torch.float32)
                else:
                    feature_tensors[key] = torch.tensor(feature[key], dtype=torch.float32)
        
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return feature_tensors, label_tensor

class ValueAgent:
    def __init__(self, device: torch.device, value_network: ValueNetwork, training_agent: LCQOAgent):
        self.device = device
        self.value_network = value_network
        self.training_agent = training_agent
        self.criterion = nn.MSELoss()

    def update(self, sql_samples, default_plan_features, false_samples = None, num_epochs = 50, batch_size = 256): # sql_samples : {sql_id:experiences}
        for param in self.value_network.planNet.parameters():
            param.requires_grad = False
        for param in self.value_network.out_head.parameters():
            param.requires_grad = True
        self.optimizer = optim.AdamW(self.value_network.out_head.parameters(), lr=1e-3)
        methods = ['diversity','loss']
        weights = {'gradient': 0.0, 'influence': 0.0, 'loss': 0.3,'diversity': 0.7}
        scores = self.compute_samples_scores(sql_samples, default_plan_features, methods=methods, weights=weights)
        X = []
        Y = []
        for sql_id, feature in default_plan_features.items():
            X.append(feature)
            Y.append(scores[sql_id])
        true_samples_num = len(Y)
        print('True Samples in Value Agent:', true_samples_num)
        if false_samples is not None:
            for feature, sql_infos in false_samples[-round(1.5*true_samples_num):]:
                X.append(feature)
                Y.append(0.0)
            # false_samples_num = len(Y) - true_samples_num
        print('False Samples in Value Agent:', len(false_samples) if false_samples is not None else 0)
        Y = np.array(Y)
        Y_normalized = normalize_narray(Y)
        dataset = ValueDataset(X, Y_normalized)
        self.value_network.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            data_loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True,
                collate_fn=self._collate_fn 
            )
            for batch_features, batch_labels in data_loader:
                batch_labels = batch_labels.unsqueeze(1).to(self.device)
                self.optimizer.zero_grad()
                predictions = self.value_network(batch_features)
                loss = self.criterion(predictions, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            if (epoch + 1) % 10 == 0:
                print(f"Value Model Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        torch.save(self.value_network.state_dict(), './ckpt/sql_value_network.pth')
    
    def _collate_fn(self, batch):
        """自定义批处理函数，用于处理字典格式的特征"""
        features_list, labels_list = zip(*batch)
        
        batch_features = {}
        for key in ['x', 'attn_bias', 'heights']:
            if key in features_list[0]:
                batch_features[key] = torch.stack([f[key] for f in features_list]).to(self.device)
        
        batch_labels = torch.stack(labels_list)
        
        return batch_features, batch_labels

    def gradient_and_influence_valuation(self, sql_samples):
        n_samples = len(sql_samples)
        gradient_scores = {}
        influence_scores = {}
        sql_ids = list(sql_samples.keys())
        
        self.training_agent.benefit_model.train()
        
        for idx in range(n_samples):
            sql_id = sql_ids[idx]
            sample_data = list(sql_samples[sql_id].values())
            if len(sample_data) == 0:
                gradient_scores[sql_id] = 0.0
                influence_scores[sql_id] = 0.0
                continue
            
            # 将所有样本数据转换为tensor，一次性处理
            sample_dataset = QDataset(sample_data)
            
            # 手动批处理所有样本
            state_lists = {}
            target_list = []
            weights_list = []
            for i in range(len(sample_dataset)):
                states_i, target_q_values_i, weights_i = sample_dataset[i]
                for k, v in states_i.items():
                    if k not in state_lists:
                        state_lists[k] = []
                    state_lists[k].append(v.unsqueeze(0))
                target_list.append(target_q_values_i.unsqueeze(0))
                weights_list.append(weights_i.unsqueeze(0))
            
            states = {k: torch.cat(v_list, dim=0).to(self.device) for k, v_list in state_lists.items()}
            target_q_values = torch.cat(target_list, dim=0).to(self.device)
            weights = torch.cat(weights_list, dim=0).to(self.device)
            
            self.training_agent.optimizer.zero_grad()
            
            predicted_q_values = self.training_agent.benefit_model(states)
            
            loss_mask = states['action_mask'].bool()
            squared_error = (predicted_q_values - target_q_values) ** 2
            masked_error = squared_error * loss_mask.float() * weights.float()
            
            if loss_mask.sum() > 0:
                loss = masked_error.sum() / loss_mask.sum()
            else:
                loss = torch.tensor(0.0, requires_grad=True).to(self.device)

            loss.backward()
            
            n_effective_samples = len(sample_data)

            grad_norm = 0.0
            grads = []
            for param in self.training_agent.benefit_model.parameters():
                if param.grad is not None:
                    grad_norm += (param.grad ** 2).sum().item()
                    grads.append(param.grad.view(-1).detach().cpu().numpy().copy())
            
            # 按样本数量平均，使不同SQL可比
            gradient_scores[sql_id] = math.sqrt(grad_norm) / n_effective_samples
            
            # 计算影响力分数（influence score）
            if len(grads) > 0:
                gradient_vector = np.concatenate(grads)
                # 按样本数量平均
                influence_scores[sql_id] = np.linalg.norm(gradient_vector) / n_effective_samples
            else:
                influence_scores[sql_id] = 0.0
        
        return gradient_scores, influence_scores
    
    
    def diversity_based_valuation(self, default_plan_features):
        diversity_scores = {sql_id: 0.0 for sql_id in default_plan_features.keys()}
        
        self.training_agent.benefit_model.eval()
        feature_vectors = []
        with torch.no_grad():
            for state in default_plan_features.values():
                state = {k: torch.tensor(v).unsqueeze(0).to(self.device) for k, v in state.items()}
                state_vector = self.training_agent.benefit_model.getStateVector(state).squeeze(0)
                feature_vectors.append(state_vector.detach().cpu().numpy())
        
        feature_matrix = np.array(feature_vectors)

        distances = pairwise_distances(feature_matrix, metric='euclidean')
    
        for sql_id, distance in zip(default_plan_features.keys(), distances):
            diversity_scores[sql_id] = np.mean(distance)
        
        diversity_scores = {sql_id: (diversity_scores[sql_id] - min(diversity_scores.values())) / (max(diversity_scores.values()) - min(diversity_scores.values()) + 1e-8) for sql_id in default_plan_features.keys()}
        
        return diversity_scores
    
    def loss_based_valuation(self, sql_samples):
        n_samples = len(sql_samples)
        loss_scores = {}
        sql_ids = list(sql_samples.keys())
        for idx in range(n_samples):
            sql_id = sql_ids[idx]
            sample_data = list(sql_samples[sql_id].values())
            sample_dataset = QDataset(sample_data)
            loss, _ = self.training_agent.obtain_loss_vector_per_sql(sample_dataset)
            avg_loss = sum(loss) / len(loss)
            loss_scores[sql_id] = 1.0 / (avg_loss + 1e-6)
        return loss_scores
    
    def compute_samples_scores(self, sql_samples, default_plan_features, methods, weights = None):
        if weights is None:
            weights = {method: 1.0 / len(methods) for method in methods}
        results = {method: {} for method in methods}
        if 'gradient' in methods or 'influence' in methods:
            if weights['gradient'] > 1e-6 or weights['influence'] > 1e-6:
                results['gradient'], results['influence'] = self.gradient_and_influence_valuation(sql_samples)  
        if 'diversity' in methods and weights['diversity'] > 1e-6:
            results['diversity'] = self.diversity_based_valuation(default_plan_features)
        if 'loss' in methods and weights['loss'] > 1e-6:
            results['loss'] = self.loss_based_valuation(sql_samples)
        combined_scores = {sql_id: 0.0 for sql_id in sql_samples.keys()}
        for method, scores in results.items():
            score_lists = list(scores.values())
            min_score = min(score_lists)
            max_score = max(score_lists)
            for sql_id, score in scores.items():
                normalized_score = (score - min_score) / (max_score - min_score + 1e-8)
                combined_scores[sql_id] += weights[method] * normalized_score
        return combined_scores
    
    def validate(self, X_test, Y_test, batch_size = 256):
        self.value_network.eval()
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_X = X_test[i:i+batch_size]

                # 准备批次输入
                batch_input = {}
                for key in batch_X[0].keys():
                    if key in ['x', 'attn_bias', 'heights']:
                        batch_values = []
                        for state in batch_X:
                            tensor = torch.tensor(state[key], dtype=torch.float32)
                            batch_values.append(tensor)
                        batch_input[key] = torch.stack(batch_values).to(self.device)
                
                predictions = self.value_network(batch_input)
                all_predictions.extend(predictions.cpu().numpy())
        
        all_predictions = np.array(all_predictions).flatten()
        
        print(Y_test)
        print(all_predictions)
        # 计算相关性
        
        pearson_corr, _ = pearsonr(Y_test, all_predictions)
        spearman_corr, _ = spearmanr(Y_test, all_predictions)
        
        print(f"\n模型评估结果:")
        print(f"  Pearson 相关系数:  {pearson_corr:.4f}")
        print(f"  Spearman 相关系数: {spearman_corr:.4f}")
        print(f"  MSE: {np.mean((Y_test - all_predictions)**2):.6f}")