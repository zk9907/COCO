import torch
import torch.nn as nn
import torch.nn.functional as F
from config.LCQOConfig import Config as LCQOConfig
from config.SQLGenConfig import Config as SQLGenConfig
from LCQO.Constant import *
from model.PlanNet import PlanNetwork
from model.SQLNet import SQLStateEncoder
lcqo_config = LCQOConfig()
sqlgen_config = SQLGenConfig()
class BenefitNetwork(nn.Module):
    def __init__(self):
        super(BenefitNetwork, self).__init__()

        self.planStateEncoder = PlanNetwork()
        self.actionStateEncoder = nn.Sequential(
            nn.Linear(len(HINT2POS), 16),  # 7 is action_code dimension
            nn.GELU()
        )
        self.planStateEncoder.node_encoder.type_embed

        self.qhead = nn.Sequential(
            nn.LayerNorm(self.planStateEncoder.hidden_dim + 16),
            nn.Linear(self.planStateEncoder.hidden_dim + 16, lcqo_config.mlp_dim),
            nn.GELU(),
            nn.Dropout(lcqo_config.mlp_dropout),
            nn.Linear(lcqo_config.mlp_dim, lcqo_config.mlp_dim // 2),
            nn.GELU(),
            nn.Linear(lcqo_config.mlp_dim // 2, lcqo_config.mlp_dim // 4),
            nn.GELU(),
            nn.Linear(lcqo_config.mlp_dim // 4, len(HINT2POS)),
        )
    
    def getStateVector(self, state):
        embed_plan = {
            "x": state["x"],
            "attn_bias": state["attn_bias"],
            "heights": state["heights"]
        }
        planStateFeatures = self.planStateEncoder(embed_plan)[:, 0, :]
        return planStateFeatures

    def forward(self, state):
        embed_plan = {
            "x": state["x"],
            "attn_bias": state["attn_bias"],
            "heights": state["heights"]
        }

        planStateFeatures = self.planStateEncoder(embed_plan)[:, 0, :]
        actionStateFeatures = self.actionStateEncoder(state["action_code"])
        stateFeatures = torch.cat([planStateFeatures, actionStateFeatures], dim=1)
        values = self.qhead(stateFeatures)
        # latency = self.lhead(planStateFeatures)
        return values
    
class ValueNetwork(nn.Module):
    def __init__(self, planNet = None):
        super(ValueNetwork, self).__init__()
        self.planNet = PlanNetwork() if planNet is None else planNet
        self.out_head = nn.Sequential(
            nn.LayerNorm(lcqo_config.hidden_dim),
            nn.Linear(lcqo_config.hidden_dim, lcqo_config.mlp_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(lcqo_config.mlp_dim * 2, lcqo_config.mlp_dim),
            nn.GELU(),
            nn.Linear(lcqo_config.mlp_dim, lcqo_config.mlp_dim // 2),
            nn.GELU(),
            nn.Linear(lcqo_config.mlp_dim // 2, 1)
        )
    def only_vector(self, x):
        embedding = self.planNet(x)
        return embedding[:, 0, :]
    def only_value(self, x):
        value = self.out_head(x)
        return value
    def forward(self, x):
        embedding = self.planNet(x)
        value = self.out_head(embedding[:, 0, :])
        return value

class GenerationNetwork(nn.Module):
    def __init__(self, hidden_dim=512, layer_norm=False):
        super(GenerationNetwork, self).__init__()
        self.state_encoder = SQLStateEncoder()
        self.action_state_dim = sqlgen_config.model_config['action_state_dim']
        self.action_type_embed = nn.Embedding(30, self.action_state_dim)
        
        self.fc_mean = nn.Linear(self.state_encoder.output_dim, self.state_encoder.output_dim)
        self.fc_log_std = nn.Linear(self.state_encoder.output_dim, self.state_encoder.output_dim)
        policy_layers = []
        policy_layers.append(nn.Linear(self.state_encoder.output_dim + self.action_state_dim, hidden_dim))
        if layer_norm:
            policy_layers.append(nn.LayerNorm(hidden_dim))
        policy_layers.append(nn.Tanh())
        policy_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if layer_norm:
            policy_layers.append(nn.LayerNorm(hidden_dim))
        policy_layers.append(nn.Tanh())
        policy_layers.append(nn.Linear(hidden_dim, hidden_dim))
        if layer_norm:
            policy_layers.append(nn.LayerNorm(hidden_dim))
        policy_layers.append(nn.Tanh())
        policy_layers.append(nn.Linear(hidden_dim, sqlgen_config.model_config['action_dim']))
        
        self.policy = nn.Sequential(*policy_layers)
        
    def forward(self, obs_or_representation, action_mask=None, get_log_prob=False, deterministic=False,get_all_probs_log_probs=False, add_noise=True, action_type=None):
        if isinstance(obs_or_representation, dict):
            representation = self.state_encoder(obs_or_representation)
            action_type_tensor = obs_or_representation["action_type"] if action_type is None else action_type
        else:
            representation = obs_or_representation
            action_type_tensor = action_type

        if action_type_tensor is None:
            raise ValueError("action_type must be provided when passing a precomputed representation to PolicyNetwork")

        mean = self.fc_mean(representation)
        if not add_noise:
            z = mean
        else:
            log_std = self.fc_log_std(representation).clamp(min=-5.0, max=1.0) 
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            z = mean + eps * std  # reparameterization trick
        
        # Get action type embedding
        action_type_embed = self.action_type_embed(action_type_tensor.long().squeeze(-1))
        
        # Concatenate the embeddings
        combined_obs = torch.cat((z, action_type_embed), dim=1)
        
        # Get raw logits
        logits = self.policy(combined_obs)
        
        # Apply mask: set logits for invalid actions to -inf
        if action_mask is not None:
            mask = action_mask.bool()
            logits = torch.where(mask, logits, torch.tensor(-1e38, device=logits.device))
        
        # Get action probabilities
        probs = F.softmax(logits, dim=-1)
        if get_all_probs_log_probs:
            log_probs_all = F.log_softmax(logits, dim=-1)
            return probs, log_probs_all
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = None 
        else:
            distribution = torch.distributions.Categorical(probs=probs)
            action = distribution.sample()
            
            # Calculate log probability if needed
            if get_log_prob:
                log_prob = distribution.log_prob(action)
                return action, log_prob
        
        return action, probs
