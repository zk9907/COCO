import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import pickle
from config.SQLGenConfig import Config
config = Config()

class Column(nn.Module):
    def __init__(self):
        super(Column, self).__init__()
        self.feature_dict = config.model_config['column']
        self.device = config.device
        self.column_feature = pickle.load(open(config.column_feature_path, 'rb'))
        # self.semantic_column_feature = pickle.load(open(config.semantic_column_path, 'rb'))
        for key in self.column_feature:
            if key == 'colDtype' or key == 'distribution':
                self.column_feature[key] = torch.tensor(self.column_feature[key], dtype=torch.long)
            else:
                self.column_feature[key] = torch.tensor(self.column_feature[key], dtype=torch.float)
        # self.column_feature['semantic'] = torch.tensor(self.semantic_column_feature, dtype=torch.float)
        self._cached_device = None
        self.feature_modules = nn.ModuleDict()
        self.linear_size = 0
        if 'colDtype' in self.feature_dict:
            input_dim, out_dim = self.feature_dict['colDtype']
            self.feature_modules['colDtype'] = nn.Embedding(input_dim, out_dim)
            self.linear_size += out_dim
        if 'semantic' in self.feature_dict:
            input_dim, out_dim = self.feature_dict['semantic']
            self.feature_modules['semantic'] = nn.Linear(input_dim, out_dim)
            self.linear_size += out_dim
        if 'distribution' in self.feature_dict:
            input_dim, out_dim = self.feature_dict['distribution']
            self.feature_modules['distribution'] = nn.Embedding(input_dim, out_dim)# DistFeature(input_dim, out_dim)
            self.linear_size += out_dim
        if 'meta' in self.feature_dict:
            input_dim, out_dim = self.feature_dict['meta']
            self.feature_modules['meta'] = nn.Sequential(
                # nn.LayerNorm(input_dim),
                nn.Linear(input_dim, out_dim),
                nn.GELU()
            )
            self.linear_size += out_dim
        
        self.final_linear = nn.Sequential(
            nn.LayerNorm(self.linear_size), 
            nn.Linear(self.linear_size, self.feature_dict['size']),
            nn.GELU()
        )
        
    def _ensure_on_device(self):
        target_device = next(self.parameters()).device
        if self._cached_device != target_device:
            for key in self.column_feature:
                tensor = self.column_feature[key]
                if isinstance(tensor, torch.Tensor):
                    self.column_feature[key] = tensor.to(device=target_device)
            self._cached_device = target_device

    def forward(self, column_idx):
        self._ensure_on_device()
        column_idx = column_idx.long()
        embeddings = []
        for key, module in self.feature_modules.items():
            feat = self.column_feature[key][column_idx]
            embeddings.append(module(feat))
        combined = torch.cat(embeddings, dim=-1)
        return self.final_linear(combined).squeeze(-2)
    
class Table(nn.Module):
    def __init__(self, columnNet = None):
        super(Table, self).__init__()
        self.feature_dict = config.model_config['table']
        self.device = config.device
        self.table_feature = pickle.load(open(config.table_feature_path, 'rb'))
        # self.semantic_table_feature = pickle.load(open(config.semantic_table_path, 'rb'))
        self.table_feature['tableMeta'] = torch.tensor(self.table_feature['tableMeta'], dtype=torch.float)
        # self.table_feature['tableEmbed'] = torch.tensor(self.table_feature['tableEmbed'], dtype=torch.float, device=self.device)
        # self.table_feature['tableEmbed'] = torch.tensor(self.semantic_table_feature, dtype=torch.float)
        # self.table_feature['columnIdx'] = torch.tensor(self.table_feature['columnIdx'], dtype=torch.long, device=self.device)
        self._cached_device = None
        self.final_input_dim = 0
        self.feature_modules = nn.ModuleDict()
        if 'tableMeta' in self.feature_dict:
            input_dim, out_dim = self.feature_dict['tableMeta']
            self.feature_modules['tableMeta'] = nn.Sequential(
                # nn.LayerNorm(input_dim),
                nn.Linear(input_dim, out_dim),
                nn.GELU()
            )
            self.final_input_dim += out_dim
        if 'tableEmbed' in self.feature_dict:
            input_dim, out_dim = self.feature_dict['tableEmbed']
            self.feature_modules['tableEmbed'] = nn.Linear(input_dim, out_dim)
            self.final_input_dim += out_dim

        # if 'columnIdx' in self.feature_dict:
        #     input_dim, out_dim = self.feature_dict['columnIdx']
        #     self.feature_modules['columnIdx'] = columnNet
        #     self.final_input_dim += out_dim
            
        self.final_linear = nn.Sequential(
            nn.LayerNorm(self.final_input_dim),
            nn.Linear(self.final_input_dim, self.feature_dict['size']),
            nn.GELU()
        )

    def _ensure_on_device(self):
        target_device = next(self.parameters()).device
        if self._cached_device != target_device:
            for key in list(self.table_feature.keys()):
                tensor = self.table_feature[key]
                if isinstance(tensor, torch.Tensor):
                    self.table_feature[key] = tensor.to(device=target_device)
            self._cached_device = target_device

    def forward(self, table_idx):
        self._ensure_on_device()
        table_idx = table_idx.long().squeeze(-1)
        embeddings = []
        for key, module in self.feature_modules.items():
            if key == 'columnIdx':
                feat = self.table_feature[key][table_idx]
                embeddings.append(module(feat).mean(dim=-2))
            else:
                feat = self.table_feature[key][table_idx]
                embeddings.append(module(feat))
        combined = torch.cat(embeddings, dim=-1)
        return self.final_linear(combined).squeeze(-2)
    
class NodeEncoder(nn.Module):
    def __init__(self):
        super(NodeEncoder, self).__init__()
        self.feature_dict = config.model_config['node']
        self.value_embed = nn.Linear(self.feature_dict['value'][0], self.feature_dict['value'][1])
        self.operator_embed = nn.Embedding(self.feature_dict['operator'][0], self.feature_dict['operator'][1])
        self.aggregate_embed = nn.Embedding(self.feature_dict['aggregate'][0], self.feature_dict['aggregate'][1])
        self.cond_embed = nn.Embedding(self.feature_dict['cond'][0], self.feature_dict['cond'][1])
        self.column_embed = Column()
        self.table_embed = Table()
        self.predicate_dim = self.feature_dict['value'][1] * 3 + self.feature_dict['operator'][1] * 3 + self.feature_dict['cond'][1] * 2
        self.final_dim = config.model_config['table']['size'] + config.model_config['column']['size'] + self.feature_dict['predicate'] + self.feature_dict['aggregate'][1] + 2
        self.predicate_linear = nn.Sequential(
            nn.LayerNorm(self.predicate_dim), 
            nn.Linear(self.predicate_dim, self.feature_dict['predicate']), 
            nn.GELU())
        self.final_linear = nn.Sequential(
            # nn.LayerNorm(self.final_dim),
            nn.Linear(self.final_dim, self.feature_dict['final']),
            nn.GELU()
        )
    def forward(self, x):
        table, column, predicate, group_by, aggregate, indicator = torch.split(x, [1, 1, 35, 1, 1, 1], dim=-1)
        table_feat = self.table_embed(table) 
        column_feat = self.column_embed(column)
        predicate_feat = self.get_predicate_embed(predicate)
        aggregate_feat = self.aggregate_embed(aggregate.long().squeeze(-1))
        combined = torch.cat((table_feat, column_feat, predicate_feat, group_by, aggregate_feat, indicator), dim=-1)
        return self.final_linear(combined)
    
    def get_predicate_embed(self, predicate):
        op_1, value_1, cond1, op_2, value_2, cond2, op_3, value_3 = torch.split(
            predicate, [1, 10, 1, 1, 10, 1, 1, 10], dim=-1
        )

        # Process components: Apply embeddings/linear layers
        # Convert indices to long type and squeeze the last dim for embedding layers
        op_1_feat = self.operator_embed(op_1.long().squeeze(-1)) # Shape: [..., 5]
        op_2_feat = self.operator_embed(op_2.long().squeeze(-1)) # Shape: [..., 5]
        op_3_feat = self.operator_embed(op_3.long().squeeze(-1)) # Shape: [..., 5]
        value_1_feat = self.value_embed(value_1)   # Shape: [..., 5] 
        value_2_feat = self.value_embed(value_2)             # Shape: [..., 5]
        value_3_feat = self.value_embed(value_3)              # Shape: [..., 5]
        cond1_feat = self.cond_embed(cond1.long().squeeze(-1))   # Shape: [..., 2]
        cond2_feat = self.cond_embed(cond2.long().squeeze(-1))   # Shape: [..., 2]

        # Concatenate processed features
        # Total size = 5+5+2 + 5+5+2 + 5+5 = 34
        combined_features = torch.cat(
            (op_1_feat, value_1_feat, cond1_feat,
             op_2_feat, value_2_feat, cond2_feat,
             op_3_feat, value_3_feat),  
            dim = -1
        )
        predicate_feat = self.predicate_linear(combined_features)
        return predicate_feat
        
class SQLStateEncoder(nn.Module):
    def __init__(self):
        super(SQLStateEncoder, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.hidden_dim
        self.input_dropout = nn.Dropout(config.dropout)
        encoders = [
            EncoderLayer(self.hidden_dim, config.ffn_dim, config.dropout, config.dropout,
                         config.head_size) for _ in range(config.num_layers)
        ]
        self.layers         = nn.ModuleList(encoders)
        self.final_ln       = nn.LayerNorm(self.hidden_dim)
        self.node_encoder   = NodeEncoder()
        # self.attn_bias_param = nn.Parameter(torch.ones(1, config.head_size, 1, 1))
        self.super_token_virtual_distance = Parameter(torch.randn(1, config.head_size, 1) * 0.02) 
        self.super_token = nn.Embedding(1, self.hidden_dim)  # the same dimension as node_encoder

    def forward(self, batched_data):
        attn_bias   = batched_data['attn_bias']
        x_features  = batched_data['node_vector']
        # context     = batched_data['buffer_vector']
        n_batch = x_features.size(0)
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, config.head_size, 1, 1)
        # tree_attn_bias = torch.where(tree_attn_bias > -1, tree_attn_bias * self.attn_bias_param, tree_attn_bias) 
        # reset rel pos here
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + self.super_token_virtual_distance   
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + self.super_token_virtual_distance
        
        node_encoded = self.node_encoder(x_features)  # (n_batch*num_nodes, final_dim)
        node_feature = node_encoded.view(n_batch, config.max_column_num, -1)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim = 1)   
        node_feature = self.input_dropout(super_node_feature)
        
        for enc_layer in self.layers:
            node_feature = enc_layer(node_feature, tree_attn_bias)
        # output = self.final_ln(node_feature)
        return node_feature[:, 0, :] #torch.cat([node_feature[:, 0, :], context], dim = -1)

class FeedForwardNetwork(nn.Module):

    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = hidden_size // head_size
        self.scale = self.att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * self.att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * self.att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * self.att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * self.att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias # attn_bias vector
            # x = x * attn_bias
        # print(x)
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):

    def __init__(self, hidden_size, ffn_size, dropout_rate,
                 attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size,
                                                 attention_dropout_rate,
                                                 head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x) 
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
