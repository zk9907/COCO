import torch
import torch.nn as nn
import pickle
from config.LCQOConfig import Config
from LCQO.Constant import *
config = Config()

class Column(nn.Module):
    def __init__(self):
        super(Column, self).__init__()
        self.feature_dict = config.model_config['column']
        self.device = config.device
        self.column_feature = pickle.load(open(config.column_feature_path, 'rb'))
        for key in self.column_feature:
            if key == 'colDtype' or key == 'distribution':
                self.column_feature[key] = torch.tensor(self.column_feature[key], dtype=torch.long, device=self.device)
            else:
                self.column_feature[key] = torch.tensor(self.column_feature[key], dtype=torch.float, device=self.device)
        self.feature_modules = nn.ModuleDict()
        self.linear_size = 0
        if 'colDtype' in self.feature_dict:
            input_dim, out_dim = self.feature_dict['colDtype']
            self.feature_modules['colDtype'] = nn.Embedding(input_dim, out_dim)
            self.linear_size += out_dim
        if 'distribution' in self.feature_dict:
            input_dim, out_dim = self.feature_dict['distribution']
            self.feature_modules['distribution'] = nn.Embedding(input_dim, out_dim) #DistFeature(input_dim, out_dim)
            self.linear_size += out_dim
        if 'meta' in self.feature_dict:
            input_dim, out_dim = self.feature_dict['meta']
            self.feature_modules['meta'] = nn.Sequential(
                nn.Linear(input_dim, out_dim),
                # nn.LayerNorm(out_dim),
                nn.GELU()
            )
            self.linear_size += out_dim
        
        self.final_linear = nn.Sequential(
            nn.Linear(self.linear_size, self.feature_dict['size']),
            nn.LayerNorm(self.feature_dict['size']), 
            nn.GELU(),
            # nn.Dropout(config.dropout)
        )
        
    def forward(self, column_idx):
        column_idx = column_idx.long()
        embeddings = []
        for key, module in self.feature_modules.items():
            feat = self.column_feature[key][column_idx]
            embeddings.append(module(feat))
        combined = torch.cat(embeddings, dim=-1)
        return self.final_linear(combined)

    def get_all_column_embedding(self):
        """Return embeddings for all columns."""
        self.eval()
        with torch.no_grad():
            embeddings = []
            for key, module in self.feature_modules.items():
                feat = self.column_feature[key]
                embeddings.append(module(feat))
            combined = torch.cat(embeddings, dim=-1)
            return self.final_linear(combined)
    
class Table(nn.Module):
    def __init__(self,columnNet = None):
        super(Table, self).__init__()
        self.feature_dict = config.model_config['table']
        self.device = config.device
        self.table_feature = pickle.load(open(config.table_feature_path, 'rb'))
        self.table_feature['tableMeta'] = torch.tensor(self.table_feature['tableMeta'], dtype=torch.float, device=self.device)
        # self.table_feature['tableEmbed'] = torch.tensor(self.table_feature['tableEmbed'], dtype=torch.float, device=self.device)
        # self.table_feature['columnIdx'] = torch.tensor(self.table_feature['columnIdx'], dtype=torch.long, device=self.device)
        self.final_input_dim = 0
        self.feature_modules = nn.ModuleDict()
        if 'tableMeta' in self.feature_dict:
            input_dim, out_dim = self.feature_dict['tableMeta']
            self.feature_modules['tableMeta'] = nn.Sequential(
                nn.Linear(input_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU()
            )
            self.final_input_dim += out_dim
            
        self.final_linear = nn.Sequential(
            nn.Linear(self.final_input_dim, self.feature_dict['size']),
            nn.LayerNorm(self.feature_dict['size']),
            nn.GELU(),
            # nn.Dropout(config.dropout)
        )

    def forward(self, table_idx):
        table_idx = table_idx.long().squeeze(-1)
        embeddings = []
        for key, module in self.feature_modules.items():
            if key == 'columnIdx':
                # Ensure the module for 'columnIdx' exists (it might not if columnNet was None)
                if 'columnIdx' in self.feature_modules:
                    feat = self.table_feature[key][table_idx]
                    embeddings.append(module(feat).mean(dim=-2))
            else:
                feat = self.table_feature[key][table_idx]
                embeddings.append(module(feat))
        combined = torch.cat(embeddings, dim=-1)
        return self.final_linear(combined).squeeze(-2)

class NodeFeature(nn.Module):
    def __init__(self):
        super(NodeFeature, self).__init__()
        self.featureDict = config.model_config
        self.output_dim = config.node_dim # Target output dimension

        # Baseline features
        self.baselineDim = 0
        self.type_embed = None
        self.pos_embed = None
        self.db_linear = None
        # self.token_type_embed = nn.Embedding(6, self.output_dim)
        self.final_input_dim = 0
        if 'type' in self.featureDict:
            self.type_embed = nn.Embedding(self.featureDict['type'][0], self.featureDict['type'][1])
            self.final_input_dim += self.featureDict['type'][1]
        if 'pos' in self.featureDict:
            self.pos_embed = nn.Embedding(self.featureDict['pos'][0], self.featureDict['pos'][1])
            # self.baselineDim += self.featureDict['pos'][1]
            self.final_input_dim += self.featureDict['pos'][1]
        if 'dbEst' in self.featureDict:
            self.db_linear = nn.Sequential(
                nn.Linear(self.featureDict['dbEst'][0], self.featureDict['dbEst'][1]),
                nn.LayerNorm(self.featureDict['dbEst'][1]),
                nn.GELU()
            )
            self.final_input_dim += self.featureDict['dbEst'][1]
        
        # self.baseline_projector = nn.Linear(self.baselineDim, self.output_dim) if self.baselineDim > 0 else None
    
        self.column = None
        if 'column' in self.featureDict:
            self.column = Column()
        
        self.has_filter = 'filter' in self.featureDict
        # self.filter_projector = None
        # self.filterDim was storing the output dimension of get_filter previously
        self.filterDim_output_from_get_filter = 0 
        if self.has_filter:
            self.op_embed = nn.Embedding(self.featureDict['filter']['op'][0], self.featureDict['filter']['op'][1])
            self.dtype_embed = nn.Embedding(self.featureDict['filter']['dtype'][0], self.featureDict['filter']['dtype'][1])
            
            # Calculate the input dimension to the self.linear_filter layer
            filter_input_to_linear_filter_dim = self.featureDict['filter']['op'][1] + self.featureDict['filter']['dtype'][1] + 2
            if self.column is not None: # Guarding access to self.column.feature_dict
                filter_input_to_linear_filter_dim += self.column.feature_dict['size']

            self.linear_filter = nn.Sequential(
                nn.Linear(filter_input_to_linear_filter_dim, self.featureDict['filter']['size']),
                nn.LayerNorm(self.featureDict['filter']['size']),
                nn.GELU()
            )
            self.final_input_dim += self.featureDict['filter']['size']
            # self.filterDim_output_from_get_filter = self.featureDict['filter']['size'] # This is the output dim of get_filter
            # self.filter_projector = nn.Linear(self.filterDim_output_from_get_filter, self.output_dim)
        
        self.has_join = 'join' in self.featureDict
        self.join_projector = None
        # self.joinDim was storing the output dimension of get_join_embed previously
        self.joinDim_output_from_get_join = 0
        if self.has_join and self.column is not None: # Guarding for self.column
            self.linear_join1 = nn.Sequential(
                nn.Linear(self.featureDict['column']['size'] * 2, self.featureDict['column']['size']),
                nn.GELU()
            )
            self.linear_join2 = nn.Sequential(
                nn.Linear(self.featureDict['column']['size'] * (MAXJOIN // 2), self.featureDict['join']['size']),
                nn.GELU()
            )
            self.final_input_dim += self.featureDict['join']['size']
            # self.joinDim_output_from_get_join = self.featureDict['join']['size'] # This is the output dim of get_join_embed
            # self.join_projector = nn.Linear(self.joinDim_output_from_get_join, self.output_dim)
        # Note: self.joinDim is not explicitly set to 0 here if conditions fail,
        # but joinDim_output_from_get_join remains 0, and projector is None.
        # The original self.joinDim variable might be unused now.

        self.has_table = 'table' in self.featureDict
        self.table_projector = None
        # self.tableDim was storing the output dimension of get_table previously
        self.tableDim_output_from_get_table = 0
        if self.has_table:
            self.table = Table() # self.column could be None
            self.final_input_dim += self.featureDict['table']['size']
            # self.tableDim_output_from_get_table = self.featureDict['table']['size'] # This is output of get_table
            # self.table_projector = nn.Linear(self.tableDim_output_from_get_table, self.output_dim)

        self.final_linear = nn.Sequential(
            nn.Linear(self.final_input_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU()
        )
        # self.final_norm = nn.LayerNorm(self.output_dim)
        # self.final_activation = nn.GELU()
        # Removed self.finalLinear and totalDim calculation
    
    def forward(self, features):
        baseline_in, table_in, join_in, joinmask_in = torch.split(features, [6, 1, MAXJOIN, MAXJOIN], dim = -1)# , MAXFILTER * 6
        
        N_effective = baseline_in.shape[0] # Assuming first dim is batch/effective batch

        active_projected_features = []

        if self.final_input_dim > 0:
            baseline_emb = self.get_baseline(baseline_in)
            # projected_baseline = self.baseline_projector(baseline_emb)
            active_projected_features.append(baseline_emb)
        
        # if self.has_filter:
        #     filter_emb = self.get_filter(filter_in)
        #     # projected_filter = self.filter_projector(filter_emb)
        #     # filter_emb = self.token_type_embed(torch.tensor([3], device=filter_emb.device)) + filter_emb
        #     active_projected_features.append(filter_emb)

        if self.has_join and self.column is not None:
            join_emb = self.get_join_embed(join_in, joinmask_in)
            # projected_join = self.join_projector(join_emb)
            # join_emb = self.token_type_embed(torch.tensor([4], device=join_emb.device)) + join_emb
            active_projected_features.append(join_emb)

        if self.has_table:
            # self.table might have been initialized with self.column = None.
            # The Table class's forward method should be robust to this.
            table_emb = self.get_table(table_in) 
            # projected_table = self.table_projector(table_emb)
            # table_emb = self.token_type_embed(torch.tensor([5], device=table_emb.device)) + projected_table
            active_projected_features.append(table_emb)

        if not active_projected_features:
            # Return a zero tensor if no features are active to be processed
            return torch.zeros(N_effective, self.output_dim, device=features.device)
        # for emb in active_projected_features:
        #     print(emb.shape)
        return self.final_linear(torch.cat(active_projected_features, dim=-1))
    
    def get_baseline(self, baseline):
        type_id, pos_id, db_est,_ = torch.split(baseline, [1, 1, 2,2], dim=-1)
        outputs = []
        if self.type_embed is not None:
            type_emb = self.type_embed(type_id.long()).squeeze(2)
            # type_emb = self.token_type_embed(torch.tensor([0], device=type_emb.device)) + type_emb
            outputs.append(type_emb)
        if self.pos_embed is not None:
            pos_emb = self.pos_embed(pos_id.long()).squeeze(2)
            # pos_emb = self.token_type_embed(torch.tensor([1], device=pos_emb.device)) + pos_emb
            outputs.append(pos_emb)
        if self.db_linear is not None:
            db_est_emb = self.db_linear(db_est)
            # db_est_emb = self.token_type_embed(torch.tensor([2], device=db_est_emb.device)) + db_est_emb
            outputs.append(db_est_emb)
        return torch.cat(outputs, dim=-1)
    
    def get_table(self, table):
        return self.table(table)
    
    def get_filter(self, filter_feat):
        op, dtype, column, isInMCV, isInHist, mask = torch.split(filter_feat , [MAXFILTER, MAXFILTER, MAXFILTER, MAXFILTER, MAXFILTER, MAXFILTER], dim=-1)
        op_emb = self.op_embed(op.long())
        dtype_emb = self.dtype_embed(dtype.long())
        isInMCV = isInMCV.unsqueeze(-1)
        isInHist = isInHist.unsqueeze(-1)
        stats = torch.cat((isInMCV, isInHist), dim=-1)

        if self.column is not None:
            column_feat = self.column(column)
            concat = torch.cat((column_feat, op_emb, dtype_emb, stats), dim=-1)
        else:
            concat = torch.cat((op_emb, dtype_emb, stats), dim=-1)
        concat = self.linear_filter(concat)

        mask = mask.bool()
        concat = concat * mask.unsqueeze(-1).float()
        num_filters = mask.sum(dim=-1, keepdim=True).float() + 1e-10
        avg = concat.sum(dim=-2) / num_filters
        return avg
    
    def get_join_embed(self, join, joinmask):
        assert self.column is not None, "self.column should be initialized and not None when calling get_join_embed"
            
        join_column = self.column(join) # self.column is now asserted to be not None
        joinmask = joinmask.bool()
        join_column = join_column * joinmask.unsqueeze(-1).float()

        join_flat = join_column.view(join_column.size(0), join_column.size(1), -1, 2 * self.column.feature_dict['size'])
        join_hidden = self.linear_join1(join_flat)
        join_hidden = join_hidden.view(join_hidden.size(0), join_hidden.size(1), -1)
        join_out = self.linear_join2(join_hidden)
        return join_out

class PlanNetwork(nn.Module):
    def __init__(self):
        super(PlanNetwork, self).__init__()
        self.hidden_dim = config.hidden_dim

        self.height_encoder = nn.Embedding(config.model_config['height'][0], config.model_config['height'][1], padding_idx=0)
        
        # Distance embedding for attention bias
        self.distance_embed = nn.Embedding(22, 1)
        
        self.input_dropout = nn.Dropout(config.dropout)
        encoders = [
            EncoderLayer(self.hidden_dim, config.ffn_dim, config.dropout, config.dropout,
                         config.head_size) for _ in range(config.num_layers)
        ]
        self.layers         = nn.ModuleList(encoders)
        self.final_ln       = nn.LayerNorm(self.hidden_dim)
        self.node_encoder   = NodeFeature()
        # Use imported Parameter
        # self.attn_bias_param = Parameter(torch.empty(1, config.head_size, 1, 1).uniform_(0.5, 3)) 
    
    def forward(self, batched_data):
        attn_bias = batched_data['attn_bias']
        x_features = batched_data['x']   # dict 格式
        heights = batched_data['heights'].long()
        n_batch = heights.size(0)
        
        mask = (attn_bias > -1)
        tree_attn_bias = torch.full_like(attn_bias, -1e9, dtype=torch.float)
        if mask.any():
            valid_distances = attn_bias[mask].long()
            valid_distances = torch.clamp(valid_distances, 0, self.distance_embed.num_embeddings - 1)
            embedded_distances = self.distance_embed(valid_distances.long()).squeeze(-1)
            tree_attn_bias[mask] = embedded_distances
        
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, config.head_size, 1, 1)
        # tree_attn_bias = torch.where(tree_attn_bias > -1, tree_attn_bias * self.attn_bias_param, tree_attn_bias)

        node_encoded = self.node_encoder(x_features)  # (n_batch*num_nodes, final_dim)
        node_encoded = node_encoded.view(n_batch, MAXNODE, -1)

        height_feat = self.height_encoder(heights)
        
        node_feature = torch.cat([node_encoded, height_feat], dim=-1)
        # node_feature = self.input_dropout(node_feature)
        
        for enc_layer in self.layers:
            node_feature = enc_layer(node_feature, tree_attn_bias)
        output = self.final_ln(node_feature)

        return output

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
            x = x + attn_bias

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