from config.GenConfig import GenConfig
import torch
from LCQO.Constant import *

class Config(GenConfig):
    def __init__(self):
        # ======    LQO Model    ========
        self.dropout = 0.3
        self.mlp_dropout = 0.3 
        self.head_size = 1
        self.ffn_dim = 256
        self.num_layers = 1
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") 
        self.model_config = {
            'device': self.device,
            'type': (len(TYPE2IDX) + 1, 32),  # (input_dim, embed_dim)
            'pos': (5, 16),
            'dbEst': (2, 16),
            'height': (HEIGHTSIZE, 16),
            'column': {
                'colDtype': (5, 16),
                # 'semantic': (16, 16),
                'distribution': (9, 32),
                'meta': (18 + 4, 32),  # n_distinct, n_null, is_index, is_primary, is_foreign
                'size':32
            },
            'join': {'size':32},
            # 'table': {
            #     'tableMeta': (3, 16), # table_size, num_columns, num_references, num_indexes
            #     # 'columnIdx': (16, 16),
            #     # 'tableEmbed': (16, 16),
            #     'size':16
            # }
        }
        self.hidden_dim = 128
        self.node_dim = self.hidden_dim - self.model_config['height'][1] # self.hidden_dim - self.model_config['height'][1]
        # self.node_input = 6 + 1 + MAXFILTER * 6 + MAXJOIN * 2
        self.mlp_dim = 128   
        self.lqo_agent_path = self.checkpoint_dir
        self.test_sql_path = ['./TestWorkload/test_job_sql.pkl',
                                './TestWorkload/test_jobext_sql.pkl',
                                './TestWorkload/test_stats_sql.pkl',
                                './TestWorkload/test_tpcds_sql.pkl']

        # # ======   LoRA Defaults   ======
        self.lora_enable = False
        self.lora_r = 8
        self.lora_alpha = 8
        self.lora_dropout = 0.05
        self.lora_targets = ['planStateEncoder', 'qhead']
        self.lora_freeze_base = True