import os
from config.GenConfig import GenConfig
from SQLGen.Constant import *
import torch
class Config(GenConfig):
    def __init__(self):
        
        self.max_value_num = 10
        self.operator_num = len(OPERATORDICT)
        self.aggregator_num = len(AGGREGATORDICT)
        self.logical_operator_num = len(LOGICALOPERATORDICT)
        self.max_column_num = AgentActionType.COL_END - AgentActionType.COL_START
        self.trigger_action_num =  TriggerActionType.AFTER_ONE_PROJECTION.value + 2

        # ======    SQLGen Model    ========
        self.dropout = 0.2
        self.head_size = 4
        self.hidden_dim = 512
        self.ffn_dim = 512
        self.num_layers = 4
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") 
        self.model_config = {
            'device': self.device,
            'column': {
                'colDtype': (6, 16),
                # 'semantic': (16, 16),
                'distribution': (10, 16),
                'meta': (18 + 4, 32),  # n_distinct, n_null, is_index, is_primary, is_foreign
                'size':64
            },
            'table': {
                'tableMeta': (3, 8),
                # 'columnIdx': (16, 16),
                # 'tableEmbed': (16, 16),
                'size':16
            },
            'node':{
                'value': (10, 32),
                'operator': (14, 32),
                'aggregate': (9, 32),
                'cond': (3, 16),
                'predicate': 64, # 5 * 3 + 5 * 3 + 2 * 2
                'final': self.hidden_dim, # 16 + 16 + 10 + 5 = 47
            },
            'action_state_dim':16,
            "action_dim": AgentActionType.COL_END + 1
        }
         # self.model_config['height'][1] +
        self.generation_params = {
            "max_no_joins": 14,
            "max_no_predicates": 6,
            "max_no_aggregates": 3,
            "max_no_group_by": 2,
            "max_predicate_per_col": 3,
            "max_value_one_predicate": 10,
            "max_group_by_per_col": 1,
            "max_aggregates_per_col": 1,
        }
        self.train_params = {
           'group_size': 8,
           'num_groups': 8,
           'update_epochs': 8,
           'gamma': 1.0,
           'clip_epsilon': 0.4,
           'entropy_coef': 0.03,
           'max_grad_norm': 1.0,
           'learning_rate': 1e-4,
           'device': self.device
        }
        self.max_update_times_per_iter = 5
        self.sqlgen_agent_path = os.path.join(self.checkpoint_dir, 'sqlgen_agent')