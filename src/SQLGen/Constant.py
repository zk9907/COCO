from enum import Enum
from enum import IntEnum

class Datatype(Enum):
    INT = 'int'
    FLOAT = 'float'
    CATEGORICAL = 'categorical'
    # STRING = 'string'
    MISC = 'misc'
    DATETIME = 'datetime'
    NULL = 'null'

    def __str__(self):
        return self.value

class Operator(Enum):
    NEQ = '!='
    EQ = '='
    LEQ = '<='
    GEQ = '>='
    LT = '<'
    GT = '>'
    LIKE = 'LIKE'
    NOT_LIKE = 'NOT LIKE'
    IS_NOT_NULL = 'IS NOT NULL'
    IS_NULL = 'IS NULL'
    IN = 'IN'
    NOT_IN = 'NOT IN'
    BETWEEN = 'BETWEEN'

    def __str__(self):
        return self.value
    
OPERATORDICT = {str(Operator.LIKE): 1, str(Operator.NOT_LIKE): 2, str(Operator.IS_NULL): 3, str(Operator.IS_NOT_NULL): 4, str(Operator.BETWEEN): 5, str(Operator.IN): 6,
                str(Operator.EQ): 7, str(Operator.NEQ): 8, str(Operator.GT): 9, str(Operator.LT): 10, str(Operator.GEQ): 11, str(Operator.LEQ): 12,str(Operator.NOT_IN): 13}

class LogicalOperator(Enum):
    AND = 'AND'
    OR = 'OR'
    def __str__(self):
        return self.value
LOGICALOPERATORDICT = {str(LogicalOperator.AND): 1, str(LogicalOperator.OR): 2}
class Aggregator(Enum):
    NONE = 'NONE'
    AVG = 'AVG'
    SUM = 'SUM'
    COUNT = 'COUNT'
    MIN = 'MIN'
    MAX = 'MAX'
    COUNTDISTINCT = 'COUNT DISTINCT'
    
    def __str__(self):
        return self.value

AGGREGATORDICT = {str(Aggregator.AVG): 1, str(Aggregator.SUM): 2, str(Aggregator.COUNT): 3, 
                  str(Aggregator.MIN): 4, str(Aggregator.MAX): 5, str(Aggregator.NONE): 6, str(Aggregator.COUNTDISTINCT): 7}

class TriggerActionType(Enum):
    SELECT_LEFT_JOIN_COLUMN = 0
    SELECT_RIGHT_JOIN_COLUMN = 1
    SELECT_PREDICATE_COLUMN = 2
    SELECT_PROJECTION_COLUMN = 3
    SELECT_OPERATOR = 4
    SELECT_BINOP_OPERATOR = 5
    SELECT_IN_OPERATOR = 6
    SELECT_NOT_IN_OPERATOR = 7
    SELECT_NOT_LIKE_OPERATOR = 8
    SELECT_LIKE_OPERATOR = 9
    SELECT_BETWEEN_OPERATOR = 10
    SELECT_VALUE = 11
    SELECT_VALUE_WITH_IN_OPERATOR = 12
    SELECT_AGG_FUNCTION = 13
    SELECT_GROUP_BY_COLUMN = 14
    SELECT_HAVING_COLUMN = 15
    AFTER_ONE_PREDICATE = 16
    AFTER_ONE_JOIN = 17
    AFTER_ONE_PROJECTION = 18
    AFTER_ONE_GROUP_BY = 19

    IS_INCLUDE_GROUP_BY = 20
    
class AgentActionType(IntEnum):
    END_JOIN = 0
    END_PREDICATE = 1
    END_PROJECTION = 2
    END_IN_OP = 3
    END_GROUP_BY = 4
    
    OP_EQUAL = 5
    OP_GREATER = 6
    OP_LESS = 7
    OP_GREATER_EQUAL = 8
    OP_LESS_EQUAL = 9
    OP_NOT_EQUAL = 10
    OP_IN = 11
    OP_NOT_IN = 12
    OP_LIKE = 13
    OP_NOT_LIKE = 14
    OP_IS_NULL = 15
    OP_IS_NOT_NULL = 16
    OP_BETWEEN = 17

    IS_INCLUDE_GROUP_BY = 18
    NOT_INCLUDE_GROUP_BY = 19

    COND_AND = 20
    COND_OR = 21
    
    FUNCTION_MIN = 22
    FUNCTION_AVG = 23
    FUNCTION_SUM = 24
    FUNCTION_COUNT = 25
    FUNCTION_COUNTDISTINCT = 26
    FUNCTION_MAX = 27
    FUNCTION_NONE = 28

    VALUE_START = 29
    VALUE_END = 38
    COL_START = 39
    COL_END = 350 + 39 # 356

    
    @classmethod
    def get_binop_operators(cls):
        return [cls.OP_EQUAL, cls.OP_GREATER, cls.OP_LESS, 
                cls.OP_GREATER_EQUAL, cls.OP_LESS_EQUAL, cls.OP_NOT_EQUAL]
    @classmethod
    def is_aggfunction(cls, value):
        return cls.FUNCTION_MIN <= value <= cls.FUNCTION_NONE
    
    @classmethod
    def get_column_ids(cls):
        return list(range(cls.COL_START, cls.COL_END + 1))
        
    @classmethod
    def is_column(cls, value):
        return cls.COL_START <= value <= cls.COL_END
        
    @classmethod
    def is_binop(cls, value):
        return cls.OP_EQUAL <= value <= cls.OP_NOT_EQUAL
    
    @classmethod
    def is_value(cls, value):
        return cls.VALUE_START <= value <= cls.VALUE_END
    @classmethod
    def is_include_group_by(cls, value):
        return cls.IS_INCLUDE_GROUP_BY <= value <= cls.NOT_INCLUDE_GROUP_BY

class Domain:
    def __init__(self, intervals, default_interval):
        self.intervals = intervals or default_interval
        self.default_interval = default_interval

    def __and__(self, other):
        result = []
        for a_low, a_high in self.intervals:
            for b_low, b_high in other.intervals:
                low = max(a_low, b_low)
                high = min(a_high, b_high)
                if low <= high:
                    result.append((low, high))
        if len(result) == 0:
            return None
        return Domain(result, self.default_interval)

    def __or__(self, other):
        all_intervals = self.intervals + other.intervals
        all_intervals.sort()
        merged = []
        for low, high in all_intervals:
            if not merged or merged[-1][1] < low:
                merged.append((low, high))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        return Domain(merged, self.default_interval)
    def get_min_max(self):
        return self.intervals[0][0], self.intervals[-1][1]
    def max_value(self):
        return self.intervals[-1][1]
    def min_value(self):
        return self.intervals[0][0]
    def get_in_range(self):
        in_range = set()
        for low, high in self.intervals:
            for i in range(low, high + 1):
                if i >= self.default_interval[0] and i <= self.default_interval[1]:
                    in_range.add(i)
        return in_range
    def __str__(self):
        return " ∪ ".join(f"({l}, {h})" for l, h in self.intervals)

    def __repr__(self):
        return str(self)