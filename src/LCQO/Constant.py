# Hint Type
LEADINGHINT = 0
RULEHINT = 1
# Postgres Data Type
PGDATATYPE = ['smallint','integer','bigint','decimal','numeric','real',
                'double precision','smallserial','serial','bigserial']
PGCHARTYPE = ['character','character varying','text','bytea']
PGDATETYPE = ['date','timestamp','interval','timestamp without time zone','time without time zone']

JOINTYPE = ["Nested Loop", "Hash Join", "Merge Join"]
SCANTYPE = ['Index Only Scan', 'Seq Scan', 'Index Scan', 'Bitmap Heap Scan','Tid Scan']
CONDTYPE = ['Hash Cond','Join Filter','Index Cond','Merge Cond','Recheck Cond']  # 'Filter' # not consider filter
BINOP = [' >= ',' <= ',' = ',' > ',' < ']
OP2IDX ={'=ANY': 0,'>=':1,'<=':2,'>': 3,'=': 4,'<': 5,'NA':6,'IS NULL':7,'IS NOT NULL':8, '<>':9,'~~':10,'!~~':11, '~~*': 12,'<>ALL':13}
TYPE2IDX = {
            "Nested Loop": 1,
            "Hash Join": 2,
            "Merge Join": 3,
            "Seq Scan": 4,
            "Index Scan": 5,
            "Index Only Scan": 6,
            "Bitmap Heap Scan": 7,
            'Tid Scan': 8
        }

ALLRULES = [
    "enable_nestloop", "enable_hashjoin", "enable_mergejoin",
    "enable_seqscan", "enable_indexscan", "enable_indexonlyscan"]

OPERATOR2HINT = {'Index Only Scan':'IndexOnlyScan', 'Seq Scan':'SeqScan', 
                     'Index Scan':'IndexScan', 'Bitmap Heap Scan':'BitmapScan','Tid Scan':'TidScan',
                     'Hash Join':'HASHJOIN','Merge Join': 'MERGEJOIN','Nested Loop':'NESTLOOP'}

ACTION2HINT = {
            0:'SET enable_nestloop TO off;',
            1:'SET enable_hashjoin TO off;',
            2:'SET enable_mergejoin TO off;',
            3:'SET enable_seqscan TO off;',
            4:'SET enable_indexscan TO off;',
            5:'SET enable_indexonlyscan TO off;'}

HINT2POS = {"NESTLOOP":0, "HASHJOIN":1, "MERGEJOIN":2, "SeqScan":3, "IndexScan":4, "IndexOnlyScan":5}

PLANMAXTIMEOUT = 3e5
MAXNODE = 40
HEIGHTSIZE = 20
MAXFILTER = 15
MAXJOIN = 10
MAXDISTANCE = 20