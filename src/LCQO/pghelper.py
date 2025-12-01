import psycopg2
from psycopg2 import extensions, errors
import pandas as pd
import time
from LCQO.Constant import *
class PGHelper:
    def __init__(self, dbConfig, heartbeat_interval=300):
        self.heartbeat_interval = heartbeat_interval
        self.con = psycopg2.connect(database=dbConfig["database"], 
                                    user=dbConfig["user"],
                                    password=dbConfig["password"], 
                                    port=dbConfig["port"],
                                    host=dbConfig["host"],
                                    connect_timeout=2000,      
                                    keepalives=1,            
                                    keepalives_idle=30,     
                                    keepalives_interval=60,  
                                    keepalives_count=50)
        self.cur = self.con.cursor()
        self.dbConfig = dbConfig
        self.cur.execute("SET geqo to off;")
        self.curDbName = dbConfig["database"]

    def _ensure_connection_ready(self):
        try:
            status = self.con.get_transaction_status()
        except Exception:
            self.reconnect(self.curDbName)
            return

        if status == extensions.TRANSACTION_STATUS_INERROR:
            self.con.rollback()
        elif status == extensions.TRANSACTION_STATUS_UNKNOWN:
            self.reconnect(self.curDbName)

    def get_table_info(self):
        self.cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        self.tableNames = [name[0] for name in self.cur.fetchall()]
        self.tableNum   = len(self.tableNames)
        return self.tableNum

    def reconnect(self, dbName):
        if self.con:
            self.con.close()
        self.con = psycopg2.connect(database=dbName, 
                                    user=self.dbConfig["user"],
                                    password=self.dbConfig["password"], 
                                    port=self.dbConfig["port"],
                                    host=self.dbConfig["host"],
                                    connect_timeout=2000,      
                                    keepalives=1,            
                                    keepalives_idle=30,     
                                    keepalives_interval=60,   
                                    keepalives_count=50)
        self.cur = self.con.cursor()
        self.refreshPgState()
        self.curDbName = dbName
        self.get_table_info()

    def transform_hint(self, hintcode):
        binary_indices = [j for j in range(hintcode.bit_length()) if (hintcode >> j) & 1]
        hints = []
        for j in binary_indices:
            hints.append(ACTION2HINT[j])
        return hints
    
    def get_latency(self, hint, sql, timeout = PLANMAXTIMEOUT, hintstyle = LEADINGHINT, dbName = None):
        start_ts = time.time()
        if dbName != None and dbName != self.curDbName:
            self.reconnect(dbName)
        self._ensure_connection_ready()
        timeout_ms = max(1, int(timeout))
        try:
            self.cur.execute(f"SET statement_timeout = {timeout_ms};")
            if hintstyle == LEADINGHINT:
                self.cur.execute(hint + sql)
            elif hintstyle == RULEHINT:
                self.refreshPgState()
                for h in hint:
                    self.cur.execute(h)
                self.cur.execute(sql)
            istimeout = False
            exectime = round((time.time() - start_ts) * 1000, 3)
        except KeyboardInterrupt:
            raise
        except errors.QueryCanceled:
            istimeout = True
            self.con.rollback()
            exectime = timeout_ms
        except Exception:
            istimeout = True
            self.con.rollback()
            exectime = timeout_ms
        return (exectime, istimeout)
    
    def get_latency_plan(self, hint, sql, dbName, timeout = PLANMAXTIMEOUT, hintstyle = RULEHINT):
        if dbName != None and dbName != self.curDbName:
            self.reconnect(dbName)
        self._ensure_connection_ready()
        timeout_ms = max(1, int(timeout))
        try:
            self.cur.execute(f"SET statement_timeout = {timeout_ms};")
            if hintstyle == LEADINGHINT:
                self.cur.execute(hint + ' EXPLAIN (ANALYZE, FORMAT JSON) ' + sql)
                rows = self.cur.fetchall()
            elif hintstyle == RULEHINT:
                self.refreshPgState()
                for h in hint:
                    self.cur.execute(h)
                self.cur.execute('EXPLAIN (ANALYZE, FORMAT JSON) ' + sql)
                rows = self.cur.fetchall()
            planJson = rows[0][0][0]
            exectime = planJson['Execution Time']
            istimeout = False
        except KeyboardInterrupt:
            raise
        except errors.QueryCanceled:
            self.con.rollback()
            planJson = self.get_cost_plan(hint, sql, hintstyle, dbName=dbName)
            exectime = timeout_ms
            istimeout = True
        except Exception:
            self.con.rollback()
            planJson = self.get_cost_plan(hint, sql, hintstyle, dbName=dbName)
            exectime = timeout_ms
            istimeout = True

        return (exectime, istimeout, planJson)

        
    def refreshPgState(self):
        self._ensure_connection_ready()
        for option in ALLRULES:
            self.cur.execute(f"SET {option} TO on")
        self.cur.execute("SET geqo to off;")

    def get_cost_plan(self, hint, sql, hintStyle, dbName = None):
        if dbName != None and dbName != self.curDbName:
            self.reconnect(dbName)
        self._ensure_connection_ready()
        startTime = time.time()
        timeout_ms = max(1, int(PLANMAXTIMEOUT))
        self.cur.execute(f"SET statement_timeout = {timeout_ms};")
        try:
            if hintStyle == LEADINGHINT:
                self.cur.execute(hint + " explain (COSTS, FORMAT JSON) " + sql)
                rows = self.cur.fetchall()
            elif hintStyle == RULEHINT:
                self.refreshPgState()
                for h in hint:
                    self.cur.execute(h)
                self.cur.execute("explain (COSTS, FORMAT JSON) " + sql) 
                rows = self.cur.fetchall()
            planJson = rows[0][0][0]
            planJson['Planning Time'] = time.time() - startTime
            return planJson
        except Exception as e:
            print(e,flush=True)
            self.con.rollback()
            return None
    
    def get_result(self, sql, dbName, timeout = PLANMAXTIMEOUT):
        if dbName != None and dbName != self.curDbName:
            self.reconnect(dbName)
        self._ensure_connection_ready()
        timeout_ms = max(1, int(timeout))
        try:
            self.cur.execute(f"SET statement_timeout = {timeout_ms};")
            self.refreshPgState()
            startTime = time.time()
            self.cur.execute(sql)
            exectime = (time.time() - startTime) * 1000
            rows = self.cur.fetchall()
            has_result = False
            for row in rows:
                if not has_result:
                    for r in row:
                        if r != 0 and r != None:
                            has_result = True
                            break
            return has_result, exectime 
        except KeyboardInterrupt:
            raise
        except errors.QueryCanceled:
            self.con.rollback()
            return None, timeout_ms
        except Exception:
            self.con.rollback()
            return None, timeout_ms
        
    def get_min_max_values(self,tableName, columnName):
        self.cur.execute(f"SELECT MIN({columnName}), MAX({columnName}) FROM {tableName};")
        minVal, maxVal = self.cur.fetchone()
        if minVal != None and maxVal != None:
            maxVal = float(maxVal)
            minVal = float(minVal)
        return minVal,maxVal
    
    def get_column_data_properties(self):
        columnDataProperties = {}
        for tableName in self.tableNames:
            self.cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{tableName}';")
            for columnName, dataType in self.cur.fetchall():
                if dataType in PGDATATYPE:
                    min_val, max_val = self.get_min_max_values(tableName, columnName)
                    columnDataProperties[tableName + '.' + columnName] = (min_val, max_val)
        return columnDataProperties
    
    def get_column_type(self):
        columnType = {}
        for tableName in self.tableNames:
            self.cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{tableName}';")
            for columnName, dataType in self.cur.fetchall():
                columnType[tableName + '.' + columnName] = dataType
        return columnType
    
    def table_column_idx(self):
        table2idx = {}
        column2idx = {}
        for tableName in self.tableNames:
            self.cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{tableName}';")
            table2idx[tableName] = len(table2idx) + 1
            for columnName in self.cur.fetchall():
                column2idx[tableName + '.' + columnName[0]] = len(column2idx) + 1
        return table2idx, column2idx
    
    def get_statistics(self, old_metaInfo = None, accuracy_mode = True):
        def read_str(vals):
            result = []
            current = []
            in_quotes = False
            for char in vals:
                if char == '"':
                    in_quotes = not in_quotes
                    current.append(char)
                elif char == ',' and not in_quotes:
                    result.append(''.join(current))
                    current = []
                else:
                    current.append(char)
            if current:
                result.append(''.join(current))
            return result
        
        
        self.tableSize = {}
        self.tableMetadata = {}
        for tableName in self.tableNames:
            if old_metaInfo is None:
                self.cur.execute(f"SELECT COUNT(*) FROM \"{tableName}\";")
                self.tableSize[tableName] = self.cur.fetchall()[0][0]
            else:
                self.tableSize[tableName] = old_metaInfo['tableAttr'][tableName]["table_size_ori"]
            # Get number of columns
            self.cur.execute(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{tableName}';")
            num_columns = self.cur.fetchone()[0]
            
            # Get primary key information
            self.cur.execute(f"""
                SELECT a.attname
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = '\"{tableName}\"'::regclass
                AND i.indisprimary;
            """)
            primary_keys = [row[0] for row in self.cur.fetchall()]
            
            # Get foreign key information
            self.cur.execute(f"""
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM
                    information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                      AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                      AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = '{tableName}';
            """)
            foreign_keys = []
            for row in self.cur.fetchall():
                foreign_keys.append({
                    'column': row[0],
                    'references_table': row[1],
                    'references_column': row[2]
                })
            
            # Get indexes information
            self.cur.execute(f"""
                SELECT
                    i.relname AS index_name,
                    a.attname AS column_name
                FROM
                    pg_class t,
                    pg_class i,
                    pg_index ix,
                    pg_attribute a
                WHERE
                    t.oid = ix.indrelid
                    AND i.oid = ix.indexrelid
                    AND a.attrelid = t.oid
                    AND a.attnum = ANY(ix.indkey)
                    AND t.relkind = 'r'
                    AND t.relname =  '{tableName}'
                ORDER BY
                    i.relname;
            """)
            indexes = []
            for row in self.cur.fetchall():
                index_name, column_name = row
                indexes.append(column_name)
            # Store all metadata for this table
            self.tableMetadata[tableName] = {
                'num_columns': num_columns,
                'primary_keys': primary_keys,
                'foreign_keys': foreign_keys,
                'indexes': indexes
            }
        
        
        all_statistics = None
        all_statistics_dict = {'colAttr':{},'tableAttr':{},'dbAttr':{}}
        for table_name in self.tableNames:
            all_statistics_dict['tableAttr'][table_name] = {}
            all_statistics_dict['tableAttr'][table_name]["table_size_ori"] = self.tableSize[table_name]
            all_statistics_dict['tableAttr'][table_name]["num_columns"] = self.tableMetadata[table_name]["num_columns"]
            # all_statistics_dict['tableAttr'][table_name]["num_references"] = len(self.tableMetadata[table_name]["foreign_keys"])
            all_statistics_dict['tableAttr'][table_name]["num_indexes"] = len(self.tableMetadata[table_name]["indexes"])

            table_statistics = pd.read_sql(f"SELECT tablename, attname, most_common_vals, histogram_bounds, n_distinct, null_frac, most_common_freqs, avg_width, correlation FROM pg_stats WHERE tablename = '{table_name}';", self.con)
            if table_statistics.empty:
                self.cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}';")
                for columnName in self.cur.fetchall(): 
                    new_row = pd.DataFrame({
                        'tablename': [table_name],
                        'attname': [columnName[0]], 
                        'most_common_vals': [None],
                        'histogram_bounds': [None],
                        'n_distinct': [1],
                        'null_frac': [0],
                        'most_common_freqs': [[1]]
                    })
                    if all_statistics is None:
                        all_statistics = new_row
                    else:
                        all_statistics = pd.concat([all_statistics, new_row])
            else:
                if all_statistics is None:
                    all_statistics = table_statistics
                else:
                    all_statistics = pd.concat([all_statistics, table_statistics])
        

        for idx, row in all_statistics.iterrows():
            # if row["tablename"] not in all_statistics_dict:
            #     all_statistics_dict[row["tablename"]] = {}
            colmn_name = row["tablename"] + '.' + row["attname"]
            if colmn_name not in all_statistics_dict['colAttr']:
                all_statistics_dict['colAttr'][colmn_name] = {}
                all_statistics_dict['colAttr'][colmn_name]["mcv_vals"] = []
                all_statistics_dict['colAttr'][colmn_name]["hist_vals"] = []
            # if row["most_common_vals"]:
            #     all_statistics_dict['colAttr'][colmn_name]["mcv_vals"] = (
            #         read_str(row["most_common_vals"][1:-1])
            #     )
            # if row["histogram_bounds"]:
            #     # print(row["histogram_bounds"])
            #     all_statistics_dict['colAttr'][colmn_name]["hist_vals"] = (
            #         read_str(row["histogram_bounds"][1:-1])
            #     )
            # if row["most_common_freqs"]:
            #     all_statistics_dict['colAttr'][colmn_name]["mcf_vals"] = sum(row["most_common_freqs"])
            # else:
            #     row["most_common_freqs"] = 0
            

            col_name = row["attname"]
            table_name = row['tablename']
            if accuracy_mode:
                if old_metaInfo is None:
                    query = f"""
                            SELECT count(distinct \"{col_name}\")
                            FROM \"{table_name}\";
                            """
                    self.cur.execute(query)
                    real_n_distinct = self.cur.fetchall()[0][0]
                    # print(real_n_distinct)# self.cur.fetchall()[0][0]
                    if self.tableSize[row["tablename"]] == 0:
                        distinct = 1
                    else:
                        distinct = real_n_distinct / self.tableSize[row["tablename"]]
                else:
                    distinct = old_metaInfo['colAttr'][colmn_name]["n_distinct"]
            else:
                distinct = abs(row["n_distinct"]) if row["n_distinct"] < 0.0 else row["n_distinct"] / (self.tableSize[row["tablename"]] + 1e-6)
            all_statistics_dict['colAttr'][colmn_name]["n_distinct_ori"] = distinct
            # print(distinct,row["n_distinct"],self.tableSize[row["tablename"]])
            # print(distinct+ row['null_frac'],self.tableSize[row["tablename"]])
            if distinct < 1e-1:
                all_statistics_dict['colAttr'][colmn_name]["n_distinct"] = 1
            elif distinct > 1-1e-1:
                all_statistics_dict['colAttr'][colmn_name]["n_distinct"] = 2
            else:
                all_statistics_dict['colAttr'][colmn_name]["n_distinct"] = 3
            # all_statistics_dict['colAttr'][colmn_name]["n_distinct"] = distinct
            all_statistics_dict['colAttr'][colmn_name]["null_frac_ori"] = row['null_frac']
            if row['null_frac'] < 1e-8:
                all_statistics_dict['colAttr'][colmn_name]["null_frac"] = 1
            elif row['null_frac'] > 0.5:
                all_statistics_dict['colAttr'][colmn_name]["null_frac"] = 2
            else:
                all_statistics_dict['colAttr'][colmn_name]["null_frac"] = 3
            # all_statistics_dict['colAttr'][colmn_name]["null_frac"] = row['null_frac']
            # all_statistics_dict['colAttr'][colmn_name]["null_frac"] = row['null_frac']

            all_statistics_dict['colAttr'][colmn_name]["avg_width"] = row['avg_width'] if row['avg_width'] > 0 else 0
            correlation = abs(row['correlation']) if row['correlation'] else 0
            if correlation < 1e-1:
                all_statistics_dict['colAttr'][colmn_name]["correlation"] = 1
            elif correlation > 1-1e-1:
                all_statistics_dict['colAttr'][colmn_name]["correlation"] = 2
            else:
                all_statistics_dict['colAttr'][colmn_name]["correlation"] = 3
            # all_statistics_dict['colAttr'][colmn_name]["correlation"] = abs(row['correlation'] if row['correlation'] else 0)
  
            all_statistics_dict['colAttr'][colmn_name]["is_primary_key"] = col_name in self.tableMetadata[row["tablename"]]["primary_keys"]
            all_statistics_dict['colAttr'][colmn_name]["is_foreign_key"] = col_name in [fk['column'] for fk in self.tableMetadata[row["tablename"]]["foreign_keys"]]
            all_statistics_dict['colAttr'][colmn_name]["has_index"] = col_name in self.tableMetadata[row["tablename"]]["indexes"]
        return all_statistics_dict

    def close(self):
        self.con.close()
        self.cur.close()
        print("[INFO] Connection closed.")