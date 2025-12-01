import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import pandas as pd
import psycopg2
import json
import datetime as dt
from SQLGen.Constant import *
import random
import numpy as np
from config.GenConfig import GenConfig
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Datatype):
            return str(obj)
        else:
            return super(CustomEncoder, self).default(obj)

def get_columns_info(conn, table):
    cur = conn.cursor()
    query = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}';"
    try:
        cur.execute(query)
    except Exception as e:
        print(f"Error retrieving columns for table {table}: {e}")
        return []
    rows = cur.fetchall()
    timestamp_types = {"timestamp without time zone", "timestamp with time zone", "date",'time without time zone'}
    character_types = {"character varying", "varchar", "character", "text","bytea"}
    numeric_types = {"integer", "numeric", "smallint", "bigint", "double precision", "real", "decimal"}
    timestamp_columns = []
    character_columns = []
    numeric_columns = []
    for row in rows: 
        if row[1] in timestamp_types:
            timestamp_columns.append(row[0])
        elif row[1] in character_types:
            if check_is_timestamp(conn, table, row[0]):
                # print(f"Column {row[0]} is a timestamp")
                timestamp_columns.append(row[0])
            else:
                character_columns.append(row[0])
        elif row[1] in numeric_types:
            numeric_columns.append(row[0])
        else:
            print(f"Unknown column type: {row[1]} for column {row[0]}")
    return timestamp_columns, character_columns, numeric_columns

def check_is_timestamp(conn, table, column):
    cur = conn.cursor()
    try:
        query = f"""
        SELECT \"{column}\"
        FROM \"{table}\" 
        WHERE \"{column}\" IS NOT NULL  
        LIMIT 100
        """
        cur.execute(query)
        rows = cur.fetchall()
        if not rows:
            return False
        
        # Extract values from rows
        samples = [row[0] for row in rows]
        
        try:
            # Try to convert all samples to datetime
            pd.to_datetime(samples)
            return True
        except (ValueError, TypeError):
            # If conversion fails, it's not a timestamp
            return False
            
    except Exception as e:
        print(f"Error sampling data from {table}.{column}: {e}")
        return False
    finally:
        cur.close()
def get_all_tables(conn):
    cur = conn.cursor()
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
    cur.execute(query)
    tables = [row[0] for row in cur.fetchall()]
    return tables
def get_data_from_column(conn, table, column, datatype, sample_size = 10000000):
    cur = conn.cursor()
    try:
        query = f"""
        SELECT \"{column}\"
        FROM \"{table}\" 
        WHERE \"{column}\" IS NOT NULL  
        ORDER BY RANDOM()
        LIMIT {sample_size}
        """
        cur.execute(query)
        rows = cur.fetchall()
        sample_data = pd.Series([row[0] for row in rows])
        if datatype == 'numeric':
            sample_data = pd.to_numeric(sample_data, errors='coerce')
        elif datatype == 'timestamp':
            sample_data = sample_data.astype(str)
        else:
            pass
        return sample_data
    except Exception as e:
        print(f"Error sampling data from {table}.{column}: {e}")
    finally:
        cur.close()

def get_numeric_samples(conn, table, column, sample_size=10000000):
    value_counts = get_data_from_column(conn, table, column, sample_size)
    try:
        value_counts = {float(x):value_counts[x] for x in value_counts if x is not None}
        print(f"Successfully sampled {len(value_counts)} numeric values from {table}.{column}")
    except Exception as e:
        print(f"Could not convert data to float for {table}.{column}: {e}")
    return value_counts

def get_character_samples(conn, table, column, sample_size=10000000):
    value_counts = get_data_from_column(conn, table, column, sample_size)
    print(f"Successfully sampled {len(value_counts)} character values from {table}.{column}")
    return value_counts

def get_timestamp_samples(conn, table, column, sample_size=10000000):
    value_counts = get_data_from_column(conn, table, column, sample_size)
    epoch = dt.datetime(1970, 1, 1)  
    converted_value_counts = {}
    
    for x, count in value_counts.items():
        if x is not None:
            ts = convert_to_datetime(x)
            if ts is not None:
                seconds = (ts - epoch).total_seconds()
                converted_value_counts[seconds] = count
    
    print(f"Successfully sampled {len(converted_value_counts)} timestamp values from {table}.{column}")
    return converted_value_counts

def convert_to_datetime(d):
    try:
        if isinstance(d, dt.datetime):
            return d
        elif isinstance(d, dt.date):
            return dt.datetime.combine(d, dt.time())
        elif isinstance(d, dt.time):
            return dt.datetime.combine(dt.date(1970, 1, 1), d)
        elif isinstance(d, str):
            try:
                return dt.datetime.fromisoformat(d)
            except Exception:
                from dateutil import parser
                return parser.parse(d)
        else:
            raise Exception(f"Unknown timestamp data type: {type(d)}")
    except Exception as e:
        print(f"Error converting to timestamp {d}: {e}")
        return None
    
def get_sample_for_generate(samples, N_split, max_sample_size, mode, min_occurance = 1):
    """
    
    """
    if len(samples) == 0:
        return []
    
    
    samples_per_bin = max_sample_size // N_split
    if mode == 'sort_by_value':
        unique_samples = pd.Series(samples.unique())
        sorted_samples = unique_samples.sort_values()
        if len(sorted_samples) <= N_split:
            return sorted_samples.tolist()
        elif len(sorted_samples) <= max_sample_size:
            bin_size = len(sorted_samples) // N_split
            result = []
            for i in range(N_split):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size if i < N_split - 1 else len(sorted_samples)
                bin_samples = sorted_samples.iloc[start_idx:end_idx]
                result.append(bin_samples.tolist())
            return result
        else:

            bin_size = len(sorted_samples) // N_split
            result = []
            for i in range(N_split):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size if i < N_split - 1 else len(sorted_samples)
                bin_samples = sorted_samples.iloc[start_idx:end_idx]
                
                if len(bin_samples) > samples_per_bin:
                    bin_samples = bin_samples.sample(samples_per_bin)
                
                result.append(bin_samples.sort_values().tolist())
            
            return result
    
    elif mode == 'sort_by_occurance':
        value_counts = samples.value_counts(ascending=True)
        value_counts = value_counts[value_counts > min_occurance]
        all_result = value_counts.index.tolist()
        if len(all_result) <= N_split:
            return all_result
        elif len(all_result) <= max_sample_size:
            bin_size = len(all_result) // N_split
            result = []
            for i in range(N_split):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size if i < N_split - 1 else len(all_result)
                bin_samples = all_result[start_idx:end_idx]
                result.append(bin_samples)
            return result
        else:

            bin_size = len(all_result) // N_split
            result = []
            for i in range(N_split):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size if i < N_split - 1 else len(all_result)
                bin_values = all_result[start_idx:end_idx]
                
                if len(bin_values) > samples_per_bin:
                    
                    bin_values = random.sample(bin_values, samples_per_bin)
                
                result.append(bin_values)
        
        return result

def get_freq_words(samples, min_word_length=3):
    if len(samples) == 0:
        return pd.Series([])
    
    string_samples = samples[samples.apply(lambda x: isinstance(x, str))]
    if len(string_samples) == 0:
        return pd.Series([])
    
    possible_words = []
    try:
        for text in string_samples:
            
            words = text.split()
            if len(words) > 1:  
                for i, word in enumerate(words):
                    if len(word) >= min_word_length:
                        if i == 0:
                            possible_words.append(word + '%')
                        elif i == len(words) - 1:
                            possible_words.append('%' + word)
                        else:
                            possible_words.append('%' + word)
                            possible_words.append(word + '%')
                            possible_words.append('%' + word + '%')
            else:
                current_type = None
                current_word = ""
                last_pos = 0
                len_text = len(text)
                for current_pos, char in enumerate(text):
                    if char.isalpha():
                        char_type = "alpha"
                    elif char.isdigit():
                        char_type = "digit"
                    else:
                        char_type = "special"
                    if char == "'":
                        char = "\\'"
                    if current_type is None:
                        current_type = char_type
                        current_word = char
                    elif char_type != current_type:
                        if len(current_word) >= min_word_length:
                            if last_pos == 0 and last_pos + len(current_word) != len_text:
                                possible_words.append(current_word + '%')
                            elif last_pos == 0 and last_pos + len(current_word) == len_text:
                                pass
                            elif last_pos != 0 and last_pos + len(current_word) == len_text:
                                possible_words.append('%' + current_word)
                            else:
                                possible_words.append('%' + current_word + '%')
                        last_pos = current_pos
                        current_type = char_type
                        current_word = char
                    else:
                        current_word += char
                if len(current_word) >= min_word_length and last_pos != 0:
                    possible_words.append('%' + current_word)
        possible_words = pd.Series(possible_words)
        return possible_words
    except Exception as e:
        return pd.Series([])
    

# Main function: Process numeric, categorical and timestamp columns
def generate_column_stats(dbname, metaInfo, save_dir, sample_size=1000000, N_split=10):
    categorical_threshold = 10000
    max_sample_size = 10000
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # if os.path.exists(os.path.join(save_dir, f'{dbname}_column_statistics.json')):
    #     print(f"Column distribution data already exists for {dbname}")
    #     return None
    # List the database configurations
    config = {**GenConfig.db_config, 'database': dbname}
    # print(f"Processing database: {dbname}")

    try:
        conn = psycopg2.connect(database=config["database"],
                                user=config["user"],
                                password=config["password"],
                                host=config.get("host", "localhost"),
                                port=config.get("port", "5432"))
    except Exception as e:
        print(f"Error connecting to database {dbname}: {e}")
        raise e
    
    tables = get_all_tables(conn)
    if not tables:
        print(f"No tables found in database {dbname}")
        conn.close()
        raise Exception(f"No tables found in database {dbname}")
    column_stats_path = os.path.join(save_dir, f'{dbname}_column_statistics.json')
    joint_column_stats = dict()

    for table in tables:
        column_stats_table = dict()
        table_size = metaInfo['tableAttr'][table]['table_size_ori']
        columns_info = get_columns_info(conn, table)
        timestamp_columns, character_columns, numeric_columns = columns_info
        for column in character_columns + numeric_columns + timestamp_columns:
            nan_ratio = metaInfo['colAttr'][table+'.'+column]['null_frac_ori']
            n_distinct = metaInfo['colAttr'][table+'.'+column]['n_distinct_ori']
            num_unique = round(n_distinct * table_size)
            print(f"Processing table: {table}, column: {column}, num_unique: {num_unique}, n_distinct: {n_distinct}, table_size: {table_size}")
            stats = dict(nan_ratio=nan_ratio, 
                         n_distinct=n_distinct, 
                         num_unique=num_unique)
            if column in numeric_columns:
                samples = get_data_from_column(conn, table, column, datatype = 'numeric', sample_size = sample_size)
            elif column in timestamp_columns:
                samples = get_data_from_column(conn, table, column, datatype = 'timestamp', sample_size = sample_size)
            else:
                samples = get_data_from_column(conn, table, column, datatype = 'string', sample_size = sample_size)
            if len(samples) == 0:
                stats.update(dict(datatype=Datatype.NULL))
            elif samples.dtype == float:
                sample_data = get_sample_for_generate(samples, N_split, max_sample_size, mode = 'sort_by_value')
                stats.update(dict(
                    datatype=Datatype.FLOAT,
                    max=samples.max(),
                    min=samples.min(),
                    mean=samples.mean(),
                    sample_data=sample_data))
            elif samples.dtype == int:
                sample_data = get_sample_for_generate(samples, N_split, max_sample_size, mode = 'sort_by_value')
                stats.update(dict(
                    datatype=Datatype.INT,
                    max=samples.max(),
                    min=samples.min(),
                    mean=samples.mean(),
                    sample_data=sample_data))
            elif samples.dtype == object:
                if column in timestamp_columns:
                    stats.update(dict(datatype=Datatype.DATETIME))
                elif num_unique > categorical_threshold:
                    stats.update(dict(datatype=Datatype.MISC))
                else:
                    stats.update(dict(datatype=Datatype.CATEGORICAL))
                

                freq_words = get_freq_words(samples, min_word_length=3)
                freq_words = get_sample_for_generate(freq_words, N_split, max_sample_size, mode='sort_by_occurance')
                sample_data = get_sample_for_generate(samples, N_split, max_sample_size, mode='sort_by_occurance')
                
                stats.update(dict(
                    sample_data=sample_data,
                    freq_words=freq_words
                ))
            column_stats_table[column] = stats
        joint_column_stats[table] = column_stats_table
        # print(timestamp_columns, character_columns, numeric_columns)
    conn.close()

    # Save the results to a JSON file
    with open(column_stats_path, 'w') as outfile:
        # workaround for numpy and other custom datatypes
        json.dump(joint_column_stats, outfile, cls=CustomEncoder)


if __name__ == "__main__":
    meta_info = pickle.load(open(GenConfig.meta_info_path, 'rb'))
    for dbname in GenConfig.databases:
        generate_column_stats(dbname, meta_info[dbname], save_dir = GenConfig.column_stats_dir,
                            sample_size = 1000000, N_split = 10)