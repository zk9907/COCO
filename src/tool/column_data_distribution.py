import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import psycopg2
import json
import numpy as np
# from datetime import datetime
from scipy.optimize import curve_fit
import datetime as dt
from scipy.stats import median_abs_deviation
from config.GenConfig import GenConfig
def get_bin_distribution_features(values, normalize_method='zscore'):
    if len(values) < 2:
        return [0.0] * 6  

    raw_values = np.array(values, dtype=np.float64)
    values = raw_values.copy()
    
    raw_mean = np.mean(raw_values)
    raw_std = np.std(raw_values)
    if raw_std == 0:
        return [1.0, 0.0, 0.0, 0.0, 1.0, 1]
 
    mad = median_abs_deviation(raw_values, scale='normal')
    dispersion = mad / np.median(raw_values) if np.median(raw_values) != 0 else 0.0
    
    valid_normalization = True
    if normalize_method == 'zscore':
        if raw_std > 1e-8:  
            values = (raw_values - raw_mean) / raw_std
        else:
            valid_normalization = False
    elif normalize_method == 'minmax':
        data_range = np.ptp(raw_values)
        if data_range > 1e-8:
            values = (raw_values - np.min(raw_values)) / data_range
        else:
            valid_normalization = False
    elif normalize_method == 'log':
        if np.min(raw_values) >= 0:
            values = np.log1p(raw_values)
        else:
            valid_normalization = False
    else:
        valid_normalization = False
    
    if not valid_normalization:
        values = raw_values  

    n_bins = int(np.ceil(np.log2(len(values)) + 1)) if len(values) > 0 else 1
    hist, bin_edges = np.histogram(values, bins=n_bins, density=True)
    hist = hist / (np.sum(hist) + 1e-10)  
    
    nonzero_hist = hist[hist > 1e-10]
    if len(nonzero_hist) < 2:
        uniformity = 1.0
    else:
        entropy = -np.sum(nonzero_hist * np.log2(nonzero_hist))
        max_entropy = np.log2(len(nonzero_hist))
        uniformity = entropy / max_entropy if max_entropy > 0 else 1.0

    def _safe_fit(func, xdata, ydata, p0):
        try:
            popt, _ = curve_fit(func, xdata, ydata, p0=p0, maxfev=2000)
            y_pred = func(xdata, *popt)
            ss_res = np.sum((ydata - y_pred)**2)
            ss_tot = np.sum((ydata - np.mean(ydata))**2)
            return max(0.0, min(1.0, 1 - ss_res/(ss_tot + 1e-10)))
        except:
            return 0.0

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    gauss_r2 = _safe_fit(
        lambda x, a, b: a * np.exp(-(x - b)**2 / 0.1),  
        bin_centers, hist, p0=[1.0, 0.5]
    )
    
    power_r2 = _safe_fit(
        lambda x, a: a * np.power(np.clip(x, 1e-5, None), -2),
        bin_centers, hist, p0=[1.0]
    )

    if valid_normalization:
        skew = np.mean(((values - np.mean(values)) / (np.std(values) + 1e-10)) ** 3)
        kurt = np.mean(((values - np.mean(values)) / (np.std(values) + 1e-10)) ** 4) - 3
    else:
        skew = 0.0
        kurt = 0.0
    
    skew_norm = 1 / (1 + np.exp(-abs(skew)))
    kurt_norm = 1 / (1 + np.exp(-abs(kurt)))
    fit_quality = max(gauss_r2, power_r2)
    if gauss_r2 == 0 and power_r2 == 0:
        dist_type = 0
    elif gauss_r2 > power_r2:
        dist_type = 1
    elif gauss_r2 < power_r2:
        dist_type = 2
    else:
        dist_type = 3

    return [
        round(uniformity, 4),
        round(dispersion, 4),
        round(skew_norm, 4),
        round(kurt_norm, 4),
        round(fit_quality, 4),
        dist_type
    ]

def get_columns_dtype(conn, table):
    cur = conn.cursor()
    query = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}';"
    try:
        cur.execute(query)
    except Exception as e:
        print(f"Error retrieving columns for table {table}: {e}")
        return []
    rows = cur.fetchall()
    timestamp_types = {"timestamp without time zone", "timestamp with time zone", "date",'time without time zone'}
    character_types = {"character varying", "varchar", "character", "text"}
    numeric_types = {"integer", "numeric", "smallint", "bigint", "double precision", "real", "decimal"}
    timestamp_columns = []
    character_columns = []
    numeric_columns = []
    for row in rows:
        if row[1] in timestamp_types:
            timestamp_columns.append(row[0])
        elif row[1] in character_types:
            character_columns.append(row[0])
        elif row[1] in numeric_types:
            numeric_columns.append(row[0])
        else:
            print(f"Unknown column type: {row[1]} for column {row[0]}")
            # raise Exception(f"Unknown column type: {row[1]} for column {row[0]}")
    return timestamp_columns, character_columns, numeric_columns

def get_data_from_table(conn, table, column, sample_size = 10000000):
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
        
        
        # Process the rows to get value counts
        value_counts = {}
        for row in rows:
            value = row[0]
            if value is not None:
                if value in value_counts:
                    value_counts[value] += 1
                else:
                    value_counts[value] = 1
        
        return value_counts
    except Exception as e:
        print(f"Error sampling data from {table}.{column}: {e}")
    finally:
        cur.close()

def get_numeric_samples(conn, table, column, sample_size=10000000):
    value_counts = get_data_from_table(conn, table, column, sample_size)
    try:
        value_counts = {float(x):value_counts[x] for x in value_counts if x is not None}
        print(f"Successfully sampled {len(value_counts)} numeric values from {table}.{column}")
    except Exception as e:
        print(f"Could not convert data to float for {table}.{column}: {e}")
    return value_counts

# Function to sample categorical data from a column
def get_character_samples(conn, table, column, sample_size=1000):
    value_counts = get_data_from_table(conn, table, column, sample_size)
    print(f"Successfully sampled {len(value_counts)} character values from {table}.{column}")
    return value_counts

def get_timestamp_samples(conn, table, column, sample_size=1000):
    value_counts = get_data_from_table(conn, table, column, sample_size)
    
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

   

def get_data_features_with_boundaries(value_counts, N_split, isString=False):
    """
    Split data into N_split bins with approximately equal number of data points,
    ensuring same values stay in the same bin, and return features with bin boundaries.
    
    Args:
        data: List of numeric values (already sorted)
        N_split: Number of bins to split data into
        original_values: For categorical data, the original string values (if applicable)
        
    Returns:
        Tuple of (bin_features, bin_boundaries)
        - bin_features: List of feature vectors for each bin
        - bin_boundaries: List of boundary information for each bin
    """
    data_features = []
    bin_boundaries = []
    
    if not value_counts:
        return [[0.0] * 6] * N_split, [[]] * N_split
    if isString:
        unique_values = list(dict(sorted(value_counts.items(), key=lambda item: item[1])).keys())
        # print(unique_values)
    else:
        unique_values = sorted(value_counts.keys())
    
    # Calculate target bin size
    total_points = sum(value_counts.values())
    
    # Create bins by determining appropriate boundaries
    bins = []
    current_bin = []
    current_count = 0
    remaining_bins = N_split
    remaining_points = total_points

    for index, val in enumerate(unique_values): #time
        # Get count for this value
        count = value_counts[val]
        if remaining_bins > 1:
            target_for_current_bin = remaining_points / remaining_bins
            if current_bin and current_count + count > target_for_current_bin:
                # if current_count + count > target_bin_size * 1.2:
                remaining_bins -= 1
                remaining_points -= current_count

                bins.append(current_bin)
                if isString:
                    bin_boundaries.append([unique_values[current_bin[0]], unique_values[current_bin[-1]]])
                else:
                    bin_boundaries.append([current_bin[0], current_bin[-1]])
                current_bin = []
                current_count = 0
            
            # Add all instances of this value to the current bin
        if isString:
            current_bin.extend([index] * count)
        else:
            current_bin.extend([val] * count)
        current_count += count

    # Add the last bin
    if current_bin:
        bins.append(current_bin)
        if isString:
            bin_boundaries.append([unique_values[current_bin[0]], unique_values[current_bin[-1]]])
        else:
            bin_boundaries.append([current_bin[0], current_bin[-1]]) 
    
    # Compute features for each bin
    for bin_data in bins:
        features = get_bin_distribution_features(bin_data)
        data_features.append(features)
    
    # Handle case where we couldn't create N_split bins
    while len(data_features) < N_split:
        data_features.append([0.0] * 6)
        bin_boundaries.append([])
    
    return data_features, bin_boundaries


# Main function: Process numeric, categorical and timestamp columns
def generate_column_data_distribution(dbname, dbInfo, save_dir, sample_size=10000000, N_split=10):
    
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(os.path.join(save_dir, f'{dbname}.json')):
        print(f"Column distribution data already exists for {dbname}")
        return None
    # List the database configurations
    config = {**GenConfig.db_config, 'database':dbname}

    # Initialize the result dictionary with both features and boundaries
    column_data_distribution = {
        "features": {},
        "boundaries": {}
    }

    dbname = config["database"]
    print(f"Processing database: {dbname}")
    try:
        conn = psycopg2.connect(database=config["database"],
                                user=config["user"],
                                password=config["password"],
                                host=config.get("host", "localhost"),
                                port=config.get("port", "5432"))
    except Exception as e:
        print(f"Error connecting to database {dbname}: {e}")
        raise e
    
    tables = dbInfo.keys()
    if not tables:
        print(f"No tables found in database {dbname}")
        conn.close()
        raise Exception(f"No tables found in database {dbname}")
    
    for table in tables:
        print(f"Processing table: {table}")
        # Get column types
        timestamp_columns, character_columns, numeric_columns = get_columns_dtype(conn, table)
        print(f"Numeric columns: {numeric_columns}")
        print(f"Character columns: {character_columns}")
        print(f"Timestamp columns: {timestamp_columns}")
        
        # Process numeric columns
        for col in numeric_columns:
            key = f"{dbname}.{table}.{col}"
            print(f"Processing numeric column: {key}")
            value_counts = get_numeric_samples(conn, table, col, sample_size=sample_size)
            if not value_counts:
                print(f"No numeric data for {key}")
                column_data_distribution["features"][key] = [[0] * 6] * N_split
                column_data_distribution["boundaries"][key] =  [[]] * N_split
                continue
            
            features, boundaries = get_data_features_with_boundaries(value_counts, N_split)
            column_data_distribution["features"][key] = features
            column_data_distribution["boundaries"][key] = boundaries
            print(f"Computed numeric features for {key} with {len(boundaries)} bins")

        # Process character columns
        for col in character_columns:
            key = f"{dbname}.{table}.{col}"
            print(f"Processing character column: {key}")
            value_counts = get_character_samples(conn, table, col, sample_size=sample_size)
            if not value_counts:
                print(f"No character data for {key}")
                column_data_distribution["features"][key] = [[0] * 6] * N_split
                column_data_distribution["boundaries"][key] = [[]] * N_split
                continue
            
            features, boundaries = get_data_features_with_boundaries(value_counts, N_split, isString=True)
            column_data_distribution["features"][key] = features
            column_data_distribution["boundaries"][key] = boundaries
            print(f"Computed character features for {key} with {len(boundaries)} bins")

        # Process timestamp columns
        for col in timestamp_columns:
            key = f"{dbname}.{table}.{col}"
            print(f"Processing timestamp column: {key}")
            value_counts = get_timestamp_samples(conn, table, col, sample_size=sample_size)
            if not value_counts:
                print(f"No timestamp data for {key}")
                column_data_distribution["features"][key] = [[0] * 6] * N_split
                column_data_distribution["boundaries"][key] =  [[]] * N_split
                continue
            
            features, boundaries = get_data_features_with_boundaries(value_counts, N_split)
            
            column_data_distribution["features"][key] = features
            column_data_distribution["boundaries"][key] = boundaries
            print(f"Computed timestamp features for {key} with {len(boundaries)} bins")

    conn.close()

    # Save the results to a JSON file
    output_file = os.path.join(save_dir, f'{dbname}.json')
    with open(output_file, "w") as f:
        json.dump(column_data_distribution, f, indent=4)
    print(f"Column distribution data saved to {output_file}")
    return column_data_distribution


if __name__ == "__main__":

    meta_info = pickle.load(open(GenConfig.meta_info_path, 'rb'))
    for dbname in GenConfig.databases:
        generate_column_data_distribution(dbname, meta_info[dbname]['tableAttr'], 
                                          save_dir = GenConfig.distribution_dir,
                                          sample_size = 1000000, N_split = GenConfig.N_bins) # higher sample_size for higher accuracy
    
    