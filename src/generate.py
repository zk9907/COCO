import argparse
import ray
import random
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SQLGen.SQLGenEnv import SQLGenEnv
from config.SQLGenConfig import Config as SQLGenConfig
from config.LCQOConfig import Config as LCQOConfig
from config.GenConfig import GenConfig
from util.SQLBuilder import SQLBuilderActor
from util.SQLBuffer import SQLBuffer, SQL
sqlgen_config = SQLGenConfig()
env = SQLGenEnv()
sql_buffer = SQLBuffer()
os.makedirs('./TestWorkload', exist_ok=True)


parser = argparse.ArgumentParser(description="Random SQL Generation")
parser.add_argument(
    "--target-per-db",
    type=int,
    default=5,
)
parser.add_argument(
    "--save-path",
    type=str,
    default=GenConfig.random_source_workloads_path,
)
args, _ = parser.parse_known_args()
sql_buffer_path = args.save_path
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)

builder_actors = []
for dbConfig in GenConfig.remote_db_config_list:
    lqo_config_instance = LCQOConfig()
    lqo_config_instance.db_config = dbConfig
    builder_actors.append(SQLBuilderActor.remote(lqo_config_instance))
generate_database = GenConfig.databases
if not builder_actors:
    print("CRITICAL ERROR: No SQLBuilderActors are configured. Exiting.")
    exit(1)

print(f"Initialized {len(builder_actors)} SQLBuilderActors.")

pending = {}  

# Initialize counts, then attempt to load and update from buffer
db_success_counts = {db_name: 0 for db_name in generate_database}
global_successful_buffered_queries = [0] # Using list for mutable integer

if os.path.exists(sql_buffer_path):
    try:
        print(f"Attempting to load SQL buffer from {sql_buffer_path}...")
        # load_state returns True on success, False if file not found (though os.path.exists should catch that)
        # or potentially raises an exception on other load errors.
        if sql_buffer.load_state(sql_buffer_path):
            print(f"Buffer file {sql_buffer_path} loaded successfully.")
            # Recalculate counts from the loaded buffer's contents
            temp_loaded_global_queries = 0
            # Ensure db_success_counts is clean before recounting
            db_success_counts = {db_name: 0 for db_name in generate_database} 
            for sql_item in sql_buffer.buffer: # Iterate through the actual list of SQL objects
                if sql_item is not None: # Buffer might not be full or contain None placeholders
                    if sql_item.dbName in db_success_counts:
                        db_success_counts[sql_item.dbName] += 1
                    else:
                        # This case should ideally not happen if generate_database covers all dbs in buffer
                        print(f"[WARN] Encountered SQL for unexpected dbName '{sql_item.dbName}' not in generate_database list during load. It will be counted.")
                        db_success_counts[sql_item.dbName] = 1 # Or handle as an error/skip
                    temp_loaded_global_queries += 1
            
            global_successful_buffered_queries[0] = temp_loaded_global_queries
            print(f"Buffer loaded. Total queries recounted from buffer: {global_successful_buffered_queries[0]}.")
            print(f"Recalculated counts per database from loaded buffer: {db_success_counts}")
        else:
            # This case implies load_state itself returned False, e.g. file existed but was perhaps empty/corrupt in a way load_state handles by returning False
            print(f"SQL buffer file {sql_buffer_path} found, but load_state indicated a problem (e.g. empty/corrupt). Starting fresh.")
            db_success_counts = {db_name: 0 for db_name in generate_database}
            global_successful_buffered_queries[0] = 0

    except Exception as e:
        print(f"Error loading or processing SQL buffer from {sql_buffer_path}: {e}. Starting with fresh counts (zeros).")
        db_success_counts = {db_name: 0 for db_name in generate_database}
        global_successful_buffered_queries[0] = 0
else:
    print(f"SQL buffer file {sql_buffer_path} not found. Starting with fresh counts (zeros).")
    # db_success_counts and global_successful_buffered_queries are already initialized to zeros

target_per_db = args.target_per_db
total_target_successful_queries = len(generate_database) * target_per_db
queries_generated_and_dispatched_attempts = 0 # Tracks total attempts dispatched

print(f"Starting interactive SQL generation and dispatch. Target: {total_target_successful_queries} successful queries in buffer.",flush=True)
print(f"Current successful queries in buffer (after potential load): {global_successful_buffered_queries[0]}.")
print(f"Current success counts per database (after potential load): {db_success_counts}.", flush=True)

def process_completed_task(future, pending_dict_ref, success_counts_per_db_ref, total_successful_counter_ref, total_target_q_gen, sql_b):
    actor_obj, c_dbName, c_submission_time, c_sql_statement = pending_dict_ref.pop(future)
    try:
        sql_obj_result = ray.get(future)
        elapsed_time = time.time() - c_submission_time
        if isinstance(sql_obj_result, SQL):
            sql_obj_result:SQL
            # Ensure the result's dbName matches the intended one; usually will.
            # Using c_dbName for tracking success for the intended database.
            print(f"  [Completed] Intended_DB: {c_dbName}, Actual_DB: {sql_obj_result.dbName}, BaseLatency: {sql_obj_result.base_latency:.2f} ms, min_latency: {sql_obj_result.min_latency:.2f} ms, BuildTime: {elapsed_time:.2f} s. SQL: {sql_obj_result.sql_statement[:60]}...",flush=True)
            sql_obj_result.update_reward(0.0)  # Placeholder reward update
            sql_b.push(sql_obj_result)  # Using 0 for position
            success_counts_per_db_ref[c_dbName] += 1
            total_successful_counter_ref[0] += 1
            print(f"    DB '{c_dbName}' successes: {success_counts_per_db_ref[c_dbName]}/{target_per_db}. Total successful: {total_successful_counter_ref[0]}/{total_target_q_gen}.",flush=True)
        else:
            # print(sql_obj_result)
            print(f"  [WARN] SQL build failed/timed out for {c_dbName}. BuildTime: {elapsed_time:.2f}s. SQL: {c_sql_statement[0][:60]}...",flush=True)
    except Exception as e:
        elapsed_time = time.time() - c_submission_time
        print(f"  [ERROR] Processing task for {c_dbName} after {elapsed_time:.2f}s: {e}. SQL: {c_sql_statement[0][:60]}...",flush=True)
    finally:
        # This finally block might not be the best place for overall progress if only successful tasks count.
        # The print inside the if sql_obj_result is not None block now gives better progress.
        print(f"  Task processed for {c_dbName}. Pending tasks: {len(pending_dict_ref)}.",flush=True)

# Main generation and dispatch loop
for dbName_idx, dbName in enumerate(generate_database):
    attempts_for_this_db = 0
    
    while db_success_counts[dbName] < target_per_db:
        attempts_for_this_db += 1
        queries_generated_and_dispatched_attempts += 1

        print(f"Generating for DB '{dbName}' (Successes: {db_success_counts[dbName]}/{target_per_db}). Attempt #{attempts_for_this_db} for this DB (Global attempt #{queries_generated_and_dispatched_attempts}).",flush=True)

        # 1. Generate SQL query and its episode data
        state, _ = env.reset(options={'dbName': dbName})
        action_mask = state['action_mask']
        done = False
        while not done:
            valid_actions = [i for i, mask_val in enumerate(action_mask) if mask_val == 1]
            action = random.choice(valid_actions) if valid_actions else 0
            next_state, reward, done, _, info = env.step(action)
            current_action_mask_for_storage = action_mask
            action_mask = next_state['action_mask']
        
            state = next_state
        sql_statement = env.get_query()
        # print(f"  Query generated: {sql_statement[:100]}...")

        # 2. Wait for an available actor if all are currently busy.
        #    Also process any completed tasks during this wait, which might fulfill the current DB's quota.
        while len(pending) >= len(builder_actors) and db_success_counts[dbName] < target_per_db:
            print(f"  All {len(builder_actors)} actors busy. DB '{dbName}' needs {target_per_db-db_success_counts[dbName]} more. Waiting for an actor. Pending tasks: {len(pending)}",flush=True)
            ready_futures, _ = ray.wait(list(pending.keys()), num_returns=1) # Blocking wait for one task
            for fut in ready_futures:
                process_completed_task(fut, pending, db_success_counts, global_successful_buffered_queries, total_target_successful_queries, sql_buffer)
        
        if db_success_counts[dbName] >= target_per_db:
            print(f"  DB '{dbName}' reached {target_per_db} successes while waiting for actor or processing other tasks. Moving to next DB or finishing.",flush=True)
            break # This DB is done, break from its attempt loop.

        # 3. Dispatch the newly generated task to an available actor
        chosen_actor = None
        shuffled_builder_actors = list(builder_actors) 
        random.shuffle(shuffled_builder_actors)
        for actor_candidate in shuffled_builder_actors:
            is_busy = any(actor_candidate == p_actor for p_actor, _,  _, _ in pending.values())
            if not is_busy:
                chosen_actor = actor_candidate
                break
        
        if chosen_actor:
            new_fut = chosen_actor.test_build_sql.remote(dbName, sql_statement[0],sql_infos = sql_statement[1])
            pending[new_fut] = (chosen_actor, dbName, time.time(), sql_statement)
            print(f"  Dispatched attempt #{attempts_for_this_db} for {dbName} (Global attempt #{queries_generated_and_dispatched_attempts}) to actor. Pending tasks: {len(pending)}.")
        else:
            # This should ideally not be reached if len(builder_actors) > 0
            # because the while loop above should free an actor or db_success_counts[dbName] should be >= 100.
            print(f"  CRITICAL LOGIC ERROR: No available actor found for dispatch. DB: {dbName}, Successes: {db_success_counts[dbName]}. Pending: {len(pending)}. This should not happen. Skipping this attempt cycle.")
            time.sleep(0.1) # Pause briefly before potentially retrying the current DB loop iteration.
        if attempts_for_this_db > 0 and attempts_for_this_db % 20 == 0:
            sql_buffer.save_state(sql_buffer_path)
# 4. After all DBs have met their 100 success quota, wait for any remaining dispatched tasks
print(f"All databases have reached their target of {target_per_db} successful SQLs. Total successful: {global_successful_buffered_queries[0]}/{total_target_successful_queries}.")
print(f"Waiting for any remaining {len(pending)} outstanding dispatched tasks to complete...")
while pending:
    num_to_wait = min(len(pending), len(builder_actors) if builder_actors else 1)
    ready_futures, _ = ray.wait(list(pending.keys()), num_returns=num_to_wait) 
    if not ready_futures and pending: 
        print(f"  Still waiting for {len(pending)} tasks, no tasks completed in last check cycle...")
        time.sleep(0.5)
        continue
    for fut in ready_futures:
        # Note: db_success_counts is still passed, though all DBs should be > 100. This is for consistency.
        process_completed_task(fut, pending, db_success_counts, global_successful_buffered_queries, total_target_successful_queries, sql_buffer)
sql_buffer.save_state(sql_buffer_path)
print(f"All SQL query processing complete. Total successful queries in buffer: {global_successful_buffered_queries[0]}/{total_target_successful_queries}.")
print(f"Success counts per database: {db_success_counts}")
print(f"Total query generation & dispatch attempts: {queries_generated_and_dispatched_attempts}")
