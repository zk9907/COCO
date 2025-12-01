import ray
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
import logging

class TaskType(Enum):
    MAIN = "main"          
    SQLGEN = "sqlgen" 

class ActorTask:
    def __init__(self, future, task_type: TaskType, task_data: Dict[str, Any]):
        self.future = future
        self.task_type = task_type
        self.task_data = task_data

class ActorManager:
    
    def __init__(self, actors: List):
        self.actors = actors
        
        self.actor_tasks: Dict[Any, Optional[ActorTask]] = {actor: None for actor in actors}
        
        self.main_results: Dict = {}
        
        self.sqlgen_results: Dict = {}
    
    def get_available_actors(self, task_type: Optional[TaskType] = None) -> List:
        available = []
        
        for actor, task in self.actor_tasks.items():
            if task is None:
                available.append(actor)
            else:
                if task_type is not None and task.task_type != task_type:
                    continue

                ready, _ = ray.wait([task.future], timeout=0)
                if ready:
                    try:
                        result = ray.get(task.future)
                        if task.task_type == TaskType.MAIN:
                            self.main_results[task.future] = (result, task.task_data)
                        elif task.task_type == TaskType.SQLGEN:
                            self.sqlgen_results[task.future] = (result, task.task_data)
                        self.actor_tasks[actor] = None
                        available.append(actor)
                        
                    except Exception as e:
                        self.actor_tasks[actor] = None
                        available.append(actor)
        
        return available
    
    def submit_main_task(self, actor, future, task_data: Dict[str, Any]) -> bool:
        if self.actor_tasks[actor] is not None:
            return False
        
        task = ActorTask(future, TaskType.MAIN, task_data)
        self.actor_tasks[actor] = task
        return True
    
    def submit_sqlgen_task(self, actor, future, task_data: Dict[str, Any]) -> bool:

        if self.actor_tasks[actor] is not None:
            return False
        
        task = ActorTask(future, TaskType.SQLGEN, task_data)
        self.actor_tasks[actor] = task
        return True
    
    def wait_any_task(self, task_type: Optional[TaskType] = None, timeout: Optional[float] = None):
        futures_info = []
        for actor, task in self.actor_tasks.items():
            if task is not None:
                if task_type is None or task.task_type == task_type:
                    futures_info.append((task.future, actor, task.task_type, task.task_data))
        
        if not futures_info:
            return None
        
        futures = [info[0] for info in futures_info]
        ready, _ = ray.wait(futures, num_returns=1, timeout=timeout)
        
        if not ready:
            return None
        
        completed_future = ready[0]
        
        for future, actor, ttype, task_data in futures_info:
            if future == completed_future:
                try:
                    result = ray.get(completed_future)
        
                    self.actor_tasks[actor] = None
                    
                    return (completed_future, ttype, task_data, result, actor)
                    
                except Exception as e:
                    self.actor_tasks[actor] = None
                    return None
        
        return None
    
    def get_main_result(self, future):
        return self.main_results.pop(future, None)
    
    def get_sqlgen_result(self, future):
        return self.sqlgen_results.pop(future, None)
    
    def get_all_pending_futures(self, task_type: Optional[TaskType] = None) -> List:
        futures = []
        for actor, task in self.actor_tasks.items():
            if task is not None:
                if task_type is None or task.task_type == task_type:
                    futures.append(task.future)
        return futures
    
    def get_busy_count(self, task_type: Optional[TaskType] = None) -> int:
        count = 0
        for actor, task in self.actor_tasks.items():
            if task is not None:
                if task_type is None or task.task_type == task_type:
                    count += 1
        return count
    
    def get_task_info(self, future) -> Optional[Tuple[TaskType, Dict[str, Any]]]:
        for actor, task in self.actor_tasks.items():
            if task is not None and task.future == future:
                return (task.task_type, task.task_data)
        return None
    
    def check_and_cache_completed_tasks(self) -> int:
        completed_count = 0
        
        for actor, task in list(self.actor_tasks.items()):
            if task is not None:
                ready, _ = ray.wait([task.future], timeout=0)
                if ready:
                    try:
                        result = ray.get(task.future)
                        
                        if task.task_type == TaskType.MAIN:
                            self.main_results[task.future] = (result, task.task_data)
                        elif task.task_type == TaskType.SQLGEN:
                            self.sqlgen_results[task.future] = (result, task.task_data)
                        self.actor_tasks[actor] = None
                        completed_count += 1
                        
                    except Exception as e:
                        self.actor_tasks[actor] = None
        
        return completed_count

