import json
import random

def load_tasks(file_path: str, num_tasks: int, seed: int, min_task_id: int, max_task_id: int) -> list:
    tasks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            task_id = data.get("id", -1)
            
            if min_task_id <= task_id <= max_task_id:
                tasks.append(data)
                
    random.seed(seed)
    random.shuffle(tasks)
    
    return tasks[:num_tasks]