import re
import os
import sys
from collections import defaultdict
import pprint

def process_log(log_content):
    # 匹配形如 [W(0)T(31)] 的标识符
    pattern = r'\[W\((\d+)\)T\((\d+)\)\]'
    warp_thread_id_ranges = []
    
    for m in re.finditer(pattern, log_content):
        start_pos = m.start()          
        end_pos = m.end()
        warp_id, thread_id = map(int, m.groups())
        prefix_range = (start_pos, end_pos)
        warp_thread_id = (warp_id, thread_id)
        warp_thread_id_ranges.append((prefix_range, warp_thread_id))
    
    # 按warp分组，每个warp包含多个线程的日志
    warp_logs = defaultdict(dict)  # {warp_id: {thread_id: log_content}}
    
    for i in range(len(warp_thread_id_ranges)):
        warp_id, thread_id = warp_thread_id_ranges[i][1]
        end_pos = warp_thread_id_ranges[i][0][1]
        
        # 确定当前日志的结束位置
        if i != len(warp_thread_id_ranges) - 1:
            next_start_pos = warp_thread_id_ranges[i + 1][0][0]
        else:
            next_start_pos = len(log_content)
        
        # 提取当前线程的日志内容
        thread_log = log_content[end_pos:next_start_pos]
        
        # 将日志添加到对应的warp和线程
        if thread_id not in warp_logs[warp_id]:
            warp_logs[warp_id][thread_id] = ""
        warp_logs[warp_id][thread_id] += thread_log

    # 为每个warp创建一个文件，按线程ID顺序输出
    for warp_id, threads in warp_logs.items():
        filename = f"W{warp_id}.log"
        with open(filename, 'w') as f:
            # 按线程ID从小到大排序
            for thread_id in sorted(threads.keys()):
                f.write(f"=== Thread {thread_id} ===\n")
                f.write(threads[thread_id])
                f.write(f"\n=== End Thread {thread_id} ===\n\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
        with open(log_file, 'r') as f:
            log_content = f.read()
    else:
        log_content = sys.stdin.read()
    
    process_log(log_content)