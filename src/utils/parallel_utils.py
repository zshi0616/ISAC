import os 
import threading
import time 
from queue import Queue

def run_command(command):
    start_time = time.time()
    stdout = os.popen(command)
    stdout = stdout.readlines()
    runtime = time.time() - start_time
    return stdout, runtime

def job(cmd, thread_idx, ret_info):
    stdout, runtime = run_command(cmd)
    ret_info[thread_idx] = (stdout, runtime)

def pararun_command(cmd_list):
    threads = []
    no_threads = len(cmd_list)
    ret_info = [[] for _ in range(no_threads)]
    assert no_threads < 32, 'Too many threads'
    stdout_list = []
    runtime_list = []
    
    # Run in parallel
    for thread_idx, cmd in enumerate(cmd_list):
        thread = threading.Thread(target=job, args=(cmd, thread_idx, ret_info, ))
        threads.append(thread)
    for i in range(0, len(threads), no_threads):
        for thread in threads[i:i+no_threads]:
            thread.start()
        for thread in threads[i:i+no_threads]:
            thread.join()
    
    # Results 
    for thread_idx, thread in enumerate(threads):
        stdout, runtime = ret_info[thread_idx]
        stdout_list.append(stdout)
        runtime_list.append(runtime)
    
    return stdout_list, runtime_list
        
if __name__ == "__main__":
    cmd_list = [
        '/Users/zhengyuanshi/studio/retrieval_sat/src/kissat/build/kissat -q /Users/zhengyuanshi/studio/dataset/LEC/all_case_cnf/mult_op_DEMO1_11_11_TOP10.cnf', 
        '/Users/zhengyuanshi/studio/retrieval_sat/src/kissat/build/kissat -q /Users/zhengyuanshi/studio/dataset/LEC/all_case_cnf/f27.cnf', 
        '/Users/zhengyuanshi/studio/retrieval_sat/src/kissat/build/kissat -q /Users/zhengyuanshi/studio/dataset/LEC/all_case_cnf/c27.cnf', 
        '/Users/zhengyuanshi/studio/retrieval_sat/src/kissat/build/kissat -q /Users/zhengyuanshi/studio/dataset/LEC/all_case_cnf/e18.cnf', 
    ]

    start_time = time.time()
    stdout_list, runtime_list = run_mult_command(cmd_list)
    runtime = time.time() - start_time
    print('Parallel: {:.2f}s'.format(runtime))
    
    # Serial 
    start_time = time.time()
    for cmd in cmd_list:
        stdout, runtime = run_command(cmd)
    runtime = time.time() - start_time
    print('Serial: {:.2f}s'.format(runtime))
    