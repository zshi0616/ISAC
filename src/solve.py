import os 
import glob 
import numpy as np
import random
import time 
from datetime import datetime

import utils.utils as utils 

KISSAT_ARGS = 'kissat_args.csv'
KISSAT = './src/kissat/build/kissat'
TIMEOUT = 1000
FLIP_BITS = 1

CASE_DIR = '/Users/zhengyuanshi/studio/dataset/LEC/all_case_cnf'
CASE_LIST = [
    'f28', 'h21', 'f13', 'aa1', 'ac3', 'ac1', 'ad14'
    # 'mult_op_DEMO1_12_12_TOP18', 'mult_op_DEMO1_12_12_TOP17', 'mult_op_DEMO1_12_12_TOP13', 
    
    # 'mult_op_DEMO1_9_9_TOP12', 'mult_op_DEMO1_9_9_TOP11', 
    # 'mult_op_DEMO1_10_10_TOP13', 'mult_op_DEMO1_10_10_TOP12', 'mult_op_DEMO1_10_10_TOP14'
]
# CASE_DIR = './case'
# CASE_LIST = [
#     'mchess16-mixed-45percent-blocked'
# ]

# config = '0 1 1 1 0 1 0 1 1 0 1 1 0 1 0 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 0 '         # Mul-Adder
config = '0 1 1 1 1 0 0 0 1 0 0 0 0 1 1 1 1 0 1 0 1 0 1 0 0 1 0 0 1 1 1 1 '         # Multipler
 
if __name__ == '__main__':
    # Parse kissat args 
    bool_args = []
    kissat_help = open('kissat_args.csv', 'r')
    lines = kissat_help.readlines()
    kissat_help.close()
    for line in lines:
        arr = line.split(',')
        if arr[1] == 'bool':
            bool_args.append(arr[0])
    
    for case_name in CASE_LIST:
        cnf_path = os.path.join(CASE_DIR, '{}.cnf'.format(case_name))
        if not os.path.exists(cnf_path):
            print('[WRAN] Case not found: {:}'.format(cnf_path))
            continue
        # Parse configuration 
        config_list = config.split(' ')
        # flip_indexs = np.random.choice(np.arange(len(config_list)), size=FLIP_BITS, replace=True)
        # for flip_idx in flip_indexs:
        #     print('Flip {} from {} to {}'.format(
        #         bool_args[flip_idx], config_list[flip_idx], '1' if config_list[flip_idx] == '0' else '0'
        #     ))
        #     config_list[flip_idx] = '1' if config_list[flip_idx] == '0' else '0'
        
        # Solve 
        solve_cmd = '{} -q {} '.format(KISSAT, cnf_path)
        solve_cmd += '--time={} '.format(TIMEOUT)
        baseline_cmd = solve_cmd
        for args_k, args_name in enumerate(bool_args):
            solve_cmd += '--{}={} '.format(args_name, config_list[args_k])
        stdout, solve_time = utils.run_command(solve_cmd)
        no_dec = int(stdout[-1].replace(' ', '').replace('\n', '').split(':')[-1])
        print('Case: {:}, Solve Time: {:.2f}s, # DEC: {}'.format(case_name, solve_time, no_dec))
        
        # Baseline 
        stdout, baseline_time = utils.run_command(baseline_cmd)
        baseline_no_dec = int(stdout[-1].replace(' ', '').replace('\n', '').split(':')[-1])
        print('Case: {:}, Baseline Time: {:.2f}s, # DEC: {}'.format(case_name, baseline_time, baseline_no_dec))
        print('Speedup: {:.2f}%'.format((baseline_time - solve_time) * 100 / baseline_time))
        print('# DEC Red.: {:.2f}%'.format((baseline_no_dec - no_dec) * 100 / baseline_no_dec))
        print()
    