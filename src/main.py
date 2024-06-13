import os 
import glob 
import numpy as np
import random
import time 
from datetime import datetime

import utils.utils as utils 
from utils.parallel_utils import pararun_command

KISSAT_ARGS = 'kissat_args.csv'
KISSAT = './src/kissat/build/kissat'
TIMEOUT = 10
TO_DEC = 1e9

CASE_DIR = '/Users/zhengyuanshi/studio/dataset/LEC/all_case_cnf'
CASE_LIST = [
    'ab9', 'aa9', 'a1', 
    'b2', 'a11', 'h7', 
    'a5', 'a28', 'd26', 
    'ad37', 'c13', 'f7', 
    
    # 'mult_op_DEMO1_12_12_TOP23', 'mult_op_DEMO1_12_12_TOP24', 'mult_op_DEMO1_12_12_TOP9', 
    # 'mult_op_DEMO1_12_12_TOP8', 'mult_op_DEMO1_12_12_TOP7', 'mult_op_DEMO1_12_12_TOP6'
    
]
# CASE_DIR = './case'
# CASE_LIST = [
#     'mchess16-mixed-45percent-blocked'
# ]

class GA():
    def __init__(self, 
                 nums = 100,    # number of individuals 
                 cross_rate = 0.8, # mating probability
                 mutation_rate_dict = {}, # mutation probability
                 args_list = [], # list of kissat args
                 deft_set = [], # default setting of kissat args
                 log_dir = './log', # log file dir, 
                 case_list = [], # list of case name
                 case_dir = './case', # case dir
                 ) -> None:
        # GA Initialization
        self.nums = nums
        self.dna_size = len(args_list)
        self.cross_rate = cross_rate
        self.mutation_rate_dict = mutation_rate_dict
        self.mutation_rate = self.mutation_rate_dict[0]
        self.args_list = args_list
        self.pop = np.random.randint(2, size=(nums-1, self.dna_size))
        self.pop = np.vstack([np.array(deft_set), self.pop])
        self.parent_split = int(self.nums * (1-self.cross_rate))
        
        # Head of log file
        self.log_path = os.path.join(log_dir, 'log_{:}_{:}_{:}_{:}_{:}.txt'.format(
            datetime.now().month, datetime.now().day, datetime.now().hour, 
            datetime.now().minute, datetime.now().second
        ))
        self.log_file = open(self.log_path, 'w')
        self.log_file.write('################################### \n')
        self.log_file.write('# Initial population: {:} \n'.format(self.nums))
        self.log_file.write('# Crossover rate: {:} \n'.format(self.cross_rate))
        self.log_file.write('# Mutation rate: {:} \n'.format(self.mutation_rate))
        self.log_file.write('################################### \n')
        self.log_file.write('\n')
        self.log('Generation, # DEC, Best, Worst, Mean, Std, Time')
        self.generation = 0
        
        # Record best
        self.best_idv = []
        self.best_fit_val = 0
        
        # CNF Initialization
        self.cnf_paths = []
        self.baseline_no_dec = []
        self.baseline_time = []
        for case_name in case_list:
            _cnf_path = os.path.join(case_dir, '{}.cnf'.format(case_name))
            if not os.path.exists(_cnf_path):
                print('[WRAN] Case not found: {:}'.format(_cnf_path))
                continue
            self.cnf_paths.append(_cnf_path)
            self.baseline_no_dec.append(TO_DEC)
            self.baseline_time.append(TIMEOUT)
        self.init_solve()
            
    def init_solve(self):
        for case_k, case in enumerate(self.cnf_paths):
            solve_cmd = '{} -q {} '.format(KISSAT, case)
            solve_cmd += '--time={}'.format(TIMEOUT)
            stdout, solve_time = utils.run_command(solve_cmd)
            if solve_time >= TIMEOUT:
                print('[WARN] Timeout: {:}'.format(case))
                continue
            no_dec = stdout[-1].replace(' ', '').replace('\n', '').split(':')[-1]
            no_dec = int(no_dec)
            self.baseline_no_dec[case_k] = no_dec
            self.baseline_time[case_k] = solve_time
        
    def get_fitness(self):
        fit_val = []    # Average reduction of variable decision times
        # Solving 
        for i in range(self.nums):
            red_list = []
            cmd_list = []
            for case_k, case in enumerate(self.cnf_paths):
                case_TO = min(int(np.ceil(self.baseline_time[case_k] * 10)), TIMEOUT)
                idv = self.pop[i]
                solve_cmd = '{} -q {} '.format(KISSAT, case)
                for args_k, args_name in enumerate(self.args_list):
                    solve_cmd += '--{}={} '.format(args_name, idv[args_k])
                solve_cmd += '--time={}'.format(case_TO)
                cmd_list.append(solve_cmd)
            stdout_list, runtime_list = pararun_command(cmd_list)
            for case_k, case in enumerate(self.cnf_paths):
                stdout = stdout_list[case_k]
                solve_time = runtime_list[case_k]
                if solve_time >= case_TO:
                    no_dec = TO_DEC
                else:
                    no_dec = stdout[-1].replace(' ', '').replace('\n', '').split(':')[-1]
                    no_dec = int(no_dec)
                no_dec_red = (self.baseline_no_dec[case_k] - no_dec) / self.baseline_no_dec[case_k]
                red_list.append(no_dec_red)
            fit_val.append(np.mean(red_list))
        fit_val = np.array(fit_val)
        self.fitness_dec_red = fit_val.copy()
        
        # Find best 
        for i in range(self.nums):
            if fit_val[i] >= self.best_fit_val:
                self.best_fit_val = fit_val[i]
                self.best_idv = self.pop[i]
        # Calculate fitness
        fit_val = (fit_val - np.min(fit_val)) / (np.max(fit_val) - np.min(fit_val) + 1e-10)
        fit_val += 1e-10
        # fit_val = np.exp(fit_val) / np.sum(np.exp(fit_val)) # softmax
        self.fitness = fit_val
        return self.fitness
    
    def select(self):
        fitness = self.get_fitness()
        self.pop = self.pop[np.random.choice(np.arange(self.nums), size=self.nums, replace=True, p=fitness/fitness.sum())]
    
    def crossover(self):
        for i in range(self.nums):
            if np.random.rand() < self.cross_rate:
                j = np.random.randint(0, self.nums)
                cross_points = np.random.randint(0, 2, self.dna_size)
                self.pop[i, cross_points] = self.pop[j, cross_points]
    
    def mutate(self):
        if self.generation in self.mutation_rate_dict.keys():
            self.mutation_rate = self.mutation_rate_dict[self.generation]
            print('[INFO] Mutation rate changed: {:.4f}'.format(self.mutation_rate))
        for i in range(self.nums):
            for j in range(self.dna_size):
                if np.random.rand() < self.mutation_rate:
                    self.pop[i, j] = 1 if self.pop[i, j] == 0 else 0
    
    def evolve(self, n_generations = 1000, n_best = 5):
        self.generation = 0
        while self.generation < n_generations:
            start_time = time.time()
            self.select()
            self.crossover()
            self.mutate()
            gen_time = time.time() - start_time
            self.generation += 1
            dec_red = self.fitness_dec_red  # Not real fitness value, Average reduction of variable decision times
            self.log(
                '{:}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2f}s'.format(
                    self.generation, np.max(dec_red), np.min(dec_red), np.mean(dec_red), np.std(dec_red), gen_time
                )
            )
        # Save best 
        self.get_fitness()
        dec_red = self.fitness_dec_red # Not real fitness value, Average reduction of variable decision times
        self.log_file.write('\n')
        self.log_file.write('################################### \n')
        best_index = np.argsort(dec_red)[-n_best:]
        for k in range(n_best):
            self.log_file.write('Best {:} [{:.4f}]: '.format(k+1, dec_red[best_index[k]]))
            for i in range(self.dna_size):
                self.log_file.write('{} '.format(self.pop[best_index[k]][i]))
            self.log_file.write('\n')
        self.log_file.write('################################### \n')
        self.log_file.write('Best: ')
        for i in range(self.dna_size):
            self.log_file.write('{} '.format(self.best_idv[i]))
        self.log_file.write('\n')
    
    def log(self, txt=''):
        self.log_file.write(txt)
        self.log_file.write('\n')
        self.log_file.flush()
        print(txt)
    
if __name__ == '__main__':
    # Parse kissat args 
    bool_args = []
    deft_set = []
    kissat_help = open('kissat_args.csv', 'r')
    lines = kissat_help.readlines()
    kissat_help.close()
    for line in lines:
        arr = line.replace('\n', '').split(',')
        if arr[1] == 'bool':
            bool_args.append(arr[0])
            if arr[-1] == 'true':
                deft_set.append(1)
            else:
                deft_set.append(0)
            
    # GA
    ga = GA(
        nums=20, cross_rate=0.7, 
        mutation_rate_dict={0: 0.005, 20: 0.01}, 
        args_list=bool_args, deft_set=deft_set,
        case_list=CASE_LIST, case_dir=CASE_DIR
    )
    ga.evolve(100, 5)
    print()
    