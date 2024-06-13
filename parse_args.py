import numpy as np 

if __name__ == '__main__':
    kissat_help = open('kissat_help.txt', 'r')
    lines = kissat_help.readlines()
    kissat_help.close()
    
    # Save 
    help_name = []
    help_type = []
    help_desp = []
    help_deft = []
    
    # Parse the help file
    for line in lines:
        if line.startswith('--'):
            line = line.replace('\n', '')
            line = line[2:]
            help_name.append(line.split('=')[0])
            args_type = line.split('=')[1].split(' ')[0]
            if 'bool' in args_type:
                help_type.append('bool')
            else:
                help_type.append(args_type)
            
            line = line[25:]
            help_desp.append(line.split('[')[0])
            args_deft = line.split('[')[1].split(']')[0]
            help_deft.append(args_deft)
    
    # Save to csv 
    output_path = 'kissat_args.csv'
    f = open(output_path, 'w')
    for i in range(len(help_name)):
        f.write(help_name[i] + ',' + help_type[i] + ',' + help_desp[i] + ',' + help_deft[i] + '\n')
    f.close()