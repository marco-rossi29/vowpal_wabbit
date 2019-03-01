import numpy as np
import multiprocessing, os, sys
from subprocess import check_output, STDOUT

class Command:
    def __init__(self, base, cb_type=None, marginal_list=None, ignore_list=None, interaction_list=None, regularization=None, learning_rate=None, power_t=None, clone_from=None):
        self.base = base
        self.loss = np.inf

        if clone_from is not None:
            # Clone initial values
            self.cb_type = clone_from.cb_type
            self.marginal_list = set(clone_from.marginal_list)
            self.ignore_list = set(clone_from.ignore_list)
            self.interaction_list = set(clone_from.interaction_list)
            self.learning_rate = clone_from.learning_rate
            self.regularization = clone_from.regularization
            self.power_t = clone_from.power_t
        else:
            # Initialize all values to vw default
            self.cb_type = 'ips'
            self.marginal_list = set()
            self.ignore_list = set()
            self.interaction_list = set()
            self.learning_rate = 0.5
            self.regularization = 0
            self.power_t = 0.5

        # Update non-None values (for set we are doing the union not a replacement)
        if cb_type is not None:
            self.cb_type = cb_type
        if marginal_list is not None:
            self.marginal_list.update(marginal_list)
        if ignore_list is not None:
            self.ignore_list.update(ignore_list)
        if interaction_list is not None:
            self.interaction_list.update(interaction_list)
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if regularization is not None:
            self.regularization = regularization
        if power_t is not None:
            self.power_t = power_t

        # Create full_command
        self.full_command = self.base
        if self.learning_rate != 0.5:
            self.full_command += " -l {:g}".format(self.learning_rate)
        self.full_command += " --cb_type {}".format(self.cb_type)
        for interaction in self.interaction_list:
            self.full_command += " -q {}".format(interaction)
        if self.regularization != 0:
            self.full_command += " --l1 {:g}".format(self.regularization)
        if self.power_t != 0.5:
            self.full_command += " --power_t {:g}".format(self.power_t)
        if self.marginal_list:
            self.full_command += " --marginal {}".format(''.join(self.marginal_list))
        for ignored_namespace in self.ignore_list:
            self.full_command += " --ignore {}".format(ignored_namespace)

    def prints(self):
        print("cb type: {}".format(self.cb_type))
        print("marginals: {}".format(self.marginal_list))
        print("ignore list: {}".format(self.ignore_list))
        print("interactions: {}".format(self.interaction_list))
        print("learning rate: {:g}".format(self.learning_rate))
        print("regularization: {:g}".format(self.regularization))
        print("power_t: {:g}".format(self.power_t))
        print("overall command: {0}".format(self.full_command))
        print("loss: {}".format(self.loss))

def run_experiment(args_tuple):

    ml_args, num_actions, base_cost, pStrategy, rnd_seed = args_tuple

    cmd_list = ['C:\\work\\bin\\Simulator_action-per-context_SINGLE\\PerformanceConsole.exe', ml_args]
    cmd_list += ('{} 0.03 0.04 {} {} 100000 10000 {}'.format(num_actions, base_cost, pStrategy, rnd_seed)).split(' ')
    
    try:
        return check_output(cmd_list, stderr=STDOUT, universal_newlines=True)
    except Exception as e:
        print("Error for command {}: {}".format(' '.join(cmd_list), e))
    
def run_experiment_set(command_list, n_proc, fp):
    print('Num of sim:',len(command_list))
    if len(command_list) > 0:
        # Run the experiments in parallel using n_proc processes
        p = multiprocessing.Pool(n_proc)
        results = p.map(run_experiment, command_list, chunksize=1)
        p.close()
        p.join()
        del p
        result_writer(results, fp)

def result_writer(results, fp):
    with open(fp, 'a') as experiment_file:
        for result in results:
            experiment_file.write(result)
        experiment_file.flush()

if __name__ == '__main__':

    fp = r'C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\pippo.txt'

    if os.path.isfile(fp):
        l = ['\t'.join(x.split('\t')[:9]) for i,x in enumerate(open(fp)) if i > 0]
        print(len(l))
        
        already_done = set(l)
        print(len(already_done))
    else:
        already_done = set()
    
    #"--cb_explore_adf --epsilon 0.05 -l 0.005 --cb_type dr -q UB"
    
    base_cmd = '--cb_explore_adf --epsilon 0.05'
    command = Command(base_cmd, learning_rate=0.005, power_t=0.5, cb_type='dr', interaction_list=['UB'])
    
    run_experiment_set([(command.full_command, 5, 1.0, 7, 24), (command.full_command, 5, 0.0, 7, 24)], 20, fp)
    
    sys.exit()
    
    
    fpi = r'C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\cb_hyper_simulate_input.csv'
    if True:
        l = [x.strip() for x in open(fpi)][1:]
        
        skipped = 0
        command_list = []
        for rnd_seed in range(10222):
            for x in l:
                recorded_prob_type,zero_one_cost,power_t,cb_type,lr = x.split(',')[:5]
                power_t = 0 if power_t == '0.0' else float(power_t)
                lr = float(lr)
                zero_one_cost = int(zero_one_cost)
                recorded_prob_type = int(recorded_prob_type)
            
                command = Command(base_cmd, regularization=0, learning_rate=lr, power_t=power_t, cb_type=cb_type)

                s = '\t'.join(map(str, command.full_command.split(' ')[1::2] + [recorded_prob_type, rnd_seed, zero_one_cost]))
                if s not in already_done:
                    command_list.append((command.full_command, recorded_prob_type, rnd_seed, zero_one_cost))
                else:
                    skipped += 1
                    # print(s)
                    # raw_input()

                if len(command_list) == 450:
                    run_experiment_set(command_list, 45, fp)
                    command_list = []
        print(len(command_list),skipped)
        # print(command_list)
        # sys.exit()
        run_experiment_set(command_list, 45, fp)
        
        rnd_seed = 10222
        command_list = []
        while True:
            for x in l:
                recorded_prob_type,zero_one_cost,power_t,cb_type,lr = x.split(',')[:5]
                power_t = 0 if power_t == '0.0' else float(power_t)
                lr = float(lr)
                zero_one_cost = int(zero_one_cost)
                recorded_prob_type = int(recorded_prob_type)
            
                command = Command(base_cmd, regularization=0, learning_rate=lr, power_t=power_t, cb_type=cb_type)
                command_list.append((command.full_command, recorded_prob_type, rnd_seed, zero_one_cost))
        
                if len(command_list) == 450:
                    run_experiment_set(command_list, 45, fp)
                    command_list = []

            rnd_seed += 1

    else:
        recorded_prob_types = [0, 1, 2, 6, 13]
        zero_one_costs = [1, 0]
        learning_rates = [1e-5, 1e-3, 1e-2, 1e-1, 0.5, 1, 10]
        regularizations = [0]
        power_t_rates = [0.5]
        cb_types = ['dr', 'ips', 'dm']
        
        # Regularization, Learning rates, and Power_t rates grid search for both ips and mtr
        command_list = []
        skipped = 0
        for rnd_seed in range(275):
            for zero_one_cost in zero_one_costs:
                for recorded_prob_type in recorded_prob_types:
                    for regularization in regularizations:
                        for cb_type in cb_types:
                            for power_t in power_t_rates:
                                for learning_rate in learning_rates:
                                    command = Command(base_cmd, regularization=regularization, learning_rate=learning_rate, power_t=power_t, cb_type=cb_type)
                                    
                                    s = '\t'.join(map(str, command.full_command.split(' ')[1::2] + [recorded_prob_type, rnd_seed, zero_one_cost]))
                                    if s not in already_done:
                                        command_list.append((command.full_command, recorded_prob_type, rnd_seed, zero_one_cost))
                                    else:
                                        skipped += 1
                                        # print(s)
                                        # raw_input()
        
                                    if len(command_list) == 450:
                                        run_experiment_set(command_list, 45, fp)
                                        command_list = []
        print(len(command_list),skipped)
        # sys.exit()
        run_experiment_set(command_list, 45, fp)
        
        print('Start while loop...')
        rnd_seed = 275
        command_list = []
        while True:
            for zero_one_cost in zero_one_costs:
                for recorded_prob_type in recorded_prob_types:
                    for regularization in regularizations:
                        for cb_type in cb_types:
                            for power_t in power_t_rates:
                                for learning_rate in learning_rates:
                                    command = Command(base_cmd, regularization=regularization, learning_rate=learning_rate, power_t=power_t, cb_type=cb_type)

                                    command_list.append((command.full_command, recorded_prob_type, rnd_seed, zero_one_cost))
                                    
                                    if len(command_list) == 450:
                                        run_experiment_set(command_list, 45, fp)
                                        command_list = []
            
            rnd_seed += 1

