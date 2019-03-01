import numpy as np
import multiprocessing, os, sys, argparse
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--num_actions', type=int, required=True)
    parser.add_argument('-p','--num_proc', type=int, required=True)
    parser.add_argument('--fp', help="output file path", required=True)
    parser.add_argument('-b','--base_cmd', help="base command (default: --cb_explore_adf --epsilon 0.05)", default='--cb_explore_adf --epsilon 0.05')
    parser.add_argument('-r','--rnd_seed_start_while_loop', type=int, default=0)
    parser.add_argument('--fpi', help="input file path", default='')
    parser.add_argument('--dry_run', help="print which blobs would have been downloaded, without downloading", action='store_true')
    
    # Parse input and create variables
    args_dict = vars(parser.parse_args())   # this creates a dictionary with all input CLI
    for x in args_dict:
        locals()[x] = args_dict[x]  # this is equivalent to foo = args.foo

    if os.path.isfile(fp):
        l = [x.split(',"Iter"',1)[0] for x in open(fp)]
        print(len(l))
        
        already_done = set(l)
        print(len(already_done))
        # print(list(already_done)[:5])
    else:
        already_done = set()
    
    # test a single experiment set
    # command = Command(base_cmd, learning_rate=0.005, power_t=0.5, cb_type='dr', interaction_list=['UB'])
    # run_experiment_set([(command.full_command, 5, 1.0, 7, 24), (command.full_command, 5, 0.0, 7, 24)], 20, fp)
    # sys.exit()
    
    num_sim = num_proc * 50
    
    #fpi = r'C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\cb_hyper_simulate_input.csv'
    if os.path.isfile(fpi):
        l = [x.strip() for x in open(fpi)][1:]
        
        skipped = 0
        command_list = []
        for rnd_seed in range(rnd_seed_start_while_loop):
            for x in l:
                pStrategy,baseCost,cb_type,lr = x.split(',')[:4]
                lr = float(lr)
                baseCost = float(baseCost)
                pStrategy = int(pStrategy)
            
                command = Command(base_cmd, learning_rate=lr, cb_type=cb_type, interaction_list=['UB'])

                s = '{{"ml_args":"{}","numActions":{},"baseCost":{},"pStrategy":{},"rewardSeed":{}'.format(command.full_command, num_actions, baseCost, pStrategy, rnd_seed)
                if s not in already_done:
                    command_list.append((command.full_command, num_actions, baseCost, pStrategy, rnd_seed))
                else:
                    skipped += 1

                if len(command_list) == num_sim and not dry_run:
                    run_experiment_set(command_list, num_proc, fp)
                    command_list = []
        print(len(command_list),skipped)
        # print(command_list)
        if dry_run:
            sys.exit()
        run_experiment_set(command_list, num_proc, fp)
        
        rnd_seed = rnd_seed_start_while_loop
        command_list = []
        while True:
            for x in l:
                pStrategy,baseCost,cb_type,lr = x.split(',')[:4]
                lr = float(lr)
                baseCost = float(baseCost)
                pStrategy = int(pStrategy)
            
                command = Command(base_cmd, learning_rate=lr, cb_type=cb_type, interaction_list=['UB'])
                command_list.append((command.full_command, num_actions, baseCost, pStrategy, rnd_seed))
        
                if len(command_list) == num_sim:
                    run_experiment_set(command_list, num_proc, fp)
                    command_list = []

            rnd_seed += 1

    else:
        recorded_prob_types = [0, 1, 2, 6, 7]
        baseCosts = [1.0, 0.0]
        learning_rates = [1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2, 1e-1, 0.5, 1, 10]
        cb_types = ['dr', 'ips', 'dm', 'mtr']
        
        # Regularization, Learning rates, and Power_t rates grid search for both ips and mtr
        command_list = []
        skipped = 0
        for rnd_seed in range(rnd_seed_start_while_loop):
            for baseCost in baseCosts:
                for pStrategy in recorded_prob_types:
                    for cb_type in cb_types:
                        for lr in learning_rates:                                   
                            command = Command(base_cmd, learning_rate=lr, cb_type=cb_type, interaction_list=['UB'])

                            s = '{{"ml_args":"{}","numActions":{},"baseCost":{},"pStrategy":{},"rewardSeed":{}'.format(command.full_command, num_actions, baseCost, pStrategy, rnd_seed)
                            # print(s)
                            # input()
                            if s not in already_done:
                                command_list.append((command.full_command, num_actions, baseCost, pStrategy, rnd_seed))
                            else:
                                skipped += 1

                            if len(command_list) == num_sim and not dry_run:
                                run_experiment_set(command_list, num_proc, fp)
                                command_list = []
        print(len(command_list),skipped)
        if dry_run:
            for x in command_list:
                print(x)
                input()
            sys.exit()
        run_experiment_set(command_list, num_proc, fp)
        
        print('Start while loop...')
        rnd_seed = rnd_seed_start_while_loop
        command_list = []
        while True:
            for baseCost in baseCosts:
                for pStrategy in recorded_prob_types:
                    for cb_type in cb_types:
                        for learning_rate in learning_rates:  
                            command = Command(base_cmd, learning_rate=lr, cb_type=cb_type, interaction_list=['UB'])

                            command_list.append((command.full_command, num_actions, baseCost, pStrategy, rnd_seed))
                            
                            if len(command_list) == num_sim:
                                run_experiment_set(command_list, num_proc, fp)
                                command_list = []
            
            rnd_seed += 1

