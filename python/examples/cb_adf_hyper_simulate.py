import numpy as np
import multiprocessing, os, sys, argparse, gzip
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

    # cmd_list = ['C:\\work\\bin\\Simulator_action-per-context_SINGLE\\PerformanceConsole.exe', ml_args]
    cmd_list = ['C:\\work\\bin\\Simulator_action-per-context_SINGLE_submodule94e2dbe9_Prob1_error_fix\\PerformanceConsole.exe', ml_args]
    cmd_list += ('{} 0.03 0.04 {} {} 1000000 50000 {}'.format(num_actions, base_cost, pStrategy, rnd_seed)).split(' ')
    
    try:
        x = check_output(cmd_list, stderr=STDOUT, universal_newlines=True)
        # if not x.startswith('{'):
            # x = x.split('\n',1)[1]
        return x
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
        print(results[-1].split('\n')[-2])

def result_writer(results, fp):
    fp_all = fp + '.allLines.txt'
    with (gzip.open(fp_all, 'at') if fp_all.endswith('.gz') else open(fp_all, 'a')) as f:
        for result in results:
            f.write(result)
    with (gzip.open(fp, 'at') if fp.endswith('.gz') else open(fp, 'a')) as f:
        for result in results:
            z = [x for x in result.split('\n') if ',1000000,' in x]
            f.write('\n'.join(z)+'\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-a','--num_actions', type=int, default=-1)
    parser.add_argument('-p','--num_proc', type=int, required=True)
    parser.add_argument('-n','--num_sim', type=int, required=True)
    parser.add_argument('--fp', help="output file path", required=True)
    parser.add_argument('-b','--base_cmd', help="base command (default: --cb_explore_adf --epsilon 0.05)", default='--cb_explore_adf --epsilon 0.05')
    parser.add_argument('-r','--rnd_seed_start_while_loop', type=int, default=0)
    parser.add_argument('--fpi', help="input file path", default='')
    parser.add_argument('--dry_run', help="print which blobs would have been downloaded, without downloading", action='store_true')
    
    # Parse input and create variables
    args_dict = vars(parser.parse_args())   # this creates a dictionary with all input CLI
    for x in args_dict:
        locals()[x] = args_dict[x]  # this is equivalent to foo = args.foo

    already_done = set()
    if os.path.isfile(fp):
        lines = [0,0]
        for x in (gzip.open(fp, 'rt') if fp.endswith('.gz') else open(fp)):
            lines[1] += 1
            
            if ',1000000,' in x:
                lines[0] += 1
                already_done.add(x.split(',1000000,',1)[0])
        print('Total lines: {}\nIter1M lines: {}\nIter1M unique: {}'.format(lines[1],lines[0],len(already_done)))
    else:
        with (gzip.open(fp, 'at') if fp.endswith('.gz') else open(fp, 'a')) as f:
            f.write('MLargs,numActions,baseCost,pStrategy,rewardSeed,Iter,CTR,GoodActions,dA\n')
    
    # test a single experiment set
    # command = Command(base_cmd, learning_rate=0.005, power_t=0.5, cb_type='dr')
    # run_experiment_set([(command.full_command, 5, 1.0, 7, 24), (command.full_command, 5, 0.0, 7, 24)], 20, fp)
    # sys.exit()
    
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
            
                command = Command(base_cmd, learning_rate=lr, cb_type=cb_type)

                s = '{{"ml_args":"{}","numActions":{},"baseCost":{},"pStrategy":{},"rewardSeed":{}'.format(command.full_command, num_actions, baseCost, pStrategy, rnd_seed)
                if s not in already_done:
                    command_list.append((command.full_command, num_actions, baseCost, pStrategy, rnd_seed))
                else:
                    skipped += 1

                if len(command_list) == num_sim and not dry_run:
                    run_experiment_set(command_list, num_proc, fp)
                    command_list = []
        print('len(command_list): {}\nskipped: {}'.format(len(command_list),skipped))
        # print(command_list)
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
            for x in l:
                pStrategy,baseCost,cb_type,lr = x.split(',')[:4]
                lr = float(lr)
                baseCost = float(baseCost)
                pStrategy = int(pStrategy)
            
                command = Command(base_cmd, learning_rate=lr, cb_type=cb_type)
                command_list.append((command.full_command, num_actions, baseCost, pStrategy, rnd_seed))
        
                if len(command_list) == num_sim:
                    run_experiment_set(command_list, num_proc, fp)
                    command_list = []

            rnd_seed += 1

    else:
        recorded_prob_types = [0]
        baseCosts = [1, 0]
        learning_rates = [1e-4, 5e-4, 1e-3, 2e-3, 2.5e-3, 5e-3, 1e-2, 2e-2, 2.5e-2, 5e-2, 1e-1, 2.5e-1, 0.5, 1, 2.5, 5, 10, 100]
        bag_vector = [0, 5, 10, 15]    # bag=0 -> --epsilon 0.05
        power_t_vec = [0, None]
        cb_types = ['mtr', 'dr']
        base_cmd_list = ['--cb_explore_adf --ignore XA -q UB', '--cb_explore_adf --ignore ABU']
        
        # # Regularization, Learning rates, and Power_t rates grid search for both ips and mtr
        # command_list = []
        # skipped = 0
        # for rnd_seed in range(rnd_seed_start_while_loop):
            # for bag in bag_vector:
                # for baseCost in baseCosts:
                    # for pStrategy in recorded_prob_types:
                        # for cb_type in cb_types:
                            # for lr in learning_rates:                                   
                                # command = Command(base_cmd, learning_rate=lr, cb_type=cb_type)

                                # s = '{{"ml_args":"{}","numActions":{},"baseCost":{},"pStrategy":{},"rewardSeed":{}'.format(command.full_command, num_actions, baseCost, pStrategy, rnd_seed)
                                # # print(s)
                                # # input()
                                # if s not in already_done:
                                    # command_list.append((command.full_command, num_actions, baseCost, pStrategy, rnd_seed))
                                # else:
                                    # skipped += 1

                                # if len(command_list) == num_sim and not dry_run:
                                    # run_experiment_set(command_list, num_proc, fp)
                                    # command_list = []
        # print(len(command_list),skipped)
        # if dry_run:
            # for x in command_list:
                # print(x)
                # input()
            # sys.exit()
        # run_experiment_set(command_list, num_proc, fp)
        
        print('Start while loop...')
        rnd_seed = rnd_seed_start_while_loop
        command_list = []
        skipped = 0
        while True:
            for base_cmd in base_cmd_list:
                marginal_list_vec = [None, ['X']] if ' --ignore ABU' in base_cmd else [None]
                for mar in marginal_list_vec:
                    for bag in bag_vector:
                        for baseCost in baseCosts:
                            for pStrategy in recorded_prob_types:
                                for cb_type in cb_types:
                                    for lr in learning_rates:
                                        for pt in power_t_vec:
                                        
                                            base_cmd2 = base_cmd
                                            if bag > 0:
                                                base_cmd2 += ' --bag ' + str(bag)
                                            command = Command(base_cmd2, learning_rate=lr, cb_type=cb_type, power_t=pt, marginal_list=mar)

                                            s = ','.join(map(str, [command.full_command, num_actions, baseCost, pStrategy, rnd_seed]))
                                            # print(s)
                                            # input()
                                            if s not in already_done:
                                                command_list.append((command.full_command, num_actions, baseCost, pStrategy, rnd_seed))
                                            else:
                                                skipped += 1

                                            if len(command_list) == num_sim and not dry_run:
                                                run_experiment_set(command_list, num_proc, fp)
                                                command_list = []
            
            rnd_seed += 1
            
            if dry_run and rnd_seed == 16:
                print(len(command_list),skipped)
                for x in command_list:
                    print(x)
                    input()
                break

            
# python C:\work\vw\python\examples\cb_adf_hyper_simulate.py -a 2 -p 43 -n 430 -r 10000 --fp "C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\myFile_Actions2.txt.gz"
# python C:\work\vw\python\examples\cb_adf_hyper_simulate.py -p 43 -n 430 -b "--cb_explore_adf --epsilon 0.05 --marginal A" --fp "C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions2-10_marginal_noQ.txt"

# python C:\work\vw\python\examples\cb_adf_hyper_simulate.py -p 43 -n 430 -b "--cb_explore_adf --ignore ABU" --fp "C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_marginal_pt_bag.txt" -a 10
# python C:\work\vw\python\examples\cb_adf_hyper_simulate.py -a 10 -p 44 -n 440 -b "--cb_explore_adf --ignore XA -q UB" --fp "C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_ALL.txt"
