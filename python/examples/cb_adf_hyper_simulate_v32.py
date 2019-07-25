import numpy as np
import multiprocessing, os, sys, argparse, gzip, datetime
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

def get_cmd_str(cmd_list):
    return cmd_list[0] + ' \"{}\" '.format(cmd_list[1]) + ' '.join(cmd_list[2:])

def run_experiment(cmd_list):
    
    try:
        x = check_output(cmd_list, stderr=STDOUT, universal_newlines=True)
        return get_cmd_str(cmd_list) + '\n' + x + '\n'
    except Exception as e:
        err_str = 'Error for command {}: {}'.format(get_cmd_str(cmd_list), e)
        print(err_str)
        return err_str
    
def run_experiment_set(command_list, n_proc, fp):
    print('Num of sim:',len(command_list))
    if len(command_list) > 0:
        t0 = datetime.datetime.now()
        print('Current time: {}'.format(t0))
        last_cmd = command_list[-1]
        command_list.sort(key=lambda x : x[2], reverse=True)    # sort descending by number of actions

        # Run the experiments in parallel using n_proc processes
        p = multiprocessing.Pool(n_proc)
        results = p.map(run_experiment, command_list, chunksize=1)
        p.close()
        p.join()
        del p
        result_writer(results, fp)
        print('Elapsed: {} - Last cmd: {}'.format(datetime.datetime.now()-t0, get_cmd_str(last_cmd)))

def result_writer(results, fp):
    with (gzip.open(fp, 'at') if fp.endswith('.gz') else open(fp, 'a')) as f:
        for result in results:
            if result.startswith('Error for command'):
                with open(fp+'.errors.txt', 'a') as fe:
                    fe.write(result+'\n')
            else:
                f.write(result)
                                

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--num_proc', type=int, required=True)
    parser.add_argument('-n','--num_sim', type=int, required=True)
    parser.add_argument('--fo', help="output file path", required=True)
    parser.add_argument('--dry_run', help="print which blobs would have been downloaded, without downloading", action='store_true')
    
    # Parse input and create variables
    args_dict = vars(parser.parse_args())   # this creates a dictionary with all input CLI
    for x in args_dict:
        locals()[x] = args_dict[x]  # this is equivalent to foo = args.foo
    
    # test a single experiment set
    # command = Command(base_cmd, learning_rate=0.005, power_t=0.5, cb_type='dr')
    # run_experiment_set([(command.full_command, 5, 1.0, 7, 24), (command.full_command, 5, 0.0, 7, 24)], 20, fp)
    # sys.exit()
    
    already_done = set()
    if os.path.isfile(fo):
        lines = [0,0]
        for x in (gzip.open(fo, 'rt') if fo.endswith('.gz') else open(fo)):
            lines[1] += 1
            
            if 'simulator.exe' in x:
                lines[0] += 1
                already_done.add(x.strip().split('simulator.exe',1)[1])
        print('Total lines: {}\nCMD lines: {}\nCMD unique: {}'.format(lines[1],lines[0],len(already_done)))
        if lines[0] > len(already_done):
            print('Duplicates CMD lines!')
            input()
    
    files = [x.path+'\\simulator.exe' for x in os.scandir(r'C:\work\bin\cs_sim_SINGLE') if x.name.startswith('Simulator_v') and 'Simulator_v35_870_82da0b7797abe' in x.name]
    simulator_path = files[0]
    # fo = r"C:\\Users\\marossi\\OneDrive - Microsoft\\Data\\cb_hyperparameters\\Actions_Contexts_matrix.5M.swap2M.txt"

    noClick = 0
    click = -1
    ctr_min = 0.03
    pStrategy = 0

    num_iter = 5000000
    num_iter_swap = 2000000
    iter_mod = 10000
    
    actions_v = [500, 10, 50, 100, 250]
    contexts_v = [1, 10, 50, 100, 250, 500]
    # actions_contexts_d = {10:  [1, 10],
                          # 50:  [1, 10, 50],
                          # 100: [1, 10, 50, 100],
                          # 250: [1, 10, 50, 100, 250],
                          # 500: [1, 10, 50, 100, 250, 500]}
    ctr_max_v = [0.1, 0.04, 0.05]
    namespace_str_v = [' --ignore A -q UB']
    clip_p_v = [' --clip_p 0.1', ' --clip_p 0.5', '']
    learning_rate_v = [' --coin', ' -l 1e-3 --power_t 0']
    
    max_seed_d = {10: 100, 50: 100, 100: 100, 250: 64, 500: 11}

    actions_counter = {x: 0 for x in sorted(actions_v)}
    # seed = -1
    command_list = []
    skipped = 0
    # while True:
        # seed += 1
    for actions in actions_v:
        for seed in list(range(max_seed_d[actions])):
            for ctr_max in ctr_max_v:
                for namespace_str in namespace_str_v:
                    for clip in clip_p_v:
                        for learning_rate in learning_rate_v:
                            ml_args = '--cb_explore_adf --cb_type mtr{}{}{}'.format(namespace_str, learning_rate, clip)
                            for contexts in contexts_v:
                                
                                args = '{} {} {} {} {} {} {} {} {} {} {}'.format(actions,contexts,ctr_min,ctr_max,noClick,click,pStrategy,num_iter,iter_mod,seed,num_iter_swap)
                                cmd_list = [simulator_path, ml_args] + args.split(' ')
                                
                                if get_cmd_str(cmd_list).split('simulator.exe',1)[1] not in already_done:
                                    command_list.append(cmd_list)
                                    actions_counter[actions] += 1
                                else:
                                    skipped += 1

                                if len(command_list) == num_sim:
                                    if dry_run:
                                        print(seed,len(command_list),skipped,actions_counter)
                                    else:
                                        run_experiment_set(command_list, num_proc, fo)
                                    command_list = []
                                    actions_counter = {x: 0 for x in sorted(actions_v)}

            
        # if seed == 13:
            # clip_p_v = [' --clip_p 0.1', '']
            # ctr_max_v = [0.05, 0.1]
        # if seed == 10:
            # actions_v = [10, 50, 100, 250]

    if dry_run:
        exit()

    actions_v = [10, 50, 100]

    print('Start while loop')
    seed = 100
    command_list = []
    skipped = 0
    while True:
        for actions in actions_v:
            for namespace_str in namespace_str_v:
                for ctr_max in ctr_max_v:
                    for clip in clip_p_v:
                        for learning_rate in learning_rate_v:
                            ml_args = '--cb_explore_adf --cb_type mtr{}{}{}'.format(namespace_str, learning_rate, clip)
                            for contexts in contexts_v:
                                
                                args = '{} {} {} {} {} {} {} {} {} {} {}'.format(actions,contexts,ctr_min,ctr_max,noClick,click,pStrategy,num_iter,iter_mod,seed,num_iter_swap)
                                cmd_list = [simulator_path, ml_args] + args.split(' ')
                                
                                if get_cmd_str(cmd_list).split('simulator.exe',1)[1] not in already_done:
                                    command_list.append(cmd_list)
                                    actions_counter[actions] += 1
                                else:
                                    skipped += 1

                                if len(command_list) == num_sim:
                                    if dry_run:
                                        print(seed,len(command_list),skipped,actions_counter)
                                    else:
                                        run_experiment_set(command_list, num_proc, fo)
                                    command_list = []
                                    actions_counter = {x: 0 for x in sorted(actions_v)}
        seed += 1

# python "C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\cb_adf_hyper_simulate_v32.py" -p 45 -n 45 --fo "C:\\Users\\marossi\\OneDrive - Microsoft\\Data\\cb_hyperparameters\\Actions_Contexts_matrix.5M.swap2M.txt"
