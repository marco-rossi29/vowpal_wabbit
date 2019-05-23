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

def run_experiment(args_tuple):

    ml_args, num_actions, base_cost, delta_cost, pStrategy, rnd_seed = args_tuple
    num_contexts = num_actions

    # cmd_list = ['C:\\work\\bin\\cs_sim_SINGLE\\Simulator_v2_840_msft_3b64d7f7e2\\simulator.exe', ml_args]
    # cmd_list = ['C:\\work\\bin\\Simulator_action-per-context_SINGLE_submodule94e2dbe9_Prob1_error_fix\\PerformanceConsole.exe', ml_args]
    # cmd_list = ['C:\\work\\bin\\cs_sim_SINGLE\\Simulator_v2_861_latest2_24e4ea4b7a41\\simulator.exe', ml_args]
    # cmd_list = ['C:\\work\\bin\\cs_sim_SINGLE\\Simulator_v2_861_prob1_error_fix_23ed513705397\\simulator.exe', ml_args]
    cmd_list = ['C:\\work\\bin\\cs_sim_SINGLE\\Simulator_v21_861_2ff79cd518e896\\simulator.exe', ml_args]

    cmd_list += ('{} {} 0.03 0.04 {} {} {} 10000000 1000000 {}'.format(num_actions, num_contexts, base_cost, delta_cost, pStrategy, rnd_seed)).split(' ')
    
    try:
        x = check_output(cmd_list, stderr=STDOUT, universal_newlines=True)
        # if not x.startswith('{'):
            # x = x.split('\n',1)[1]
        return x
    except Exception as e:
        err_str = 'Error for command {}: {}'.format(' '.join(cmd_list), e)
        print(err_str)
        return err_str
    
def run_experiment_set(command_list, n_proc, fp):
    print('Num of sim:',len(command_list))
    if len(command_list) > 0:
        t0 = datetime.datetime.now()
        print('Current time: {}'.format(t0))
        command_list.sort(key=lambda x : (x[1], max((int(x[0].split(e,1)[1].split(' ',1)[0]) if e in x[0] else 1) for e in [' --bag ',' --cover '])), reverse=True)

        # Run the experiments in parallel using n_proc processes
        p = multiprocessing.Pool(n_proc)
        results = p.map(run_experiment, command_list, chunksize=1)
        p.close()
        p.join()
        del p
        result_writer(results, fp)
        print('Elapsed: {} - Last cmd: {}'.format(datetime.datetime.now()-t0, results[-1].splitlines()[-1]))

def result_writer(results, fp):
    fp_all = fp + '.allLines.txt'
    with (gzip.open(fp_all, 'at') if fp_all.endswith('.gz') else open(fp_all, 'a')) as f, (gzip.open(fp, 'at') if fp.endswith('.gz') else open(fp, 'a')) as f2:
        for result in results:
            if result.startswith('Error for command'):
                with open(fp+'.errors.txt', 'a') as fe:
                    fe.write(result+'\n')
            else:
                f.write(result)
                z = [x for x in result.splitlines() if ',10000000,' in x]
                f2.write('\n'.join(z)+'\n')


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
            
            if ',10000000,' in x:
                lines[0] += 1
                already_done.add(x.split(',10000000,',1)[0])
        print('Total lines: {}\nIter1M lines: {}\nIter1M unique: {}'.format(lines[1],lines[0],len(already_done)))
        if lines[0] > len(already_done):
            unique_lines = set()
            cnt = 0
            with open(fp+'.deDup.txt','w') as fd:
                for x in (gzip.open(fp, 'rt') if fp.endswith('.gz') else open(fp)):
                    if x in unique_lines:
                        continue
                    unique_lines.add(x)
                    fd.write(x)
                    cnt += 1
            print('Total deDup lines: {}\nUniqueLines: {}'.format(cnt,len(unique_lines)))
            input()

    else:
        with (gzip.open(fp, 'at') if fp.endswith('.gz') else open(fp, 'a')) as f:
            f.write('MLargs,numActions,baseCost,deltaCost,pStrategy,rewardSeed,Iter,CTR,GoodActions,dA\n')
    
    # test a single experiment set
    # command = Command(base_cmd, learning_rate=0.005, power_t=0.5, cb_type='dr')
    # run_experiment_set([(command.full_command, 5, 1.0, 7, 24), (command.full_command, 5, 0.0, 7, 24)], 20, fp)
    # sys.exit()
    
    
    base_cmd_list = ['--cb_explore_adf --ignore XA -q UB']#,'--cb_explore_adf --ignore XA -q UB --ignore_linear UB']#, '--cb_explore_adf --ignore ABU']  
    #base_cmd_list = ['--cb_explore_adf --ignore ABU']
    # learning_rates = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 2.5e-3, 5e-3, 1e-2, 2e-2, 2.5e-2, 5e-2, 1e-1, 2.5e-1, 0.5, 1, 2.5, 5, 10, 100, 1000]
    # learning_rates = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 2.5e-3, 5e-3, 1e-2, 2e-2, 2.5e-2, 5e-2, 1e-1, 2.5e-1, 0.5, 1, 2.5, 5, 10, 100, 1000]
    # learning_rates = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 10, 100, 1000]
    # learning_rates = [5e-2, 1e-1, 2.5e-1, 0.5, 1, 2.5, 5, 10, 100, 1000]
    # learning_rates = [0.5, 1, 2.5, 5, 10, 100, 1000]#[5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 2.5e-3, 5e-3, 1e-2]
    recorded_prob_types = [0,1]
    cb_types = ['mtr','dr']
    # costTuple_d = {x:[(1, 1), (1, 10), (0,1), (0,10), (10,1)] for x in cb_types}   # (baseCost,deltaCost) tuples    
    power_t_vec = {x:[0] for x in cb_types}
    
    params_bag = [' --bag {}{}{}'.format(N,n,m) for N in [2] for n in [' --greedify'] for m in ['',' --epsilon 0.05']]
    params_cover = [' --cover {}{}'.format(N,n) for N in [2] for n in [' --nounif']]
    params_softmax = [' --softmax{}{}'.format(n,m) for n in ['', ' --lambda 0.5', ' --lambda 2', ' --lambda 3'] for m in ['',' --epsilon 0.05']]
    
    mel1_vec = [' --mellowness 0.001', ' --mellowness 0.01', ' --mellowness 0.1', ' --mellowness 0.5', ' --mellowness 0.75', ' --mellowness 1.0']
    mel2_vec = [' --mellowness 0.01']
    
    params_regcb = [' {}{}{}'.format(x,n,m) for x in ['--regcbopt'] for m in ['',' --epsilon 0.05'] for n in mel2_vec]
    psi_vec_cover = [0.01]
    
    # exploration_d = {'mtr': [''] + params_bag + params_regcb,
                     # 'dr':  [''] + params_bag + params_cover + params_regcb}

    exploration_d = {x: [''] for x in cb_types}

    print('Start while loop...')
    rnd_seed = rnd_seed_start_while_loop
    command_list = []
    skipped = 0
    while True:
        for base_cmd in base_cmd_list:
            marginal_list_vec = [None]#, ['X']] if ' --ignore ABU' in base_cmd else [None]
            for mar in marginal_list_vec:
                for cb_type in cb_types:
                    for exploration in exploration_d[cb_type]:
                        psi_vec = psi_vec_cover if '--cover' in exploration else [None]
                        for psi in psi_vec:
                            costTuple_vec = [(0,1),(1,10)]
                            # if rnd_seed > 5:
                                # if 'regcb' in exploration:
                                    # costTuple_vec = [(1,1), (1,10)]
                                # elif 'softmax' in exploration:
                                    # costTuple_vec = [(1,10), (0,10)]
                            for costTuple in costTuple_vec:
                                for pt in power_t_vec[cb_type]:
                                    for pStrategy in recorded_prob_types:
                                        if rnd_seed > 2 and 'softmax' in exploration:
                                            learning_rates = [5e-2, 1e-1, 2.5e-1, 0.5, 1, 2.5, 5, 10, 100, 1000]
                                        else:
                                            learning_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5]
                                        for lr in learning_rates:

                                            base_cmd2 = base_cmd + exploration
                                            if psi is not None:
                                                base_cmd2 += ' --psi ' + str(psi)

                                            command = Command(base_cmd2, learning_rate=lr, cb_type=cb_type, power_t=pt, marginal_list=mar)

                                            s = ','.join(map(str, [command.full_command, num_actions, costTuple[0], costTuple[1], pStrategy, rnd_seed]))
                                            if s not in already_done:
                                                command_list.append((command.full_command, num_actions, costTuple[0], costTuple[1], pStrategy, rnd_seed))
                                            else:
                                                skipped += 1

                                            if len(command_list) == num_sim and not dry_run:
                                                run_experiment_set(command_list, num_proc, fp)
                                                command_list = []
                    
        if dry_run:
            print(rnd_seed,len(command_list),skipped)
            # for x in command_list:
                # print(x)
                # input()
            if rnd_seed == 15:
                break
            
        rnd_seed += 1
        # if rnd_seed == 5:
            # costTuple_d = {x:[(1, 1), (1, 10)] for x in cb_types}
            # learning_rates = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 2.5e-3, 5e-3, 1e-2]
            
            # params_regcb = [' --regcbopt{}{}'.format(n,m) for n in [' --mellowness 0.5', ' --mellowness 0.75'] for m in ['']]
                     
            # exploration_d = {x: params_regcb for x in cb_types}
            
            # recorded_prob_types = [0,1]
            
            # psi_vec_cover = [0, 0.01]
            # params = [' --cover {}{}'.format(N,n) for N in [2, 5] for n in ['', ' --nounif']]
            # exploration_d = {x: params for x in cb_types}
            
# python C:\work\vw\python\examples\cb_adf_hyper_simulate.py -a 2 -p 43 -n 430 -r 10000 --fp "C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\myFile_Actions2.txt.gz"
# python C:\work\vw\python\examples\cb_adf_hyper_simulate.py -p 43 -n 430 -b "--cb_explore_adf --epsilon 0.05 --marginal A" --fp "C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions2-10_marginal_noQ.txt"

# python C:\work\vw\python\examples\cb_adf_hyper_simulate.py -p 43 -n 430 -b "--cb_explore_adf --ignore ABU" --fp "C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_marginal_pt_bag.txt" -a 10
# python "C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\cb_adf_hyper_simulate.py" -a 10 -p 44 -n 440 --fp "C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_ALL-Simulator_v21_861_2ff79cd518e896.txt"
