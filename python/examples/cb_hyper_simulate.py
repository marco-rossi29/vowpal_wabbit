import multiprocessing, os, sys
from string import ascii_lowercase
from scipy.stats import beta
import numpy as np

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
        self.full_command += " --cb_type {}".format(self.cb_type)
        if self.marginal_list:
            self.full_command += " --marginal {}".format(''.join(self.marginal_list))
        for ignored_namespace in self.ignore_list:
            self.full_command += " --ignore {}".format(ignored_namespace)
        for interaction in self.interaction_list:
            self.full_command += " -q {}".format(interaction)
        self.full_command += " -l {:g}".format(self.learning_rate)
        self.full_command += " --l1 {:g}".format(self.regularization)
        self.full_command += " --power_t {:g}".format(self.power_t)

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

    num_customers, num_actions, ctr_good, ctr_bad, rnd_seed = args_tuple

    np.random.seed(rnd_seed)

    customer_types = {x:i for i,x in enumerate(ascii_lowercase[:num_actions])}
    c_list = list(customer_types.keys())
    
    p_uniform_random = 1.0/float(num_actions)
    p_list = [p_uniform_random for _ in customer_types]

    counters = {x: [[1,1] for _ in range(num_actions)] for x in customer_types}
    
    ctr_all = [ctr_good, ctr_bad]
    
    num_clicks = 0
    h = [0, 0]
    
    output = []
    
    for icustomer in range(num_customers):

        # Get customer
        customer_type = np.random.choice(c_list, p=p_list)

        # Ask which action to show this customer
        action = max(((i, beta.ppf(.95, x[0]+1, x[1])) for i,x in enumerate(counters[customer_type])),key=lambda y : y[1])[0]

        # Did the customer click?
        if customer_types[customer_type] == action:
            ctr = ctr_all[0]
            h[0] += 1
        else:
            ctr = ctr_all[1]
            h[1] += 1

        clicked = np.random.choice([True, False], p=[ctr, 1.0 - ctr])

        # update stats
        if clicked:
            counters[customer_type][action][0] += 1
            num_clicks += 1
        else:
            counters[customer_type][action][1] += 1
            
        if (icustomer+1) % 50000 == 0:
            output.append(','.join(map(str, [rnd_seed, num_clicks, icustomer+1, sum(ctr_all[i]*float(x)/float(icustomer+1) for i,x in enumerate(h)), float(num_clicks)/float(icustomer+1)] + h + [float(x)/float(icustomer+1) for x in h])))

    return '\n'.join(output)
    
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
            experiment_file.write(result + "\n")
        experiment_file.flush()

if __name__ == '__main__':

    # fp = r'/mnt/d/data/vw-python-bug/sim_code_good_v4_p0.04_2users_totalClip.txt'
    # fp = r'/mnt/c/Users/marossi/OneDrive - Microsoft/Data/cb_hyperparameters/sim_code_good_v4_p0.04_2users_totalClip.txt'
    fp = r'C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_UCB.txt'
    
    num_customers = 1000000
    num_actions = 10
    ctr_good = 0.04
    ctr_bad = 0.03
    num_proc = 43
    
    rnd_seed = 1
    command_list = []
    while True:
        command_list.append((num_customers, num_actions, ctr_good, ctr_bad, rnd_seed))
        
        if len(command_list) == num_proc*3:
            print('Run iter:',rnd_seed)
            run_experiment_set(command_list, num_proc, fp)
            command_list = []
        
        rnd_seed += 1
        


    if os.path.isfile(fp):
        l = ['\t'.join(x.split('\t')[:9]) for i,x in enumerate(open(fp)) if i > 0]
        print(len(l))
        
        already_done = set(l)
        print(len(already_done))
    else:
        with open(fp, 'a') as experiment_file:
            experiment_file.write('Actions\tEps\tCb_type\tLearningRate\tL1-Reg\tPowerT\tForced\tSeed\tCost_0_1\tCTR Math\tCTR\tClicks\tIters\tGoodActions\tBadActions\n')
        already_done = set()
    
    base_cmd = '--cb_explore 2 --epsilon 0.05'
    rnd_seed_start = 1
    num_proc = 40
    
    fpi = r'/mnt/c/Users/marossi/OneDrive - Microsoft/Data/cb_hyperparameters/cb_hyper_simulate_input.csv'
    if False:
        l = [x.strip() for x in open(fpi)][1:]
        
        skipped = 0
        command_list = []
        for rnd_seed in range(rnd_seed_start):
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
                    run_experiment_set(command_list, num_proc, fp)
                    command_list = []
        print(len(command_list),skipped)
        # print(command_list)
        # sys.exit()
        run_experiment_set(command_list, num_proc, fp)
        
        rnd_seed = rnd_seed_start
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
                    run_experiment_set(command_list, num_proc, fp)
                    command_list = []

            rnd_seed += 1

    else:
        recorded_prob_types = [0, 1, 2, 6, 13]
        zero_one_costs = [1, 0]
        learning_rates = [1e-2, 2.5e-2, 5e-2, 1e-1, 0.5]
        regularizations = [0]
        power_t_rates = [0.5]
        cb_types = ['dr', 'ips']
        
        # Regularization, Learning rates, and Power_t rates grid search for both ips and mtr
        command_list = []
        skipped = 0
        for rnd_seed in range(rnd_seed_start):
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
                                        run_experiment_set(command_list, num_proc, fp)
                                        command_list = []
        print(len(command_list),skipped)
        # sys.exit()
        run_experiment_set(command_list, num_proc, fp)
        
        print('Start while loop...')
        rnd_seed = rnd_seed_start
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
                                        run_experiment_set(command_list, num_proc, fp)
                                        command_list = []
            
            rnd_seed += 1

