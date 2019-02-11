import numpy as np
from vowpalwabbit.pyvw import vw
import multiprocessing, os, sys

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
        self.full_command += " -l {}".format(self.learning_rate)
        self.full_command += " --l1 {}".format(self.regularization)
        self.full_command += " --power_t {}".format(self.power_t)

    def prints(self):
        print("cb type: {0}".format(self.cb_type))
        print("marginals: {0}".format(self.marginal_list))
        print("ignore list: {0}".format(self.ignore_list))
        print("interactions: {0}".format(self.interaction_list))
        print("learning rate: {0}".format(self.learning_rate))
        print("regularization: {0}".format(self.regularization))
        print("power_t: {0}".format(self.power_t))
        print("overall command: {0}".format(self.full_command))
        print("loss: {0}".format(self.loss))

def run_experiment(args_tuple):

    cmd_str, recorded_prob_type, rnd_seed, zero_one_cost = args_tuple

    np.random.seed(rnd_seed)

    num_customers = 100000
    customer_types = {'a':1, 'b':2}
    ctr_all = [0.04, 0.03]
    
    actions = [i+1 for i in range(len(customer_types))]
    p_uniform_random = 1.0/float(len(customer_types))

    model = vw(cmd_str+' --quiet')
        
    num_clicks = 0
    h = [0, 0]
    for icustomer in range(num_customers):

        # Get customer
        customer_type = np.random.choice(customer_types.keys(), p=[p_uniform_random for _ in customer_types])

        # Ask VW which action to show this customer
        s = '| {}'.format(customer_type)
        ex = model.example(s)
        probs = model.predict(ex)
        ex.finish()
        
        s = sum(probs)
        probs = [prob/s for prob in probs] # Necessary b/c need to sum to 1 for python

        ichoice = np.random.choice(range(len(actions)), p=probs)
        action = actions[ichoice]
        prob = probs[ichoice]

        # Did the customer click?
        if customer_types[customer_type] == action:
            ctr = ctr_all[0]
            h[0] += 1
        else:
            ctr = ctr_all[1]
            h[1] += 1

        clicked = np.random.choice([True, False], p=[ctr, 1.0 - ctr])


        # Learn
        if recorded_prob_type == 1:
            prob = p_uniform_random
        elif recorded_prob_type == 2:
            prob = max(prob, 0.5)
        elif recorded_prob_type == 3:
            prob = max(prob, 0.2)
        elif recorded_prob_type == 4:
            prob = max(prob, 0.75)
        elif recorded_prob_type == 6:
            prob = max(prob, 0.9)
        elif recorded_prob_type == 7:
            prob = .9
        elif recorded_prob_type == 5:
            if prob > 0.5:
                prob = 0.6
            else:
                prob = 0.4
        elif recorded_prob_type == 8:
            if prob > 0.5:
                prob = 0.8
            else:
                prob = 0.2
        
        if zero_one_cost == 1:
            cost = 0.0 if clicked else 1.0
        else:
            cost = -1.0 if clicked else 0.0

        event = '{}:{}:{} | {}'.format(action, cost, prob, customer_type)
        ex = model.example(event)
        model.learn(ex)
        ex.finish()

        # Save results
        if clicked:
            num_clicks += 1
            
        # if (icustomer+1) % 100000 == 0:
            # print(num_clicks, icustomer+1, sum(ctr_all[i]*float(x)/float(icustomer+1) for i,x in enumerate(h)), float(num_clicks)/float(icustomer+1), h, [float(x)/float(icustomer+1) for x in h])

        # x = sorted([(model.get_weight(i),i) for i in range(model.num_weights()) if np.abs(model.get_weight(i)) > 1e-5])
        # print('---------------------------------------')
        # for y in x:
            # print(y)
        # print('---------------------------------------')
    
    s = '\t'.join(map(str, cmd_str.split(' ')[1::2] + [recorded_prob_type, rnd_seed, zero_one_cost, sum(ctr_all[i]*float(x)/float(icustomer+1) for i,x in enumerate(h)), float(num_clicks)/float(icustomer+1), num_clicks, icustomer+1] + h))
    print(s)
    model.finish()

    return s
    
def run_experiment_set(command_list, n_proc, fp):
    print('Num of sim:',len(command_list))
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
    fp = r'/mnt/c/Users/marossi/OneDrive - Microsoft/Data/cb_hyperparameters/sim_code_good_v4_p0.04_2users_totalClip.txt'

    if os.path.isfile(fp):
        l = ['\t'.join(x.split('\t')[:9]) for i,x in enumerate(open(fp)) if i > 0]
        print(len(l))
        
        already_done = set(l)
        print(len(already_done))
    else:
        already_done = set()
    

    recorded_prob_types = [0, 1, 4, 5, 6, 8]
    zero_one_costs = [1, 0]
    learning_rates = [1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1, 0.5]
    regularizations = [0]
    power_t_rates = [0.5]
    cb_types = ['ips', 'dr', 'dm']
    
    # Regularization, Learning rates, and Power_t rates grid search for both ips and mtr
    command_list = []
    skipped = 0
    for rnd_seed in range(1002):
        for zero_one_cost in zero_one_costs:
            for recorded_prob_type in recorded_prob_types:
                for regularization in regularizations:
                    for cb_type in cb_types:
                        for power_t in power_t_rates:
                            for learning_rate in learning_rates:
                                command = Command('--cb_explore 2 --epsilon 0.05', regularization=regularization, learning_rate=learning_rate, power_t=power_t, cb_type=cb_type)
                                
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
    
    
    rnd_seed = 1002
    while True:
        command_list = []
        for zero_one_cost in zero_one_costs:
            for recorded_prob_type in recorded_prob_types:
                for regularization in regularizations:
                    for cb_type in cb_types:
                        for power_t in power_t_rates:
                            for learning_rate in learning_rates:
                                command = Command('--cb_explore 2 --epsilon 0.05', regularization=regularization, learning_rate=learning_rate, power_t=power_t, cb_type=cb_type)
                                
                                # s = '\t'.join(map(str, command.full_command.split(' ')[1::2] + [recorded_prob_type, rnd_seed, zero_one_cost]))
                                
                                # if s not in already_done:
                                command_list.append((command.full_command, recorded_prob_type, rnd_seed, zero_one_cost))
    
        run_experiment_set(command_list, 45, fp)
        
        rnd_seed += 1
                        
'''

import pandas as pd
import matplotlib.pyplot as plt

fp = r'c:\Users/marossi/OneDrive - Microsoft/Data/cb_hyperparameters/sim_code_good_v4_p0.04_2users_totalClip.txt'

cols = ['Forced','Cost_0_1','Power','Cb_type','LearningRate']

df = pd.read_csv(fp, sep='\t')

df.groupby(by=cols)['CTR Math'].quantile([.1]).sort_values(ascending=False)

print(df.groupby(by=cols)['CTR Math'].mean().sort_values(ascending=False).head(30).to_string())

print(df.groupby(by=cols)['Seed'].max().to_string())

for i,x in df.groupby(by=cols):
    m = x['CTR Math'].mean()
    if m > 0.039:
        print(i,'\t{:.3%}\t'.format(m),'\t'.join('{:.3%}'.format(y) for y in x['CTR Math'].quantile([.1, .25, .5, .75, .9])))

df[((df.Forced == 0) | (df.Forced == 1) | (df.Forced == 6)) & ((df.LearningRate >= 0.01) & (df.LearningRate <= 0.1)) & (df.Power == 0.5)].boxplot(by=['Cb_type','Forced','Cost_0_1'], column='CTR Math')

inner_cols = ['Cost_0_1','Cb_type','Forced']
fig, axes = plt.subplots(nrows=2, ncols=3)
for i,lr in enumerate([0.01, 0.025, 0.05, 0.075, 0.1, 0.5]):
    x = df[((df.Forced == 0) | (df.Forced == 1) | (df.Forced == 6)) & (df.LearningRate == lr) & (df.Power == 0.5) & (df.Cb_type != 'dm')]
    x.boxplot(by=inner_cols, column=['CTR Math'], ax=axes[int(i/3),i%3], rot=90)
    axes[int(i/3),i%3].set_title('lr = '+str(lr))
    z = [y['CTR Math'].mean() for _,y in x.groupby(by=inner_cols)]
    axes[int(i/3),i%3].scatter(list(range(1,len(z)+1)),z)
plt.show()

inner_cols = ['Cb_type', 'LearningRate']
fig, axes = plt.subplots(nrows=2, ncols=3)
for i,cost in enumerate([0, 1]):
    for j,forced in enumerate([0, 1 ,6]):
        x = df[(df.Forced == forced) & (df.Cost_0_1 == cost) & (df.LearningRate >= 0.01) & (df.LearningRate <= 0.5) & (df.Power == 0.5) & (df.Cb_type != 'dm')]
        x.boxplot(by=inner_cols, column=['CTR Math'], ax=axes[i, j], rot=90)
        axes[i, j].set_title('Cost_0_1 = '+str(cost)+' Forced = '+str(forced))
        axes[i, j].set_xlabel('')
        
        z = [y['CTR Math'].mean() for _,y in x.groupby(by=inner_cols)]
        axes[i, j].scatter(list(range(1,len(z)+1)),z)
plt.show()

inner_cols = ['LearningRate','Cb_type']
fig, axes = plt.subplots(nrows=2, ncols=3)
for i,cost in enumerate([0, 1]):
    for j,forced in enumerate([0, 1 ,6]):
        x = df[(df.Forced == forced) & (df.Cost_0_1 == cost) & (df.LearningRate >= 0.001) & (df.LearningRate <= 0.5) & (df.Power == 0.5) & (df.Cb_type != 'dm')]
        x.boxplot(by=inner_cols, column=['CTR Math'], ax=axes[i, j], rot=90)
        axes[i, j].set_title('Cost_0_1 = '+str(cost)+' Forced = '+str(forced))
        axes[i, j].set_xlabel('')
        
        z = [y['CTR Math'].mean() for _,y in x.groupby(by=inner_cols)]
        axes[i, j].scatter(list(range(1,len(z)+1)),z)
plt.show()

'''
