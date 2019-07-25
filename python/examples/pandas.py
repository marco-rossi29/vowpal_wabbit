import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools, os, gzip
from cycler import cycler

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'per_%s' % n
    return percentile_


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
s,c = zip(*itertools.product(['-', '--', ':', '-.',(0, (1, 1))], colors[:6]))
plt.rc('axes', prop_cycle=(cycler('color', c) + cycler('linestyle', s)))

plt.rcParams.update({'font.size': 18})

fp = r'c:\Users/marossi/OneDrive - Microsoft/Data/cb_hyperparameters/sim_code_good_v4_p0.04_10users_totalClip.txt'

cols = ['Forced','Cost_0_1','Power','Cb_type','LearningRate']

df = pd.read_csv(fp, sep='\t')

df.groupby(by=cols)['CTR Math'].quantile([.1]).sort_values(ascending=False)

print(df.groupby(by=cols)['CTR Math'].mean().sort_values(ascending=False).head(30).to_string())

print(df.groupby(by=cols)['CTR Math'].agg(['mean', 'median', 'min', 'max']).sort_values(by='median', ascending=False).head(30).to_string())

print(df.groupby(by=cols)['Seed'].max().to_string())

for i,x in df.groupby(by=cols):
    m = x['CTR Math'].mean()
    if m > 0.039:
        print(i,'\t{:.3%}\t'.format(m),'\t'.join('{:.3%}'.format(y) for y in x['CTR Math'].quantile([.1, .25, .5, .75, .9])))

df[((df.Forced == 0) | (df.Forced == 1) | (df.Forced == 6)) & ((df.LearningRate >= 0.01) & (df.LearningRate <= 0.1)) & (df.Power == 0.5)].boxplot(by=['Cb_type','Forced','Cost_0_1'], column='CTR Math')

inner_cols = ['Cost_0_1','Forced']
fig, axes = plt.subplots(nrows=2, ncols=3)
for i,lr in enumerate([0.01, 0.025, 0.05, 0.075, 0.1, 0.5]):
    x = df[(df.LearningRate == lr) & (df.Power == 0.5) & (df.Cb_type == 'dr')]
    x.boxplot(by=inner_cols, column=['CTR Math'], ax=axes[int(i/3),i%3], rot=90)
    axes[int(i/3),i%3].set_title('lr = '+str(lr))
    axes[int(i/3),i%3].set_xlabel('')
    
    z = [y['CTR Math'].mean() for _,y in x.groupby(by=inner_cols)]
    axes[int(i/3),i%3].scatter(list(range(1,len(z)+1)),z)
plt.show()

inner_cols = ['Cb_type', 'LearningRate']
fig, axes = plt.subplots(nrows=2, ncols=3)
for i,cost in enumerate([0, 1]):
    for j,forced in enumerate([0, 1, 6, 5, 8]):
        x = df[(df.Forced == forced) & (df.Cost_0_1 == cost) & (df.LearningRate >= 0.01) & (df.LearningRate <= 0.5) & (df.Power == 0.5) & (df.Cb_type != 'dm')]
        x.boxplot(by=inner_cols, column=['CTR Math'], ax=axes[i, j], rot=90)
        axes[i, j].set_title('Cost_0_1 = '+str(cost)+' Forced = '+str(forced))
        axes[i, j].set_xlabel('')
        
        z = [y['CTR Math'].mean() for _,y in x.groupby(by=inner_cols)]
        axes[i, j].scatter(list(range(1,len(z)+1)),z)
plt.show()

inner_cols = ['LearningRate']
fig, axes = plt.subplots(nrows=2, ncols=3)
for i,cost in enumerate([0, 1]):
    for j,forced in enumerate([0, 1, 2, 3, 8]):
        x = df[(df.Forced == forced) & (df.Cost_0_1 == cost) & (df.LearningRate >= 0.001) & (df.LearningRate <= 0.5) & (df.Power == 0.5) & (df.Cb_type == 'dr')]
        x.boxplot(by=inner_cols, column=['CTR Math'], ax=axes[i, j], rot=90)
        axes[i, j].set_title('Cost_0_1 = '+str(cost)+' Forced = '+str(forced))
        axes[i, j].set_xlabel('')
        
        z = [y['CTR Math'].mean() for _,y in x.groupby(by=inner_cols)]
        axes[i, j].scatter(list(range(1,len(z)+1)),z)
plt.show()


forced_vec = [0, 1, 6, 11, 4]#, 2, 3, 4, 5, 6, 8, 9, 10]
inner_cols = ['LearningRate']
fig, axes = plt.subplots(nrows=2, ncols=len(forced_vec), sharey=True, sharex=True)
for i,cost in enumerate([0, 1]):
    for j,forced in enumerate(forced_vec):
        x = df[(df.Forced == forced) & (df.Cost_0_1 == cost) & (df.LearningRate >= 0.001) & (df.LearningRate <= 0.5) & (df.Power == 0.5) & (df.Cb_type == 'dr')]
        x.boxplot(by=inner_cols, column=['CTR Math'], ax=axes[i, j], rot=90)
        axes[i, j].set_title('Cost_0_1 = '+str(cost) + ' - Forced = '+str(forced))
        axes[i, j].set_xlabel('')
        
        z = [y['CTR Math'].mean() for _,y in x.groupby(by=inner_cols)]
        axes[i, j].scatter(list(range(1,len(z)+1)),z)
#plt.tight_layout()
plt.show()

inner_cols = ['LearningRate', 'Forced']
fig, axes = plt.subplots(nrows=2, ncols=1)
for i,cost in enumerate([0, 1]):
    x = df[((df.Forced == 0) | (df.Forced == 1)) & (df.Cost_0_1 == cost) & (df.LearningRate >= 0.001) & (df.LearningRate <= 1) & (df.Power == 0.5) & (df.Cb_type == 'dr')]
    x.boxplot(by=inner_cols, column=['CTR Math'], ax=axes[i], rot=90)
    axes[i].set_title('Cost_0_1 = '+str(cost))
    axes[i].set_xlabel('')
    
    z = [y['CTR Math'].mean() for _,y in x.groupby(by=inner_cols)]
    axes[i].scatter(list(range(1,len(z)+1)),z)
plt.show()

x.groupby('Forced')['GoodActions'].plot(kind='hist', density=True, cumulative=True, bins=100, histtype='step')

print(x.groupby(by=cols)['GoodActions'].agg(['mean', 'min', percentile(5), percentile(10), percentile(25), 'median', percentile(75), percentile(90), percentile(95), 'max', 'count']).sort_values(by='min', ascending=False).head(50).to_string())

------------------------------------------------------------------------------------------------------------------
def boxplot_sorted(df, by, column, rot=90):
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    z = df2.mean().sort_values()
    df2[z.index].boxplot(rot=rot)
    means = df2[z.index].mean()
    plt.scatter(list(range(1,len(means)+1)), means)
    plt.show()

def boxplot(df, by, column, rot=60):
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby(by)})
    ax = df2.boxplot(rot=rot)
    means = df2.mean()
    ax.scatter(list(range(1,len(means)+1)), means)
    return ax

def print_stats(do_plot=False, update=False, sort_by_str='mean', cols=['Forced','Cost_0_1','Power','Cb_type','LearningRate'], fp=r'c:\Users/marossi/OneDrive - Microsoft/Data/cb_hyperparameters/sim_code_good_v4_p0.04_2users_totalClip.txt', lr=0.025, cost=1, power_t=0.5, cb_type='dr'):
    if update:
        df = pd.read_csv(fp, sep='\t')
    
    x = df[(df.Cb_type == cb_type) & (df.Forced <= 2)]
    x = x.replace({"Forced":{0:'Original',1:'[0.5,0.5]', 2:'max(p,0.5)', 3:'max(p,0.2)', 4:'max(p,0.75)', 6: 'max(p,0.9)', 9:'min(p,0.5)', 5:'[0.4,0.6]', 7:'[0.9,0.9]', 8:'[0.2,0.8]', 10:'[0.45,0.55]', 11:'[0.4,0.5]', 12:'[0.5,0.6]'}})
    
    print(x.groupby(by=cols)['GoodActions'].agg(['mean', 'min', percentile(5), percentile(10), percentile(25), 'median', percentile(75), percentile(90), percentile(95), 'max', 'count']).sort_values(by=sort_by_str, ascending=False).head(80).to_string())
    
    x = df[(df.Cost_0_1 == cost) & (df.LearningRate == lr) & (df.Power == power_t) & (df.Cb_type == cb_type)]
    x = x.replace({"Forced":{0:'Original',1:'Uniform', 2:'max(p,0.5)', 3:'max(p,0.2)', 4:'max(p,0.75)', 6: 'max(p,0.9)', 9:'min(p,0.5)', 5:'[0.4,0.6]', 7:'[0.9,0.9]', 8:'[0.2,0.8]', 10:'[0.45,0.55]', 11:'[0.4,0.5]', 12:'[0.5,0.6]', 13:'0.5'}}})
    
    print(x.groupby(by=cols)['GoodActions'].agg(['mean', 'min', percentile(5), percentile(10), percentile(25), 'median', percentile(75), percentile(90), percentile(95), 'max', 'count']).sort_values(by=sort_by_str, ascending=False).head(25).to_string())
    
    if do_plot:
        x.boxplot(by=['LearningRate','Forced'], column=['GoodActions'], rot=30)
        #boxplot_sorted(x, ['LearningRate','Forced'], 'GoodActions')

------------------------------------------------------------------------------------------------
fp = r'c:\Users/marossi/OneDrive - Microsoft/Data/cb_hyperparameters/myFile_Actions2.txt'

def load_df_end(fp, rename_pStrategy=True, fpi=None, grep_fpi=b'"Iter":1000000,'):
    if fpi and os.path.isfile(fpi):
        if os.path.isfile(fp):
            if input('File already exist. Press ENTER to overwrite...' ) != '':
                return None
        print('Grepping lines with:',grep_fpi)
        with open(fp,'wb') as f:
            for i,x in enumerate(gzip.open(fpi)):
                if (i+1) % 10000 == 0:
                    ds_parse.update_progress(i+1,prefix=fpi+' - ')
                if b'"Iter":1000000,' in x or b'"Iter":950000,' in x:
                    f.write(x)
            len_text = ds_parse.update_progress(i+1,prefix=fpi+' - ')
            sys.stdout.write("\r" + " "*len_text + "\r")
            sys.stdout.flush()

    print('Loading input file')
    df = pd.read_json(fp, lines=True)
    print('Creating columns')
    df['Cb_type'] = df.apply(lambda row: row['ml_args'].split(' --cb_type ',1)[1].split(' ',1)[0], axis=1)
    df['LearningRate'] = df.apply(lambda row: float(row['ml_args'].split(' -l ',1)[1].split(' ',1)[0]), axis=1)
    #df['eps'] = df.apply(lambda row: row['ml_args'].split(' --epsilon ',1)[1].split(' ',1)[0], axis=1)
    df['y'] = df['goodActions']/df['Iter']
    if rename_pStrategy:
        print('Rename pStrategy col')
        df = df.replace({"pStrategy":{0:'Original',1:'Uniform', 2:'max(p,0.5)', 3:'max(p,0.2)', 4:'max(p,0.75)', 6: 'max(p,0.9)', 9:'min(p,0.5)', 5:'[0.4,0.6]', 7:'[0.9,0.9]', 8:'[0.2,0.8]', 10:'[0.45,0.55]', 11:'[0.4,0.5]', 12:'[0.5,0.6]', 13:'0.5'}})
    return df
    
def semilogx(df, cols=['pStrategy','baseCost','Cb_type','LearningRate']):
    fig, ax = plt.subplots(figsize=(8,6))
    df.groupby(by=cols)['goodActions'].agg('mean').unstack().transpose().plot(ax=ax)
    ax.set_xscale('log')
    plt.show()

    
cols = ['pStrategy','baseCost','Cb_type','LearningRate']

print(df.groupby(by=cols)['goodActions'].agg(['mean', 'min', percentile(5), percentile(10), percentile(25), 'median', percentile(75), percentile(90), percentile(95), 'max', 'count']).sort_values(by='mean', ascending=False).head(50).to_string())

print(df[(df.baseCost == 0.0) & (df.pStrategy == 'Original')].groupby(by=cols)['goodActions'].agg(['mean', 'min', percentile(5), percentile(10), percentile(25), 'median', percentile(75), percentile(90), percentile(95), 'max', 'count']).to_string())


df[df.Iter == 500000].groupby(by=cols)['goodActions'].agg('mean').unstack(level=[0,1,2]).fillna(0)
print(df.groupby(by=cols)['goodActions'].agg('mean').unstack(level=[0,1,2]).fillna(0).agg(max).sort_values(ascending=False).to_string())
print(df.groupby(by=cols)['goodActions'].agg('count').unstack(level=[0,1,2]).transpose().to_string())
print(df[df.Iter == 1000000].groupby(by=cols)['goodActions'].agg('mean').unstack(level=[0,1,2]).transpose().to_string())


--------------------------------------------------------------

df = pd.read_csv(r"C:\Users\Marco\Documents\Work\MS\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_marginal_pt_bag.txt")

df2 = df[df.Iter == 1000000]
df2['bag'] = df2.apply(lambda row: int(row['MLargs'].split(' --bag ',1)[1].split(' ',1)[0]) if ' --bag ' in row['MLargs'] else 0, axis=1)
df2['marginal'] = df2.apply(lambda row: 1 if ' --marginal ' in row['MLargs'] else 0, axis=1)
df2['pt0'] = df2.apply(lambda row: 1 if ' --power_t 0' in row['MLargs'] else 0, axis=1)
df2['cb_type'] = df2.apply(lambda row: (row['MLargs'].split(' --cb_type ',1)[1].split(' ',1)[0]), axis=1)
df2['lr'] = df2.apply(lambda row: float(row['MLargs'].split(' -l ',1)[1].split(' ',1)[0]), axis=1)

print(df2.groupby(by=['bag','cb_type','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('max').unstack(level=[0,1,2,3,4,5]).fillna(0).agg(max).sort_values(ascending=False).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['bag','cb_type','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('max').unstack(level=[0,1,2,3,4,5]).fillna(0).agg(max).sort_values(ascending=False).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['bag','cb_type','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('max').unstack(level=[0,1,2]).fillna(0).agg(max).sort_values(ascending=False).to_string())
print(df2.groupby(by=['bag','cb_type','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('max').unstack(level=[0,1,2]).fillna(0).agg(max).sort_values(ascending=False).to_string())
print(df2.groupby(by=['bag','cb_type','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('max').unstack(level=[0,1,2,3]).fillna(0).agg(max).sort_values(ascending=False).to_string())
print(df2.groupby(by=['bag','cb_type','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('mean').unstack(level=[0,1,2,3]).fillna(0).agg(max).sort_values(ascending=False).to_string())
print(df2.groupby(by=['bag','cb_type','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('count').unstack(level=[0,1,2,3]).fillna(0).agg(max).sort_values(ascending=False).to_string())
print(df2.groupby(by=['bag','cb_type','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('mean').unstack(level=[0,1,2,3]).fillna(0).agg(max).sort_values(ascending=False).to_string())
print(df2.groupby(by=['bag','cb_type','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('mean').unstack(level=[0,1,2,3]).fillna(0).agg(max).to_string())
print(df2.groupby(by=['cb_type','bag','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('mean').unstack(level=[0,1,2,3]).fillna(0).agg(max).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['bag','cb_type','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('max').unstack(level=[0,1,2]).fillna(0).agg(max).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['bag','cb_type','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('max').unstack(level=[0,1,2,3]).fillna(0).agg(max).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['bag','cb_type','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('max').unstack(level=[0,1,2,3,4]).fillna(0).agg(max).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['bag','cb_type','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('max').unstack(level=[0,1,2,3,4,5]).fillna(0).agg(max).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['cb_type','bag','pt0','marginal','baseCost','pStrategy','lr'])['CTR'].agg('max').unstack(level=[0,1,2,3,4,5]).fillna(0).agg(max).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['cb_type','pt0','marginal','baseCost','pStrategy','bag','lr'])['CTR'].agg('max').unstack(level=[0,1,2,3,4,5]).fillna(0).agg(max).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','bag','lr'])['CTR'].agg('max').unstack(level=[0,1,2]).fillna(0).agg(max).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','bag','lr'])['CTR'].agg('max').unstack(level=[0,1,2,3]).fillna(0).agg(max).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','bag','lr'])['CTR'].agg('max').unstack(level=[0,1,2,3,4]).fillna(0).agg(max).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','bag','lr'])['CTR'].agg('max').unstack(level=[0,1,2,4]).fillna(0).agg(max).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','bag','lr'])['CTR'].agg('max').unstack(level=[0,1,2,5]).fillna(0).agg(max).to_string())
df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','bag','lr'])['CTR'].agg('max').unstack(level=[0,1,2,5]).fillna(0).agg(max).plot()
plt.show()

plt.show()
df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','bag','lr'])['CTR'].agg('mean').unstack(level=[0,1,2,5]).fillna(0).agg(max).plot()
plt.show()
df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','bag','lr'])['CTR'].agg('mean').unstack(level=[0,1,2,5]).fillna(0).agg(max)
df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','bag','lr'])['CTR'].agg('mean').unstack(level=[0,1,2,5,6]).fillna(0).agg(max)
df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','bag','lr'])['CTR'].agg('mean').unstack(level=[0,1,2,3,5,6]).fillna(0).agg(max)
print(df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','bag','lr'])['CTR'].agg('mean').unstack(level=[0,1,2,3,5,6]).fillna(0).agg(max).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','bag','lr'])['CTR'].agg('mean').unstack(level=[0,1,2,3,6,5]).fillna(0).agg(max).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','lr','bag'])['CTR'].agg('mean').unstack(level=[0,1,2,3,4,5,6]).fillna(0).agg(max).to_string())
print(df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','lr','bag'])['CTR'].agg('mean').to_string())
df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','lr','bag'])['CTR'].agg('mean').plot()
plt.show()
df2[df2.cb_type != 'ips'].groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','lr','bag'])['CTR'].agg('mean').plot()
plt.show()
df2.groupby(by=['cb_type','baseCost','pStrategy','marginal','pt0','lr','bag'])['CTR'].agg('mean').plot()
plt.show()
df2[df2.cb_type != 'ips'].groupby(by=['marginal','cb_type','baseCost','pStrategy','pt0','lr','bag'])['CTR'].agg('mean').plot()
plt.show()
df2[df2.cb_type != 'ips'].groupby(by=['marginal','pt0','cb_type','baseCost','pStrategy','lr','bag'])['CTR'].agg('mean').plot()
plt.show()
df2[df2.cb_type != 'ips'].groupby(by=['lr','marginal','pt0','cb_type','baseCost','pStrategy','bag'])['CTR'].agg('mean').plot()
plt.show()
df2[df2.cb_type != 'ips'].groupby(by=['bag','marginal','cb_type','baseCost','pStrategy','pt0','lr'])['CTR'].agg('mean').plot()
plt.show()
df2[(df2.cb_type != 'ips') & (df2.bag == 5)].groupby(by=['bag','marginal','cb_type','baseCost','pStrategy','pt0','lr'])['CTR'].agg('mean').plot()
plt.show()
df2[(df2.cb_type != 'ips') & (df2.bag == 5)].groupby(by=['bag','marginal','cb_type','baseCost','pStrategy','pt0','lr'])['CTR'].agg('mean').plot()
plt.show()
df2[(df2.cb_type != 'ips') & (df2.bag == 5)].groupby(by=['bag','marginal','cb_type','baseCost','pStrategy','pt0','lr'])['CTR'].agg('mean')
%history

----------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = [(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_marginal_pt_bag.txt", 'csv'),
         (r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_marginal_pt_epsilon005.txt", 'csv'),
         (r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_marginal_pt_epsilon005_quad.txt", 'csv'),
         (r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions2-10_marginal_I1M.txt", 'json'),
         (r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_I1M.txt", 'json'),
         (r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_ABU.txt", 'csv')]

files = [(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_marginal_pt_eps_bag.ALL.temp.txt", 'csv'),
         (r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions2-10_marginal_I1M.txt", 'json'),
         (r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_I1M.txt", 'json')]
         
aggfunc_list = ['mean','min','median','max','count']
cols = ['pStrategy','baseCost','cb_type','ignore','quad','bag','pt0','marginal','lr']
         
df_list = []
for fp,ext in files:
    print(fp)
    if ext == 'json':
        df = pd.read_json(fp, lines=True)
        df = df[df.numActions == 10]
        col_name = 'ml_args'

    else:
        df = pd.read_csv(fp)
        col_name = 'MLargs'
    print('Adding columns...')
    df['bag'] = df.apply(lambda row: int(row[col_name].split(' --bag ',1)[1].split(' ',1)[0]) if ' --bag ' in row[col_name] else 0, axis=1)
    df['marginal'] = df.apply(lambda row: row[col_name].split(' --marginal ',1)[1].split(' ',1)[0] if ' --marginal ' in row[col_name] else 'No', axis=1)
    df['ignore'] = df.apply(lambda row: row[col_name].split(' --ignore ',1)[1].split(' ',1)[0] if ' --ignore ' in row[col_name] else 'No', axis=1)
    df['quad'] = df.apply(lambda row: row[col_name].split(' -q ',1)[1].split(' ',1)[0] if ' -q ' in row[col_name] else 'No', axis=1)
    df['pt0'] = df.apply(lambda row: 1 if ' --power_t 0' in row[col_name] else 0, axis=1)
    df['cb_type'] = df.apply(lambda row: (row[col_name].split(' --cb_type ',1)[1].split(' ',1)[0]), axis=1)
    df['lr'] = df.apply(lambda row: float(row[col_name].split(' -l ',1)[1].split(' ',1)[0]), axis=1)
    
    print('Adding pivot_columns...')
    df_list.append(df.pivot_table(index=cols,values='CTR', aggfunc=aggfunc_list))
df = pd.concat(df_list)

df.to_csv(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_ALL.stats9.csv")

fpo = r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions11.temp.csv"

------------------------------ MERGED ----------------------------------
l861 = [x for x in open(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_10M_ALL-Simulator_v21_861_2ff79cd518e896.txt.allLines.txt")]
l870 = [x for x in open(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_10M_ALL-Simulator_v3_870_eddda453a0.txt.allLines.txt")]

l = [x for x in l861[1:] if 'warning' not in x and ' dr ' not in x and ' --bag ' not in x and ' --cover ' not in x and ' --soft' not in x and ' --regcb' not in x]

ls = set(l870)

l2 = []
cnt = 0
for x in l:
	temp = x.split(',')
	s = ','.join([temp[0],'10,10',temp[2],str(int(temp[2])-int(temp[3]))]+temp[4:])
	if s not in ls:
		l2.append(s)
	else:
		cnt += 1

fpA = r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_10M_ALL-Simulator_v3_870_eddda453a0.merged.txt.allLines.txt"
fp = r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_10M_ALL-Simulator_v3_870_eddda453a0.merged.txt"
with open(fp,'w') as f, open(fpA,'w') as fA:
	for x in l870+l2:
		fA.write(x)
		if 'MLargs' in x or ',10000000,' in x:
			f.write(x)

-------------------------------------------------------------------------------

def create_csv_pt(fp, fpo=None, col_name='MLargs', best_idx_fp=None, idx_margin=0, maxRewardSeed=None):

    if not fpo:
        fpo = fp+'.pivot.csv'

    print('Load:',fp)
    df = pd.read_csv(fp)
    
    print('Filtering rows...')
    old_rows = len(df)
    df = df[~df[col_name].str.contains('warning')]
    print('Total rows:',len(df))
    print('Filtered rows:',old_rows-len(df))
    
    if maxRewardSeed is not None:
        df = df[df.rewardSeed <= maxRewardSeed]
    
    print('Creating columns - Loop')
    expl_l = []
    marginal_l = []
    ignore_l = []
    q_l = []
    pt_l = []
    cb_l = []
    lr_l = []
    eps_l = []
    psi_l = []
    coin_l = []
    ignore_lin_l = []
    const_l = []
    for s in df[col_name]:
        if ' --bag ' in s:
            expl = 'bag'+(s.split(' --bag ',1)[1].split(' ',1)[0]) + ('g' if ' --greedify ' in s else '')
        elif ' --cover ' in s:
            expl = 'cover'+(s.split(' --cover ',1)[1].split(' ',1)[0]) + ('n' if ' --nounif ' in s else '')
            # expl += '_psi'+(s.split(' --psi ',1)[1].split(' ',1)[0] if ' --psi ' in s else '1')
        elif ' --softmax ' in s:
            expl = 'softmax'+(s.split(' --lambda ',1)[1].split(' ',1)[0] if ' --lambda ' in s else '1')
        elif ' --regcb' in s:
            expl = 'regcbopt' if 'regcbopt' in s else 'regcb'
            expl += (s.split(' --mellowness ',1)[1].split(' ',1)[0] if ' --mellowness ' in s else '0.1')
        else:
            expl = 'epsG'
        expl_l.append(expl)
        eps_l.append(float(s.split(' --epsilon ',1)[1].split(' ',1)[0]) if ' --epsilon ' in s else (0.05 if expl == 'epsG' else 0.0))
        psi_l.append(float(s.split(' --psi ',1)[1].split(' ',1)[0]) if ' --psi ' in s else 'No')
        marginal_l.append(s.split(' --marginal ',1)[1].split(' ',1)[0] if ' --marginal ' in s else 'No')
        ignore_l.append(s.split(' --ignore ',1)[1].split(' ',1)[0] if ' --ignore ' in s else 'No')
        ignore_lin_l.append(s.split(' --ignore_linear ',1)[1].split(' ',1)[0] if ' --ignore_linear ' in s else 'No')
        q_l.append(s.split(' -q ',1)[1].split(' ',1)[0] if ' -q ' in s else 'No')
        if ' --power_t ' in s:
            pt_l.append(s.split(' --power_t ',1)[1].split(' ',1)[0])
        else:
            pt_l.append('0.5')
        cb_l.append(s.split(' --cb_type ',1)[1].split(' ',1)[0])
        lr_l.append(float(s.split(' -l ',1)[1].split(' ',1)[0]))
        const_l.append('n' if ' --noconstant' in s else 'y')
        coin_l.append('y' if ' --coin' in s else 'n')

    print('Creating columns - Assign')
    df['expl'] = expl_l
    df['eps'] = eps_l
    df['psi'] = psi_l
    df['marginal'] = marginal_l
    df['ignore'] = ignore_l
    df['quad'] = q_l
    df['pt'] = pt_l
    df['cb_type'] = cb_l
    df['lr'] = lr_l
    df['ignore_lin'] = ignore_lin_l
    df['const'] = const_l
    df['coin'] = coin_l
    if 'clickCost' in df:
        df['costVec'] = '['+df['noClickCost'].astype(str)+','+df['clickCost'].astype(str)+']'
    else:
        df['costVec'] = '['+df['baseCost'].astype(str)+','+(df['baseCost']-df['deltaCost']).astype(str)+']'
        
        
    print('Create pivot table csv file')
    aggfunc_list = ['mean','min','median','max','count','std']
    cols = ['pStrategy','costVec','cb_type','ignore','quad','ignore_lin','expl','psi','eps','const','coin','pt','marginal','lr','Iter']
    p = df.pivot_table(index=cols,values=['GoodActions','dA','CTR'], aggfunc=aggfunc_list).rename(columns={'GoodActions':'GA'})
    p.columns = [x+'_'+y for x,y in zip(p.columns.get_level_values(1),p.columns.get_level_values(0))]
    if 'GA_std' not in p:
        p['GA_std'] = 0
        p['dA_std'] = 0
    p.to_csv(fpo)
    
    # if best_idx_fp:
        # print('Create best indices csv file')
        # df2 = df.pivot_table(index=cols,values='CTR', aggfunc='mean')
        # idx = df2.groupby(cols[:-1])['CTR'].transform(max) <= df2['CTR']+idx_margin
        # df2[idx].to_csv(best_idx_fp)
        
        # l = [x for x in open(best_idx_fp)]            
        # with open(best_idx_fp, 'w') as f:
            # f.write('MLargs,'+l[0])
            # for row in l[1:]:
                # f.write(MLargs_from_row(row) + ',' + row)

def create_csv_pt_lr(fp, fpo=None, col_name='MLargs'):

    if not fpo:
        fpo = fp+'.pivot.csv'

    print('Load:',fp)
    df = pd.read_csv(fp)
    print('Creating columns - Loop')
    lr_l = []
    VWargs_l = []
    for s in df[col_name]:
        temp = s.split(' -l ',1)
        temp2 = temp[1].split(' ',1)
        lr_l.append(float(temp2[0]))
        VWargs_l.append(temp[0] + ' ' + temp2[1])

    print('Creating columns - Assign')
    df['lr'] = lr_l
    df['VWargs'] = VWargs_l
        
    print('Create pivot table csv file')
    aggfunc_list = ['mean','min','median','max','count','std']
    cols = ['VWargs','pStrategy','baseCost','deltaCost','lr']
    p = df.pivot_table(index=cols,values='CTR', aggfunc=aggfunc_list)
    p.columns = p.columns.get_level_values(0)
    p.to_csv(fpo)
                

def MLargs_from_row(row):
    cols = ['pStrategy','baseCost','cb_type','ignore','quad','ignore_lin','expl','eps','pt','marginal','lr']
    s = {cols[i]: item for i,item in enumerate(row.split(',')) if 2 <= i < len(cols)}
    
    MLargs = '--cb_explore_adf'
    if s['ignore'] != 'No':
        MLargs += ' --ignore ' + s['ignore']
    if s['quad'] != 'No':
        MLargs += ' -q ' + s['quad']
    if s['ignore_lin'] != 'No':
        MLargs += ' --ignore_linear ' + s['ignore_lin']
    if s['bag'] != '0':
        MLargs += ' --bag ' + str(s['bag'])
        if s['greedify'] == 'Yes':
            MLargs += ' --greedify'
        if s['eps'] != '0':
            MLargs += ' --epsilon ' + s['eps']
    MLargs += ' -l ' + s['lr']
    MLargs += ' --cb_type ' + s['cb_type']
    MLargs += ' --power_t ' + s['pt']
    if s['marginal'] != 'No':
        MLargs += ' --marginal ' + s['marginal']
    return MLargs

--------------------------------------------------
files = [x.path+'\\simulator.exe' for x in os.scandir(r'C:\work\bin\cs_sim_SINGLE') if x.name.startswith('Simulator_v') and 'stopwatch' in x.name and ('_v21_' in x.name or '_v3' in x.name)]

files = [x for x in files if '_v3' in x and 'v3_' not in x]

files = [x for x in files if 'e6db601668567' in x or '7d3cf61e876' in x]

ml_args = '--cb_explore_adf --ignore XA -q UB --epsilon 0.2 -l 0.5 --cb_type mtr --power_t 0'
base_args = '10 10 0.03 0.04'
num_actions = 10
num_contexts = 10
min_p = 0.03
max_p = 0.04
baseCost = 1
deltaCost = 1
num_iter = 100000
pStrategy = 2
seed = 0

for seed in range(100):
    for fp in files:
        if '7d3cf61e876' in fp and pStrategy in {2,14}:
            clip_p_str = ' --clip_p '+('0.1' if pStrategy == 14 else '0.5')
            args = ' \"{}\" {} {} {} 0 {} {} {} {}'.format(ml_args+clip_p_str, base_args, baseCost,baseCost-deltaCost,num_iter,num_iter,seed,num_iter+1)
            print(fp+args)
            os.system(fp+args)
    
        if '_v2' in fp:
            args = ' \"{}\" {} {} {} {} {} {} {}'.format(ml_args, base_args, baseCost,deltaCost,pStrategy,num_iter,num_iter,seed)
        else:
            args = ' \"{}\" {} {} {} {} {} {} {} {}'.format(ml_args, base_args, baseCost,baseCost-deltaCost,pStrategy,num_iter,num_iter,seed,num_iter+1)
        print(fp+args)
        os.system(fp+args)

--------------------------------------------------

from subprocess import check_output, STDOUT
import os
files = [x.path+'\\simulator.exe' for x in os.scandir(r'C:\work\bin\cs_sim_SINGLE') if x.name.startswith('Simulator_v') and 'Simulator_v33_870_ba8f1d69fcbc' in x.name]
fp = files[0]
fo = r"C:\\Users\\marossi\\OneDrive - Microsoft\\Data\\cb_hyperparameters\\Actions_Contexts_matrix.5M.swap2M.txt"

noClick = 0
click = -1
num_iter = 5000000
num_iter_swap = 2000000
iter_mod = 10000
actions_v = [10, 50, 100, 250, 500]
contexts_v = [1, 10, 50, 100, 500]
ctr_max_v = [0.05, 0.04, 0.1]
ctr_min = 0.03
pStrategy = 0

seed = 0
while True:
    for namespace_str in [' --ignore A -q UB', ' --ignore B -q UA']:
        for clip in [' --clip_p 0.1', ' --clip_p 0.5', '']:
            for learning_rate in [' --coin', ' -l 1e-3 --power_t 0']:
                ml_args = '--cb_explore_adf --cb_type mtr{}{}{}'.format(namespace_str, learning_rate, clip)
                for ctr_max in ctr_max_v:
                    for actions in actions_v:
                        for contexts in contexts_v:
                            if contexts > actions:
                                continue
                            
                            args = '{} {} {} {} {} {} {} {} {} {} {}'.format(actions,contexts,ctr_min,ctr_max,noClick,click,pStrategy,num_iter,iter_mod,seed,num_iter_swap)
                            cmd_str = fp + ' \"{}\" '.format(ml_args) + args
                            cmd_list = [fp, ml_args] + args.split(' ')
                            print(cmd_str)
                            x = check_output(cmd_list, stderr=STDOUT, universal_newlines=True)
                            
                            with open(fo, 'a') as f:
                                f.write(cmd_str+'\n')
                                f.write(x+'\n')
    seed += 1

------------------------------------------------------

fp = r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\Actions_Contexts_matrix.5M.swap2M.txt"

def create_5Mcsv(fp, fpo=None):
    
    if fpo is None:
        fpo = fp+".5M.csv"

    print('Reading file...')
    l = open(fp).read().split('\n\n')

    print('Starting loop...')
    data = []
    for y in l:
        lines = y.split('\n')
        if len(lines) > 1:
            z = lines[1].split(',10000,',1)[0]
            if ' 0.03 0.04 ' in lines[0]:
                z += ',0.03,0.04'
            elif ' 0.03 0.05 ' in lines[0]:
                z += ',0.03,0.05'
            elif ' 0.03 0.1 ' in lines[0]:
                z += ',0.03,0.1'
            else:
                print('Error')
            if '--coin' in lines[0]:
                z += ',y'
            else:
                z += ',n'
            if '--clip_p 0.1' in lines[0]:
                z += ',0.1'
            elif '--clip_p 0.5' in lines[0]:
                z += ',0.5'
            else:
                z += ',0.0'
            if ' --ignore A -q UB' in lines[0]:
                z += ',B'
            else:
                z += ',A'
            data.append((lines[0],list(zip(*[map(float, x.split(',')[-4:]) for x in lines[1:]])), z))

    print('CMD loaded:',len(data))
    print('Writing csv...')
    with open(fpo, 'w') as f:
        f.write('cmd,Mlargs,actions,contexts,noClick,click,pStrategy,Seed,minP,maxP,coin,Clip,NS,Iter,CTR,GA,dA\n')

        index_2M = int(2000000/10000)-1
        index_5M = int(5000000/10000)-1
        for x in data:
            f.write(','.join(map(str,[x[0], x[2]]+[y[index_2M] for y in x[1]]))+'\n')
            f.write(','.join(map(str,[x[0], x[2]]+[y[index_5M] for y in x[1]]))+'\n')
            
    return data

def plot_5M(in_list=['" 100 1 ', '0.03 0.04 ', '5000000 10000 0 2000000', '--ignore A'], not_in_list=[], leg_size=12, bta=(0, 0, 0, 0), loc=3, ncol=1):
    for x in data:
        # if '" 100 1 ' in x[0] and '0.03 0.04 ' in x[0] and '5000000 10000 0 2000000' in x[0] and '--ignore A' in x[0]:
        if all(y in x[0] for y in in_list) and all(y not in x[0] for y in not_in_list):
            print(x[0])
            plt.plot(x[1][0],x[1][3], label=x[0].split('.exe ',1)[1])
    # plt.title('Actions: 500 Contexts: 1')
    # plt.legend(loc=8, prop={'size': leg_size})
    plt.legend(bbox_to_anchor=bta, loc=loc, ncol=ncol, mode="expand", prop={'size': leg_size})
    plt.margins(0)
    plt.show()
    
------------------------------------------------------

import pandas as pd
import os
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\Actions_Contexts_matrix.5M.swap2M.txt.5M.csv")

# df[(df.NS == 'B') & (df.actions == 10) & (df.contexts == 10)].pivot_table(index=['Iter','actions','contexts','maxP','Clip','coin'],values='CTR', aggfunc=['mean','min','median','max','std','count'])

# plt.rc('font', **{'size':20})

column = 'CTR'
rot = 60
dpi = 500
for Iter in [2000000, 5000000]:
    for contexts in [10, 100]:
        for maxP in [0.05, 0.04, 0.1]:
                    
            df2 = df[(df.NS == 'B') & (df.contexts == contexts) & (df.maxP == maxP) & (df.Iter == Iter)]
            for i,groupby in enumerate([['coin','Clip','actions'], ['Clip','coin','actions']]):
                print(groupby, contexts, maxP, Iter)
                # ax = df2.boxplot(column=column, by=groupby, rot=rot)

                df2 = pd.DataFrame({col:vals[column] for col, vals in df2.groupby(groupby)})
                ax = df2.boxplot(rot=rot)
                
                # Adding scatter for mean
                means = df2.mean()
                ax.scatter(list(range(1,len(means)+1)), means, s=10)
                
                plt.title('Contexts: {} - MaxP: {} - Iter: {}M'.format(contexts, maxP, int(Iter/1000000)))
                ax.set_ylabel('CTR')
                ax.set_xlabel('groupby: ['+','.join(groupby)+']')
                ax.set_ylim([0.0297,maxP])

                fig = plt.gcf()
                fn = os.path.join(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\figures", 'boxplot_v3_groupby{}-actions-contexts{}-maxP{}-iter{}M.png'.format(i,contexts, maxP, int(Iter/1000000)))

                # plt.figure(figsize=(800/dpi, 800/dpi), dpi=dpi)

                fig.set_size_inches((18, 11), forward=False)
                fig.savefig(fn, dpi=dpi, bbox_inches='tight')
                # plt.show()
                plt.close()
            


------------------------------------------------------
df[df.Iter == 1000000].pivot_table(columns=['MLargs','pStrategy','baseCost'],values='CTR', aggfunc=['mean','count'])
df = pd.read_csv(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_marginal_pt_bag.txt")
df.describe()
df.columns()
df.columns
pd.pivot_table(df[df.Iter == 1000000], columns=['MLargs','pStrategy','baseCost'],values='CTR', aggfunc=['mean','count'])
pd.pivot_table(df[df.Iter == 1000000], index=['MLargs','pStrategy','baseCost'],values='CTR', aggfunc=['mean','count'])
df[df.Iter == 1000000].pivot_table(index=['MLargs','pStrategy','baseCost'],values='CTR', aggfunc=['mean','count'])
df[df.Iter == 1000000].pivot_table(index=['MLargs','pStrategy','baseCost'],values='CTR', aggfunc=['mean','min','median','max','count']).to_csv(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_marginal_pt_bag.txt.csv")
df2 = pd.read_csv(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_marginal_pt_epsilon005.txt")
df2[df2.Iter == 1000000].pivot_table(index=['MLargs','pStrategy','baseCost'],values='CTR', aggfunc=['mean','min','median','max','count']).to_csv(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_marginal_pt_epsilon005.txt.csv")
df2[df2.Iter == 1000000].pivot_table(index=['MLargs','pStrategy','baseCost'],values='CTR', aggfunc=['mean','min','median','max','count']).to_csv(r)
df3 = pd.read_csv(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_marginal_pt_epsilon005_quad.txt")
df3[df3.Iter == 1000000].pivot_table(index=['MLargs','pStrategy','baseCost'],values='CTR', aggfunc=['mean','min','median','max','count']).to_csv(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_marginal_pt_epsilon005_quad.txt.csv")
df0 = pd.read_json(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions2-10_marginal_I1M.txt", lines=True)
df0.columns
df0[df0.numActions == 10].pivot_table(index=['ml_args','pStrategy','baseCost'],values='CTR', aggfunc=['mean','min','median','max','count']).to_csv(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_marginal.txt.csv")
df00 = pd.read_json(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_I1M.txt", lines=True)
df00[df00.numActions == 10].pivot_table(index=['ml_args','pStrategy','baseCost'],values='CTR', aggfunc=['mean','min','median','max','count']).to_csv(r"C:\Users\marossi\OneDrive - Microsoft\Data\cb_hyperparameters\CTR-4-3_Actions10_I1M.txt.csv")



1) difference between --ignore ABU and --ignore XA -q UB
2) difference between baseCost 0 and 1
3) Difference between p=0.1 and p=0.9 adapting lr (it should be same at least for mtr)
4) Marginal:
    - --ignore X: not beneficial
    - --ignore ABU: sometimes beneficial (dr, pt 0) sometime detrimental (mtr)

