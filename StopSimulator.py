import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class StopSimulator(object):
    def __init__(self):
        default_params = {
                'correct_params': {'drift': .14, 'thresh': 42, 'nondec': 200},
                'incorrect_params': {'drift': .075, 'thresh': 42, 'nondec': 200},
                'stop_params': {'drift': .14, 'thresh': 42, 'nondec': 50}}
        self.__dict__.update(**default_params)
        self.SSDList = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        
    def random_wald(self, drift, thresh):
        """ returns wald
        
        from http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
        """
        mu = thresh/drift
        lam = thresh**2
        v = np.random.rand(); z = np.random.rand()
        y = v**2
        x = mu + (mu**2)*y/(2*lam) \
            - (mu/(2*lam))*(4*mu*lam*y  \
            + (mu**2)*(y**2))**.5;
        if z <= mu/(mu + x):
            wald = x;
        else:
            wald = (mu**2)/x
        return wald
        
    def get_go_trials(self, ntrials=100):
        go_rts = []
        for params in [self.correct_params, self.incorrect_params]:
            drift = params['drift']
            thresh = params['thresh']
            nondec = params['nondec']
            rts = [self.random_wald(drift, thresh) + nondec for _ in range(ntrials)]
            go_rts.append(rts)
        df = pd.DataFrame(go_rts, index = ['correct_RT', 'incorrect_RT']).T
        df.insert(2, 'correct', df.apply(lambda x: x[0]<x[1], axis=1))
        df.insert(0, 'rt', df.apply(lambda x: min(x.correct_RT, x.incorrect_RT), axis=1))
        df = df.reindex(sorted(df.columns), axis=1)
        return df
    
    def get_SS_trials(self, ntrials=100):
        stop_df = pd.DataFrame()
        for SSD in self.SSDList:
            df = self.get_go_trials(ntrials)
            drift, thresh, nondec = [self.stop_params[i] for i in ['drift','thresh','nondec']]
            ssrt = [self.random_wald(drift, thresh) + nondec for _ in range(ntrials)]
            # add stopping columns
            df.insert(0, 'SSD', SSD)
            df.insert(0, 'SSRT', ssrt)
            df.insert(0, 'stop_time', df.SSD + df.SSRT)
            # stopped if SSRT is the fastest RT
            stopped = df.apply(lambda x: min(x.stop_time, x.correct_RT, x.incorrect_RT) == x.stop_time, axis=1)
            df.insert(0, 'stopped', stopped)
            # otherwise calculate RT
            df.rt = df.apply(lambda x: x.rt if not x.stopped else np.nan, axis=1)
            df = df.reindex(sorted(df.columns), axis=1)
            stop_df = pd.concat([stop_df, df], axis=0)
        return stop_df
            
    def get_all_trials(self, ntrials=100):
        go_df = self.get_go_trials(ntrials)
        stop_df = exp.get_SS_trials(ntrials)
        all_trials = pd.concat([go_df, stop_df], axis=0)
        all_trials.SSD.replace(np.nan, -1, inplace=True)
        return all_trials

np.random.seed(5)
exp = StopSimulator()
all_trials = exp.get_all_trials(1000)

# Plots
sns.set_context('poster')

# SSD cumulative density plots
plt.figure(figsize=(12,8))
for SSD, rt in all_trials.query('SSD>-1').groupby('SSD').rt:
    rt = [i for i in rt if not pd.isnull(i)]
    sns.kdeplot(rt, cumulative=True, label=SSD, linewidth=3)
sns.kdeplot(all_trials.query('SSD==-1').rt, cumulative=True, label='Go', 
            linewidth=5, linestyle='..', color='red')
plt.legend(title = 'SSD')
plt.xlabel('Time (ms)')
plt.ylabel('Density')
plt.title('Cumulative RT Density')

# plot inhibition function
plt.figure(figsize=(12,8))
all_trials.groupby('SSD').stopped.apply(np.mean).plot(marker='o',linewidth=3)
plt.ylabel('Probability of Stoping')
plt.title('Inhibition Function')
