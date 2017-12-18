import numpy as np
import pandas as pd

class StopSimulator(object):
    def __init__(self):
        default_params = {'correct_params': {'drift': .14, 'thresh': 42},
                          'incorrect_params': {'drift': .075, 'thresh': 42}}
        self.__dict__.update(**default_params)
        
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
        
    def get_go_rt(self, ntrials=100):
        go_rts = []
        for params in [self.correct_params, self.incorrect_params]:
            drift = params['drift']
            thresh = params['thresh']
            rts = [self.random_wald(drift, thresh) for _ in range(ntrials)]
            go_rts.append(rts)
        df = pd.DataFrame(go_rts, index = ['correct_RT', 'incorrect_RT']).T
        df.insert(2, 'correct', df.apply(lambda x: x[0]<x[1], axis=1))
        return df
            

exp = StopSimulator()
go_df = exp.get_go_rt(1000)


