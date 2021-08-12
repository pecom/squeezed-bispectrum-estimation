import numpy as np
import scipy.stats as stats
import scipy.signal as signal

class SampleLine:
    
    def __init__(self, N, c):
        self.Ndata = N
        self.ncopy = c
        self.dsize = N*c
        self.half_n = int(N/2) + 1
        rng = np.random.default_rng()
        self.rng = rng
        line_data = rng.standard_normal(N)
        full_data = np.tile(line_data, c)
        self.line_data = line_data
        self.full_data = full_data
    
    
    def basic_psest(self):
        rstart = self.rng.integers(0, self.dsize - self.Ndata)
        line_subset = self.full_data[rstart:rstart+self.Ndata]

        f_line = np.fft.fft(line_subset)
        corrs = signal.correlate(f_line, f_line)
        corrs = corrs/np.max(corrs)
        corr_lags = signal.correlation_lags(len(f_line), len(f_line))

        pos_lags = corr_lags >= 0

        f_corr = np.fft.fft(corrs)
        corr_k = np.arange(self.half_n).astype(float)
        corr_k[0] = np.inf
        self.corr_k = corr_k

        squared_fcor = np.abs(f_corr[:self.half_n])**2
        return squared_fcor
    
    def multi_psest(self, N=1000):
        fin_est = []
        for i in range(N):
            fin_est.append(self.basic_psest())
        fin_est = np.array(fin_est)
        meaned_fest = np.mean(fin_est, axis=0)
        return meaned_fest
    
    def create_ps_theory(self, power, amp, ds):
        kvec = np.fft.fftfreq(ds) * ds
        if power > 0:
            kvec[0] = np.inf
        knorm = np.abs(kvec)

        ps = amp/(knorm**power)
        return knorm, ps