import numpy as np
import scipy.stats as stats
import scipy.signal as signal


class SampleLine:
    
    def __init__(self, N, power=-1, amp=-1):
        self.N = N
        rng = np.random.default_rng()
        self.rng = rng
        
        self.full_data = rng.standard_normal(N)
        
        if power==-1:
            self.ps_power = rng.uniform()*3 + 1
        else:
            self.ps_power = power
        if amp==-1:
            self.ps_amp = rng.uniform()*5
        else:
            self.ps_amp = amp
        self.kn, self.ps = self.create_ps(self.ps_power, self.ps_amp, N)
        self.ps_sqrt = np.sqrt(self.ps)
        
    def create_ps(self, power, amp, ds):
        kvec = np.fft.fftfreq(ds) * ds
        if power > 0:
            kvec[0] = np.inf
        knorm = np.abs(kvec)

        ps = amp/(knorm**power)
        return knorm, ps
    
    def create_sample_spec(self):
        line_subset = self.rng.standard_normal((self.N, 2)).view(dtype=complex).flatten()
        f_spec = line_subset * self.ps_sqrt
        return f_spec
    
    def create_dsample(self, verbose=False):
        fspec = self.create_sample_spec()
        ifspec = np.fft.irfft(fspec)
        return ifspec
                  
    # Estimate power spectrum given realizations. Likely complete nonsense/garbage below:
    def basic_psest(self):
        dat = self.create_dsample(verbose=False)
        scaled_fcor = (np.abs(np.fft.rfft(dat))**2)
        return scaled_fcor
    
    def multi_psest(self, N=1000):
        fin_est = []
        for i in range(N):
            fin_est.append(self.basic_psest())
        fin_est = np.array(fin_est)
        meaned_fest = np.mean(fin_est, axis=0)
        return meaned_fest
                  
        
    
 