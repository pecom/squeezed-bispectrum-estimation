import numpy as np
import scipy.stats as stats
import scipy.signal as signal


class SampleLine:
    
    def __init__(self, N, c, power=-1, amp=-1):
        self.N = N
        self.copies = c
        dsize = int(N*c)
        rng = np.random.default_rng()
        self.rng = rng
        line_data = rng.standard_normal(N)
        self.full_data = np.tile(line_data, c)
        self.dsize = dsize
        
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
        rstart = self.rng.integers(0, self.dsize-self.N)
        line_subset = self.full_data[rstart:rstart+self.N]
        ff_line = np.fft.fft(line_subset)
        f_spec = ff_line * self.ps_sqrt
        return f_spec
    
    def create_dsample(self, verbose=False):
        fspec = self.create_sample_spec()
        ifspec = np.fft.ifft(fspec)
        dset = np.real(ifspec)
        if verbose:
            print("Imaginary part magnitude: ", np.mean(np.abs(ifspec.imag)))
        return dset
                  
        
    # Estimate power spectrum given realizations. Likely complete nonsense/garbage below:
    def basic_psest(self):
        dat = self.create_dsample(verbose=False)
        t1 = np.fft.fft(dat)
        good_stop = int(len(t1)/2)

        corrs = signal.correlate(t1, t1)
        corr_lags = signal.correlation_lags(len(t1), len(t1))
        corrs = corrs/np.max(corrs)

        pos_lags = corr_lags >= 0

        f_corr = np.fft.fft(corrs)
        corr_k = np.arange(int(np.sum(pos_lags)/2)).astype(float)
        corr_k[0] = np.inf

        self.corr_k = corr_k
        scaled_fcor = np.abs(f_corr[pos_lags][:good_stop])**2/corr_k**2
        return scaled_fcor
    
    def multi_psest(self, N=1000):
        fin_est = []
        for i in range(N):
            fin_est.append(self.basic_psest())
        fin_est = np.array(fin_est)
        meaned_fest = np.mean(fin_est, axis=0)
        return meaned_fest
                  
        
    
 