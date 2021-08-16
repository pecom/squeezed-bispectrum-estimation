import numpy as np
import scipy.stats as stats
import scipy.signal as signal


class SampleLine:
    
    def __init__(self, N, periodic=False, power=-1, amp=-1):
        self.N = N
        rng = np.random.default_rng()
        periodic = int(N/2)
        self.dsize = int(N/2 + 1)
        self.rng = rng
        
        if periodic:
            smaldat = rng.standard_normal(periodic)
            fdat = np.concatenate((smaldat, smaldat[::-1]))
            self.full_data = fdat
        
        if power==-1:
            self.ps_power = rng.uniform()*3 + 1
        else:
            self.ps_power = power
        if amp==-1:
            self.ps_amp = rng.uniform()*5
        else:
            self.ps_amp = amp
        
        self.kn, self.ps = self.create_ps(self.ps_power, self.ps_amp, self.N)
        self.ps_sqrt = np.sqrt(self.ps)
        self.mask = self.create_mask()
        
    def create_ps(self, power, amp, ds):
        kvec = np.fft.fftfreq(ds) * ds
        if power > 0:
            kvec[0] = np.inf
        knorm = np.abs(kvec)
        knorm = knorm[:self.dsize]

        ps = amp/(knorm**power)
        return knorm, ps
    
    def create_sample_spec(self):
        line_subset = self.rng.standard_normal((self.dsize, 2)).view(dtype=complex).flatten()
        f_spec = line_subset * self.ps_sqrt
        return f_spec
    
    def create_dsample(self, verbose=False):
        fspec = self.create_sample_spec()
        ifspec = np.fft.irfft(fspec)
        return ifspec
                  
    def basic_psest(self):
        dat = self.create_dsample(verbose=False)
        scaled_fcor = (np.abs(np.fft.rfft(dat))**2)
        return scaled_fcor
    
    def masked_basic_psest(self):
        dat = self.create_dsample(verbose=False)
        masked_dat = dat * self.mask
        scaled_fcor = (np.abs(np.fft.rfft(masked_dat))**2)
        return scaled_fcor
                  
    def create_mask(self, n_holes=2):
        mask = np.ones(self.N)
        rstart = np.sort(self.rng.integers(0, self.N, n_holes))
        
        hsize = 32
        # For now make fixed hole size:
        for r in rstart:
            mask[r:r+hsize] = 0
        return mask
        
    
 