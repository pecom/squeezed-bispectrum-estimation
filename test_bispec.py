import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt
import bispec_1d
from bispec_1d import SampleLine

kpow = 2.1
Ndata = 512
test_line = SampleLine(Ndata, 2)
meaned_fest = test_line.multi_psest(1000)
kn, ps = test_line.create_ps_theory(kpow, 1, Ndata)


fig, ax = plt.subplots(1, figsize=(6,6))
ax.plot(kn, ps, label="Theory")
ax.plot(test_line.corr_k, meaned_fest/test_line.corr_k**kpow, label="Data")
ax.semilogx()
ax.semilogy()
ax.legend()
plt.show()