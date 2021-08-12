import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt
import bispec_1d
from bispec_1d import SampleLine

test_line = SampleLine(512, 2, power=2.1)
meaned_fest = test_line.multi_psest(1000)


fig, ax = plt.subplots(1, figsize=(6,6))
ax.plot(test_line.kn, test_line.ps, label="Theory")
ax.plot(test_line.corr_k, meaned_fest, label="Data")
ax.semilogx()
ax.semilogy()
ax.legend()
plt.show()