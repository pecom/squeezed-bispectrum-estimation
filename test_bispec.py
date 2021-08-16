import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt
import bispec_1d
from bispec_1d import SampleLine

Ndata = 512
test_line = SampleLine(Ndata)
print(test_line.create_dsample())

