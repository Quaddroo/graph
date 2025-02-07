# %%
%load_ext autoreload
%autoreload 2

from resampling_opengl_2 import resample_opengl
from graph import LineGraphSequential
from utils import generate_random_walk
import numpy as np
from time import perf_counter_ns

# %%
data = generate_random_walk(1000000, step_size=0.5)

t0 = perf_counter_ns()
resample_1 = resample_opengl(data, 4)
t1 = perf_counter_ns()
resample_2 = LineGraphSequential.resample(None, data, 4)
t2 = perf_counter_ns()

assert np.all(resample_1[:, 1] == resample_2[:, 1])
# the timestamps have rounding issues, I consider this a sufficient test, but ideally should test those also eventually.

print(f"""
Test passed. 
OpenGL resample time: {t1 - t0} ns
Numba + np resample time: {t2 - t1} ns
OpenGL/Numba: {(t1-t0)/(t2-t1)}
""")

# %%

import os
os.environ["PAGER"] = "cat" # avoids it using less to page shit
%load_ext line_profiler
%lprun -f my_func my_func()


# %%
