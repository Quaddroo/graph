# %%
%load_ext autoreload
%autoreload 2

from resampling_opengl import resample_opengl, setup_environment, setup_pygame, generate_initial_data_textures, prep_data_for_texture, split_timestamps_into_f32, get_resulting_pixeldata, resample_opengl_1M, setup_glfw, prep_timestamps_for_texture, prep_timestamps_for_texture_alt
from graph import LineGraphSequential
from utils import generate_random_walk
import numpy as np
from time import perf_counter_ns

# %%


# setup_pygame() # This sets up an opengl environment. Since it must occur no matter what when launching Graph, it is unfair to include in the performance comparison.

resampling_ns = range(4, 11)
data_amounts = (543643, 1000000, 10000000)
total_num_tests = len(resampling_ns) * len(data_amounts)
num_passes = 0
num_fails = 0
for resampling_n in resampling_ns:
    for data_amount in data_amounts:
        passed = "passed"
        data = generate_random_walk(data_amount, step_size=0.5)

        t0 = perf_counter_ns()
        setup_glfw() # need this after all, but basically no perf impact
        resample_1 = resample_opengl(data, resampling_n)
        t1 = perf_counter_ns()
        resample_2 = LineGraphSequential.resample(None, data, resampling_n)
        t2 = perf_counter_ns()

        try:
            assert np.all(resample_1[:, 1] == resample_2[:, 1])
            assert np.argmax(resample_2[:, 0] - resample_1[:, 0] > 10) == 0
            num_passes += 1
        except Exception as e:
            print("failed")
            print(e)
            passed = "FAILED"
            num_fails += 1
        # the timestamps have rounding issues, I consider this a sufficient test, but ideally should test those also eventually.

        print(f"""
        Test {passed} for {resampling_n} n.
        Data amount: {data_amount}
        OpenGL resample time: {t1 - t0} ns
        Numba + np resample time: {t2 - t1} ns
        OpenGL/Numba: {(t1-t0)/(t2-t1)}

        total: {num_passes}p {num_fails}f / {total_num_tests}
        """)
# 
# %%
# # 
# import os
# os.environ["PAGER"] = "cat" # avoids it using less to page shit
# %load_ext line_profiler
# %lprun -f resample_opengl_1M resample_opengl(data, resampling_n)
# # 
# 
def debug_shit():
    diff_index = np.argmin(resample_1[:, 1] == resample_2[:, 1])
    print(diff_index)
    group = diff_index // 4
    group_index = group * 4

    (resample_1[:, 1] == resample_2[:, 1])[group_index:group_index + 4]

    resample_1[:, 1][group_index:group_index+4] # group number 166
    resample_2[:, 1][group_index:group_index+4] # group number 166
    data[:, 1][group*resampling_n:group*resampling_n+resampling_n] # each group is 6 elements long


    sum([len(r) for r in resampled_batches])

# %%
num_fails = 0
num_passes = 0
total_num_tests = 0
passed = "passed"

data_amount = 10000000
resampling_n = 60
data = generate_random_walk(data_amount, step_size=0.5)

t0 = perf_counter_ns()
setup_glfw() # need this after all, but basically no perf impact
resample_1 = resample_opengl(data, resampling_n)
t1 = perf_counter_ns()
resample_2 = LineGraphSequential.resample(None, data, resampling_n)
t2 = perf_counter_ns()

np.argmax(resample_2[:, 0] - resample_1[:, 0] > 10) == 0

# %%

try:
    assert np.all(resample_1[:, 1] == resample_2[:, 1])
    num_passes += 1
except Exception as e:
    print("failed")
    print(e)
    passed = "FAILED"
    num_fails += 1
# the timestamps have rounding issues, I consider this a sufficient test, but ideally should test those also eventually.

print(f"""
Test {passed} for {resampling_n} n.
Data amount: {data_amount}
OpenGL resample time: {t1 - t0} ns
Numba + np resample time: {t2 - t1} ns
OpenGL/Numba: {(t1-t0)/(t2-t1)}

total: {num_passes}p {num_fails}f / {total_num_tests}
""")
# 
# %%
# # 
import os
os.environ["PAGER"] = "cat" # avoids it using less to page shit
%load_ext line_profiler
%lprun -f prep_timestamps_for_texture_alt resample_opengl(data, 10)
# # 
# 
# %%
