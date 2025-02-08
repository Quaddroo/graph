# %%
%load_ext autoreload
%autoreload 2

from resampling_opengl import resample_opengl, setup_environment, setup_pygame, generate_initial_data_textures, prep_data_for_texture, split_timestamps_into_f32, get_resulting_pixeldata, resample_opengl_1M
from graph import LineGraphSequential
from utils import generate_random_walk
import numpy as np
from time import perf_counter_ns

# %%
for resampling_n in range(4, 11):
#     resampling_n = 5
    passed = "passed"
    data = generate_random_walk(10000000, step_size=0.5)

    setup_pygame() # This sets up an opengl environment. Since it must occur no matter what when launching Graph, it is unfair to include in the performance.

    t0 = perf_counter_ns()
    resample_1 = resample_opengl(data, resampling_n)
    t1 = perf_counter_ns()
    resample_2 = LineGraphSequential.resample(None, data, resampling_n)
    t2 = perf_counter_ns()

    try:
        assert np.all(resample_1[:, 1] == resample_2[:, 1])
    except Exception as e:
        print("failed")
        print(e)
        passed = "FAILED"
    # the timestamps have rounding issues, I consider this a sufficient test, but ideally should test those also eventually.

    print(f"""
    Test {passed} for {resampling_n} n.
    OpenGL resample time: {t1 - t0} ns
    Numba + np resample time: {t2 - t1} ns
    OpenGL/Numba: {(t1-t0)/(t2-t1)}
    """)
# 
# # %%
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

    # %%
    (resample_1[:, 1] == resample_2[:, 1])[group_index:group_index + 4]

    # %%
    resample_1[:, 1][group_index:group_index+4] # group number 166
    # %%
    resample_2[:, 1][group_index:group_index+4] # group number 166
    # %%
    data[:, 1][group*resampling_n:group*resampling_n+resampling_n] # each group is 6 elements long


[len(r) for r in resampled_batches]


# %%
resampling_n = 6
data = generate_random_walk(10000000, step_size=0.5)

t0 = perf_counter_ns()
resample_1 = resample_opengl(data, resampling_n)
t1 = perf_counter_ns()
resample_2 = LineGraphSequential.resample(None, data, resampling_n)
t2 = perf_counter_ns()
# %%
