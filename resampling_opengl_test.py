from resampling_opengl_2 import resample_opengl
from graph import LineGraphSequential

resample_1 = resample_opengl(random_walk_data, 4)
resample_2 = LineGraphSequential.resample(None, random_walk_data, 4)

assert np.all(resample_1[:, 1] == resample_2[:, 1])

# the timestamps have rounding issues, I consider this a sufficient test, but ideally should test those also eventually.

