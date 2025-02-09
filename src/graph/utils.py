import numpy as np
import time

def generate_climbing_walk(steps, start_value=0, step_size=1):
    x_values = np.arange(int(time.time()), int(time.time()) + steps)
    y_values = np.arange(0, 0+steps)
#     y_values = np.cumsum(np.random.choice([-step_size, step_size], size=steps)) + start_value
    return np.column_stack((x_values, y_values))

def generate_random_walk(steps, start_value=0, step_size=1):
    x_values = np.arange(int(time.time()), int(time.time()) + steps)
    y_values = np.cumsum(np.random.choice([-step_size, step_size], size=steps)) + start_value
    return np.column_stack((x_values, y_values))

