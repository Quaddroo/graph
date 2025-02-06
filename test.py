# %%
from graph import Graph
import numpy as np
import time

def generate_random_walk(steps, start_value=0, step_size=1):
    x_values = np.arange(int(time.time()), int(time.time()) + steps)
    y_values = np.cumsum(np.random.choice([-step_size, step_size], size=steps)) + start_value
    return np.column_stack((x_values, y_values))

random_walk_data = generate_random_walk(10000000, step_size=0.5)


# %%
g = Graph(run_in_new_thread=False)
g.add_line_seq(random_walk_data)

g.start()

# %%
