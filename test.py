# %%
from graph import Graph
import numpy as np
import time
from utils import generate_random_walk

random_walk_data = generate_random_walk(10000000, step_size=0.5)

# %%
g = Graph(run_in_new_thread=False)
g.add_line_seq(random_walk_data)

g.start()

# %%
