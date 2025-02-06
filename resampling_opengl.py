# %%
import glfw
from OpenGL.GL import *

def initialize_headless_glfw():
   if not glfw.init():
       raise RuntimeError("Failed to initialize GLFW")

   # Set window hints for creating an OpenGL core profile context
   glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
   glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
   glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
   glfw.window_hint(glfw.VISIBLE, GL_FALSE)  # Do not show window
   glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)  # Required on macOS

   # Create an offscreen render context with no visible window
   window = glfw.create_window(1, 1, "", None, None)
   if not window:
       glfw.terminate()
       raise RuntimeError("Failed to create GLFW headless window")

   # Make the context current
   glfw.make_context_current(window)

   return window

# Initialize the headless GLFW context
window = initialize_headless_glfw()

# Now proceed with setting up your shaders and buffers

# %%

from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np


# %%
def create_buffer(data):
   buffer = glGenBuffers(1)
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer)
   data_array = np.array(data, dtype=np.float32)
   glBufferData(GL_SHADER_STORAGE_BUFFER, data_array.nbytes, data_array, GL_STATIC_DRAW)
   glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer)  # Bind to binding point 0
   return buffer

# %%
x_data = np.random.rand(10000000).astype(np.float32)  # Example large dataset
y_data = np.random.rand(10000000).astype(np.float32)

# %%
x_buffer = create_buffer(x_data)
y_buffer = create_buffer(y_data)

# %%
def compile_compute_shader(source):
    shader = compileShader(source, GL_COMPUTE_SHADER)
    program = compileProgram(shader)
    return program

# %%
with open("resampling_glsl.c", "r") as f:
    compute_shader_source = f.read()

# %%
compute_program = compile_compute_shader(compute_shader_source)

# %%
def execute_shader(program, num_groups):
    glUseProgram(program)
    glDispatchCompute(num_groups, 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

# %%
# Setup and parameters
chunk_size = 256
num_elements = len(y_data)
num_groups = (num_elements + chunk_size - 1) // chunk_size

# %%
# Create output buffer
ohlc_buffer = glGenBuffers(1)
ohlc_data = np.empty(4 * num_groups, dtype=np.float32)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, ohlc_buffer)
glBufferData(GL_SHADER_STORAGE_BUFFER, ohlc_data.nbytes, None, GL_DYNAMIC_COPY)
glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ohlc_buffer)

# %%
# Set uniform `n`
glUseProgram(compute_program)
glUniform1i(glGetUniformLocation(compute_program, "n"), chunk_size)

# %%
# Run the shader
execute_shader(compute_program, num_groups)

# %%
# Read back data
glBindBuffer(GL_SHADER_STORAGE_BUFFER, ohlc_buffer)
result = np.frombuffer(glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, ohlc_data.nbytes), dtype=np.float32)

# %%
 # Terminate GLFW on program exit
 glfw.destroy_window(window)
 glfw.terminate()
