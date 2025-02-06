# %%
import math
import pygame
from pygame.locals import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from OpenGL.GL.framebufferobjects import *

import time

# %%
def setup_pygame():
    # This is done to set the opengl context.
    pygame.init()

    # Set OpenGL attributes
    pygame.display.gl_set_attribute(GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(GL_CONTEXT_MINOR_VERSION, 3)
#     pygame.display.gl_set_attribute(GL_CONTEXT_PROFILE_MASK, GL_CONTEXT_PROFILE_CORE)

    # Create a Pygame window with OpenGL context
    screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)

    # Ensure OpenGL functions are initialized
    pygame.display.set_caption('OpenGL Context with Pygame')
    return

# %%
def generate_random_walk(steps, start_value=0, step_size=1):
    x_values = np.arange(int(time.time()), int(time.time()) + steps)
    y_values = np.cumsum(np.random.choice([-step_size, step_size], size=steps)) + start_value
    return np.column_stack((x_values, y_values))

# %%

def set_up_floating_point_framebufer():
    """
        This is necessary so the output colors are not rounded to 256 values, for higher precision (f32)
    """

    max_texture_size = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
    width = max_texture_size
#     height = max_texture_size
    height = 1
#     height = 1 # can this be more? probably not?
#     height = 2 # can this be more? probably not?

    framebuffer = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)

    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        raise Exception("Framebuffer is not complete")
    return framebuffer

def create_texture(data, texture_unit):
    # Convert data to float32 for texture compatibility
    data = np.array(data, dtype=np.float32)

    # Generate texture ID
    texture_id = glGenTextures(1)
    
    # Bind the texture
    glActiveTexture(GL_TEXTURE0 + texture_unit)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    
    # Create texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, len(data), 1, 0, GL_RED, GL_FLOAT, data)
    
    return texture_id

def compile_shader_program(vertex_source, fragment_source):
    vertex_shader = compileShader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_source, GL_FRAGMENT_SHADER)
    program = compileProgram(vertex_shader, fragment_shader)
    return program

def bind_and_set_textures(shader_program, x_texture_id, y_texture_id):
    # Use the shader program
    glUseProgram(shader_program)

    # Activate and bind the first texture
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, x_texture_id)
    # Set the sampler for x_texture in the shader to use texture unit 0
    glUniform1i(glGetUniformLocation(shader_program, "x_texture"), 0)

    # Activate and bind the second texture
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, y_texture_id)
    # Set the sampler for y_texture in the shader to use texture unit 1
    glUniform1i(glGetUniformLocation(shader_program, "y_texture"), 1)

    # Example: setting other uniforms, like the texture dimensions and n (group size)
    tex_dimensions = (len(x), 1)  # Assuming 1D texture with width equal to the length of x
    glUniform2iv(glGetUniformLocation(shader_program, "tex_dimensions"), 1, tex_dimensions)
    glUniform1i(glGetUniformLocation(shader_program, "n"), 4)  # Adjust 'n' as needed

def render_fullscreen_quad():
    # Bind VAO
    glBindVertexArray(vao)

    # Use the shader program containing your fragment shader
    glUseProgram(shader_program) # Make sure shader_program is your compiled shader program ID
    n_location = glGetUniformLocation(shader_program, "n")

    n_value = 4  # Set this to the desired group size
    glUniform1i(n_location, n_value)

    # Draw the quad
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    # Unbind VAO for cleanliness
    glBindVertexArray(0)

# %%
glGetError()


# %%

setup_pygame()

# %%
framebuffer = set_up_floating_point_framebufer()

# %%
# Create textures for x and y
random_walk_data = generate_random_walk(10000000, step_size=0.5)
x = random_walk_data[:, 0]
y = random_walk_data[:, 1]
n = 4  # Example group size
y_length = len(y) 
# Width is the number of OHLC values (4 per group)
width = math.ceil(y_length / n) * 4
height = 1


# %%

max_texture_size = glGetIntegerv(GL_MAX_TEXTURE_SIZE)


x_texture_id = create_texture(x[0:max_texture_size], 0)


y_texture_id = create_texture(y[0:max_texture_size], 1)


# %%

fragment_shader_code = """
#version 330 core

precision highp float; // supposed to increase precision
// Texture inputs
uniform sampler2D x_texture;
uniform sampler2D y_texture;

// Output color
out vec4 FragColor;

// Define the group size (n)
uniform int n;

// Define texture dimensions
uniform ivec2 tex_dimensions;

void main() {
    // Calculate current fragment index
    int index = int(gl_FragCoord.x);

    // Calculate group index and starting position
    int group = index / n;
    int start_idx = group * n;

    // Ensure we don't overshoot
    if (start_idx >= tex_dimensions.x) {
        discard; // Avoid processing out of bounds
    }

    // Calculate OHLC values
    float open_val = texelFetch(y_texture, ivec2(start_idx, 0), 0).r;
    float high_val = open_val;
    float low_val = open_val;
    float close_val;
    
    for (int i = 1; i < n && (start_idx + i) < tex_dimensions.x; ++i) {
        float y_val = texelFetch(y_texture, ivec2(start_idx + i, 0), 0).r;
        high_val = max(high_val, y_val);
        low_val = min(low_val, y_val);
    }
    close_val = texelFetch(y_texture, ivec2(min(start_idx + n - 1, tex_dimensions.x - 1), 0), 0).r;

    // Midpoint index for x
    int mid_index = start_idx + n / 2;
    float mid_x = texelFetch(x_texture, ivec2(mid_index, 0), 0).r;

    // Output OHLC as a color/vector
    if (index % 4 == 0) {
        FragColor = vec4(0.1, 0.1, 0.1, 0.1);
        // FragColor = vec4(mid_x, open_val, 0.0, 1.0);
    } else if (index % 4 == 1) {
        FragColor = vec4(0.1, 0.1, 0.1, 0.1);
        // FragColor = vec4(mid_x, high_val, 0.0, 1.0);
    } else if (index % 4 == 2) {
        FragColor = vec4(0.2, 0.2, 0.2, 0.2);
        // FragColor = vec4(mid_x, low_val, 0.0, 1.0);
    } else {
        FragColor = vec4(3, 3, 3, 3);
        // FragColor = vec4(mid_x, close_val, 0.0, 1.0);
    }
}

"""

vertex_shader_code = """
#version 330 core

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position.xy, 0.0, 1.0);
}
"""



# %%

reference_results = glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT)

# %%

shader_program = compile_shader_program(vertex_shader_code, fragment_shader_code)
# glUseProgram(shader_program) # TODO: maybe this can be done here only

# %%

# Vertex coordinates for a fullscreen quad
vertices = np.array([
    -1.0, -1.0,  # Bottom-left
     1.0, -1.0,  # Bottom-right
    -1.0,  1.0,  # Top-left
     1.0,  1.0   # Top-right
], dtype=np.float32)

# Create VAO and VBO
vao = glGenVertexArrays(1)
vbo = glGenBuffers(1)

# %%
bind_and_set_textures(shader_program, x_texture_id, y_texture_id)

# %%
glBindVertexArray(vao)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
glViewport(0, 0, width, height)

# %%
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# %%

# Enable vertex attribute (position)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)



render_fullscreen_quad()


# Unbind for cleanliness; they should be bound when rendering
glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindFramebuffer(GL_FRAMEBUFFER, 0)
glBindVertexArray(0)

# %%
# 
# height = 1
# width - len(y)
# 


glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
results = glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT)
glBindFramebuffer(GL_FRAMEBUFFER, 0)

# results = glReadPixels(0, 0, width, height, GL_RGBA32F, GL_FLOAT)


# %%

np.all(results == reference_results) # returns false, so there is a change


# %%
for i, result in enumerate(results):
    if result[0][0] == 0.:
        print("found it")
        print(i)
        print(result)
        break
# %%

# Reshape the results
result_array = np.frombuffer(results, dtype=np.float32).reshape(-1, 4)

# Extract Separate OHLC Values
mid_x_values = result_array[:, 0]
open_values = result_array[:, 1]
high_values = result_array[:, 2]
low_values = result_array[:, 3]
close_values = result_array[:, 3]  # Similar since stored in FragColor.w

# Combine into a structured array, or use as needed
ohlc_data = np.column_stack((mid_x_values, open_values, high_values, low_values, close_values))

# Example of using OHLC data
print("OHLC data:\n", ohlc_data)


# %%
