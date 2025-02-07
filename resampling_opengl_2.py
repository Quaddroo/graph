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
def split_timestamps_into_f32(timestamps):
#     test_timestamps = random_walk_data[:, 0][0:5]/100000
    divided_timestamps = timestamps/100000
    m = np.modf(divided_timestamps)
    r1 = m[0].astype('float32')
    r2 = m[1].astype('float32')
    return r1, r2


def combine_timestamps_into_f64(timestamps1, timestamps2):
    timestamps1 = timestamps1.astype('float64') * 100000
    timestamps2 = timestamps2.astype('float64') * 100000
    return np.add(timestamps1, timestamps2)

def check_texture_underlying_data(textureID, width, height):
    # Assuming `textureID` is your texture ID
    glBindTexture(GL_TEXTURE_2D, textureID)

    # Prepare an array to store the pixel data
    data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT)

    # Convert to a numpy array (assuming your texture uses GL_FLOAT for red channel)
#     pixel_data = np.frombuffer(data, dtype=np.float32).reshape((height, width))

    return data

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

def find_optimal_texture_size():
    """
    It is sad, but this is the only robust approach.
    However, because of stability issues, I'm going to have to assume a fixed available texture size for now.
    """
    return 1000, 1000 # width, height
    max_texture_side_size = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
    try:
        for i in range(100):
            width = max_texture_side_size
            height = 100*(i+1)

            framebuffer = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)

            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
    except GLError as err:
        import pdb
        pdb.set_trace()


def set_up_floating_point_framebufer(width, height):
    """
        This is necessary so the output colors are not rounded to 256 values, for higher precision (f32)
    """

    texture0 = create_texture(None, texture_unit=3, fallback_width=width, fallback_height=height)
    texture1 = create_texture(None, texture_unit=4, fallback_width=width, fallback_height=height) # not sure what texture unit is

    framebuffer = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture0, 0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, texture1, 0)
    draw_buffers = (GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1)
    glDrawBuffers(2, draw_buffers)

    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        raise Exception("Framebuffer is not complete")
    return framebuffer

def prep_data_for_texture(data):
    # Convert data to float32 for texture compatibility
    data = np.array(data, dtype=np.float32)
    data = np.stack([data, np.zeros_like(data), np.zeros_like(data), np.zeros_like(data)], axis=-1)
    return data

def prep_timestamps_for_texture(data):
    t1, t2 = split_timestamps_into_f32(data)    
    data = np.stack([t1, t2, np.zeros_like(data), np.zeros_like(data)], axis=-1)
    return data

def create_texture(data, texture_unit, fallback_width=None, fallback_height=None):
    if data is None:
        if not fallback_width or not fallback_height:
            raise Exception("No data provided to create_texture means it needs fallback_width or fallback_height")
        width = fallback_width
        height = fallback_height
    else:
        width = data.shape[0]
        height = data.shape[1]

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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, data)
#     glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, data)

#     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
    
    return texture_id

def compile_shader_program(vertex_source, fragment_source):
    vertex_shader = compileShader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_source, GL_FRAGMENT_SHADER)
    program = compileProgram(vertex_shader, fragment_shader)
    return program

def bind_and_set_textures(shader_program, x_texture_id, y_texture_id, width, height):
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
#     tex_dimensions = (len(x), 1)  # Assuming 1D texture with width equal to the length of x
    tex_dimensions = (width, height)
    glUniform2iv(glGetUniformLocation(shader_program, "tex_dimensions"), 1, tex_dimensions)
    glUniform1i(glGetUniformLocation(shader_program, "n"), 4)  # Adjust 'n' as needed

def render_fullscreen_quad(vao, shader_program, n_value):
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

def setup_environment():
    glGetError()

    # setup_pygame() < OMITTED because of how it will usually be used.

    width, height = find_optimal_texture_size()

    framebuffer = set_up_floating_point_framebufer(width, height)

    return width, height, framebuffer

def create_vao_vbo():
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

    return vao, vbo, vertices

def generate_initial_data_textures(random_walk_data, width, height):
    x = random_walk_data[:, 0]
    y = random_walk_data[:, 1]
    y_length = len(y) 

    x = prep_timestamps_for_texture(x)
    y = prep_data_for_texture(y)
    x_reshaped = x.reshape(width, height, 4)
    y_reshaped = y.reshape(width, height, 4)

    x_texture_id = create_texture(x_reshaped, 0)

    y_texture_id = create_texture(y_reshaped, 1)

    return x_texture_id, y_texture_id

def prepare_environment_for_drawing(vao, vbo, framebuffer, width, height, vertices):
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
    glViewport(0, 0, width, height)

    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Enable vertex attribute (position)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)

def cleanup_environment_after_drawing():
    # Unbind for cleanliness; they should be bound when rendering
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glBindVertexArray(0)

def get_resulting_pixeldata(framebuffer, width, height):
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)

    glReadBuffer(GL_COLOR_ATTACHMENT0)
    results_0 = glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT)

    glReadBuffer(GL_COLOR_ATTACHMENT1)
    results_1 = glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return results_0, results_1

fragment_shader_code = """
#version 330 core
#extension GL_ARB_draw_buffers : enable
precision highp float;

// Texture inputs
uniform sampler2D x_texture;
uniform sampler2D y_texture;

// Output color
layout(location = 0) out vec4 FragColor0;
layout(location = 1) out vec4 FragColor1;

// Define the group size (n)
uniform int n;
uniform ivec2 tex_dimensions;

void main() {
    // Current fragment index
    int fragX = int(gl_FragCoord.x);
    int fragY = int(gl_FragCoord.y);

    // Calculate 1D index
    int uniqueIndex = fragY * tex_dimensions.x + fragX;

    // Calculate starting indices for the group
    int start_idx = uniqueIndex * n;
    int start_idx_x = start_idx % tex_dimensions.x;
    int start_idx_y = start_idx / tex_dimensions.x;

    // OHLC calculation
    float open_val = texelFetch(y_texture, ivec2(start_idx_x, start_idx_y), 0).r;

    float high_val = open_val;
    float low_val = open_val;
    float close_val;

    for (int i = 1; i < n && (start_idx_x + i) < tex_dimensions.x; ++i) {
        float y_val = texelFetch(y_texture, ivec2(start_idx_x + i, start_idx_y), 0).r;
        high_val = max(high_val, y_val);
        low_val = min(low_val, y_val);
    }

    close_val = texelFetch(y_texture, ivec2(min(start_idx_x + n - 1, tex_dimensions.x - 1), start_idx_y), 0).r;

    // Midpoint index for x
    int mid_index_x = start_idx_x + n / 2;
    float mid_x_0 = texelFetch(x_texture, ivec2(mid_index_x, start_idx_y), 0).r;
    float mid_x_1 = texelFetch(x_texture, ivec2(mid_index_x, start_idx_y), 0).g;

    FragColor0 = vec4(mid_x_0, mid_x_1, 0.0, 0.0);
    FragColor1 = vec4(open_val, high_val, low_val, close_val);
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

def resample_opengl(data, n_value):
    glDisable(GL_BLEND) # Might not be necessary

    width, height, framebuffer = setup_environment()

    x_texture_id, y_texture_id = \
        generate_initial_data_textures(data, width, height)

    shader_program = compile_shader_program(vertex_shader_code, fragment_shader_code)
    # glUseProgram(shader_program) # TODO: maybe this can be done here only

    vao, vbo, vertices = create_vao_vbo()

    bind_and_set_textures(shader_program, x_texture_id, y_texture_id, width, height)

    prepare_environment_for_drawing(vao, vbo, framebuffer, width, height, vertices)

    render_fullscreen_quad(vao, shader_program, n_value)

    cleanup_environment_after_drawing()

    results_0, results_1 = get_resulting_pixeldata(framebuffer, width, height)

    new_ys = np.frombuffer(results_1, dtype=np.float32).reshape(-1, 4).reshape(-1, 1)[0:1000000]

    new_xs_0 = np.frombuffer(results_0, dtype=np.float32).reshape(-1, 4)[:, 1].reshape(-1, 1)

    new_xs_1 = np.frombuffer(results_0, dtype=np.float32).reshape(-1, 4)[:, 0].reshape(-1, 1)

    new_xs = combine_timestamps_into_f64(new_xs_0, new_xs_1)[0:1000000//n_value]

    new_xs = np.repeat(new_xs, 4)

    final_resampled_array = np.column_stack((new_xs, new_ys))

    return final_resampled_array

