# %%
from pygame.locals import *
from resampling_opengl import resample_opengl, setup_glfw
import pygame
import numpy as np
import traceback
from numba import jit
import _thread #
# import pygame.gfxdraw
import textwrap
from functools import partial
import time

import math
from datetime import timezone

from multiprocessing import Process, Manager, Value

class GraphObject:
    global_x = 0
    global_y = 0
    values = []
    def __init__(self, data, color="black", size=4, label=None, point_labels=[]):
        """
        data must be np.array [[x0, y0], [x1, y1], ...]
        point_labels must be empty / same length as data, with one label for each (x, y) pair.
        """

        indices = np.arange(len(data))
        self.initial_data = np.column_stack((data, indices))
        """                     ^^
        The only reason for the ^^ column-stack is the need to be able
        to find point_labels for a random bucket of data points from initial_data.
        If you don't have the original indices, it is impossible to find which
        point_labels correspond to the data points. You might argue that a
        cleaner solution is to just include the point labels in the original
        data array, however, numpy does not support multiple datatypes in an
        array. An alternative is to use numpy structured arrays, however, numba
        does not work with them without heavy workarounds. Implementing said
        workarounds incurs a performance penalty.
        """

        self.data_max_x = self.initial_data[:, 0].max()
        self.data_max_y = self.initial_data[:, 1].max()
        self.data_min_x = self.initial_data[:, 0].min()
        self.data_min_y = self.initial_data[:, 1].min()

        """
        the following fields are used in transformations that occur in 
        prepare_for_interval:

        initial_data is maybe_resample_data into uncut_resampled_data.
        uncut_resampled_data is maybe_cut_to_interval into current_render_data.
        current_render_data is gen_display_corrupted_data into display_corrupted_data.

        """
        self.uncut_resampled_data = np.copy(data)
        self.current_render_data = np.copy(data)
        self.current_render_data_min_x = self.data_min_x #performance enchancement
        self.current_render_data_max_x = self.data_max_x
        self.current_render_data_min_y = self.data_min_y
        self.current_render_data_max_y = self.data_max_y
        self.current_render_data_empty = False

        self.display_corrupted_data = np.copy(self.current_render_data)
    
        self.color = color
        self.size = size
        self.drawn_rect = None

        self.tmi_screen_width_multiplier = 2
        self.cut_extra_screen_width_multiplier = 1 #must be smaller than self.tmi_screen_width_multiplier

        self.label = label

        self.point_labels = np.array(point_labels)
        if self.point_labels.size != 0:
            assert len(self.point_labels) == len(self.initial_data), "point label length must match data point amount"

    def cut_data_creating_render(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        self.current_render_data = self._cut_data(self.uncut_resampled_data, startx, starty, endx, endy, chart_width_px, chart_height_px)

        self.current_render_data_min_x = startx
        self.current_render_data_max_x = endx
        self.current_render_data_min_y = starty
        self.current_render_data_max_y = endy

        self.current_render_data_empty = self.current_render_data.size == 0


        # TODO: implement the below code (speed optimisation for extremely large datasets):
        """
        have_precut_region = self.precut_resampled_data and
                             self.precut_resampled_data_n == self.data_sampling_n and
                             self.precut_resampled_data_startx < startx and
                             endx < self.precut_resampled_data_endx

        if have_precut_region:
            self.current_render_data = self._cut_data(self.precut_resampled_data, startx, starty, endx, endy, chart_width_px, chart_height_px)
        else:
            self.current_render_data = self._cut_data(self.uncut_resampled_data, startx, starty, endx, endy, chart_width_px, chart_height_px)
            self.make_precut_region(startx, starty, endx, endy, chart_width_px, chart_height_px)
        """

    def maybe_resample_data(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        # TODO: implement Ramer-Douglas-Peucker Algorithm (rdp library) or
        # Visvalingam-Whyatt Algorithm (simplification library). ChatGPT helps.
        return False #"did not resample"

    def draw(self):
        raise NotImplementedError()
    def clear(self):
        raise NotImplementedError()
    def delete(self):
        raise NotImplementedError()

    def prepare_for_quit():
        pass

    def identify(self, x0, y0, x1, y1, x_is_date):
#         relevant_data = self._cut_data(self.initial_data, startx = x0, starty = y0, endx = x1, endy = y1, chart_width_px=None, chart_height_px=None)
        relevant_data = GraphObject._cut_data(self, self.initial_data, startx = x0, starty = y0, endx = x1, endy = y1, chart_width_px=None, chart_height_px=None)
        # NOTE: ^^ provides more accurate cutting for this usecase.
        if relevant_data.size == 0:
            return None #not identifiable in this range.
        min_x_report = relevant_data[:, 0].min()
        max_x_report = relevant_data[:, 0].max()

        if x_is_date:
            min_x_report = repr(np.datetime64(int(min_x_report * 1000), 'ms').astype("datetime64[ms]"))
            max_x_report = repr(np.datetime64(int(max_x_report * 1000), 'ms').astype("datetime64[ms]"))

        label_indicies = relevant_data[:, 2]
        if self.point_labels.size != 0:
            try:
                relevant_point_labels = self.point_labels[label_indicies.astype(int)]
                relevant_point_labels = "\n".join(relevant_point_labels)
            except:
                print(label_indicies)
                raise
        else:
            relevant_point_labels = None
        
        # Trust me, no indentation here is the best solution. Otherwise, you must also indent relevant_point_labels in other places, which have no idea about this function, or must write some custom asdjflksadjflksadfa.
        info = f"""\
({type(self).__name__}), label:
{self.label}
_
min x:
{min_x_report}
max x:
{max_x_report}
min_y, max_y = ({relevant_data[:, 1].min()} , {relevant_data[:, 1].max()})
_
point labels:
{relevant_point_labels} """

        # info = textwrap.dedent(info) # << dropped support for this, see comment above.
        return info

    def prepare_for_interval(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        resampled = self.maybe_resample_data(startx, starty, endx, endy, chart_width_px, chart_height_px)
        if resampled:
            # TODO: this is not optimal, results in 2 cuts as the next maybe will cut
            self.cut_data_creating_render(startx, starty, endx, endy, chart_width_px, chart_height_px)
        else:
            self.maybe_cut_data_creating_render(startx, starty, endx, endy, chart_width_px, chart_height_px)

        self.gen_display_corrupted_data(startx, starty, endx, endy, chart_width_px, chart_height_px)


    @staticmethod
    @jit(nopython=True)
    def should_cut_data(startx, starty, endx, endy, chart_width_px, chart_height_px, screen_width, screen_height, visible_data_min_y, visible_data_max_y, visible_data_min_x, visible_data_max_x, data_min_x, data_min_y, data_max_x, data_max_y, tmi_screen_width_multiplier):
        """
        preliminary unit tests show this to be around 3x slower.
        (in maybe_cut_data_creating_render_new vs maybe_cut_data_creating_render)
        preliminary integration tests show about a 50% slowdown.
        I guess because it is simple math, no vectorisation is possible and numba just adds an useless overhead.
        
        """
        left_tmi_boundary = startx - screen_width*tmi_screen_width_multiplier
        right_tmi_boundary = endx + screen_width*tmi_screen_width_multiplier
        top_tmi_boundary = endy + screen_height
        bottom_tmi_boundary = starty - screen_height

        return  (visible_data_min_x > startx and not data_min_x > startx) or\
            visible_data_min_x < left_tmi_boundary or\
            (visible_data_max_x < endx and not data_max_x < endx) or\
            visible_data_max_x > right_tmi_boundary or\
            (visible_data_min_y > starty and not data_min_y > starty) or\
            visible_data_min_y < bottom_tmi_boundary or\
            (visible_data_max_y < endy and not data_max_y < endy) or\
            visible_data_max_y > top_tmi_boundary

    def maybe_cut_data_creating_render_new(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        """
        DEPRECATED for now
        see docstring for should_cut_data
        """
        screen_width = endx - startx
        screen_height = endy - starty
#         if not self.current_render_data[:, 0].any() or not self.current_render_data[:, 1].any():
#         if len(self.current_render_data) == 0:
#         if len(self.current_render_data) == 0: # 14%
#         if self.current_render_data.size == 0: # 13.1%
        if self.current_render_data_empty:
            # TODO: potential optimization: only checking every N frames
            # This means we are probably out of bounds. To reduce CPU usage, should check if we are within bounds to decide on whether to keep cutting or not:
            if  self.data_max_x > startx and\
                self.data_min_x < endx and\
                self.data_max_y > starty and\
                self.data_min_y < endy:

                left_cut_boundary = startx - screen_width * self.cut_extra_screen_width_multiplier
                right_cut_boundary = endx + screen_width * self.cut_extra_screen_width_multiplier
                top_cut_boundary = endy + screen_height/2
                bottom_cut_boundary = starty - screen_height/2

    #             print("cutting data")
                self.cut_data_creating_render(startx = left_cut_boundary, 
                              starty = bottom_cut_boundary, 
                              endx = right_cut_boundary, 
                              endy = top_cut_boundary, 
                              chart_width_px = chart_width_px, 
                              chart_height_px = chart_height_px)

            return
#         visible_data_min_x = self.current_render_data[:, 0].min()
#         visible_data_min_y = self.current_render_data[:, 1].min()
#         visible_data_max_x = self.current_render_data[:, 0].max()
#         visible_data_max_y = self.current_render_data[:, 1].max()
#         print(f"{visible_data_min_x=} {visible_data_max_x=} {visible_data_min_y=} {visible_data_max_y=} ")

        # This might be a bit hard to read. It basically checks if the current
        # screen encapsulates the visible data. If there is an overflow (too
        # much data visible or too little), it will cut; An exception included
        # within is that if the absolute data overflows past the axies, then
        # visible data overflowing does not matter as there is nothing more to
        # display.
        if GraphObject.should_cut_data(startx = startx, 
                                       starty = starty, 
                                       endx =  endx, 
                                       endy =  endy, 
                                       chart_width_px =  chart_width_px, 
                                       chart_height_px =  chart_height_px, 
                                       screen_width =  screen_width, 
                                       screen_height =  screen_height, 
                                       visible_data_min_y =  self.current_render_data_min_y, 
                                       visible_data_max_y =  self.current_render_data_max_y, 
                                       visible_data_min_x =  self.current_render_data_min_x, 
                                       visible_data_max_x =  self.current_render_data_max_x, 
                                       data_min_x =  self.data_min_x, 
                                       data_min_y =  self.data_min_y, 
                                       data_max_x =  self.data_max_x, 
                                       data_max_y =  self.data_max_y,
                                       tmi_screen_width_multiplier = self.tmi_screen_width_multiplier):

# 
#             print("cutting data")
#             print(f"""
#             {visible_data_min_x > startx}
#             {visible_data_min_x < left_tmi_boundary} {visible_data_min_x=} {left_tmi_boundary=} {startx=} {screen_width=}
#             {visible_data_max_x < endx}
#             {visible_data_max_x > right_tmi_boundary}
#             {visible_data_min_y > starty}
#             {visible_data_min_y < bottom_tmi_boundary}
#             {visible_data_max_y < endy}
#             {visible_data_max_y > top_tmi_boundary}
#             """)
# 
            left_cut_boundary = startx - screen_width * self.cut_extra_screen_width_multiplier
            right_cut_boundary = endx + screen_width * self.cut_extra_screen_width_multiplier
            top_cut_boundary = endy + screen_height/2
            bottom_cut_boundary = starty - screen_height/2

#             print("cutting data")
            self.cut_data_creating_render(startx = left_cut_boundary, 
                          starty = bottom_cut_boundary, 
                          endx = right_cut_boundary, 
                          endy = top_cut_boundary, 
                          chart_width_px = chart_width_px, 
                          chart_height_px = chart_height_px)

    def maybe_cut_data_creating_render(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        screen_width = endx - startx
        screen_height = endy - starty
        # Speed optimization journey:
        # if not self.current_render_data[:, 0].any() or not self.current_render_data[:, 1].any():
        # if len(self.current_render_data) == 0:
        # if len(self.current_render_data) == 0: # 14%
        # if self.current_render_data.size == 0: # 13.1%
        if self.current_render_data_empty:
            # TODO: potential optimization: only checking every N frames
            # This means we are probably out of bounds. To reduce CPU usage, should check if we are within bounds to decide on whether to keep cutting or not:
            if  self.data_max_x > startx and\
                self.data_min_x < endx and\
                self.data_max_y > starty and\
                self.data_min_y < endy:

                left_cut_boundary = startx - screen_width * self.cut_extra_screen_width_multiplier
                right_cut_boundary = endx + screen_width * self.cut_extra_screen_width_multiplier
                top_cut_boundary = endy + screen_height/2
                bottom_cut_boundary = starty - screen_height/2

                self.cut_data_creating_render(startx = left_cut_boundary, 
                              starty = bottom_cut_boundary, 
                              endx = right_cut_boundary, 
                              endy = top_cut_boundary, 
                              chart_width_px = chart_width_px, 
                              chart_height_px = chart_height_px)

            return

        left_tmi_boundary = startx - screen_width*self.tmi_screen_width_multiplier
        right_tmi_boundary = endx + screen_width*self.tmi_screen_width_multiplier
        top_tmi_boundary = endy + screen_height
        bottom_tmi_boundary = starty - screen_height

        visible_data_min_y = self.current_render_data_min_y
        visible_data_max_y = self.current_render_data_max_y
        visible_data_min_x = self.current_render_data_min_x
        visible_data_max_x = self.current_render_data_max_x
#         visible_data_min_x = self.current_render_data[:, 0].min()
#         visible_data_min_y = self.current_render_data[:, 1].min()
#         visible_data_max_x = self.current_render_data[:, 0].max()
#         visible_data_max_y = self.current_render_data[:, 1].max()
#         print(f"{visible_data_min_x=} {visible_data_max_x=} {visible_data_min_y=} {visible_data_max_y=} ")

        # This might be a bit hard to read. It basically checks if the current
        # screen encapsulates the visible data. If there is an overflow (too
        # much data visible or too little), it will cut; An exception included
        # within is that if the absolute data overflows past the axies, then
        # visible data overflowing does not matter as there is nothing more to
        # display.
        if  (visible_data_min_x > startx and not self.data_min_x > startx) or\
            visible_data_min_x < left_tmi_boundary or\
            (visible_data_max_x < endx and not self.data_max_x < endx) or\
            visible_data_max_x > right_tmi_boundary or\
            (visible_data_min_y > starty and not self.data_min_y > starty) or\
            visible_data_min_y < bottom_tmi_boundary or\
            (visible_data_max_y < endy and not self.data_max_y < endy) or\
            visible_data_max_y > top_tmi_boundary:
# 
#             print("cutting data")
#             print(f"""
#             {visible_data_min_x > startx}
#             {visible_data_min_x < left_tmi_boundary} {visible_data_min_x=} {left_tmi_boundary=} {startx=} {screen_width=}
#             {visible_data_max_x < endx}
#             {visible_data_max_x > right_tmi_boundary}
#             {visible_data_min_y > starty}
#             {visible_data_min_y < bottom_tmi_boundary}
#             {visible_data_max_y < endy}
#             {visible_data_max_y > top_tmi_boundary}
#             """)
# 
            left_cut_boundary = startx - screen_width * self.cut_extra_screen_width_multiplier
            right_cut_boundary = endx + screen_width * self.cut_extra_screen_width_multiplier
            top_cut_boundary = endy + screen_height/2
            bottom_cut_boundary = starty - screen_height/2

            self.cut_data_creating_render(startx = left_cut_boundary, 
                          starty = bottom_cut_boundary, 
                          endx = right_cut_boundary, 
                          endy = top_cut_boundary, 
                          chart_width_px = chart_width_px, 
                          chart_height_px = chart_height_px)

#     def cut_data_creating_render(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        # TODO: intelligent cutting (max out points that are connected to
        # points that are OOB)
#         self.current_render_data = self.uncut_resampled_data[(startx<=self.uncut_resampled_data[:,0])&(self.uncut_resampled_data[:,0]<=endx)&(self.uncut_resampled_data[:,1]<=endy)&(starty<=self.uncut_resampled_data[:,1])]
#         self.current_render_data = self._cut_data(self.uncut_resampled_data, startx, starty, endx, endy, chart_width_px, chart_height_px)

    def _cut_data(self, data_to_cut, startx, starty, endx, endy, chart_width_px, chart_height_px):
        # TODO: remove the _cut_Data chart dimension args? not needed?
        return GraphObject._cut_data_numba(data_to_cut, startx, starty, endx, endy, chart_width_px, chart_height_px)

    def _cut_data_old(self, data_to_cut, startx, starty, endx, endy, chart_width_px, chart_height_px):
        return data_to_cut[(startx<=data_to_cut[:,0])&(data_to_cut[:,0]<=endx)&(data_to_cut[:,1]<=endy)&(starty<=data_to_cut[:,1])]

    @staticmethod
    @jit(nopython=True)
    def _cut_data_numba(data_to_cut, startx, starty, endx, endy, chart_width_px, chart_height_px):
        return data_to_cut[(startx<=data_to_cut[:,0])&(data_to_cut[:,0]<=endx)&(data_to_cut[:,1]<=endy)&(starty<=data_to_cut[:,1])]


    def gen_display_corrupted_data(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        return self.gen_display_corrupted_data_v3(startx, starty, endx, endy, chart_width_px, chart_height_px)

    def gen_display_corrupted_data_v1(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        """
        DEPRECATED
        """
        if not self.current_render_data.any():
            self.display_corrupted_data = np.empty_like(self.current_render_data)
            return

        self.display_corrupted_data = np.copy(self.current_render_data)

        self.display_corrupted_data[:, 0] = (self.current_render_data[:, 0] - startx) * (chart_width_px) / (endx - startx)  

        self.display_corrupted_data[:, 1] = (-endy + self.current_render_data[:, 1]) * (chart_height_px) / (starty - endy)

    def gen_display_corrupted_data_v2(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        """
        DEPRECATED
        This is basically just this:
          def pos_to_pygame_pos(self, x, y):
             Xs = self.x_axis_min
             Xe =  self.x_axis_max
             Xe2 = self.width
             x2 = ((-1 * x * Xe2) + (Xs * Xe2)) / (Xs-Xe)
     
             # y2 is not vetted.
             Ys = self.y_axis_min
             Ye =  self.y_axis_max
             Ys2 = self.width
             y2 = ((-1 * Ye * Ys2) + (y * Ys2)) / (Ys-Ye)
     
        """
        if self.current_render_data_empty:
            return
        if self.display_corrupted_data.shape != (self.current_render_data.shape[0], 2):
            # only initialize if the shape does not match (speed optimization)
            self.display_corrupted_data = np.empty((self.current_render_data.shape[0], 2), dtype=self.current_render_data.dtype)
        
        display_corrupted_data_x, display_corrupted_data_y =\
        GraphObject.gen_display_corrupted_data_numba(
             current_render_data_x = self.current_render_data[:, 0], 
             current_render_data_y = self.current_render_data[:, 1], 
             startx = startx, 
             starty = starty, 
             endx = endx, 
             endy = endy, 
             chart_width_px = chart_width_px, 
             chart_height_px = chart_height_px)

        self.display_corrupted_data[:,0] = display_corrupted_data_x
        self.display_corrupted_data[:,1] = display_corrupted_data_y

    def gen_display_corrupted_data_v3(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        """
        This is basically just this:
          def pos_to_pygame_pos(self, x, y):
             Xs = self.x_axis_min
             Xe =  self.x_axis_max
             Xe2 = self.width
             x2 = ((-1 * x * Xe2) + (Xs * Xe2)) / (Xs-Xe)
     
             # y2 is not vetted.
             Ys = self.y_axis_min
             Ye =  self.y_axis_max
             Ys2 = self.width
             y2 = ((-1 * Ye * Ys2) + (y * Ys2)) / (Ys-Ye)
     
        """
#         if len(self.current_render_data) == 0: # 14%
#         if self.current_render_data.size == 0: # 13.1%
        if self.current_render_data_empty:
#             self.display_corrupted_data = []
            return
        if self.display_corrupted_data.shape != (self.current_render_data.shape[0], 2):
            # only initialize if the shape does not match (speed optimization)
            self.display_corrupted_data = np.empty((self.current_render_data.shape[0], 2), dtype=self.current_render_data.dtype)
        
        self.display_corrupted_data =\
        GraphObject.gen_display_corrupted_data_numba_2(
             current_render_data = self.current_render_data,
             startx = startx, 
             starty = starty, 
             endx = endx, 
             endy = endy, 
             chart_width_px = chart_width_px, 
             chart_height_px = chart_height_px)

    @staticmethod
    @jit(nopython=True)
    def gen_display_corrupted_data_numba(current_render_data_x, current_render_data_y, startx, starty, endx, endy, chart_width_px, chart_height_px):
        display_corrupted_data_x = (current_render_data_x - startx) * (chart_width_px) / (endx - startx)  
        display_corrupted_data_y = (-endy + current_render_data_y) * (chart_height_px) / (starty - endy)

        return display_corrupted_data_x, display_corrupted_data_y

    @staticmethod
    @jit(nopython=True)
    def gen_display_corrupted_data_numba_2(current_render_data, startx, starty, endx, endy, chart_width_px, chart_height_px):
        data = current_render_data.copy() # necessary, numba does not isolate the fn
        data[:, 0] = (data[:, 0] - startx) * (chart_width_px) / (endx - startx)  
        data[:, 1] = (-endy + data[:, 1]) * (chart_height_px) / (starty - endy)

        return data


    @staticmethod
    @jit(nopython=True)
    def gen_display_corrupted_data_numba_3(current_render_data, startx, starty, endx, endy, chart_width_px, chart_height_px):
        # Same speed as numba_2
        data = current_render_data.copy()
        data_x = data[:, 0]
        data_y = data[:, 1]
        data_x = (data_x - startx) * (chart_width_px) / (endx - startx)  
        data_y = (-endy + data_y) * (chart_height_px) / (starty - endy)

        return data


    def get_visible_data(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        """ Used from the outside mostly, e. g. to autoscale axies """
#         if self.current_render_data_empty:
#             return []
        visible_portion = self._cut_data(self.current_render_data, startx, starty, endx, endy, chart_width_px, chart_height_px)

        return visible_portion


# %%
class PointGraph(GraphObject):
    def __init__(self, data, color="black", size=4, label=None, line_size=0, point_labels=[]):
        super().__init__(data = data, color=color, size=size, label=label, point_labels=point_labels)
        self.line_size = line_size

    def draw(self, screen):
#         if len(self.display_corrupted_data) == 0:
        if self.current_render_data_empty:
            return
        self.drawn_rect = pygame.draw.circle(surface=screen, 
                                             color=self.color, 
                                             center=self.display_corrupted_data[0], 
                                             radius=self.size, 
                                             width= self.line_size)
        
class PointsGraph(PointGraph):
    def __init__(self, *args, **kwargs):
        PointGraph.__init__(self, *args, **kwargs)

    def draw(self, screen):
        if self.current_render_data_empty:
            return

        screen.lock() # speed optimization
        for point_coordinates in self.display_corrupted_data:
            pygame.draw.circle(surface=screen, 
                             color=self.color, 
                             center=point_coordinates,
                             radius=self.size, 
                             width= self.line_size)
        screen.unlock() # speed optimization

class LineGraph(GraphObject):

    def cut_data_creating_render(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        super().cut_data_creating_render(startx, starty, endx, endy, chart_width_px, chart_height_px)
        self.current_render_data_empty = self.current_render_data.size < 4
        # ^ can't draw a line from 1 point

    def draw(self, screen):
#         if len(self.display_corrupted_data[:, 0]) < 2:
#         if len(self.display_corrupted_data) < 2:
        if self.current_render_data_empty:
            return
        self.drawn_rect = pygame.draw.lines(surface=screen, 
                                            color = self.color, 
                                            closed = False, 
                                            points = self.display_corrupted_data, 
                                            width = self.size,
                                            )

#       # the following does NOT perform better:
#         self.drawn_rect = pygame.gfxdraw.polygon(screen, self.display_corrupted_data, pygame.Color("#000000"))


#     def maybe_resample_data(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
#         # TODO: implement Ramer-Douglas-Peucker Algorithm (rdp library) or
#         # Visvalingam-Whyatt Algorithm (simplification library). ChatGPT helps.
#         return False #"did not resample"
# 

class LineGraphSequential(LineGraph):
    def __init__(self, data, color="black", size=4, label=None, dont_resample=False, point_labels=[]):
        super().__init__(data = data, color=color, size=size, label=label, point_labels = point_labels)
        self.data_sampling_n = 1

        # TODO: automatic or more systematic or something:
        self.data_sampling_choices = [
            1,
            # 5, Sampling close to 4 = len(O, H, L, C) doesn't actually affect data density
            5 * 4, # this is real slow, openGL makes it decent
#             40,
            15 * 4,
            60 * 4,
            60 * 4 * 4,
            60 * 12 * 4,
            60 * 24 * 4,
            60 * 24 * 7 * 4,
            60 * 24 * 7 * 5 * 4,
            60 * 24 * 7 * 5 * 12 * 4
            ]
        
#         self.data_sampling_choices = np.array(self.data_sampling_choices)
        ## Adjust the data sampling choices so it never tries to sample more than the available data per candle:
        data_sampling_choice_max = LineGraphSequential.find_closest_number_to(len(self.initial_data), self.data_sampling_choices)
        data_sampling_choice_max_index = self.data_sampling_choices.index(data_sampling_choice_max)
        self.data_sampling_choices = self.data_sampling_choices[0:data_sampling_choice_max_index]

        self.previous_chart_width = None

        self.resample_threshold = 1.05 
        # ^^^ TODO: remove? is chart width + snap to tf sufficient? divergence
        # between optimum sampling and current
        self.chart_width_upper = 2
        self.chart_width_lower = 0.1

        self.previous_n = None
# 
#         # chart width: optimal sampling for chart widths larger than this
#         self.optimal_n_memory_width = min(self.chart_width_upper, self.chart_width_upper) #don't use this
#         self.optimal_n_memories = {
#                 self.data_max_x - self.data_min_x: idk,
#                 prev - memory_width: idk2,
#                 ...
#             }
# 
#         self.optimal_n_memories = {}

        self.tmi_screen_width_multiplier = 4 # TODO: is this and the bottom one basically one param? think about it.
        self.cut_extra_screen_width_multiplier = 2 #must be smaller than self.tmi_screen_width_multiplier

#         self.cached_resamples = {} # n: uncut_resampled_data

        self.cached_resamples = Manager().dict()

        self.n_per_pixel = 1
        self.n_approximation_multiplier = None

        self.dont_resample = dont_resample

#         self.terminating = Manager().bool()
        self.terminating = Value('b', False)

        self.have_set_up_glfw = False

        if self.dont_resample:
            self.maybe_resample_data = partial(GraphObject.maybe_resample_data, self)
        else:
            self.preparation_process = Process(target = self.pre_prepare_cached_resamples)
            self.preparation_process.start()
#             _thread.start_new_thread(self.pre_prepare_cached_resamples, ())

    def prepare_for_quit(self):
        self.terminating.value = True
        print("terminating preparation process")
        self.preparation_process.terminate()
        print("preparation process terminated")
        print("joining preparation process")
        self.preparation_process.join()

    def pre_prepare_cached_resamples(self):
        for data_sampling_choice in reversed(self.data_sampling_choices):
            if self.terminating.value:
                print("terminating pre_prepare_cached_resamples early")
                return
            self.resample_initial_data(n = data_sampling_choice)

    def maybe_cut_data_creating_render(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        screen_width = endx - startx
        screen_height = endy - starty
#         if not self.current_render_data[:, 0].any() or not self.current_render_data[:, 1].any():
#         if not self.current_render_data.any():
#         if len(self.current_render_data) == 0:
#         if self.current_render_data.size == 0: # 13.1%
        if self.current_render_data_empty:
            if  self.data_max_x > startx and\
                self.data_min_x < endx and\
                self.data_max_y > starty and\
                self.data_min_y < endy:

                left_cut_boundary = startx - screen_width * self.cut_extra_screen_width_multiplier
                right_cut_boundary = endx + screen_width * self.cut_extra_screen_width_multiplier
                top_cut_boundary = endy  # does nothing
                bottom_cut_boundary = starty # does nothing

                self.cut_data_creating_render(startx = left_cut_boundary, 
                              starty = bottom_cut_boundary, 
                              endx = right_cut_boundary, 
                              endy = top_cut_boundary, 
                              chart_width_px = chart_width_px, 
                              chart_height_px = chart_height_px)

            return


        left_tmi_boundary = startx - screen_width*self.tmi_screen_width_multiplier
        right_tmi_boundary = endx + screen_width*self.tmi_screen_width_multiplier

#         visible_data_min_x = self.current_render_data[:, 0].min()
#         visible_data_max_x = self.current_render_data[:, 0].max()
        visible_data_min_x = self.current_render_data_min_x
        visible_data_max_x = self.current_render_data_max_x
#         print(f"{visible_data_min_x=} {visible_data_max_x=} {visible_data_min_y=} {visible_data_max_y=} ")

        # This might be a bit hard to read. It basically checks if the current
        # screen encapsulates the visible data. If there is an overflow (too
        # much data visible or too little), it will cut; An exception included
        # within is that if the absolute data overflows past the axies, then
        # visible data overflowing does not matter as there is nothing more to
        # display.
        if  (visible_data_min_x > startx and not self.data_min_x > startx) or\
            visible_data_min_x < left_tmi_boundary or\
            (visible_data_max_x < endx and not self.data_max_x < endx) or\
            visible_data_max_x > right_tmi_boundary:
# 
#             print("cutting data")
#             print(f"""
#             {visible_data_min_x > startx}
#             {visible_data_min_x < left_tmi_boundary} {visible_data_min_x=} {left_tmi_boundary=} {startx=} {screen_width=}
#             {visible_data_max_x < endx}
#             {visible_data_max_x > right_tmi_boundary}
#             {visible_data_min_y > starty}
#             {visible_data_min_y < bottom_tmi_boundary}
#             {visible_data_max_y < endy}
#             {visible_data_max_y > top_tmi_boundary}
#             """)
# 
            left_cut_boundary = startx - screen_width * self.cut_extra_screen_width_multiplier
            right_cut_boundary = endx + screen_width * self.cut_extra_screen_width_multiplier
            top_cut_boundary = endy #does not influence shit
            bottom_cut_boundary = starty #does not influence shit

            self.cut_data_creating_render(startx = left_cut_boundary, 
                          starty = bottom_cut_boundary, 
                          endx = right_cut_boundary, 
                          endy = top_cut_boundary, 
                          chart_width_px = chart_width_px, 
                          chart_height_px = chart_height_px)

# 
    def make_precut_region(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        # TODO: implement (useful for extremely large dataset lag reduction). also see comment in cut_data_creating_render
        if len(self.uncut_resampled_data < self.precutting_data_threshold):
            # too little data for cutting to make sense
            self.precut_resampled_data = None
            return
        else:
            index_of_startx = self.find_first_gt(startx, self.uncut_resampled_data[:, 0])
            index_of_endx = self.find_first_gt(endx, self.uncut_resampled_data[:, 0])

            left_cut_side = max(index_of_startx - self.precutting_data_amount/2, 0)
            right_cut_side = min(index_of_endx + self.precutting_data_amount/2, len(self.uncut_resampled_data))

            self.precut_resampled_data = self._cut_data(self.uncut_resampled_data, startx, starty, endx, endy, chart_width_px, chart_height_px) 
            self.precut_resampled_data_n = self.data_sampling_n
            self.precut_resampled_data_startx = left_cut_side
            self.precut_resampled_data_endx = right_cut_side

    @staticmethod
    @jit(nopython=True)
    def find_first_gt(item, vec):
        """return the index when vec is greater than item"""
        for i in range(len(vec)):
            if item < vec[i]:
                return i
        return -1        

    @staticmethod
    @jit(nopython=True)
    def find_indices(startx, endx, data_to_cut):
        """return the index when data_to_cut is greater than startx"""
        index0 = -1
        index1 = -1

        i = 0
        while i < len(data_to_cut):
            if startx < data_to_cut[i]:
                index0 = i
                break
            i = i + 1

        while i < len(data_to_cut):
            if endx < data_to_cut[i]:
                index1 = i
                break
            i = i + 1

        return index0, index1
 
    def _cut_data(self, data_to_cut, startx, starty, endx, endy, chart_width_px, chart_height_px):
        # TODO: remove the _cut_Data chart dimension args? not needed?
        return self._cut_data_new(data_to_cut, startx, starty, endx, endy, chart_width_px, chart_height_px)

    def _cut_data_old(self, data_to_cut, startx, starty, endx, endy, chart_width_px, chart_height_px):
        # Somehow 5x slower than the new version, would only expect 2x increase, probably for loop/range() shit
        idx0 = LineGraphSequential.find_first_gt(startx, data_to_cut[:,0])
        idx1 = LineGraphSequential.find_first_gt(endx, data_to_cut[:,0])
        return data_to_cut[idx0:idx1]

    def _cut_data_new(self, data_to_cut, startx, starty, endx, endy, chart_width_px, chart_height_px):
        idx0, idx1 = LineGraphSequential.find_indices(startx, endx, data_to_cut[:,0])
        return data_to_cut[idx0:idx1]

    def _cut_data_old(self, data_to_cut, startx, starty, endx, endy, chart_width_px, chart_height_px):
        return data_to_cut[(startx<=data_to_cut[:,0])&(data_to_cut[:,0]<=endx)]


    def calculate_optimal_sampling_n(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
#         if not self.n_approximation_multiplier:
        self.get_optimal_sampling_n_approximation_multiplier(chart_width_px, chart_height_px)

        chart_width = endx - startx
        self.previous_chart_width = chart_width
        self.previous_n = self.approximate_optimal_sampling_n(chart_width)

        # NOTE: very cheeky optimization method. maybe too cheeky?
        self.calculate_optimal_sampling_n = self.calculate_optimal_sampling_n_p2

        return self.previous_n

    def calculate_optimal_sampling_n_p2(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
        chart_width = endx - startx

        chart_width_ratio = chart_width / self.previous_chart_width
        if  chart_width_ratio > self.chart_width_lower and\
            chart_width_ratio < self.chart_width_upper:

            return self.previous_n

        else:
            self.previous_chart_width = chart_width
#             self.previous_n = self.calculate_optimal_sampling_n_mid(startx, starty, endx, endy, chart_width_px, chart_height_px, current_chart_width = chart_width)
#             self.previous_n = self.calculate_optimal_sampling_n_slow(startx, starty, endx, endy, chart_width_px, chart_height_px, current_chart_width = chart_width)
            self.previous_n = self.approximate_optimal_sampling_n(chart_width)
            return self.previous_n

    
    def calculate_optimal_sampling_n_slow(self, startx, starty, endx, endy, chart_width_px, chart_height_px): #, current_chart_width):
        # This is the golden standard for calculating the optimal n sampling.
        # However, it's slow as all fuck because it has to cut such a huge data
        # chunk. As such, the mid algo is way faster and produces extremely
        # tiny errors.

        # However, mid is now deprecated in favour of doing this once since the result can be used to calculate future values.
        visible_raw_data_portion = self._cut_data(self.initial_data, startx, starty, endx, endy, chart_width_px, chart_height_px)

        optimal_sampling_n = int(len(visible_raw_data_portion) / chart_width_px)

        optimal_sampling_n = optimal_sampling_n * self.n_per_pixel

#         print(f"{current_chart_width=} {optimal_sampling_n=} {current_chart_width/optimal_sampling_n}")
        return optimal_sampling_n

    def get_optimal_sampling_n_approximation_multiplier(self, chart_width_px, chart_height_px):
        o_n = self.calculate_optimal_sampling_n_slow(startx = self.data_min_x, 
                                               starty = self.data_min_y, 
                                               endx = self.data_max_x, 
                                               endy = self.data_max_y, 
                                               chart_width_px = chart_width_px, 
                                               chart_height_px = chart_height_px)

        example_chart_width = self.data_max_x - self.data_min_x
        # desire:  chart_width * n_approximation_multiplier = o_n
        self.n_approximation_multiplier = o_n / example_chart_width

    def approximate_optimal_sampling_n(self, current_chart_width):
        return self.n_approximation_multiplier * current_chart_width
        
    def calculate_optimal_sampling_n_mid(self, startx, starty, endx, endy, chart_width_px, chart_height_px, current_chart_width):
#         optimal_n_memory_width = min(self.chart_width_lower, self.chart_width_upper)
#         optimal_n_memory_width = 3
#         for chart_width, n in self.optimal_n_memories.items():
#             if current_chart_width >= chart_width and current_chart_width < chart_width * optimal_n_memory_width:
#                 print("managed to use cache")
#                 return n
#         # didn't find it in memories:
#         print("didn't manage to use cache, using slow version")
#         print("recalculating optimal sampling n")
#         if self.current_render_data.any(): TODO this will break?
        if self.previous_n and self.current_render_data.size != 0:
            visible_portion = self._cut_data(self.current_render_data, startx, starty, endx, endy, chart_width_px, chart_height_px)
            optimal_sampling_n = int(len(visible_portion) * (self.previous_n/4) / chart_width_px)
            # ^ divide by 4, as values are squashed into OHLC
        else:
            visible_portion = self._cut_data(self.initial_data, startx, starty, endx, endy, chart_width_px, chart_height_px)
            optimal_sampling_n = int(len(visible_portion) / chart_width_px)
            
        optimal_sampling_n = optimal_sampling_n * self.n_per_pixel

        self.optimal_n_memories[current_chart_width] = optimal_sampling_n

        print(f"{current_chart_width=} {optimal_sampling_n=} {current_chart_width/optimal_sampling_n}")

        return optimal_sampling_n
            
            
    def maybe_resample_data(self, startx, starty, endx, endy, chart_width_px, chart_height_px):

        # TODO: test alternate methods (LTTB algorythm, min-max sampling, ...?)

        # optimal sampling is when there are as many chart x values as there are pixels on the screen.
        optimal_sampling_n = self.calculate_optimal_sampling_n(startx, starty, endx, endy, chart_width_px, chart_height_px)
        
        rounded_optimal_sampling_n = LineGraphSequential.find_closest_number_to(optimal_sampling_n, self.data_sampling_choices)

#         if max(rounded_optimal_sampling_n, self.data_sampling_n) / min(rounded_optimal_sampling_n, self.data_sampling_n) > self.resample_threshold:
        if rounded_optimal_sampling_n != self.data_sampling_n:
#             print(f"{optimal_sampling_n=} {rounded_optimal_sampling_n=}")
            self.data_sampling_n = rounded_optimal_sampling_n
            self.uncut_resampled_data = self.resample_initial_data(n = self.data_sampling_n)
#             self.uncut_resampled_data = self.resample(self.initial_data, self.data_sampling_n)
            return True
        else:
            return False

    def setup_glfw_once(self):
        if not self.have_set_up_glfw:
            setup_glfw()
        self.have_set_up_glfw = True

    @staticmethod
    def find_closest_number_to(number, numbers):
        return min(numbers, key=lambda x:abs(x-number))

    def resample_initial_data(self, n):
        print(f"resampling data to {n=}")
        if n == 1:
            print("returning initial data")
            return self.initial_data[:, 0:2]
        else:
            if n not in self.cached_resamples.keys():
                print("doing hard calculations")
                if n <= 60: # OpenGL is faster in these cases.
                    self.setup_glfw_once()
                    self.cached_resamples[n] = resample_opengl(self.initial_data, n)
                else:
                    self.cached_resamples[n] = self.resample(self.initial_data, n)
            else:
                print("using cache")
            return self.cached_resamples[n]
        
    
    def resample(self, data, n):
        x = data[:, 0]
        y = data[:, 1]
        new_data = LineGraphSequential._resample_numba(x, y, n)
        return new_data
        # return LineGraphSequential._resample_numpy(data, )

    @staticmethod
    def _resample_numpy(data, n):
        # Ensure x and y are numpy arrays
        x = data[:, 0]
        y = data[:, 1]
#         print("doing the actual resampling")
#         print(f"{len(x)=} {n=}")

        # Reshape y data into groups of `n` points for resampling
        y_resampled = y[:len(y) // n * n].reshape(-1, n)
        x_resampled = x[:len(x) // n * n].reshape(-1, n)

        # Compute OHLC for each group
        open_ = y_resampled[:, 0]
        high = y_resampled.max(axis=1)
        low = y_resampled.min(axis=1)
        close = y_resampled[:, -1]

        # Use the midpoint x value for each group and repeat it for each OHLC value
        x_mid = x_resampled[:, n // 2]
        x_ohlc = np.repeat(x_mid, 4)

        # Stack the OHLC values in the order: open, high, low, close
        y_ohlc = np.column_stack((open_, high, low, close)).ravel()

        new_data = np.column_stack((x_ohlc, y_ohlc))

#         print(f"{len(x_ohlc)=} {n=} {n*len(x_ohlc)}")
        return new_data

#     @jit(nopython=True)
    @staticmethod
    def _resample_numba(x, y, n):
        # This is only about 20% faster than the pure numpy version.

        # Pre-calculate number of groups
        num_groups = math.ceil(len(y) / n)
        
        # Initialize arrays to store OHLC and x_mid values
        open_ = np.empty(num_groups)
        high = np.empty(num_groups)
        low = np.empty(num_groups)
        close = np.empty(num_groups)
        x_mid = np.empty(num_groups)

        # Calculate OHLC values for each group
        i = 0
        for i in range(num_groups-1):
            start_idx = i * n
            end_idx = start_idx + n
            y_group = y[start_idx:end_idx]
            
            open_[i] = y_group[0]
            high[i] = y_group.max()  # Find max within each group
            low[i] = y_group.min()   # Find min within each group
            close[i] = y_group[-1]
            x_mid[i] = x[start_idx + n // 2]  # Take midpoint x
        # Last group is treated separately, as it may not be full.
        # this would make the x_mid line break, specifically. If can find a
        # fast workaround, that would make this unnecessarry.
        i = i + 1
        start_idx = i * n
        end_idx = len(y)
#         end_idx = min(start_idx + n, len(y))
        y_group = y[start_idx:end_idx]
        
        open_[i] = y_group[0]
        high[i] = y_group.max()  # Find max within each group
        low[i] = y_group.min()   # Find min within each group
        close[i] = y_group[-1]
        x_mid[i] = x[(start_idx + end_idx) // 2]  # Take midpoint x

        # Repeat x_mid for each OHLC step (O-H-L-C)
        x_ohlc = np.repeat(x_mid, 4)
        
        # Stack OHLC in order and flatten
        y_ohlc = np.empty(4 * num_groups)
        y_ohlc[0::4] = open_
        y_ohlc[1::4] = high
        y_ohlc[2::4] = low
        y_ohlc[3::4] = close

        new_data = np.column_stack((x_ohlc, y_ohlc))
        return new_data

    @staticmethod
    @jit(nopython=True)
    def _resample_numba_2(x, y, n):
        # This is slower than _resample_numpa
        # Calculate number of groups for resampling
        num_groups = len(y) // n
        
        # Truncate to make divisible by n and flatten into 1D
        y_trimmed = y[:num_groups * n]
        x_trimmed = x[:num_groups * n]
        
        # Use slicing to get open, high, low, close without specifying axis
        open_ = y_trimmed[::n]
        close = y_trimmed[n-1::n]

        # For high and low, we take max/min within each chunk of n
        high = np.array([np.max(y_trimmed[i:i + n]) for i in range(0, len(y_trimmed), n)])
        low = np.array([np.min(y_trimmed[i:i + n]) for i in range(0, len(y_trimmed), n)])

        # Midpoint x values
        x_mid = x_trimmed[n // 2::n]

        # Repeat x_mid for each OHLC step and flatten OHLC values
        x_ohlc = np.repeat(x_mid, 4)
        y_ohlc = np.empty(4 * num_groups)
        y_ohlc[0::4] = open_
        y_ohlc[1::4] = high
        y_ohlc[2::4] = low
        y_ohlc[3::4] = close

        new_data = np.column_stack((x_ohlc, y_ohlc))
        return x_ohlc, y_ohlc
# %%
class LineGraphSequentialPairs(LineGraphSequential):
    """
    connects pairs of points. meant to be faster than the single ver
    """
    def __init__(self, *args, **kwargs):
        LineGraphSequential.__init__(self, *args, **kwargs)
#         super().__init__(*args, **kwargs)
        self.current_render_data_parity = 0
        self.dont_resample = True # needs an alternative method of resampling, probably.

    def draw(self, screen):
        if self.current_render_data_empty:
            return

        reshaped_display_corrupted_data = self.display_corrupted_data[self.current_render_data_parity:]
        num_rows = reshaped_display_corrupted_data.shape[0]
        if num_rows % 2 != 0:
            reshaped_display_corrupted_data = reshaped_display_corrupted_data[:(num_rows // 2) * 2]  # truncate to nearest multiple of 2
        # TODO: move some of these  calcs out to cut or something?
        reshaped_display_corrupted_data = reshaped_display_corrupted_data.reshape(-1, 2, 2)

        screen.lock() # speed optimization #TODO: can i do this outside for all drawings in a batch?
        for point_pair in reshaped_display_corrupted_data:
#             line = [previous_point_pair, point_pair]
            pygame.draw.lines(surface=screen,  #TODO: is line faster than lines?
                              color = self.color, 
                              closed = False, 
                              points = point_pair,
                              width = self.size,
                             )
        screen.unlock()

    def cut_data_creating_render(self, startx, starty, endx, endy, chart_width_px, chart_height_px):
#         self.current_render_data, i = self._cut_data_current_render(self.uncut_resampled_data, startx, starty, endx, endy, chart_width_px, chart_height_px)
        self.current_render_data, i = self._cut_data_current_render(self.uncut_resampled_data, startx, starty, endx, endy, chart_width_px, chart_height_px)

        self.current_render_data_parity = i%2
        self.current_render_data_start_index = i

        self.current_render_data_min_x = startx
        self.current_render_data_max_x = endx
        self.current_render_data_min_y = starty
        self.current_render_data_max_y = endy

        self.current_render_data_empty = self.current_render_data.size == 0

    def _cut_data_current_render(self, data_to_cut, startx, starty, endx, endy, chart_width_px, chart_height_px):
        idx0, idx1 = LineGraphSequential.find_indices(startx, endx, data_to_cut[:,0])
        return data_to_cut[idx0:idx1], idx0

# %%

class Graph:
    def __init__(self, auto_zoom_y=True, auto_zoom_to_all_objects=True, x_is_date=True, run_in_new_thread=True): # {{{
        self.width = 1200
        self.height = 800
        self.font_size = 30
        self.background_color = pygame.Color("#ffffff")
        self.background_color_2 = pygame.Color("#a9ffdc")
        self.ask_amount_color = "red"
        self.bid_amount_color = "green"
        self.under_cursor_color = pygame.Color("#8fffe9")

        self.y_position = 0
        self.x_position = 0

        self.y_axis_min = 0
        self.y_axis_max = 500
        self.x_axis_min = 0
        self.x_axis_max = 500

        self.pan_jump_speed = 100
        self.panning_velocities = [0, 0] # x, y
        self.panning_acceleration = 0.05
        self.panning_deceleration = 0.9

        self.mouse_zoom_speed = 1.04 #fracional zoom
        self.zooming_multiplier = 5
        self.zooming_velocity = 0
        self.zooming_acceleration = 0.05
        self.zooming_deceleration = 0.9
        self.auto_zoom_y = auto_zoom_y

        self.cursor_movement_velocities = [0, 0]
        self.cursor_movement_max_speed = 10

        self.batch_size = 1
        self.batched_data = {}

        self.global_scene = []
        self.local_scene = []
        self.unbound_objects = []
        
        self.log_scale = False
        self.mode = 'normal'

        self.mouse_dragging = False
        self.mouse_dragging_x_min_offset = None
        self.mouse_dragging_y_min_offset = None
        self.mouse_dragging_x_max_offset = None
        self.mouse_dragging_y_max_offset = None
        self.update_local_scene_counter = 0
        self.update_local_scene_step = 30

        self.x_is_date = x_is_date

        self.key_down = {
            "alt": False,
            "shift": False,
            }

        pygame.init()
        pygame.freetype.init()
        self.default_font = pygame.freetype.Font(None, self.font_size)

        self.running = False
        self.auto_zoom_to_all_objects=auto_zoom_to_all_objects
        self.run_in_new_thread = run_in_new_thread

    def zoom_to(self, objects:list[GraphObject]):
        xes = []
        ys = []
        for o in objects:
            xes.extend([o.data_min_x, o.data_max_x])
            ys.extend([o.data_min_y, o.data_max_y])

        self.x_axis_min = min(xes)
        self.x_axis_max = max(xes)
        self.y_axis_min = min(ys)
        self.y_axis_max = max(ys)

    def expand_zoom_to(self, g:GraphObject):
        self.x_axis_min = min(self.x_axis_min, g.data_min_x)
        self.x_axis_max = max(self.x_axis_max, g.data_max_x)
        self.y_axis_min = min(self.y_axis_min, g.data_min_y)
        self.y_axis_max = max(self.y_axis_max, g.data_max_y)

    def add_point(self, data, color="red", size=2, line_size=0, label=None, point_labels=[]):
        o = PointGraph(data=data, color=color, size=size, line_size=line_size, label=label, point_labels=point_labels)
        self.global_scene.append(o)
#         self.expand_zoom_to(o)
        return o
    def add_points(self, data, color="red", size=2, line_size=0, label=None, point_labels=[]):
        o = PointsGraph(data=data, color=color, size=size, line_size=line_size, label=label, point_labels=point_labels)
        self.global_scene.append(o)
#         self.expand_zoom_to(o)
        return o
        
    def add_line(self, data, color="black", size=4, label=None, point_labels=[]):
        l = LineGraph(data, color, size, label=label, point_labels=point_labels)
        self.global_scene.append(l)
#         self.expand_zoom_to(l)
        return l

    def add_line_seq(self, data, color="black", size=4, label=None, point_labels=[]):
        l = LineGraphSequential(data, color, size, label=label, point_labels=point_labels)
        self.global_scene.append(l)
#         self.expand_zoom_to(l)
        return l

    def do_auto_zoom_y(self):
        if self.mouse_dragging:
            return
        minimums = []
        maximums = []
        startx = self.x_axis_min 
        starty = self.y_axis_min 
        endx = self.x_axis_max 
        endy = self.y_axis_max
        chart_width_px = self.width
        chart_height_px = self.height
        for item in self.local_scene:
            item_visible_data = item.get_visible_data(\
                                      startx = startx,
                                      starty = starty,
                                      endx = endx,
                                      endy = endy,
                                      chart_width_px = chart_width_px,
                                      chart_height_px = chart_height_px)

            if len(item_visible_data) == 0:
# #             if not item_visible_data.any():
                continue
            minimums.append(item_visible_data[:, 1].min())
            maximums.append(item_visible_data[:, 1].max())

        if len(minimums) == 0 or len(maximums) == 0:
            print("auto zoom failing, no data detected")
            return

        self.y_axis_max = max(maximums)
        self.y_axis_min = min(minimums)

    def prepare_all_items_for_new_interval(self):
        if self.update_local_scene_counter % self.update_local_scene_step == 0:
            # local scene is updating this frame, need to have all items know if they are within bounds for sure.
            for item in self.global_scene:
                item.prepare_for_interval(startx = self.x_axis_min, 
                                          starty = self.y_axis_min, 
                                          endx = self.x_axis_max, 
                                          endy = self.y_axis_max,
                                          chart_width_px = self.width,
                                          chart_height_px = self.height,)
        else:
            for item in self.local_scene:
                item.prepare_for_interval(startx = self.x_axis_min, 
                                          starty = self.y_axis_min, 
                                          endx = self.x_axis_max, 
                                          endy = self.y_axis_max,
                                          chart_width_px = self.width,
                                          chart_height_px = self.height,)
            
            
    def draw_onscreen_data(self):
        for item in self.local_scene:
            item.draw(self.screen)

    def draw_gui(self):
        status = self.default_font.render(self.mode)
        rect = self.screen.blit(status[0], (self.width - status[1][2], 0))

        if self.mode in ("select 1", "select 2"):
            color = "black" if self.mode == "select 1" else pygame.Color("#cccccc")
            points = [  [0, self.selection_starty],
                        [self.width, self.selection_starty],
                        [self.width, 0],
                        [self.selection_startx, 0],
                        [self.selection_startx, self.height]    ]
            self.drawn_rect = pygame.draw.lines(surface=self.screen, 
                                                color=color,
                                                closed = False, 
                                                points = points,
                                                width = 2,
                                                )
        if self.mode == "select 2":
            points = [  [0, self.selection_endy],
                        [self.width, self.selection_endy],
                        [self.width, 0],
                        [self.selection_endx, 0],
                        [self.selection_endx, self.height]    ]
            self.drawn_rect = pygame.draw.lines(surface=self.screen, 
                                                color="black",
                                                closed = False, 
                                                points = points,
                                                width = 2,
                                                )

    def print_selected_data(self):
        item_identities = []
        xa, ya = self.pygame_pos_to_pos(self.selection_startx, self.selection_starty)
        xb, yb = self.pygame_pos_to_pos(self.selection_endx, self.selection_endy)

        x0 = min(xa, xb)
        x1 = max(xa, xb)
        y0 = min(ya, yb)
        y1 = max(ya, yb)
        for item in self.global_scene: # TODO: local scene good idea here?
            item_identity = item.identify(x0, y0, x1, y1, self.x_is_date)
            if item_identity:
                item_identities.append(item_identity)
        
        if len(item_identities) != 0:
            print("\n===== SELECTED DATA =====")
            for item_identity in item_identities:
                print(item_identity)
                print("______________________________\n")

    def draw_axies(self):
        if self.x_is_date:
            min_x_text = str(np.datetime64(int(self.x_axis_min), 's'))
            max_x_text = str(np.datetime64(int(self.x_axis_max), 's'))
        else:
            min_x_text = str(self.x_axis_min)
            max_x_text = str(self.x_axis_max)

        min_x_text = self.default_font.render(min_x_text)
        rect = self.screen.blit(min_x_text[0], (0, self.height - min_x_text[1][1]))

        max_x_text = self.default_font.render(max_x_text)
        rect = self.screen.blit(max_x_text[0], (self.width - max_x_text[1][2] , self.height - max_x_text[1][1]))

        min_y_text = self.default_font.render(str(self.y_axis_min))
        rect = self.screen.blit(min_y_text[0], (0, self.height - min_y_text[1][1] - min_x_text[1][1]))

        max_y_text = self.default_font.render(str(self.y_axis_max))
        rect = self.screen.blit(max_y_text[0], (0, 0))
# 
#         max_x_text = self.default_font.render(str(self.x_axis_max))
#         rect = self.screen.blit(max_x_text[0], (0, 0))
# 
    def pick_point(self, event):
        x_delta = 0
        y_delta = 0
        if event.type not in (pygame.KEYDOWN, pygame.KEYUP):
            return False
        if event.mod == pygame.KMOD_LALT:
            movement_multiplier = 0.33
        else:
            movement_multiplier = 1

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_h:
                x_delta = 0

            elif event.key == pygame.K_j:
                y_delta = 0

            elif event.key == pygame.K_k:
                y_delta = 0

            elif event.key == pygame.K_l:
                x_delta = 0

            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                return x_delta, y_delta, True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_h:
                x_delta -= self.cursor_movement_max_speed * movement_multiplier

            elif event.key == pygame.K_j:
                y_delta += self.cursor_movement_max_speed * movement_multiplier

            elif event.key == pygame.K_k:
                y_delta -= self.cursor_movement_max_speed * movement_multiplier

            elif event.key == pygame.K_l:
                x_delta += self.cursor_movement_max_speed * movement_multiplier

        self.cursor_movement_velocities = [x_delta, y_delta]
        return False

    def update_local_scene(self, force=False):
        """
            updating rarely significantly improves performance while not impacting usability
            MUST run after prepare_all_items_for_new_interval.
        """
        if self.update_local_scene_counter % self.update_local_scene_step == 0 or force:
            self.local_scene = [item for item in self.global_scene if not item.current_render_data_empty]

    def update_cursor_position(self):
        if self.mode == "select 1":
            self.selection_startx += self.cursor_movement_velocities[0]
            self.selection_starty += self.cursor_movement_velocities[1]
        elif self.mode == "select 2":
            self.selection_endx += self.cursor_movement_velocities[0]
            self.selection_endy += self.cursor_movement_velocities[1]

    def start(self):
        if self.run_in_new_thread:
            _thread.start_new_thread(self.start_internal, ())
        else:
            self.start_internal()
    
    def start_internal(self):
        if self.auto_zoom_to_all_objects:
            self.zoom_to(self.global_scene)
        self.running = True
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        pygame.init()
        self.prepare_all_items_for_new_interval()
        self.update_local_scene(force=True)
        while self.running:
            try:
#                 self.zoom()

                self.update_local_scene_counter += 1

                moved_1 = self.pan_smooth()
                moved_2 = self.zoom_smooth(x_only = True)
                if moved_1 or moved_2:
                    self.prepare_all_items_for_new_interval() #NOTE: this and the next function are linked fundamentally, don't change order.
                    self.update_local_scene()
                    
                if self.auto_zoom_y:
                    self.do_auto_zoom_y()

                self.update_cursor_position()

                self.screen.fill(self.background_color)

#                 self.onscreen_data = self.calculate_onscreen_data()

                self.draw_onscreen_data()

                self.draw_axies()
                self.draw_gui()

                # flip() the display to put your work on screen
                pygame.display.flip() # TODO: is this necessary here or?

                # poll for events
                # pygame.QUIT event means the user clicked X to close your window
                for event in pygame.event.get(): # {{{
                    if event.type == pygame.QUIT:
                        print("pygame quit")
                        self.running = False

                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.mode = "normal"

                    if self.mode == "normal":
                        # scroll
                        if event.type == pygame.MOUSEWHEEL:
                            zooming_state = event.y
    #                         self.zooming_velocity = self.zooming_velocity + self.zooming_multiplier * zooming_state
                            kwargs = {"x": True, "y": True}
                            if self.key_down['shift']:
                                kwargs['x'] = False
                            if self.key_down['alt']:
                                kwargs['y'] = False

                            if zooming_state > 0:
                                self.zoom_in_mouse(**kwargs)
                            else:
                                self.zoom_out_mouse(**kwargs)

                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_LSHIFT:
                                self.key_down['shift'] = True
                                # for mouse movement

                            elif event.key == pygame.K_LALT:
                                self.key_down['alt'] = True
                                # for mouse movement

                            if event.mod == pygame.KMOD_LALT:
                                movement_multiplier = 0.1
                            else:
                                movement_multiplier = 1

                            if event.key == pygame.K_h:
                                self.panning_velocities[0] -= self.panning_acceleration * movement_multiplier

                            elif event.key == pygame.K_j:
                                self.zooming_velocity += self.zooming_acceleration * movement_multiplier

                            elif event.key == pygame.K_k:
                                self.zooming_velocity -= self.zooming_acceleration * movement_multiplier

                            elif event.key == pygame.K_l:
                                self.panning_velocities[0] += self.panning_acceleration * movement_multiplier


                        elif event.type == pygame.KEYUP:
                            if event.key == pygame.K_LSHIFT:
                                self.key_down['shift'] = False

                            elif event.key == pygame.K_LALT:
                                self.key_down['alt'] = False

                            elif event.key == pygame.K_ESCAPE:
                                self.mode = "normal"

                            elif event.key == pygame.K_s:
                                self.selection_startx = self.height/2
                                self.selection_endx = self.height/2
                                self.selection_starty = self.width/2
                                self.selection_endy = self.width/2
                                self.mode = "select 1"


                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            self.mouse_dragging = True # TODO: true
                            self.mouse_dragging_x_axis_min = self.x_axis_min # These are necessary to prevent a feedback loop between _pygame_pos_to_pos and these.
                            self.mouse_dragging_x_axis_max = self.x_axis_max
                            self.mouse_dragging_y_axis_min = self.y_axis_min
                            self.mouse_dragging_y_axis_max = self.y_axis_max

                            mouse_x, mouse_y = event.pos
                            mouse_x, mouse_y = self._pygame_pos_to_pos(mouse_x, mouse_y)
                            self.mouse_dragging_startx = mouse_x
                            self.mouse_dragging_starty = mouse_y

                            self.x_axis_min_startpos = self.x_axis_min
                            self.x_axis_max_startpos = self.x_axis_max

                            self.y_axis_min_startpos = self.y_axis_min
                            self.y_axis_max_startpos = self.y_axis_max

                        elif event.type == pygame.MOUSEBUTTONUP:
                            self.mouse_dragging = False
            #                 print(f"{event.button}, {event.pos}")
                            if event.button == 1: #left click
                                pos = pygame.mouse.get_pos()
                            elif event.button == 3: #right click
                                pass

                        elif event.type == pygame.MOUSEMOTION:
                            if self.mouse_dragging:
                                mouse_x, mouse_y = event.pos
                                mouse_x, mouse_y = self._pygame_pos_to_pos(mouse_x, mouse_y)

                                mouse_x_offset = self.mouse_dragging_startx - mouse_x
                                mouse_y_offset = self.mouse_dragging_starty - mouse_y

                                self.x_axis_min = self.x_axis_min_startpos + mouse_x_offset
                                self.y_axis_min = self.y_axis_min_startpos + mouse_y_offset
                                self.x_axis_max = self.x_axis_max_startpos + mouse_x_offset
                                self.y_axis_max = self.y_axis_max_startpos + mouse_y_offset
                                
                                self.prepare_all_items_for_new_interval()

                    elif self.mode == "select 1":
                        fin = self.pick_point(event)
                        if fin:
                            self.selection_endx = min(self.selection_startx + self.pan_jump_speed, self.width * 0.95)
                            self.selection_endy = min(self.selection_starty - self.pan_jump_speed, 0 + 0.95 * self.height)
                            self.mode = "select 2"

                    elif self.mode == "select 2":
                        fin = self.pick_point(event)
                        if fin:
                            self.print_selected_data()
                            self.mode = "normal"

                        

                self.clock.tick(60)  # limits FPS to 60

            except Exception as e:
                # NOTE: this try/except block avoids various X crashes.
                print("ERROR:")
#                 import traceback # why does this have to duplicate here?
                print(traceback.format_exc())

                import pdb
                pdb.set_trace()
                self.running = False



        for chart in self.global_scene:
            chart.prepare_for_quit()
        pygame.quit()

    def _pygame_pos_to_pos(self, x, y):
        """
        The general formula for
            x - point coordinate
            Xs - startpoint of chart for point coodinate x
            Xe - endpoint of chart for point coordinate x

            x2 - point coordinate in translated chart
            Xs2 - startpoint of translated chart housing x2
            Xe2 - endpoint of translated chart housing x2
        is:

        X2 = ((-1 * Xe * Xs2) + (x * Xs2) - (x * Xe2) + (Xs * Xe2)) / (Xs - Xe)
        
        All used formulas are derived from this. This is derived from a
        maintenance of left/right side ratios for x and x2 in each chart.
        """

        Xe = self.width
        Xe2 = self.mouse_dragging_x_axis_max # These are necessary to prevent a feedback loop between _pygame_pos_to_pos and these.
        Xs2 = self.mouse_dragging_x_axis_min
        x2 = ((-1 * Xe * Xs2) + (x * Xs2) - (x * Xe2)) / (-Xe)

        Ys = self.height
        Ye2 = self.mouse_dragging_y_axis_max
        Ys2 = self.mouse_dragging_y_axis_min
        y2 = ((y * Ys2) - (y * Ye2) + (Ys * Ye2)) / (Ys)

        return x2, y2


    def pygame_pos_to_pos(self, x, y):
        """
        The general formula for
            x - point coordinate
            Xs - startpoint of chart for point coodinate x
            Xe - endpoint of chart for point coordinate x

            x2 - point coordinate in translated chart
            Xs2 - startpoint of translated chart housing x2
            Xe2 - endpoint of translated chart housing x2
        is:

        X2 = ((-1 * Xe * Xs2) + (x * Xs2) - (x * Xe2) + (Xs * Xe2)) / (Xs - Xe)
        
        All used formulas are derived from this. This is derived from a
        maintenance of left/right side ratios for x and x2 in each chart.
        """

        Xe = self.width
        Xe2 = self.x_axis_max
        Xs2 = self.x_axis_min
        x2 = ((-1 * Xe * Xs2) + (x * Xs2) - (x * Xe2)) / (-Xe)

        Ys = self.height
        Ye2 = self.y_axis_max
        Ys2 = self.y_axis_min
        y2 = ((y * Ys2) - (y * Ye2) + (Ys * Ye2)) / (Ys)

        return x2, y2

    def pos_to_pygame_pos(self, x, y):
        """
            see pygame_pos_to_pos
        """
        Xs = self.x_axis_min
        Xe =  self.x_axis_max
        Xe2 = self.width
        x2 = ((-1 * x * Xe2) + (Xs * Xe2)) / (Xs-Xe)

        # y2 is not vetted.
        Ys = self.y_axis_min
        Ye =  self.y_axis_max
#         Ye2 = 0
        Ys2 = self.width
        y2 = ((-1 * Ye * Ys2) + (y * Ys2)) / (Ys-Ye)

        
        return x2, y2

    def calculate_onscreen_data(self):
        onscreen_data = [] # y_position, (price, amount)
        for level_index, (price, amount) in enumerate(self.batched_data.items()):
            level_y_position = self.y_position + level_index * (self.font_size )
            if level_y_position < 0:
                continue
            if level_y_position > self.height - self.font_size:
                break
            rect = pygame.Rect(0, level_y_position, self.width, self.font_size)
#             onscreen_data.append((level_y_position, (price, amount)))
            onscreen_data.append((rect, (price, amount)))
        return onscreen_data


# ALL MOTIONS:
    def pan_smooth(self):
        moved = False
        if self.panning_velocities[1]:
            chart_height = self.y_axis_max - self.y_axis_min
            self.y_axis_min += self.panning_velocities[1] * chart_height
            self.y_axis_max += self.panning_velocities[1] * chart_height
            if self.panning_velocities[1] < 0:
                self.panning_velocities[1] = min(self.panning_velocities[1] * self.panning_deceleration, 1)
            else:
                self.panning_velocities[1] = max(self.panning_velocities[1] * self.panning_deceleration, 1)
            moved = True
        if self.panning_velocities[0]:
            chart_width = self.x_axis_max - self.x_axis_min
            self.x_axis_min += self.panning_velocities[0] * chart_width
            self.x_axis_max += self.panning_velocities[0] * chart_width
            if self.panning_velocities[0] < 0:
                self.panning_velocities[0] = min(self.panning_velocities[0] * self.panning_deceleration, 0)
            else:
                self.panning_velocities[0] = max(self.panning_velocities[0] * self.panning_deceleration, 0)
            moved = True

        return moved

    def move_up(self, amount_multiplier=1):
        chart_height = self.y_axis_max - self.y_axis_min

        self.y_axis_min += amount_multiplier * self.pan_jump_speed * chart_height * 0.01
        self.y_axis_max += amount_multiplier * self.pan_jump_speed * chart_height * 0.01

        self.prepare_all_items_for_new_interval()

    def move_down(self, amount_multiplier=1):
        chart_height = self.y_axis_max - self.y_axis_min
        self.y_axis_min -= amount_multiplier * self.pan_jump_speed * chart_height * 0.01
        self.y_axis_max -= amount_multiplier * self.pan_jump_speed * chart_height * 0.01

        self.prepare_all_items_for_new_interval()

    def move_left(self, amount_multiplier=1):
        chart_width = self.x_axis_max - self.x_axis_min
        self.x_axis_min -= amount_multiplier * self.pan_jump_speed * chart_width * 0.01
        self.x_axis_max -= amount_multiplier * self.pan_jump_speed * chart_width * 0.01

        self.prepare_all_items_for_new_interval()

    def move_right(self, amount_multiplier=1):
        chart_width = self.x_axis_max - self.x_axis_min
        self.x_axis_min += amount_multiplier * self.pan_jump_speed * chart_width * 0.01
        self.x_axis_max += amount_multiplier * self.pan_jump_speed * chart_width * 0.01

        self.prepare_all_items_for_new_interval()

    def zoom_out_mouse(self, x=True, y=True):
        if y:
            y_axis_diff = self.y_axis_max - self.y_axis_min
            new_y_axis_diff = y_axis_diff * self.mouse_zoom_speed
            y_axis_addition = (new_y_axis_diff - y_axis_diff)/2

            self.y_axis_min -= y_axis_addition
            self.y_axis_max += y_axis_addition
        if x:
            x_axis_diff = self.x_axis_max - self.x_axis_min
            new_x_axis_diff = x_axis_diff * self.mouse_zoom_speed
            x_axis_addition = (new_x_axis_diff - x_axis_diff)/2

            self.x_axis_min -= x_axis_addition
            self.x_axis_max += x_axis_addition
        self.prepare_all_items_for_new_interval()

    def zoom_in_mouse(self, x=True, y=True):
        if y:
            y_axis_diff = self.y_axis_max - self.y_axis_min
            new_y_axis_diff = y_axis_diff / self.mouse_zoom_speed
            y_axis_addition = (y_axis_diff - new_y_axis_diff)/2

            self.y_axis_min += y_axis_addition
            self.y_axis_max -= y_axis_addition
        if x:
            x_axis_diff = self.x_axis_max - self.x_axis_min
            new_x_axis_diff = x_axis_diff / self.mouse_zoom_speed
            x_axis_addition = (x_axis_diff - new_x_axis_diff)/2

            self.x_axis_min += x_axis_addition
            self.x_axis_max -= x_axis_addition

        self.prepare_all_items_for_new_interval()

    def zoom_smooth(self, x_only=False):
        chart_width = self.x_axis_max - self.x_axis_min
        chart_height = self.y_axis_max - self.y_axis_min

        self.x_axis_min -= self.zooming_velocity * chart_width
        self.x_axis_max += self.zooming_velocity * chart_width


        if not x_only:
            if self.y_axis_min - self.zooming_velocity >= self.y_axis_max + self.zooming_velocity:
                print("hit y ceiling, no more zoomout")
    #             self.y_axis_min = self.y_axis_max - 1
                return

            self.y_axis_min -= self.zooming_velocity * chart_height
            self.y_axis_max += self.zooming_velocity * chart_height

        moved = self.zooming_velocity
        self.zooming_velocity = self.zooming_velocity * self.zooming_deceleration

        return moved

    @staticmethod
    def get_clicked_step(pos, rendered_rects_levels):
        for rect_level in rendered_rects_levels:
            if rect_level[0].collidepoint(pos):
                return rect_level[1]
# %%

# 
# # %%
# g = Graph(auto_zoom_y = True) #auto_zoom_y = True)
# 
# 
# test_line = np.random.rand(50000, 2) * 500
# test_line_sorted = test_line[np.lexsort((test_line[:, 1], test_line[:, 0]))]
# # l = g.add_line_seq(test_line_sorted, width=1)
# 
# 
# 
# pa = DataLoader("bybit_spot_HOTUSDT_1min").get_pa()
# pa2 = DataLoader("binance_spot_HOTUSDT_1min").get_pa()
# 
# test_coordinates = list(pa[:][int(len(pa[:])/2)][['Time', 'High']])
# test_coordinates[0] = test_coordinates[0].astype('datetime64[s]').astype(np.int64)
# # pa.set_window(2958451 - 50000, len(pa.array))
# c2 = g.add_candle_line(pa, color="red", size=1)
# c = g.add_candle_line(pa2, color="blue", size=1)
# o = g.add_point(data = np.array([test_coordinates]), color="green", size=10, line_size=3)
# 
# 
# 
# g.start()
# # %load_ext line_profiler
# # %lprun -f c.prepare_for_interval g.start()

# %%
