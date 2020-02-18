import datetime

import numpy as np
import tkinter as tk
import pyscreenshot as image_grab

import util


class GUI:

    def __init__(self, gui_config, size, sensor_readings_dimensions):
        """
        create graphical user interface / visualization for the simulator

        the following methods are 'public'
            reset(objects_info)
            draw(objects_info, texts_info)
            take_screen_shot()

        Parameters
        ----------
        gui_config : GuiConfig
            class object / data structure for all the necessary parameters (from config.py)
        sensor_readings_dimensions : int
            number of possible sensor readings
        """
        self.verbose = gui_config['verbose']

        # root
        self.tk_root = tk.Tk()
        self.tk_root.title(gui_config['title'])
        self.tk_root.resizable(gui_config['resizable'], gui_config['resizable'])

        # canvas
        self.draw_scale = gui_config['draw scale']
        canvas_width = self.draw_scale * size[0]
        canvas_height = self.draw_scale * size[1]
        self.canvas = tk.Canvas(self.tk_root,
                                width=canvas_width, height=canvas_height,
                                background=gui_config['panel background'])
        self.canvas.pack()

        self.center_X = canvas_width / 2
        self.center_Y = canvas_height / 2

        # vertical lines at an interval of "line_distance" pixel
        for x in range(gui_config['line distance'], int(canvas_width), gui_config['line distance']):
            self.canvas.create_line(x, 0, x, canvas_height, fill=gui_config['grid fill color'])
        # horizontal lines at an interval of "line_distance" pixel
        for y in range(gui_config['line distance'], int(canvas_height), gui_config['line distance']):
            self.canvas.create_line(0, y, canvas_width, y, fill=gui_config['grid fill color'])

        self.canvas.create_rectangle(1, 1, canvas_width-1, canvas_height-1, fill='', width=1)
        # self.canvas.create_rectangle(0, self.center_Y, canvas_width, canvas_height, fill='black')

        if gui_config['offset'] == 'AllDown':
            self.center_Y = canvas_height

        # marker
        self.marker_id = gui_config['marker id']
        self.marker_color = gui_config['marker color']
        self.marker_rad = self.draw_scale * gui_config['marker radius']
        self.is_marked = False
        self.marker_position = None

        # texts
        self.time_step_text = ''
        self.time_step_text_id = self.canvas.create_text(10, 10, text='', anchor=tk.W)
        self.time_step_text_color = gui_config['text color']

        self.prediction_error_text = ''
        self.prediction_error_text_id = self.canvas.create_text(10, 40, text='', anchor=tk.W)
        self.prediction_error_text_color = gui_config['text color']
        self.prediction_error_text_color_change_threshold = \
            gui_config['prediction error text color change threshold']

        self.target_error_text = ''
        self.target_error_text_id = self.canvas.create_text(10, 25, text='', anchor=tk.W)
        self.target_error_text_color = gui_config['text color']
        self.target_error_text_color_change_threshold = \
            gui_config['target error text color change threshold']

        # If true, no line from the ball position to the first prediction position will be drawn
        self.hide_first_line = False

        self.sensor_directions = util.calc_sensor_directions(sensor_readings_dimensions)
        self.thrust_directions = util.calc_thrust_directions()

        self.agent_circle_colors = gui_config['agent colors']
        self.agent_text_colors = gui_config['agent text colors']
        self.agent_text_size = gui_config['agent text size']
        self.thrust_color = gui_config['thrust color']
        self.thrust_factor = gui_config['thrust factor']

        self.prediction_circle_radius_scale = gui_config['prediction circle radius scale']
        self.prediction_circle_color = gui_config['prediction circle color']
        self.prediction_text_color = gui_config['prediction text color']
        self.prediction_text_size = gui_config['prediction text size']

        self.target_circle_radius_scale = gui_config['target circle radius scale']
        self.target_circle_color = gui_config['target circle color']
        self.target_text_color = gui_config['target text color']
        self.target_text_size = gui_config['target text size']

        self.obstacle_circle_color = gui_config['obstacle circle color']
        self.obstacle_text_color = gui_config['obstacle text color']
        self.obstacle_text_size = gui_config['obstacle text size']

        """
        variables for better book keeping of all the different gui objects
        """
        # collect ids of gui objects that are agents
        self.gui_agents_circle_idx = []
        # and text fields
        self.gui_agents_texts_idx = []
        # collect corresponding names
        self.gui_agents_idx_to_name = []

        self.gui_thrusters_idx = []
        self.gui_thrusters_idx_to_name = []

        self.gui_predictions_circle_idx = []
        self.gui_predictions_texts_idx = []
        self.gui_predictions_idx_to_name = []

        self.gui_targets_circle_idx = []
        self.gui_targets_texts_idx = []
        self.gui_targets_idx_to_name = []

        self.gui_obstacles_circle_idx = []
        self.gui_obstacles_texts_idx = []
        self.gui_obstacles_idx_to_name = []

    def reset(self, env_objects_info):
        """
        reset the gui to the default start appearance

        mostly used for a new epoch

        this method should be called before draw
        (or else the appearance is not well defined)

        Parameters
        ----------
        env_objects_info : list
            containing dictionaries with
                name : str
                type : str
                position : (2,) array
                velocity : (2,) array
                radius : float
                if type == 'agent':
                    thrust number : int
                    thruster activity : (motor_commands_dimensions,) array
                    predictions : (prediction_horizon, position_dimensions) array
                    target : (position_dimensions,) array

        Returns
        -------

        """
        self.gui_agents_circle_idx = []
        self.gui_agents_idx_to_name = []
        self.gui_agents_texts_idx = []
        self.gui_thrusters_idx = []
        self.gui_thrusters_idx_to_name = []

        self.gui_predictions_circle_idx = []
        self.gui_predictions_idx_to_name = []
        self.gui_predictions_texts_idx = []

        self.gui_targets_circle_idx = []
        self.gui_targets_idx_to_name = []
        self.gui_targets_texts_idx = []

        self.gui_obstacles_circle_idx = []
        self.gui_obstacles_texts_idx = []
        self.gui_obstacles_idx_to_name = []

        self.delete_objects()
        self.create_objects(env_objects_info)

        if self.verbose:
            print('update with env objects:')
            for eoi in env_objects_info:
                print(eoi)
            self.print_()

    def print_(self):
        print('resulting gui objects:')
        print('agent idx            ', self.gui_agents_circle_idx)
        print('agent texts idx      ', self.gui_agents_texts_idx)
        print('agent names          ', self.gui_agents_idx_to_name)
        print('thrusts idx          ', self.gui_thrusters_idx)
        print('thrusts names        ', self.gui_thrusters_idx_to_name)
        print('predictions idx      ', self.gui_predictions_circle_idx)
        print('predictions texts idx', self.gui_predictions_texts_idx)
        print('predictions names    ', self.gui_predictions_idx_to_name)
        print('targets idx          ', self.gui_targets_circle_idx)
        print('target texts idx     ', self.gui_targets_texts_idx)
        print('targets names        ', self.gui_targets_idx_to_name)
        print('obstacle idx         ', self.gui_obstacles_circle_idx)
        print('obstacle texts idx   ', self.gui_obstacles_texts_idx)
        print('obstacle names       ', self.gui_obstacles_idx_to_name)

    def delete_objects(self):
        """delete all canvas widget that are different between scenarios"""
        self.delete_objects_specified_by_tag('agent')
        self.delete_objects_specified_by_tag('agent text')
        self.delete_objects_specified_by_tag('thruster')
        self.delete_objects_specified_by_tag('prediction')
        self.delete_objects_specified_by_tag('prediction text')
        self.delete_objects_specified_by_tag('target')
        self.delete_objects_specified_by_tag('target text')
        self.delete_objects_specified_by_tag('obstacle')
        self.delete_objects_specified_by_tag('obstacle text')

    def delete_objects_specified_by_tag(self, gui_type_tag):
        """
        remove all canvas widgets with the specified tag

        Parameters
        ----------
        gui_type_tag : str
            which canvas widgets to delete

        Returns
        -------

        """
        objects_specified = self.canvas.find_withtag(gui_type_tag)
        for object_specified in objects_specified:
            self.canvas.delete(object_specified)

    def create_objects(self, env_objects_info):
        """
        create additional canvas widgets
        that are needed for visualization of the current environment state

        Parameters
        ----------
        env_objects_info : list
            containing dictionaries with
                name : str
                type : str
                position : (2,) array
                velocity : (2,) array
                radius : float
                color : str
                if type == 'agent':
                    thrust number : int
                    thruster activity : (motor_commands_dimensions,) array
                    predictions : (prediction_horizon, position_dimensions) array
                    target : (position_dimensions,) array

        Returns
        -------

        """
        for env_object_info in env_objects_info:
            if env_object_info['type'] == 'agent':
                self.create_complete_agent(env_object_info)
            elif env_object_info['type'] == 'obstacle':
                self.create_obstacle(env_object_info)
            else:
                raise RuntimeError('env object type {} not understood'.format(env_object_info['type']))

    def create_complete_agent(self, env_object_info):
        """
        create canvas widgets that are needed for one agent

        Parameters
        ----------
        env_object_info : dict
            name : str
            type : str
            position : (2,) array
            velocity : (2,) array
            radius : float
            thrust number : int
            thruster activity : (motor_commands_dimensions,) array
            predictions : (prediction_horizon, position_dimensions) array
            target : (position_dimensions,) array

        Returns
        -------

        """
        self.create_thruster(env_object_info)
        self.create_predictions(env_object_info)
        self.create_target(env_object_info)

        # sequential arrangement is important,
        # because the agent circle widget should not be overlapped by any other widget
        self.create_agent(env_object_info)

    def create_agent(self, env_object_info):
        """create canvas widget that represents the agent"""
        gui_agent_position = self.scale(env_object_info['position'])
        gui_agent_radius = env_object_info['radius'] * self.draw_scale
        if len(self.gui_agents_circle_idx) < 2:
            agent_circle_color = self.agent_circle_colors[len(self.gui_agents_circle_idx)]
            agent_text_color = \
                self.agent_text_colors[len(self.gui_agents_circle_idx)]
        else:
            raise RuntimeError('not enough colors for more than {} agents specified'
                               .format(len(self.agent_circle_colors)))
        gui_agent_id = self.draw_circle(gui_agent_position,
                                        gui_agent_radius,
                                        agent_circle_color,
                                        'agent')
        self.gui_agents_circle_idx.append(gui_agent_id)
        self.gui_agents_idx_to_name.append(env_object_info['name'])

        gui_agent_text_id = self.draw_text(gui_agent_position,
                                           agent_text_color,
                                           env_object_info['name'],
                                           self.agent_text_size,
                                           'agent text')
        self.gui_agents_texts_idx.append(gui_agent_text_id)

    def create_thruster(self, env_object_info):
        """create canvas widget that represents thruster"""
        gui_thruster_position = self.scale(env_object_info['position'])
        gui_thruster_radius = self.draw_scale * env_object_info['radius']

        for idx in range(env_object_info['thrust number']):
            gui_thruster_id = self.draw_thrusters(gui_thruster_position,
                                                  gui_thruster_radius,
                                                  self.thrust_directions[idx],
                                                  env_object_info['thruster activity'][idx],
                                                  self.thrust_factor,
                                                  self.thrust_color,
                                                  'thruster')
            self.gui_thrusters_idx.append(gui_thruster_id)
            self.gui_thrusters_idx_to_name.append(
                '{}_thruster_{}'.format(env_object_info['name'], idx))

    def create_predictions(self, env_object_info):
        """create several canvas widgets that each represents a prediction"""
        gui_prediction_radius = \
            env_object_info['radius'] * self.draw_scale * self.prediction_circle_radius_scale
        predictions_array = env_object_info['predictions']
        for idx, prediction in enumerate(predictions_array):
            gui_prediction_position = self.scale(prediction)
            gui_prediction_id = self.draw_circle(gui_prediction_position,
                                                 gui_prediction_radius,
                                                 self.prediction_circle_color,
                                                 'prediction')
            self.gui_predictions_circle_idx.append(gui_prediction_id)
            self.gui_predictions_idx_to_name.append(
                '{}_prediction_{}'.format(env_object_info['name'], idx))

            gui_prediction_text_id = \
                self.draw_text(gui_prediction_position,
                               self.prediction_text_color,
                               '{}-P{}'.format(env_object_info['name'], idx),
                               self.prediction_text_size,
                               'prediction text')
            self.gui_predictions_texts_idx.append(gui_prediction_text_id)

    def create_target(self, env_object_info):
        """create canvas widget that represents a target"""
        gui_target_position = self.scale(env_object_info['target'])
        gui_target_radius = \
            env_object_info['radius'] * self.draw_scale * self.target_circle_radius_scale
        gui_target_id = self.draw_circle(gui_target_position,
                                         gui_target_radius,
                                         self.target_circle_color,
                                         'target')
        self.gui_targets_circle_idx.append(gui_target_id)
        self.gui_targets_idx_to_name.append(
            '{}_target'.format(env_object_info['name']))

        gui_target_text_id = self.draw_text(gui_target_position,
                                            self.target_text_color,
                                            env_object_info['name'] + '-T',
                                            self.target_text_size,
                                            'target text')
        self.gui_targets_texts_idx.append(gui_target_text_id)

    def create_obstacle(self, env_object_info):
        """
        create canvas widget that represents an obstacle

        Parameters
        ----------
        env_object_info : dict
            name : str
            type : str
            position : (2,) array
            velocity : (2,) array
            radius : float
            color : str

        Returns
        -------

        """
        gui_obstacle_position = self.scale(env_object_info['position'])
        gui_obstacle_radius = env_object_info['radius'] * self.draw_scale
        gui_obstacle_id = self.draw_circle(gui_obstacle_position,
                                           gui_obstacle_radius,
                                           self.obstacle_circle_color,
                                           'obstacle')
        self.gui_obstacles_circle_idx.append(gui_obstacle_id)
        self.gui_obstacles_idx_to_name.append(env_object_info['name'])

        obstacle_text = 'O-' + env_object_info['name'].split('_')[1]
        gui_obstacle_text_id = self.draw_text(gui_obstacle_position,
                                              self.obstacle_text_color,
                                              obstacle_text,
                                              self.obstacle_text_size,
                                              'obstacle text')
        self.gui_obstacles_texts_idx.append(gui_obstacle_text_id)

    def draw_circle(self, position, radius, color, tag):
        """create circle canvas widget"""
        return self.canvas.create_oval(
            util.get_circle_points(position, radius),
            fill=color,
            outline='black',
            tags=tag
        )

    def draw_thrusters(self, position, radius, direction, activity, factor, color, tag):
        """create polygon canvas widget representing thrusters"""
        return self.canvas.create_polygon(util.get_thruster_points(np.random.normal(0, 0.1),
                                                                   position,
                                                                   radius,
                                                                   direction,
                                                                   activity,
                                                                   factor),
                                          fill=color, tags=tag)

    def draw_text(self, position, text_color, text, size, tag, anchor=tk.CENTER):
        """create text canvas widget for better labeling of different objects"""
        return self.canvas.create_text(position[0],
                                       position[1],
                                       fill=text_color,
                                       text=text,
                                       font=(None, size),
                                       tag=tag,
                                       anchor=anchor)

    def update(self, env_objects_info, texts_info):
        """
        method for drawing new time step

        all agents and obstacles should be already created

        Parameters
        ----------
        env_objects_info : []
            containing dictionaries with
                name : str
                type : str
                position : (2,) array
                velocity : (2,) array
                radius : float
                color : str

        texts_info : dict
            dictionary containing
                time step : int
                    current time step that should be displayed
                prediction error : float
                    current prediction error
                target error : float
                    current target error

        Returns
        -------

        """
        self.update_texts(texts_info)
        self.update_objects(env_objects_info)

        self.tk_root.update_idletasks()
        self.tk_root.update()

        if self.verbose:
            print('update with env objects:')
            for eoi in env_objects_info:
                print(eoi)
            self.print_()

    def update_texts(self, texts_info):
        """
        update all canvas text widgets

        Parameters
        ----------
        texts_info : dict
            time step : int
            prediction error : float
            target error : float

        Returns
        -------

        """
        self.update_time_step_text(texts_info['time step'])
        self.update_prediction_error_text(texts_info['prediction error'])
        self.update_target_error_text(texts_info['target error'])

    def update_time_step_text(self, time_step):
        """
        update time step canvas text widget

        Parameters
        ----------
        time_step : int
            current time step to display

        Returns
        -------

        """
        self.time_step_text = 't: {}'.format(time_step)
        self.canvas.itemconfigure(self.time_step_text_id, text=self.time_step_text)

    def update_prediction_error_text(self, prediction_error):
        """
        update prediction error canvas text widget

        Parameters
        ----------
        prediction_error : float
            current prediction error to display

        Returns
        -------

        """
        self.prediction_error_text = 'Prediction error: {}'.format(prediction_error)

        if prediction_error <= self.prediction_error_text_color_change_threshold:
            self.prediction_error_text_color = 'green'
        else:
            self.prediction_error_text_color = 'black'

        self.canvas.itemconfigure(self.target_error_text_id,
                                  text=self.target_error_text,
                                  fill=self.target_error_text_color)

    def update_target_error_text(self, target_error):
        """
        update target error canvas text widget

        Parameters
        ----------
        target_error : float
            current target error to display

        Returns
        -------

        """
        self.target_error_text = 'Target error: {}'.format(target_error)

        if target_error <= self.target_error_text_color_change_threshold:
            self.target_error_text_color = 'green'
        else:
            self.target_error_text_color = 'black'

        self.canvas.itemconfigure(self.prediction_error_text_id,
                                  text=self.prediction_error_text,
                                  fill=self.prediction_error_text_color)

    def update_objects(self, env_objects_info):
        """
        update all relevant canvas widgets

        that are needed for visualization of the current environment state

        Parameters
        ----------
        env_objects_info : list
            containing dictionaries with
                name : str
                type : str
                position : (2,) array
                velocity : (2,) array
                radius : float
                color : str
                if type == 'agent':
                    thrust number : int
                    thruster activity : (motor_commands_dimensions,) array
                    predictions : (prediction_horizon, position_dimensions) array
                    target : (position_dimensions,) array

        Returns
        -------

        """
        for env_object_info in env_objects_info:
            if env_object_info['type'] == 'agent':
                self.update_complete_agent(env_object_info)
            elif env_object_info['type'] == 'obstacle':
                self.update_obstacles(env_object_info)
            else:
                raise RuntimeError('env object type {} not understood'.format(env_object_info['type']))

    def update_complete_agent(self, env_object_info):
        """
        update canvas widgets that are needed for current agent

        Parameters
        ----------
        env_object_info : dict
            name : str
            type : str
            position : (2,) array
            velocity : (2,) array
            radius : float
            thrust number : int
            thruster activity : (motor_commands_dimensions,) array
            predictions : (prediction_horizon, position_dimensions) array
            target : (position_dimensions,) array

        Returns
        -------

        """
        self.update_thruster(env_object_info)
        self.update_predictions(env_object_info)
        self.update_target(env_object_info)

        # sequential arrangement is important,
        # because the agent circle widget should not be overlapped by any other widget
        self.update_agent(env_object_info)

    def update_agent(self, env_object_info):
        """
        update canvas widget that represents the agent

        Parameters
        ----------
        env_object_info : dict
            name : str
            type : str
            position : (2,) array
            velocity : (2,) array
            radius : float
            thrust number : int
            thruster activity : (motor_commands_dimensions,) array
            predictions : (prediction_horizon, position_dimensions) array
            target : (position_dimensions,) array

        Returns
        -------

        """
        gui_agent_name_id = \
            self.gui_agents_idx_to_name.index(env_object_info['name'])
        gui_agent_id = self.gui_agents_circle_idx[gui_agent_name_id]
        new_gui_agent_position = self.scale(env_object_info['position'])
        new_gui_agent_radius = env_object_info['radius'] * self.draw_scale
        new_gui_agent_points = util.get_circle_points(new_gui_agent_position,
                                                      new_gui_agent_radius)
        self.canvas.coords(gui_agent_id, new_gui_agent_points)

        gui_agent_text_id = self.gui_agents_texts_idx[gui_agent_name_id]
        self.canvas.coords(gui_agent_text_id,
                           new_gui_agent_position[0],
                           new_gui_agent_position[1])

    def update_thruster(self, env_object_info):
        """
        update canvas widget that represents thruster

        Parameters
        ----------
        env_object_info : dict
            name : str
            type : str
            position : (2,) array
            velocity : (2,) array
            radius : float
            thrust number : int
            thruster activity : (motor_commands_dimensions,) array
            predictions : (prediction_horizon, position_dimensions) array
            target : (position_dimensions,) array

        Returns
        -------

        """
        new_gui_thruster_position = self.scale(env_object_info['position'])
        new_gui_thruster_radius = self.draw_scale * env_object_info['radius']

        for idx in range(env_object_info['thrust number']):
            gui_thruster_name_id = self.gui_thrusters_idx_to_name.index(
                    '{}_thruster_{}'.format(env_object_info['name'], idx))
            gui_thruster_id = self.gui_thrusters_idx[gui_thruster_name_id]
            new_gui_thruster_points = util.get_thruster_points(np.random.normal(0, 0.1),
                                                               new_gui_thruster_position,
                                                               new_gui_thruster_radius,
                                                               self.thrust_directions[idx],
                                                               env_object_info['thruster activity'][idx],
                                                               self.thrust_factor)
            self.canvas.coords(gui_thruster_id, new_gui_thruster_points)

    def update_predictions(self, env_object_info):
        """
        update several canvas widgets that each represents a prediction

        Parameters
        ----------
        env_object_info : dict
            name : str
            type : str
            position : (2,) array
            velocity : (2,) array
            radius : float
            thrust number : int
            thruster activity : (motor_commands_dimensions,) array
            predictions : (prediction_horizon, position_dimensions) array
            target : (position_dimensions,) array

        Returns
        -------

        """
        predictions_array = env_object_info['predictions']

        for idx, prediction in enumerate(predictions_array):
            gui_prediction_name_id = self.gui_predictions_idx_to_name.index(
                '{}_prediction_{}'.format(env_object_info['name'], idx))
            gui_prediction_id = self.gui_predictions_circle_idx[gui_prediction_name_id]
            new_gui_prediction_position = self.scale(prediction)
            new_gui_prediction_radius = \
                env_object_info['radius'] * self.draw_scale * self.prediction_circle_radius_scale
            new_gui_prediction_points = util.get_circle_points(new_gui_prediction_position,
                                                               new_gui_prediction_radius)
            self.canvas.coords(gui_prediction_id, new_gui_prediction_points)

            gui_prediction_text_id = \
                self.gui_predictions_texts_idx[gui_prediction_name_id]
            self.canvas.coords(gui_prediction_text_id,
                               new_gui_prediction_position[0],
                               new_gui_prediction_position[1])

    def update_target(self, env_object_info):
        """
        update canvas widgets that represents a target

        Parameters
        ----------
        env_object_info : dict
            name : str
            type : str
            position : (2,) array
            velocity : (2,) array
            radius : float
            thrust number : int
            thruster activity : (motor_commands_dimensions,) array
            predictions : (prediction_horizon, position_dimensions) array
            target : (position_dimensions,) array

        Returns
        -------

        """
        gui_target_name_id = self.gui_targets_idx_to_name.index(
            '{}_target'.format(env_object_info['name']))
        gui_target_id = self.gui_targets_circle_idx[gui_target_name_id]
        new_gui_target_position = self.scale(env_object_info['target'])
        new_gui_target_radius = \
            env_object_info['radius'] * self.draw_scale * self.target_circle_radius_scale
        new_gui_target_points = util.get_circle_points(new_gui_target_position,
                                                       new_gui_target_radius)
        self.canvas.coords(gui_target_id, new_gui_target_points)

        gui_target_text_id = self.gui_targets_texts_idx[gui_target_name_id]
        self.canvas.coords(gui_target_text_id,
                           new_gui_target_position[0],
                           new_gui_target_position[1])

    def update_obstacles(self, env_object_info):
        """
        update canvas widget that represents an obstacle

        Parameters
        ----------
        env_object_info  : dict
            name : str
            type : str
            position : (2,) array
            velocity : (2,) array
            radius : float

        Returns
        -------

        """
        gui_obstacle_name_id = \
            self.gui_obstacles_idx_to_name.index(env_object_info['name'])
        gui_obstacle_id = self.gui_obstacles_circle_idx[gui_obstacle_name_id]
        new_gui_obstacle_position = self.scale(env_object_info['position'])
        new_gui_obstacle_radius = env_object_info['radius'] * self.draw_scale
        new_gui_obstacle_points = util.get_circle_points(new_gui_obstacle_position,
                                                         new_gui_obstacle_radius)
        self.canvas.coords(gui_obstacle_id, new_gui_obstacle_points)

        gui_obstacle_text_id = self.gui_obstacles_texts_idx[gui_obstacle_name_id]
        self.canvas.coords(gui_obstacle_text_id,
                           new_gui_obstacle_position[0],
                           new_gui_obstacle_position[1])

    def close(self):
        """close tkinter application"""
        self.tk_root.quit()

    def scale(self, coordinates):
        """
        transform environment coordinates into gui coordinates

        Parameters
        ----------
        coordinates : (2,) array

        Returns
        -------
        result : (2,) array
        """
        result = np.zeros(coordinates.shape)

        result[0] = coordinates[0] * self.draw_scale + self.center_X
        result[1] = self.center_Y - coordinates[1] * self.draw_scale

        return result

    def descale(self, coordinates):
        """
        transform gui coordinates into environment coordinates

        Parameters
        ----------
        coordinates : (2,) array

        Returns
        -------
        result : (2,) array

        """
        result = np.zeros(coordinates.shape)

        result[0] = (coordinates[0] - self.center_X) / self.draw_scale
        result[1] = -(coordinates[1] - self.center_Y) / self.draw_scale

        return result

    # TODO is this somewhere needed?
    def mark(self, position):
        """TODO docstring"""
        self.is_marked = True
        self.marker_position = self.scale(position)

    # TODO is this somewhere needed?
    def draw_marker(self):
        """TODO docstring"""
        self.marker_id = self.canvas.create_oval(
            self.marker_position[0] + self.marker_rad,
            self.marker_position[1] + self.marker_rad,
            self.marker_position[0] - self.marker_rad,
            self.marker_position[1] - self.marker_rad,
            fill=self.marker_color,
            outline='black')

    # TODO rework ?
    def make_screen_shot(self, title=None):
        """TODO docstring"""
        # Take screen shot
        x = self.canvas.winfo_rootx() + self.canvas.winfo_x()
        y = self.canvas.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        box = (x, y, x1, y1)
        if title is None:
            title = str(datetime.datetime.now()). \
                replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")
        image_grab.grab(bbox=box, childprocess=False).save("screenshots/" + title + ".png")
