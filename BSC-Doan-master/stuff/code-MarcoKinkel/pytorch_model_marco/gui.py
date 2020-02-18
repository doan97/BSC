from simulator import Simulator
import tkinter as tk
from random import random, gauss
import numpy as np
import math
from agent import Agent
import global_config as c
import datetime

import pyscreenshot as ImageGrab # For Linux

class GUI(object):

    def __init__(self):

        agents = []

        drawscale = 250.0

        width = 3.0  # with 1.5 to each side
        height = 2.0

        gui = tk.Tk()
        gui.title("Simulator")
        gui.resizable(False, False)

        panel_width = drawscale * width
        panel_height = drawscale * height
        panel = tk.Canvas(gui, width=panel_width, height=panel_height, background="white")
        panel.pack()

        # setup timer
        fps = 30.0
        dtmsec = 1000 / fps
        dtsec = dtmsec / 1000.0

        # groundcolor = 'yellow'
        groundHeight = 0

        centerX = panel_width / 2
        groundY = panel_height - groundHeight

        # Draw grid
        for step in range(0, int(panel_width+1), 20):
            panel.create_line((step, 0, step, groundY), fill='lightgray')
            panel.create_line((0, step, panel_width, step), fill='lightgray')
        
        panel.create_rectangle(1, 1, panel_width-1, groundY-1, fill='', width=10)

        # Marker
        marker_rad = drawscale * 0.03
        marker_color = 'yellow'
        marker_id = None
        is_marked = False

        # Texts
        time_step_id = panel.create_text(10, 10, text='', anchor=tk.W)
        time_step_text = ''
        
        target_error_id = panel.create_text(10, 25, text='', anchor=tk.W)
        target_error_text = ''
        target_error_text_color = 'black'
        
        prediction_error_id = panel.create_text(10, 40, text='', anchor=tk.W)
        prediction_error_text = ''
        prediction_error_text_color = 'black'

        # If true, no line from the ball position to the first prediction position will be drawn
        hide_first_line = False

        sensor_dirs = calc_sensor_dirs()

        #--------------------------
        # Export class variables
        self.agents = agents

        self.gui = gui
        self.panel = panel

        self.drawscale = drawscale
        self.dtmsec = dtmsec
        self.centerX = centerX
        self.groundY = groundY
        
        self.marker_id = marker_id
        self.marker_color = marker_color
        self.marker_rad = marker_rad
        self.is_marked = is_marked

        self.prediction_error_id = prediction_error_id
        self.target_error_id = target_error_id
        self.prediction_error_text = prediction_error_text
        self.target_error_text = target_error_text
        self.time_step_id = time_step_id
        self.time_step_text = time_step_text
        self.target_error_text_color = target_error_text_color
        self.prediction_error_text_color = prediction_error_text_color

        self.hide_first_line = hide_first_line

        self.sensor_dirs = sensor_dirs

    def register(self, agent):
        self.agents.append(agent)
     
    def mark(self, position):
        self.is_marked = True
        self.marker_pos = self.scale(position)

    def scale(self, coordinates):
        result = np.zeros_like(coordinates)

        if(coordinates.shape == (2,)):
            result[0] = self.centerX + (coordinates[0] * self.drawscale)
            result[1] = self.groundY - (coordinates[1] * self.drawscale)

            # result = [result[i].item() for i in range(len(result))]
        
        elif(len(coordinates.shape) == 2 and coordinates.shape[1] == 2):
            result[:, 0] = self.centerX + (coordinates[:, 0] * self.drawscale)
            result[:, 1] = self.groundY - (coordinates[:, 1] * self.drawscale)

            # for i in range(result.shape[0]):
            #     for j in range(result.shape[1]):
            #         result[i, j] = result[i, j].item()

        else:
            raise SystemExit("Wrong Dimensionality in gui.scale()")

        return result

    def unscale(self, gui_coordinates):
        result = np.zeros_like(gui_coordinates, dtype=np.float32)

        result[0] = (gui_coordinates[0] - self.centerX) / self.drawscale
        result[1] = (self.groundY - gui_coordinates[1]) / self.drawscale

        return result

    def update_prediction_error(self, error):
        self.prediction_error_text = 'Prediction error: ' + str(error)

        if(error <= c.PREDICTION_ERROR_BOUNDARY):
            self.prediction_error_text_color = 'green'
        else:
            self.prediction_error_text_color = 'black'

    def update_target_error(self, error):
        self.target_error_text = 'Target error: ' + str(error)

        if(error <= c.TARGET_ERROR_BOUNDARY):
            self.target_error_text_color = 'green'
        else:
            self.target_error_text_color = 'black'

    def update_time_step(self, from_step, to_step=None):
        if(to_step is None):
            self.time_step_text = 't: ' + str(from_step)
        else:
            self.time_step_text = 't: ' + str(from_step) + ' to ' + str(to_step)

    '''
    Variables that draw needs:
        ballPos
        thrust_activity: the values of the thrust activity. Depends on the motor activity.
        thrust_ids: array of ids of all thrusts. shape (num_of_thrusts)

    Constants that run needs:
        ballRad
        thrustDirections. X and y coordinates per thrust. Shape (num_of_thrusts, 2)
        thrustColor
        ballColor
        dtmsec: the update frequency
    '''
    def draw(self):

        for a in self.agents:            
            if a.gui_att.scv_points is not None and a.gui_att.show_scv_targets is True:
                a.gui_att.scv_ids = self.draw_scv_targets(a.gui_att.predictions, a.gui_att.scv_points, a.gui_att.scv_ids, a.gui_att.scv_text_ids)
            
            a.gui_att.thrust_ids = self.draw_thrusts(a.gui_att.thrust_ids, a.gui_att.num_thrusts, a.gui_att.ball_pos, a.gui_att.thrust_directions, a.gui_att.thrust_activity, a.gui_att.thrust_factor, a.gui_att.thrust_color, a.gui_att.ball_rad)
            a.gui_att.ball_id = self.draw_ball(a.gui_att.ball_id, a.gui_att.ball_pos, a.gui_att.ball_rad, a.gui_att.ball_color)
            a.gui_att.ball_real_id = self.draw_ball_real(a.gui_att.ball_real_id, a.gui_att.is_ball_real_pos_set, a.gui_att.ball_real_pos, a.gui_att.ball_rad)
            if a.gui_att.show_target is True:
                a.gui_att.target_id = self.draw_target(a.gui_att.target_id, a.gui_att.target_pos, a.gui_att.target_rad, a.gui_att.target_color)
            if a.gui_att.show_predictions is True and a.gui_att.predictions is not None:
                a.gui_att.prediction_ids, a.gui_att.prediction_point_ids = self.draw_prediction_line(a.gui_att.prediction_ids, a.gui_att.prediction_point_ids, a.gui_att.ball_pos, a.gui_att.predictions)
            if a.gui_att.show_simulated_positions is True and a.gui_att.simulated_positions is not None:
                a.gui_att.simulated_position_ids = self.draw_simulated_positions_line(a.gui_att.simulated_position_ids, a.gui_att.is_ball_real_pos_set, a.gui_att.ball_real_pos, a.gui_att.ball_pos, a.gui_att.simulated_positions)
            a.gui_att.ball_name_id = self.draw_ball_name(a.gui_att.ball_name_id, a.id, a.gui_att.ball_pos, a.gui_att.ball_rad, a.gui_att.ball_name_color)
            # if a.gui_att.sensor_activity is not None:
                # a.gui_att.sensor_ray_ids = self.draw_sensor_activity(a.gui_att.sensor_ray_ids, a.gui_att.sensor_activity, a.gui_att.ball_pos, a.gui_att.sensor_ray_length)
            if a.gui_att.sensor_predictions is not None:
                a.gui_att.sensor_prediction_ids = self.draw_sensor_predictions(a.gui_att.sensor_predictions, a.gui_att.sensor_prediction_ids, a.gui_att.predictions)

        if self.marker_id is not None:
            self.panel.delete(self.marker_id)

        if self.is_marked:
            self.marker_id = self.panel.create_oval(
                self.marker_pos[0] + self.marker_rad,
                self.marker_pos[1] + self.marker_rad,
                self.marker_pos[0] - self.marker_rad,
                self.marker_pos[1] - self.marker_rad,
                fill=self.marker_color,
                outline='black'
            )

        # Errortexts
        self.panel.itemconfigure(self.target_error_id, text=self.target_error_text, fill=self.target_error_text_color)
        self.panel.itemconfigure(self.prediction_error_id, text=self.prediction_error_text, fill=self.prediction_error_text_color)
        self.panel.itemconfigure(self.time_step_id, text=self.time_step_text)

        # Now Update the GUI
        self.gui.update_idletasks()
        self.gui.update()


    # -----------
    # Draw methods
    # -----------

    def draw_thrusts(self, thrust_ids, num_thrusts, ball_pos, thrust_directions, thrust_activity, thrust_factor, thrust_color, ball_rad):
        
        ball_pos = self.scale(np.copy(ball_pos))
        
        for i in range(num_thrusts):
            r = gauss(0, 0.1)
            points = [
                ball_pos[0] - int((1.0 + r) * thrust_directions[i, 0] * thrust_activity[i] * thrust_factor * (ball_rad * self.drawscale)),
                ball_pos[1] + int((1.0 + r) * thrust_directions[i, 1] * thrust_activity[i] * thrust_factor * (ball_rad * self.drawscale)),
                ball_pos[0],
                ball_pos[1] - 5,
                ball_pos[0],
                ball_pos[1] + 5]
            
            if thrust_ids[i] is not None :
                self.panel.delete(thrust_ids[i])

            points = [points[i].item() for i in range(len(points))]
            
            thrust_ids[i] = self.panel.create_polygon(points, fill=thrust_color)

        return thrust_ids

    def draw_ball(self, ball_id, ball_pos, ball_rad, ball_color):
        
        ball_pos = self.scale(np.copy(ball_pos))

        if ball_id is not None :
            self.panel.delete(ball_id)

        ball_id = self.panel.create_oval(
            ball_pos[0] + (ball_rad * self.drawscale),
            ball_pos[1] + (ball_rad * self.drawscale),
            ball_pos[0] - (ball_rad * self.drawscale),
            ball_pos[1] - (ball_rad * self.drawscale),
            fill=ball_color,
            outline='black'
        )


        return ball_id

    def draw_ball_name(self, ball_name_id, ball_name, ball_pos, ball_rad, ball_name_color):
        ball_pos = self.scale(np.copy(ball_pos))

        if ball_name_id is not None :
            self.panel.delete(ball_name_id)

        ball_pos = [ball_pos[i].item() for i in range(len(ball_pos))]
        
        ball_name_id = self.panel.create_text(ball_pos[0], ball_pos[1], text=ball_name, anchor=tk.CENTER)

        return ball_name_id

    def draw_ball_real(self, ball_real_id, is_ball_real_pos_set, ball_real_pos, ball_rad):

        ball_real_pos = self.scale(np.copy(ball_real_pos))

        if ball_real_id is not None :
            self.panel.delete(ball_real_id)
        
        if(is_ball_real_pos_set):
            ball_real_id = self.panel.create_oval(
                ball_real_pos[0] + (ball_rad * self.drawscale),
                ball_real_pos[1] + (ball_rad * self.drawscale),
                ball_real_pos[0] - (ball_rad * self.drawscale),
                ball_real_pos[1] - (ball_rad * self.drawscale),
                fill='',
                outline='black')

        return ball_real_id

    def draw_target(self, target_id, target_pos, target_rad, target_color):

        target_pos = self.scale(np.copy(target_pos))
    
        if target_id is not None :
            self.panel.delete(target_id)
        
        target_id = self.panel.create_oval(
            target_pos[0] + (target_rad * self.drawscale),
            target_pos[1] + (target_rad * self.drawscale),
            target_pos[0] - (target_rad * self.drawscale),
            target_pos[1] - (target_rad * self.drawscale),
            fill=target_color,
            outline='black'
        )

        return target_id

    def draw_prediction_line(self, prediction_ids, prediction_point_ids, ball_pos, predictions):

        ball_pos = self.scale(np.copy(ball_pos))
        predictions = self.scale(np.copy(predictions))
        
        if prediction_ids is not None:
            last_point = ball_pos
            for i in range(len(prediction_ids)):
                if prediction_ids[i] is not None:
                    self.panel.delete(prediction_ids[i])
                    self.panel.delete(prediction_point_ids[i])

                points = [
                    last_point[0],
                    last_point[1],
                    predictions[i, 0],
                    predictions[i, 1]
                ]

                last_point = predictions[i]

                if c.VISUAL_LEARNING_STEP_BY_STEP is True and i == 0 and self.hide_first_line is True:
                    continue

                points = [points[i].item() for i in range(len(points))]

                prediction_ids[i] = self.panel.create_line(points, fill='red')

                prediction_point_ids[i] = self.panel.create_oval(
                    last_point[0] + (0.005 * self.drawscale),
                    last_point[1] + (0.005 * self.drawscale),
                    last_point[0] - (0.005 * self.drawscale),
                    last_point[1] - (0.005 * self.drawscale),
                    fill='red',
                    outline='red'
                )

        return prediction_ids, prediction_point_ids

    def draw_scv_targets(self, predictions, scv_points, scv_ids, scv_text_ids):
        predictions = self.scale(np.copy(predictions))
        scv_points = self.scale(np.copy(scv_points))

        for t in range(len(scv_ids)-1, -1, -1):
            if scv_ids[t] is not None:
                self.panel.delete(scv_ids[t])
            
            if scv_text_ids[t] is not None:
                self.panel.delete(scv_text_ids[t])

            points = [
                predictions[t, 0],
                predictions[t, 1],
                scv_points[t, 0],
                scv_points[t, 1]
            ]

            points = [points[i].item() for i in range(len(points))]

            # scv_ids[t] = self.panel.create_line(points, fill='green')

            if c.PLOT_SCV_GOALS is True:
                scv_ids[t] = self.panel.create_oval(
                    scv_points[t, 0] + 10,
                    scv_points[t, 1] + 10,
                    scv_points[t, 0] - 10,
                    scv_points[t, 1] - 10,
                    fill='white',
                    outline='red'
                )

                text_x = scv_points[t, 0].item()
                text_y = scv_points[t, 1].item()
                scv_text_ids[t] = self.panel.create_text(text_x, text_y, text=str(t+1))

        return scv_ids

    def draw_sensor_predictions(self, sensor_predictions, sensor_prediction_ids, predictions):
        
        predictions = self.scale(np.copy(predictions))
        
        scale = 200
        proximities = np.clip(sensor_predictions * scale, 0, scale)

        for t in range(1):
        # for t in range(sensor_predictions.shape[0]):

            # x, y = get_angle_ray(0.5, proximities[t, 0], predictions[t])
            # Scale the sensor vector to the length of the proximity
            first_point = previous_point = get_angle_ray(self.sensor_dirs, 0, proximities[t, 0], predictions[t])
            lines = np.zeros([c.INPUT_SENSOR_DIM], dtype=np.int)

            for s in range(1, c.INPUT_SENSOR_DIM):
                #x, y = get_angle_ray(s+0.5, proximities[t, s], predictions[t])
                point = get_angle_ray(self.sensor_dirs, s, proximities[t, s], predictions[t])

                points = [
                    previous_point[0],
                    previous_point[1],
                    point[0],
                    point[1]
                ]

                previous_point = point

                lines[s-1] = self.panel.create_line(points, fill='brown')

            # Last line back to first point
            points = [
                point[0],
                point[1],
                first_point[0],
                first_point[1]
            ]
            lines[s] = self.panel.create_line(points, fill='brown')

            # Delete old line ids
            if sensor_prediction_ids[t] is not None:
                for s in range(sensor_prediction_ids[t].shape[0]):
                    self.panel.delete(sensor_prediction_ids[t][s])

            # Store new line ids
            sensor_prediction_ids[t] = lines

        return sensor_prediction_ids


    def draw_simulated_positions_line(self, simulated_positions_ids, is_ball_real_pos_set, ball_real_pos, ball_pos, simulated_positions):
        
        simulated_positions = self.scale(np.copy(simulated_positions))
        ball_real_pos = self.scale(np.copy(ball_real_pos))
        ball_pos = self.scale(np.copy(ball_pos))
        
        if simulated_positions_ids is not None:

            if(is_ball_real_pos_set):
                last_point = ball_real_pos
            else:
                last_point = ball_pos

            for i in range(len(simulated_positions_ids)):
                if simulated_positions_ids[i] is not None:
                    self.panel.delete(simulated_positions_ids[i])

                points = [
                    last_point[0],
                    last_point[1],
                    simulated_positions[i, 0],
                    simulated_positions[i, 1]
                ]

                last_point = simulated_positions[i]

                if c.VISUAL_LEARNING_STEP_BY_STEP is True and i == 0 and self.hide_first_line is True:
                    continue

                points = [points[i].item() for i in range(len(points))]

                simulated_positions_ids[i] = self.panel.create_line(points, fill='black')

        return simulated_positions_ids
                

    def draw_sensor_activity(self, sensor_ray_ids, sensor_activity, ball_pos, length=5):

        ball_pos = self.scale(np.copy(ball_pos))

        for i in range(len(sensor_ray_ids)):
            if sensor_ray_ids[i] is not None:
                self.panel.delete(sensor_ray_ids[i])
                sensor_ray_ids[i] = None
        
        for i in range(len(sensor_activity)):
            proximity = sensor_activity[i]
            if proximity > 0:
                # This sensor is active
                # Check if the previous one is also active. Then dont draw this one, maybe just copy its id
                if(sensor_ray_ids[2*i-1] is not None):
                    sensor_ray_ids[2*i] = sensor_ray_ids[2*i-1]

                else:
                    point = get_angle_ray(self.sensor_dirs, i, length, ball_pos)

                    # Draw it
                    points = [
                        ball_pos[0],
                        ball_pos[1],
                        point[0],
                        point[1]
                    ]

                    points = [points[i].item() for i in range(len(points))]

                    sensor_ray_ids[2*i] = self.panel.create_line(points, fill='black')

                # Now the to-ray
                point = get_angle_ray(self.sensor_dirs, i+1, length, ball_pos)

                # Draw it
                points = [
                    ball_pos[0],
                    ball_pos[1],
                    point[0],
                    point[1]
                ]
                points = [points[i].item() for i in range(len(points))]

                sensor_ray_ids[2*i+1] = self.panel.create_line(points, fill='black')

        return sensor_ray_ids

    def save_screenshot(self, title=None):
        # Take screenshot
        x=self.panel.winfo_rootx()+self.panel.winfo_x()
        y=self.panel.winfo_rooty()+self.panel.winfo_y()
        x1=x+self.panel.winfo_width()
        y1=y+self.panel.winfo_height()
        box=(x,y,x1,y1)
        if title is None:
            title = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace("-", "_").replace(".","_")
        ImageGrab.grab(bbox=box, childprocess=False).save("screenshots/"+ title +".png")

# def get_relative_coordinates(source, i, length):
#     angle = i * (2 * np.pi) / c.INPUT_SENSOR_DIM
#     x = math.cos(angle) * length
#     y = math.sin(angle) * length

#     return source[0] + x, source[1] + y

def calc_sensor_dirs():
    dirs = []
    for i in range(c.INPUT_SENSOR_DIM):
        sensor_range_rad = (2 * np.pi) / c.INPUT_SENSOR_DIM
        angle_rad = (i * sensor_range_rad + (i+1) * sensor_range_rad) / 2.
        dirs.append(np.array([np.cos(angle_rad), np.sin(angle_rad)]))

    dirs = np.asarray(dirs)
    return dirs

def get_angle_ray(dirs, i, length, ball_pos):
    i %= len(dirs)
    return ball_pos + (length * dirs[i] / np.linalg.norm(dirs[i]))

def get_angle_ray_old(i, length, ball_pos):
    angle = i * (2 * np.pi) / c.INPUT_SENSOR_DIM
    if angle > np.pi:
        angle -= 2*np.pi

    tan = np.math.tan(angle)
    
    #by squaring the angle, the information of direction is lost
    x = abs(np.math.sqrt(length**2 / (tan**2 + 1)))
    y = abs(tan * x)

    if abs(angle) > np.pi/2:
        x *= -1
        
    if angle > 0:
        # In the simulator, the y value is inverted. High value = lower coordinate
        y *= -1

    x_pos = ball_pos[0] + x
    y_pos = ball_pos[1] + y

    return x_pos, y_pos

