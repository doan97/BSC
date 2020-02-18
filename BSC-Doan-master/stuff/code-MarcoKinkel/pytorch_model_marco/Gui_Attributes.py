import numpy as np
import global_config as c

class Gui_attributes():
    def __init__(self, gui, sim, radius, color=None):

        self.sim = sim
        
        if color is None:
            if self.sim.mode == 0:
                self.ball_color = 'red'  # Rocket
                self.ball_name_color = 'white'
            elif self.sim.mode == 1:
                self.ball_color = 'green'    # Glider
                self.ball_name_color = 'black'
            elif self.sim.mode == 2:
                self.ball_color = 'yellow'   # Stepper
                self.ball_name_color = 'black'
        else:
            self.ball_color = color
            self.ball_name_color = 'black'

        self.target_color = 'red'
        self.thrust_color = 'blue'

        self.ball_id = None
        self.ball_real_id = None
        self.target_id = None
        self.thrust_ids = np.repeat(None, self.sim.numThrusts)
        self.prediction_ids = None   # We don't know the motor inference depth yet
        self.prediction_point_ids = None
        self.simulated_position_ids = None   # We don't know the motor inference depth yet
        self.ball_name_id = None
        self.sensor_ray_ids = np.repeat(None, 2*c.INPUT_SENSOR_DIM)  # twice because we draw two rays per sensor

        self.ball_pos = np.zeros(2)
        self.ball_real_pos = np.copy(self.ball_pos)
        self.is_ball_real_pos_set = False

        self.ball_rad = radius
        self.target_pos = np.zeros(2)
        self.target_rad = self.ball_rad * 0.5  # Target is half the size of the ball

        self.num_thrusts = self.sim.numThrusts
        self.thrust_directions = self.sim.thrustDirections
        self.thrust_activity = np.zeros(self.num_thrusts)
        self.thrust_factor = 4.0

        self.predictions = None
        self.simulated_positions = None

        self.show_target = True
        self.show_predictions = True
        self.show_simulated_positions = True
        self.show_scv_targets = True

        self.sensor_ray_length = 20.0
        self.sensor_activity = None

        self.scv_ids = None
        self.scv_text_ids = None
        self.scv_points = None

        self.sensor_prediction_ids = None
        self.sensor_predictions = None

    def context_change(self):
        # Delete all thrust ids from gui in case the amount changes
        for i in range(thrust_ids):
            self.panel.delete(thrust_ids[i])

        self.num_thrusts = self.sim.numThrusts
        self.thrust_ids = np.repeat(None, self.num_thrusts)
        self.thrust_activity = np.zeros(self.num_thrusts)

        self.ball_color = get_ball_color(self.sim.mode)

    '''
    position has shape (2)
    motor_commands has shape (4)
    Position is the real position as calculated by the simulator. No need to consider delta-processing
    '''
    def update_position(self, position, motor_commands=None, sensor_data=None):
        # Motorcommands need to be clamped and set to 0 depending on the mode
        if motor_commands is not None:
            self.thrust_activity = self.sim.get_thrusts_from_motor_commands(motor_commands)
        else:
            self.thrust_activity = np.zeros(c.INPUT_MOTOR_DIM)
            
        self.ball_pos = position

        if sensor_data is not None:
            self.sensor_activity = sensor_data

    def update_real_position(self, position):
        self.ball_real_pos = position
        self.is_ball_real_pos_set = True

    '''
    Target position should have shape (2)
    '''
    def update_target(self, target_position):
        self.target_pos = target_position

    '''
    Predictions have shape (time_steps, sensor_dim)
    Always put the absolute predictions in here
    '''
    def update_actinf_path(self, predictions):
        self.predictions = np.zeros_like(predictions)

        if self.prediction_ids is None:
            self.prediction_ids = np.repeat(None, predictions.shape[0])
            self.prediction_point_ids = np.repeat(None, predictions.shape[0])

        self.predictions = predictions

    '''
    Path of simulated real positions
    '''
    def update_simulated_positions_path(self, simulated_positions):
        self.simulated_positions = np.zeros_like(simulated_positions)

        if self.simulated_position_ids is None:
            self.simulated_position_ids = np.repeat(None, simulated_positions.shape[0])

        self.simulated_positions = simulated_positions

    def update_scv_lines(self, abs_targets):

        # abs_targets = abs_targets[:2]

        # self.scv_points = self.predictions + scv
        self.scv_points = abs_targets

        if self.scv_ids is None:
            self.scv_ids = np.repeat(None, abs_targets.shape[0])
            self.scv_text_ids = np.repeat(None, abs_targets.shape[0])

    def update_sensor_predictions(self, predictions):
        self.sensor_predictions = predictions

        if self.sensor_prediction_ids is None:
            self.sensor_prediction_ids = np.repeat(None, predictions.shape[0])

    
    def get_ball_color(mode):
        if mode == 0:
            return 'red'  # Rocket
        elif mode == 1:
            return 'green'    # Glider
        elif mode == 2:
            return 'yellow'   # Stepper
        
