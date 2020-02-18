import numpy as np
import datetime


class Config:
    def __init__(self):
        self.seed = 1223
        self.agents_config = Config._get_agents_config()
        self.env_config = Config._get_env_config()
        self.model_config = Config._get_model_config(self.env_config.input_dim, self.env_config.output_dim)
        self.gui_config = Config._get_gui_config()
        self.inference_config = Config._get_inference_config()
        self.train_config = Config._get_train_config()
        self.save_config = Config._get_save_config()

    @staticmethod
    def _get_agents_config():
        return AgentsConfig()

    @staticmethod
    def _get_env_config():
        return EnvConfig()

    @staticmethod
    def _get_model_config(input_dim, output_dim):
        return ModelConfig(input_dim, output_dim)

    @staticmethod
    def _get_gui_config():
        return GuiConfig()

    @staticmethod
    def _get_inference_config():
        return InferenceConfig()

    @staticmethod
    def _get_train_config():
        return TrainConfig()

    @staticmethod
    def _get_save_config():
        return SaveConfig()


class AgentsConfig:
    def __init__(self):

        self.agent_mode = -1
        self.available_modes = {
            -1: 'Alone',
            0: 'Same target',
            1: 'Chase with rnd target, stage 1',
            2: 'Chase with target line',
            3: 'Obstacle in center',
            4: 'B chases A, A wants distance',
            5: 'B wants proximity, A does not',
            6: '-1 but with loss',
            # 7x: Obstacle avoidance
            71: 'Static B',
            72: 'B is two positions behind A',
            73: 'B is opposite of A',
            74: 'B starts in middle with same goal as A',
            8: 'Uniform distribution',
            9: 'Goal directed actinf with obstacles',
            10: 'Chase with rnd target, stage 3',
            11: 'A static in center. B flies by'
        }
        assert self.agent_mode in self.available_modes.keys(), 'agent mode not available'
        print('agents mode: {}'.format(self.agent_mode))

        # TODO fill in other modes
        if self.agent_mode == -1:
            """alone"""
            self.agents = [
                Agent(name='A', initial_position=[0., 1.], mass=0.1, radius=0.06,
                      agent_type='ActiveInference', target_type='Random', save_file='')
            ]
        elif self.agent_mode == 0:
            """same target"""
            self.agents = [
                Agent(name='A', initial_position=[0., 1.], mass=0.1, radius=0.06,
                      agent_type='ActiveInference', target_type='Random', save_file=''),
                Agent(name='B', initial_position=[-1., 1.], mass=0.1, radius=0.06,
                      agent_type='ActiveInference', target_type='TargetOfOther', save_file='')
            ]
        elif self.agent_mode == 1:
            """chase with random target, stage 1"""
            self.agents = [
                Agent(name='A', initial_position=[0., 1.], mass=0.1, radius=0.06,
                      agent_type='ActiveInference', target_type='Random', save_file=''),
                Agent(name='B', initial_position=[-1., 1.], mass=0.1, radius=0.06,
                      agent_type='ActiveInference', target_type='Other', save_file='')
            ]
        elif self.agent_mode == 2:
            """chase with line target"""
            self.agents = [
                Agent(name='A', initial_position=[0., 1.], mass=0.1, radius=0.06,
                      agent_type='ActiveInference', target_type='Line', save_file=''),
                Agent(name='B', initial_position=[-1., 1.], mass=0.1, radius=0.06,
                      agent_type='ActiveInference', target_type='Other', save_file='')
            ]
        elif self.agent_mode == 3:
            """agent b as obstacle in center"""
            self.agents = [
                Agent(name='A', initial_position=[-1., 1.], mass=0.1, radius=0.06,
                      agent_type='ActiveInference', target_type=[1., 1.], save_file=''),
                Agent(name='B', initial_position=[.0, 1.], mass=0.1, radius=0.06,
                      agent_type='Obstacle', target_type=None, save_file='')
            ]
        elif self.agent_mode == 4:
            """agent b chases agent a, agent a wants distance (?)"""
            # TODO find difference between 1 and 4
            pass
        elif self.agent_mode == 5:
            """agent b wants proximity, agent a does not"""
            self.agents = [
                Agent(name='A', initial_position=[-1., 1.], mass=0.1, radius=0.06,
                      agent_type='ActiveInference', target_type='Proximity', save_file=''),
                Agent(name='B', initial_position=[.4, 1.], mass=0.1, radius=0.06,
                      agent_type='ActiveInference', target_type='NoProximity', save_file='')
            ]
        elif self.agent_mode == 6:
            """-1 but with loss (?)"""
            # TODO find difference between -1 and 6
            pass
        elif self.agent_mode == 71:
            pass
        elif self.agent_mode == 72:
            pass
        elif self.agent_mode == 73:
            pass
        elif self.agent_mode == 74:
            pass
        elif self.agent_mode == 8:
            pass
        elif self.agent_mode == 9:
            pass
        elif self.agent_mode == 10:
            pass
        elif self.agent_mode == 11:
            pass


class Agent:
    def __init__(self, name,

                 initial_position,
                 mass,
                 radius,
                 agent_type,
                 target_type,

                 save_file):
        """dummy class for saving specific agent config"""
        self.name = name

        self.initial_position = np.array(initial_position)
        self.mass = mass
        self.radius = radius

        self.agent_type = agent_type

        self.target_type = target_type

        self.save_file = save_file


class EnvConfig:
    def __init__(self):
        self.verbose = False

        self.physic_mode_names = ['Rocket', 'Glider', 'Stepper', 'Stepper2Dir']

        self.max_thrust = 1.0
        self.left_border = -1.5
        self.right_border = 1.5
        self.top_border = 3.0
        self.bot_border = 0.0
        self.size = np.array([self.right_border - self.left_border,
                              self.top_border - self.bot_border])
        self.delta_time = 1. / 30.

        self.number_of_obstacles = 0
        self.positions_of_obstacles = None
        self.radii_of_obstacles = 0.06

        self.number_of_problems = 1
        self.min_motor_value = 0
        self.max_motor_value = 1

        # Point spread function
        self.point_spread_flag = True
        self.point_spread_type = 'gauss'  # or linear
        self.point_spread_size = 0.1  # 0.2 means the signal decreases by 0.2 times the number of sensors per sensor
        self.point_spread_sigma = 1.0  # Sigma for Gaussian distribution
        self.point_spread_clip_min = 0.0
        self.point_spread_clip_max = 1.0
        self.point_spread_round_decimals = 8

        self.normalize_sensor_readings = True

        # PROXIMITY SENSORS
        self.max_distance = 10  # times the radius

        self.border_proximity_weight = 0.5  # 0.5

        self.position_dimensions = 2
        self.motor_commands_dimensions = 4
        self.sensor_readings_dimensions = 16

        # (s_t, a_t, r_t (omitted), s_(t+1), a_(t+1) (omitted))
        self.transition_data_type = np.dtype([
            # time step t
            ('old position', np.float64, (self.position_dimensions,)),
            ('old velocity', np.float64, (self.position_dimensions,)),
            ('old sensor readings', np.float64, (self.sensor_readings_dimensions,)),
            ('motor commands', np.float64, (self.motor_commands_dimensions,)),

            # time step t+1
            ('new position', np.float64, (self.position_dimensions,)),
            ('new velocity', np.float64, (self.position_dimensions,)),
            ('new sensor readings', np.float64, (self.sensor_readings_dimensions, ))
        ])

        self.input_dim = \
            self.position_dimensions * 2 + self.motor_commands_dimensions + self.sensor_readings_dimensions
        self.output_dim = \
            self.position_dimensions * 2 + self.sensor_readings_dimensions

        # self.learning_scenarios = ['alone', 'static', 'alone', 'line', 'alone',
        #                            'curve', 'alone', 'line_acc', 'alone', 'curve_acc']
        self.obstacle_transformation_names = ['static', 'line', 'curve', 'line_acc', 'curve_acc']

        self.max_absolute_obstacle_start_velocity = 0.015
        self.obstacle_line_acceleration_factor = 1.005

        self.max_absolute_obstacle_rotation_radius = 0.2
        self.obstacle_curve_acceleration_factor = 1.005

        self.default_object_mass = 0.1


class ModelConfig:
    def __init__(self, input_dim, output_dim):

        self.position_weight_learning = 100
        self.acceleration_weight_learning = 0.01
        self.sensor_weight_learning = 10.0

        self.use_sensor_sensitivity = False
        self.learn_sensor_sensitivity = False

        # optimizing config
        self.learning_rate = 0.01
        self.batch_size = 100

        # module config : simple lstm  x -> h0 -> ... -> hn -> y
        self.input_dim = input_dim  # dimension of x
        self.hidden_dim = 32  # Dimension von h
        self.num_layers = 1  # number of vertical LSTM layers
        self.output_dim = output_dim  # dimension of y


class GuiConfig:
    def __init__(self):
        self.verbose = False

        self.title = 'Simulator'
        self.resizable = False

        self.draw_scale = 250.0

        # in pixel
        self.line_distance = 10

        self.offset = 'AllDown'  # None  # 'AllDown'

        self.panel_background = 'white'
        self.grid_fill_color = 'gray'

        self.marker_id = None
        self.marker_color = 'yellow'
        self.marker_radius = 0.03

        self.text_color = 'black'
        self.prediction_error_text_color_change_threshold = 0.01
        self.target_error_text_color_change_threshold = 0.01

        # unused
        # self.frames_per_second = 30.0
        # self.time_steps_per_second = 1000

        self.agent_colors = ['red', 'blue']
        self.agent_text_colors = ['blue', 'red']
        self.agent_text_size = 10

        self.thrust_color = 'yellow'
        self.thrust_factor = 4.0

        self.prediction_circle_radius_scale = 0.6
        self.prediction_circle_color = 'black'
        self.prediction_text_color = 'white'
        self.prediction_text_size = 5

        self.target_circle_radius_scale = 0.7
        self.target_circle_color = 'green'
        self.target_text_color = 'black'
        self.target_text_size = 7

        self.obstacle_circle_color = 'gray'
        self.obstacle_text_color = 'black'
        self.obstacle_text_size = 10

        # PLOT
        self.show_sensor_plot = False

        # No visual learning -> Half duration
        self.visual_learning = False
        self.visual_learning_step_by_step = True and self.visual_learning


class ModeConfig:
    def __init__(self):
        self.id = None
        self.description = None
        self.agents = None
        self.target = None


class InferenceConfig:
    def __init__(self):

        # Amount of epochs of size NUM_MINI_EPOCHS
        self.num_all_steps = 2000

        self.inference_iterations = 5
        self.prediction_horizon = 10
        self.step_size = .1

        self.target_change_frequency = 50

        self.mask_gradients_at_proximity = False

        self.use_scv = True
        self.scv_smoothing_factor = 0.5
        self.scv_weighting_factor = 4
        self.scv_beta = 4

        self.plot_scv_goals = True

        self.position_loss_weight = 1.0
        self.sensor_readings_loss_weight = 0.0

        self.vehicle = 1


class TrainConfig:
    def __init__(self):
        # Amount of epochs of size NUM_MINI_EPOCHS
        self.num_epochs = 2

        # Amount of mini-epochs of size NUM_TIME_STEPS
        self.num_mini_epochs = 2

        # Amount of steps until a new scenario is chosen
        self.num_time_steps = 10

        # maximum size of the ring buffer that holds the sequences
        self.max_memory_size = 1000


class SaveConfig:
    def __init__(self):

        self.start_dir = '.'
        self.project_name = 'Testing'
        self.epoch_step = 1
        self.mini_epoch_step = 1

        # choose environment name
        self.env_name = '/env_xxx'

        self.load_from_data = True
        self.continue_from_epoch = False
        self.continue_epoch = 0

        # reset if the project folder already exits
        self.reset = False
        self.reset_all = False #turn False for loading

        self.date = True #date in project name
        self.time = False #time in project name
        self.new = False #create new directorys

        self.save_config = False
        self.save_agents = True
        self.save_environment = False

        self.config_name = '/config.py'

        self.project_folder = self.start_dir + '/project_' + self.project_name

        date, time = self.get_Date_Time()
        if self.date:
            self.project_folder += '_' + date
        if self.time:
            self.project_folder += '_' + time

        tmp = self.project_folder
        self.env_path = tmp + '/environment'
        self.agent_path = tmp + '/agents'
        self.conf_path = tmp + '/configs'

        self.env_path_load = tmp + '/environment/final'
        self.agent_path_load = tmp + '/agents/final'

        self.structure = {self.project_folder:
                              {'/environment': {'/final': None}, #loading_path for environment
                               '/agents': {'/final' : None}, #loading path for agents
                               '/configs':
                                   {'/gui_configs': None,
                                    '/env_configs': None}
                               }
                          }

    def get_Date_Time(self):
        time_str = str(datetime.datetime.now().time())[:5]
        date_str = str(datetime.datetime.now().date())[:10]
        return date_str , time_str