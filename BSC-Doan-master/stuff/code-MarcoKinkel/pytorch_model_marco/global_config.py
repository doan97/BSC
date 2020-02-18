# -----------------
# GENERAL
# -----------------
RINGBUFFER_SIZE = 40000

# -----------------
# RNN
# -----------------
NUM_LAYERS = 1  # number of vertical LSTM layers
HIDDEN_DIM = 36  # Dimension von h

# -----------------
# INPUT
# -----------------
INPUT_POSITION_DIM = 2
INPUT_MOTOR_DIM = 4
INPUT_SENSOR_DIM = 16       # 16
INPUT_ACCELERATION_DIM = 2
INPUT_DIM = INPUT_POSITION_DIM + INPUT_MOTOR_DIM + INPUT_SENSOR_DIM + INPUT_ACCELERATION_DIM

POSITION_DIM_START = 0
POSITION_DIM_END = POSITION_DIM_START + INPUT_POSITION_DIM
MOTOR_DIM_START = POSITION_DIM_END
MOTOR_DIM_END = MOTOR_DIM_START + INPUT_MOTOR_DIM
SENSOR_DIM_START = MOTOR_DIM_END
SENSOR_DIM_END = SENSOR_DIM_START + INPUT_SENSOR_DIM
ACCELERATION_DIM_START = SENSOR_DIM_END
ACCELERATION_DIM_END = ACCELERATION_DIM_START + INPUT_ACCELERATION_DIM

# -----------------
# OUTPUT
# -----------------
OUTPUT_POSITION_DIM = 2
OUTPUT_SENSOR_DIM = INPUT_SENSOR_DIM                # INPUT_SENSOR_DIM or 0
OUTPUT_ACCELERATION_DIM = INPUT_ACCELERATION_DIM     # 0 if acceleration should not be predicted
OUTPUT_DIM = OUTPUT_POSITION_DIM + OUTPUT_SENSOR_DIM + OUTPUT_ACCELERATION_DIM

OUTPUT_POSITION_DIM_START = 0
OUTPUT_POSITION_DIM_END = OUTPUT_POSITION_DIM_START + OUTPUT_POSITION_DIM
OUTPUT_SENSOR_DIM_START = OUTPUT_POSITION_DIM_END
OUTPUT_SENSOR_DIM_END = OUTPUT_SENSOR_DIM_START + OUTPUT_SENSOR_DIM
OUTPUT_ACCELERATION_DIM_START = OUTPUT_SENSOR_DIM_END
OUTPUT_ACCELERATION_DIM_END = OUTPUT_ACCELERATION_DIM_START + OUTPUT_ACCELERATION_DIM

#------------------
# PLOT
#------------------
SHOW_SENSOR_PLOT = False
SHOW_SENSOR_PLOT_STEP_BY_STEP = False


# -----------------
# PROXIMITY SENSORS
# -----------------
MAX_DISTANCE = 60      # times the radius

# Point spread function
POINT_SPREAD = True
POINT_SPREAD_TYPE = 'gauss' # or linear
SPREADSIZE = 0.1  # 0.2 means the signal decreases by 0.2 times the number of sensors per sensor
SIGMA = 1.0  # Sigma for Gaussian distribution

BORDER_PROXIMITY_WEIGHT = 0.  # 0.5

USE_SENSOR_SENSITIVITY = False
LEARN_SENSOR_SENSITIVITY = False

#--------------
# LEARNING
#--------------
VISUAL_LEARNING = True    # No visual learning -> Half duration
VISUAL_LEARNING_STEP_BY_STEP = True and VISUAL_LEARNING

LEARNING_SCENARIOS = ['alone', 'static', 'alone', 'line', 'alone', 'curve', 'alone', 'line_acc', 'alone', 'curve_acc']
# LEARNING_SCENARIOS = ['static']
#LEARNING_SCENARIOS = ['static', 'line', 'curve', 'line_acc', 'curve_acc']
#LEARNING_SCENARIOS = ['curve', 'line']
POSITION_WEIGHT_LEARNING = 100      # 100
SENSOR_WEIGHT_LEARNING = 1.0       # 10, 100 fail, 20 fail,
ACCELERATION_WEIGHT_LEARNING = 0.01  # 1e-5

#------------------
# ACTINF
#------------------
MODE = 10
# -1    Alone
# 0     Same target
# 1     Chase with rnd target, stage 1
# 2     Chase with target line
# 3     Obstacle in center
# 4     B chases A, A wants distance
# 5     B wants proximity, A does not
# 6     -1 but with loss
# 7x    Obstacle avoidance
#   71  Static B
#   72  B is two positions behind A
#   73  B is opposite of A
#   74  B starts in middle with same goal as A
# 8     Uniform distribution
# 9     Goal directed actinf with obstacles
# 10    Chase with rnd target, stage 3
# 11    A static in center. B flies by

MASK_GRADIENTS_AT_PROXIMITY = False

USE_SCV = False
SCV_SMOOTHING_FACTOR = 0.5  # 0.5
SCV_WEIGHTING_FACTOR = 4    # 4
SCV_BETA = 4    # 4

PLOT_SCV_GOALS = False


CLAMP_TARGET_VELOCITY = True
CLAMP_TARGET_VELOCITY_VALUE = 0.015
CLAMP_TARGET_VELOCITY_VALUE_VELINF = 0.015

POSITION_LOSS_WEIGHT_ACTINF = 1.0
SENSOR_LOSS_WEIGHT_ACTINF = 0.1

VEHICLE = 1

