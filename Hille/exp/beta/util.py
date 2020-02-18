import torch
import numpy as np


def remove_list_item(l, item):
    return [i for i in l if i != item]


def remove_dict_key(d, key):
    return {k: v for k, v in d.items() if k is not key}


def get_key_values_as_list(d, keys):
    return [v for k, v in d.items() if k in keys]


def get_split_sections(list_of_arrays, dim):
    return [l.shape[dim] for l in list_of_arrays]


def get_differences(array):
    return array[1:] - array[:-1]


# environment init
def calc_thrust_directions():
    tmp1 = np.array([1.0, 1.0])
    tmp2 = np.array([-1.0, 1.0])
    tmp3 = np.array([1.0, -1.0])
    tmp4 = np.array([-1.0, -1.0])
    tmp1_norm = tmp1 / np.linalg.norm(tmp1)
    tmp2_norm = tmp2 / np.linalg.norm(tmp2)
    tmp3_norm = tmp3 / np.linalg.norm(tmp3)
    tmp4_norm = tmp4 / np.linalg.norm(tmp4)
    return np.array([tmp1_norm, tmp2_norm, tmp3_norm, tmp4_norm])


# for gui thruster
def get_thruster_points(r, position, radius, direction, activity, factor):
    """
    create canvas coordinates for triangle at position facing direction

    activity and factor determine the scaling

    Parameters
    ----------
    r : float
        random number for nice fluctuations
    position : (2,) array
    radius : float
    direction : (2,) array
    activity : float
    factor : float

    Returns
    -------
    points : []
        coordinates

    """
    return [
        position[0] - int(1.0 + r) * direction[0] * activity * factor * radius,
        position[1] + int(1.0 + r) * direction[1] * activity * factor * radius,
        position[0],
        position[1] - 5,
        position[0],
        position[1] + 5
    ]


# for gui agent, prediction and target
def get_circle_points(position, radius):
    """
    create canvas coordinates for circle at position with radius

    Parameters
    ----------
    position : (2,) array
    radius : float

    Returns
    -------
    points : []
        coordinates

    """
    return [
        position[0] + radius,
        position[1] + radius,
        position[0] - radius,
        position[1] - radius
    ]


def roll_array_and_fill_last_entry_randomly(array, dim):
    # print(array, array.shape)
    array = torch.roll(array, -1, dim)
    pos = ','.join(['-1' if d == dim else ':' for d in range(len(list(array.shape)))])
    new = torch.rand(array.shape).float()
    exec('array[{0}]=new[{0}]'.format(pos))
    # print(array)
    return array


# used for training environment model
def get_random_motor_commands(old_motor_commands, physic_mode_id, num_thrusters):
    """
    # TODO docstring

    Parameters
    ----------
    old_motor_commands : (motor_commands_dimensions,) array
    physic_mode_id : int
    num_thrusters : int

    Returns
    -------
    new_motor_commands : (motor_commands_dimensions,) array

    """
    r = np.random.rand()

    if r < 0.7:
        # 70%: new commands
        new_motor_commands = np.zeros(len(old_motor_commands))
        r = np.random.rand()

        if physic_mode_id == 0:
            # 1 of 2 active .2
            # both active .4
            # both equal .3
            # all out .1

            if r < 0.1:
                # 10%: do nothing
                return new_motor_commands

            elif r < 0.3:
                # 20%: 1 active
                active_thrust = np.random.choice(num_thrusters)
                new_motor_commands[active_thrust] = np.random.rand()
                return new_motor_commands

            elif r < 0.7:
                # 40%: both active, but random
                return np.random.random(len(old_motor_commands))

            else:
                # 30%: both active, but equal
                thrust = np.random.rand()
                new_motor_commands[0] = thrust
                new_motor_commands[1] = thrust
                return new_motor_commands

        else:
            # 0 of 4 active .2
            # 1 of 4 active .2
            # 2 of 4 active .2
            # 3 of 4 active .2
            # 4 of 4 active .2

            if r < 0.2:
                # 20%: do nothing
                return new_motor_commands

            elif r < 0.4:
                # 20%: 1 active
                active_thrust = np.random.choice(num_thrusters)
                new_motor_commands[active_thrust] = np.random.rand()
                return new_motor_commands

            elif r < 0.6:
                # 20%: 2 active
                active_thrusts = np.random.choice(num_thrusters, 2, replace=False)
                new_motor_commands[active_thrusts[0]] = np.random.random()
                new_motor_commands[active_thrusts[1]] = np.random.random()
                return new_motor_commands

            elif r < 0.8:
                # 20%: 3 active
                active_thrusts = np.random.choice(num_thrusters, 3, replace=False)
                new_motor_commands[active_thrusts[0]] = np.random.random()
                new_motor_commands[active_thrusts[1]] = np.random.random()
                new_motor_commands[active_thrusts[2]] = np.random.random()
                return new_motor_commands

            else:
                # 20%: all active
                return np.random.random(size=len(old_motor_commands))
    else:
        # 30%: old commands
        return old_motor_commands


def convert_ray_distances_to_sensor_readings(ray_distances):
    """
    Convert the ray distances to sensor readings

    one sensor reading is the max of its two surrounding ray distances

    Parameters
    ----------
    ray_distances : (sensor_readings_dimensions,) array

    Returns
    -------
    sensor_readings : (sensor_readings_dimensions,) array

    """
    return np.maximum.reduce([np.roll(ray_distances, 1),
                              np.roll(ray_distances, -1)])


def calculate_closest_agent_distance(my_pos, other_pos, my_radius, other_radius,
                                     sensor_readings_dimensions, max_distance):
    """
    # TODO docstring

    Parameters
    ----------
    my_pos : (position_dimensions,) array
    other_pos : (position_dimensions,) array
    my_radius : float
    other_radius : float
    sensor_readings_dimensions : int
    max_distance : float

    Returns
    -------
    active_sensor_index : int
    distance : float
        or
                    : None
                    : None
        in the case distance is greater than max_distance

    """
    distance = np.linalg.norm(other_pos - my_pos)
    real_distance = max([distance - my_radius - other_radius, 0])

    distance = get_proximity_linear(real_distance, my_radius, max_distance)

    if distance is None:
        return None, None

    x_diff = other_pos[0] - my_pos[0]
    y_diff = other_pos[1] - my_pos[1]
    angle_rad = np.arctan2(y_diff, x_diff)

    if angle_rad < 0:
        angle_rad += 2 * np.pi

    angle_deg = angle_rad * 180 / np.pi

    # with 4 sensors, the range is 90 degrees or pi/2
    sensor_range_deg = 360 / sensor_readings_dimensions

    # Determine the id of the active sensor, clockwise
    active_sensor = int(angle_deg / sensor_range_deg)

    # If the angle is exactly 360 degrees, this will return an invalid index
    if active_sensor == sensor_readings_dimensions:
        active_sensor -= 1

    return active_sensor, distance


def get_proximity_tan_h(distance, my_radius, max_distance):
    """
    # TODO docstring

    Parameters
    ----------
    distance : float
    my_radius : float
    max_distance : float

    Returns
    -------
    _distance : float

    """
    _max_distance = max_distance * my_radius
    max_proximity = 1 / _max_distance
    proximity = 1 / (distance + 1e-30)

    # Ignore sensor data when distance is too high
    if distance >= _max_distance:
        _distance = 0.0
    else:
        _distance = np.tanh(proximity - max_proximity) + 1e-30
    return _distance


def get_proximity_linear(distance, radius, max_distance):
    """
    # TODO docstring

    Parameters
    ----------
    distance : float
    radius : float
    max_distance : float

    Returns
    -------
    _distance : float

    """
    _max_distance = max_distance * radius

    # Ignore sensor data when distance is too high
    if distance >= _max_distance:
        _distance = 0.0
    else:
        _distance = -1.0 * (distance / _max_distance) + 1.0
    return _distance


def point_spread(active_sensor_index, proximity,
                 function_type, sensor_readings_dimensions,
                 point_spread_size, point_spread_sigma):
    """
    calculate point spread function for given distance at active sensor index

    https://en.wikipedia.org/wiki/Point_spread_function

    Parameters
    ----------
    active_sensor_index : int
    proximity : float
    function_type : str
    sensor_readings_dimensions : int
    point_spread_size : float
    point_spread_sigma : float

    Returns
    -------

    """
    if function_type not in ['linear', 'gauss', 'linear_normalized']:
        raise RuntimeError(
            'the point spread function {} is unknown'.format(function_type))

    def linear(x):
        # f(x) = m*x + c
        # m is -1* point_spread_size
        # c is 1.0
        # So 1.0 is the maximum value
        return (-1 * point_spread_size) * abs(x) + 1.0

    def gauss(x):
        # gauss = 1./np.square_root(2*np.pi*sigma**2)
        # * math.exp(-(abs(x)**2)/(2*sigma**2))
        return np.exp(-(np.abs(x) ** 2) / (2 * point_spread_sigma ** 2))

    steps_ = np.arange(-(sensor_readings_dimensions-1)/2,
                       sensor_readings_dimensions/2)
    shift = -(sensor_readings_dimensions-1)//2 + active_sensor_index

    if function_type in ('linear', 'linear_normalized'):
        raise RuntimeError(
            'not tested because it was not explained in the paper')

    elif function_type is 'gauss':
        return gauss(np.roll(steps_, shift))*proximity

    else:
        raise RuntimeError('invalid function type should have been caught')


# TODO rework
def calc_agent_intersection_proximity(my_pos, my_radius,
                                      other_pos, other_radius,
                                      sensor_vector, max_distance):
    """
    # TODO docstring

    http://mathworld.wolfram.com/Circle-LineIntersection.html

    Parameters
    ----------
    my_pos
    my_radius
    other_pos
    other_radius
    sensor_vector
    max_distance

    Returns
    -------

    """
    # The algorithm works for a circle at (0,0)
    # So translate the other_pos to (0,0)
    my_pos_translated = my_pos - other_pos
    # other_pos_translated = other_pos - other_pos

    x_1 = my_pos_translated[0]  # start point of ray
    y_1 = my_pos_translated[1]  # start point of ray
    line_end = my_pos_translated + sensor_vector * 10  # end point of ray
    x_2 = line_end[0]
    y_2 = line_end[1]
    r = other_radius

    # x_2_retranslated = x_2 + my_pos[0]
    # y_2_retranslated = y_2 + my_pos[1]

    d_x = x_2 - x_1
    d_y = y_2 - y_1
    d_r = np.sqrt(d_x ** 2 + d_y ** 2)
    distance_ = x_1 * y_2 - x_2 * y_1

    # fig, ax = plt.subplots()
    # ax.set_xlim(-100, 100)
    # ax.set_ylim(-100, 100)

    # Draw line
    # ax.plot([my_pos[0], x_2_retranslated], [my_pos[1], y_2_retranslated], '-')

    # Draw circle
    # circle = plt.Circle(other_pos, r)
    # ax.add_artist(circle)

    if (r ** 2 * d_r ** 2 - distance_ ** 2) < 0:
        # plt.show()
        return None

    else:
        x_plus = \
            (distance_ * d_y
             + np.sign(d_y) * d_x * np.sqrt(r ** 2 * d_r ** 2 - distance_ ** 2)) / d_r ** 2
        x_minus = \
            (distance_ * d_y
             - np.sign(d_y) * d_x * np.sqrt(r ** 2 * d_r ** 2 - distance_ ** 2)) / d_r ** 2

        y_plus = \
            (-distance_ * d_x
             + abs(d_y) * np.sqrt(r ** 2 * d_r ** 2 - distance_ ** 2)) / d_r ** 2
        y_minus = \
            (-distance_ * d_x
             - abs(d_y) * np.sqrt(r ** 2 * d_r ** 2 - distance_ ** 2)) / d_r ** 2

        intersection1 = np.array([x_plus, y_plus])
        intersection2 = np.array([x_minus, y_minus])

        # determine closest of both intersection points
        dist1 = np.linalg.norm(intersection1 - my_pos_translated)
        dist2 = np.linalg.norm(intersection2 - my_pos_translated)

        if dist1 <= dist2:
            dist = dist1
            intersection = intersection1
        else:
            dist = dist2
            intersection = intersection2

        # prevent to detect intersections in wrong direction
        # if the distance of mypos to one of the intersection points gets bigger
        # when adding a dirs-Vector to mypos, then the intersection is in the wrong direction
        if ((x_2 >= intersection[0] >= x_1) or (
                x_2 <= intersection[0] <= x_1)) and \
                ((y_2 >= intersection[1] >= y_1) or (
                        y_2 <= intersection[1] <= y_1)):

            real_distance = dist - my_radius
            proximity = get_proximity_linear(real_distance, my_radius, max_distance)
            return proximity

        else:
            return None


def calculate_all_border_distances(agent_position,
                                   agent_radius,
                                   sensor_vector,
                                   max_distance,
                                   border_proximity_weight,
                                   borders):
    """
    # TODO docstring

    Parameters
    ----------
    agent_position : (2,) array
    agent_radius : float
    sensor_vector : (2,) array
    max_distance : float
    border_proximity_weight : float
    borders : (4,(2,(2,))) array

    Returns
    -------
    distances : (sensor_readings_dimensions,) array

    """
    distances = []
    for i, border in enumerate(borders):
        distances.append(
            calculate_one_border_distance(
                agent_position,
                agent_radius,
                border,
                sensor_vector,
                max_distance,
                border_proximity_weight
            )
        )
    return distances


def calculate_one_border_distance(my_pos, my_radius,
                                  border_pos, sensor_vector,
                                  max_distance, border_proximity_weight):
    """
    # TODO docstring

    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

    Parameters
    ----------
    my_pos : (2,) array
    my_radius : float
    border_pos : (2,(2,)) array
    sensor_vector : (2,) array
    max_distance : float
    border_proximity_weight : float

    Returns
    -------
    distance : float
        returns -infinity in case of invalid intersection point
        outside environment

    """
    if border_proximity_weight == 0.0:
        return 0.0

    x1 = my_pos[0]
    y1 = my_pos[1]

    line_end = my_pos + sensor_vector * 10  # end point of ray
    x2 = line_end[0]
    y2 = line_end[1]

    # plt.plot([x1, x2], [y1, y2], '-')
    x3 = border_pos[0, 0]
    y3 = border_pos[0, 1]
    x4 = border_pos[1, 0]
    y4 = border_pos[1, 1]

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    p_x = \
        ((x1 * y2 - y1 * x2) * (x3 - x4)
         - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    p_y = \
        ((x1 * y2 - y1 * x2) * (y3 - y4)
         - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    # Check if the intersection is in the direction of the sensor vector
    # If the intersection point is between mypos and L, it's cool
    if ((x2 >= p_x >= x1) or (x2 <= p_x <= x1)) and \
            ((y2 >= p_y >= y1) or (y2 <= p_y <= y1)):

        intersection = np.array([p_x, p_y])
        distance = np.linalg.norm(intersection - my_pos)

        # plt.plot(intersection[0], intersection[1], 'r*')
        # plt.show()

        real_distance = distance - my_radius
        proximity = get_proximity_linear(real_distance, my_radius, max_distance)
        return proximity * border_proximity_weight

    else:
        # plt.show()
        return -np.inf


def calc_sensor_directions(sensor_readings_dimensions):
    """
    # TODO docstring

    Parameters
    ----------
    sensor_readings_dimensions : int
        number of sensor reading directions

    Returns
    -------
    directions : (sensor_readings_dimensions,) array

    """
    directions = []
    for i in range(sensor_readings_dimensions):
        sensor_range_rad = (2 * np.pi) / sensor_readings_dimensions
        angle_rad = (i * sensor_range_rad + (i + 1) * sensor_range_rad) / 2.
        directions.append(np.array([np.cos(angle_rad), np.sin(angle_rad)]))

    return np.asarray(directions)


def get_angle_ray(dirs, i, length, ball_pos):
    """
    # TODO docstring

    Parameters
    ----------
    dirs
    i
    length
    ball_pos

    Returns
    -------

    """
    i %= len(dirs)
    return ball_pos + (length * dirs[i] / np.linalg.norm(dirs[i]))
