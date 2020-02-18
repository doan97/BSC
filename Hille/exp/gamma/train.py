import logging

import util


def run(config_file):
    """
    TODO docstring

    Parameters
    ----------
    config_file

    Returns
    -------

    """
    file_name, load, mode, config, seed = util.initialize(config_file)

    logging.info('started run')

    start_epoch, end_epoch, environment, agents, number_of_mini_epochs, number_of_time_steps = \
        util.get_setup(file_name, load, mode, config, seed)

    for epoch in range(start_epoch, start_epoch + end_epoch):
        run_epoch(environment, agents, epoch, number_of_mini_epochs, number_of_time_steps)

    util.save(file_name, start_epoch + end_epoch, environment, agents)

    logging.info('ended run')


def run_epoch(environment, agents, epoch, number_of_mini_epochs, number_of_time_steps):
    """change physic mode, perform one epoch, learn the model after each min epoch"""
    environment.change_physic_mode('Rocket')

    old_state = environment.reset(agents.reset())

    for mini_epoch in range(number_of_mini_epochs):
        old_state = run_mini_epoch(environment, agents, old_state,
                                   epoch,
                                   number_of_mini_epochs, mini_epoch,
                                   number_of_time_steps)
        agents.learn()


def run_mini_epoch(environment, agents, old_state,
                   epoch,
                   number_of_mini_epochs, mini_epoch,
                   number_of_time_steps):
    """
    perform one mini epoch with new scenario
    save transitions
    visualize current environment step

    Parameters
    ----------
    agents : < Class : Handler >
    environment : < Class : Environment >
    old_state : dict
        dictionary containing
            dictionary containing old position, old acceleration and old sensor readings
            for every agent that can be controlled
    epoch : int
        current epoch
    number_of_mini_epochs : int
    mini_epoch : int
        current mini epoch
    number_of_time_steps : int

    Returns
    -------
    old_state : dict
        dictionary containing
            dictionary containing new position, new acceleration and new sensor readings
            for every agent that can be controlled

    """
    time_step = 0

    while True:
        actions = agents.act(old_state, environment.get_physic_mode_id(), environment.numThrusts)

        new_state = environment.step(actions)

        total_step = (epoch + 1) * mini_epoch * number_of_time_steps + time_step

        agents.update(total_step, old_state, actions)

        old_state = new_state

        time_step += 1

        environment.render(sleep=0.1)

        if time_step % number_of_time_steps == 0:
            print('finished epoch {} - {}/{} (total steps: {})'
                  .format(epoch + 1, mini_epoch + 1, number_of_mini_epochs, total_step))
            break

    return old_state
