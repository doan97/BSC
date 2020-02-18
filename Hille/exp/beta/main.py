import time

import agents
import environment
import data_saver
import config as conf


class Main:
    def __init__(self, config):
        """
        train the model on the given environment

        Parameters
        ----------
        config : Config
            includes all parameters for one experiment

        """
        self.saver = data_saver.Saver()

        self.num_epochs = config.train_config.num_epochs
        self.num_mini_epochs = config.train_config.num_mini_epochs
        self.num_time_steps = config.train_config.num_time_steps

        self.agents = agents.Agents(config.agents_config,
                                    config.inference_config,
                                    config.model_config,
                                    config.train_config.max_memory_size,
                                    config.train_config.num_time_steps,
                                    config.env_config.transition_data_type)

        self.env = environment.Environment(config.env_config, config.gui_config, seed=config.seed)

    def run(self, start_epoch=0):
        """perform the specified number of epochs"""
        self.saver.save_env(self.env)
        self.agents.reset_dataset()
        for epoch in range(start_epoch, self.num_epochs):
            self.agents.save(epoch, 0)
            self.run_epoch(epoch)
        time.sleep(2)

    def run_epoch(self, epoch):
        """change physic mode, perform one epoch, learn the model after each min epoch"""
        self.env.change_physic_mode('Rocket')

        old_state = self.env.reset(self.agents.reset())

        for mini_epoch in range(self.num_mini_epochs):
            old_state = self.run_mini_epoch(epoch, mini_epoch, old_state)
            self.agents.learn()

    def run_mini_epoch(self, epoch, mini_epoch, old_state, infer_action=True):
        """
        perform one mini epoch with new scenario
        save transitions
        visualize current environment step

        Parameters
        ----------
        epoch : int
            current epoch
        mini_epoch : int
            current mini epoch
        old_state : dict
            dictionary containing
                dictionary containing old position, old acceleration and old sensor readings
                for every agent that can be controlled
        infer_action : bool
            whether to train or to infer actions

        Returns
        -------
        old_state : dict
            dictionary containing
                dictionary containing new position, new acceleration and new sensor readings
                for every agent that can be controlled

        """
        time_step = 0
        self.env.change_obstacle_transformation('static')

        # print('state', old_state)

        while True:
            if infer_action:
                actions = self.agents.get_inference_action(old_state,
                                                           self.env.get_physic_mode_id(),
                                                           self.env.numThrusts)
            else:
                actions = self.agents.get_train_action(old_state,
                                                       self.env.get_physic_mode_id(),
                                                       self.env.numThrusts)

            # print('actions', actions)

            new_state = self.env.step(actions)

            self.agents.update((epoch + 1) * mini_epoch * self.num_time_steps + time_step,
                               old_state, actions, new_state)
            # self.agents.append_transition(old_state, actions, new_state)
            old_state = new_state

            # print('state', old_state)

            time_step += 1
            self.env.render()
            # time.sleep(0.1)

            if time_step % self.num_time_steps == 0:
                print('finished {}/{} - {}/{}'.format(epoch + 1,
                                                      self.num_epochs,
                                                      mini_epoch + 1,
                                                      self.num_mini_epochs))
                break

        return old_state


def run_main(config):
    if config.save_config.load_from_data:
        import data_loader
        d = data_loader.Loader()
        train = d.create_main_object()
        if config.save_config.continue_from_epoch:
            continue_epoch = config.save_config.continue_epoch
            train.run(start_epoch=continue_epoch)
        else:
            train.run()
    else:
        Main(config).run()


if __name__ == '__main__':
    c = conf.Config()
    run_main(c)
