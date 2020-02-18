import os
import pickle
from abc import ABC

import config
import main
import gui
from active_inference_agent import ActiveInferenceAgent as Agent
from agents import Agents
from environment import Environment
from train import Training


class Loader(Environment, Agents, Agent, ABC):
    def __init__(self):
        #super().__init__()
        self.s_conf = config.SaveConfig()
        self.config = config.Config()
        self.load = self.s_conf.load_from_data
        self.path = self.s_conf.project_folder

        self.env = None
        self.gui = None
        self.training = None
        self.agents = None

    def get_environment(self):
        env: Environment = pickle.load(open(self.s_conf.env_path_load + self.s_conf.env_name, 'rb'))
        return env


    def get_agents(self):
        agents = []
        for agent_config in self.config.agents_config.agents:
            agent: Agent = Agent(agent_config,
                                 self.config.inference_config,
                                 self.config.model_config,
                                 self.config.train_config.max_memory_size,
                                 self.config.train_config.num_time_steps,
                                 self.config.env_config.transition_data_type)
            name = agent_config.name
            name = self.get_right_agent(self.s_conf.agent_path_load, name)

            agent.model.load(self.s_conf.agent_path_load + '/' + name)
            agents.append(agent)
        return agents

    def get_right_agent(self, path, starts_with):
        for (dir_path, dir_names, file_names) in os.walk(path):
            print(dir_path, file_names)
            if dir_path == self.s_conf.agent_path_load:
                for filename in file_names:
                    if str(filename).startswith(starts_with):
                        return filename

    def create_agents_object(self):
        agents_list = self.get_agents()
        agents: Agents = Agents(self.config.agents_config,
                                self.config.inference_config,
                                self.config.model_config,
                                self.config.train_config.max_memory_size,
                                self.config.train_config.num_time_steps,
                                self.config.env_config.transition_data_type)
        agents.agents = agents_list
        return agents

    def create_environment_object(self):
        env: Environment = self.get_environment()
        env_config = self.config.env_config
        gui_config = self.config.gui_config
        env.gui = gui.GUI(gui_config,
                          env_config.size,
                          env_config.sensor_readings_dimensions)
        return env

    def create_train_object(self):
        env: Environment = self.create_environment_object()
        agents: Agents = self.create_agents_object()

        training = Training(self.config)
        training.agents = agents
        training.env = env

        return training

    def create_main_object(self):
        env: Environment = self.create_environment_object()
        agents: Agents = self.create_agents_object()
        main_ = main.Main(self.config)
        main_.agents = agents
        main_.env = env
        return main_


if __name__ == "__main__":
    loader = Loader()
    print(loader.create_train_object())
    pass
