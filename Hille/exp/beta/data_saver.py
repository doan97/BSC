import pickle
import os
from environment import Environment
from active_inference_agent import ActiveInferenceAgent as Agent
import config

class Saver(Environment, Agent):
    def __init__(self):
        self.config = config.Config()
        self.conf = self.config.save_config

        self.epoch_step = self.conf.epoch_step
        self.mini_epoch_step = self.conf.mini_epoch_step
        self.id = self.conf.project_name
        self.new = self.conf.new

        self.project_folder = self.conf.project_folder
        self.structure = self.conf.structure

        self.env_path = self.conf.env_path
        self.agent_path = self.conf.agent_path
        self.conf_path = self.conf.conf_path

        if self.conf.reset_all:
            self.delete_all_projects()
        if self.conf.reset:
            self.reset_folder(self.project_folder)
        if self.conf.new:
            self.make_dirs(self.structure)
        if self.conf.save_config:
            self.save_config()



    def make_dirs(self,s, start=''):
        try:
            for folder in s:
                act_path = start + folder
                os.mkdir(act_path)
                if s[folder] is not None:
                    self.make_dirs(s[folder], act_path)
        except:
            print('Something went wrong.')

    def save_config(self):
        path = self.conf.conf_path
        name = self.conf.config_name
        path = path + name
        os.system('cp config.py ' + path)
        print(path)

    def reset_folder(self, path):
        os.system('rm -rf ' + path)

    def delete_all_projects(self):
        os.system('rm -rf ' + self.conf.start_dir + '/project*')

    def save_env(self, env: Environment):
        if self.conf.save_environment:
            name = self.conf.env_name
            tmp = env.gui
            env.gui = None
            os.system('touch ' + self.env_path + name)
            pickle.dump(env, open(self.env_path + name , 'wb'))
            env.gui = tmp

    def save_agent(self, agent: Agent, epoch, mini_epoch):
        if self.conf.save_agents:
            if epoch % self.epoch_step == 0:
                if mini_epoch % self.mini_epoch_step == 0:
                    agent_file = self.agent_path + '/' + agent.name + str(epoch) + '_' + str(mini_epoch)
                    agent.model.save(agent_file)
