
import numpy as np

import torch

import gym


from gym import envs
envids = [spec.id for spec in envs.registry.all()]
for envid in sorted(envids):
    print(envid)


class Empowerment:
    def __init__(self, envname):
        self.env = gym.make(envname)

    def run(self):
        pass


if __name__ == '__main__':
    e = Empowerment('CartPole-v1')
    e.run()
