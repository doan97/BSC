import numpy as np

# TODO write docstrings


class Target:
    def __init__(self, position, stop_iteration):
        self.position = position
        self.stop_iteration = stop_iteration

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError('create target subclass')


class ConstantTarget(Target):
    def __init__(self, position, stop_iteration):
        super().__init__(position, stop_iteration)
        self.iteration = 0

    def __next__(self):
        if self.iteration == self.stop_iteration:
            raise StopIteration
        else:
            self.iteration += 1
            return self.position


class RandomTarget(Target):
    def __init__(self, position, stop_iteration,
                 refresh_iteration, low, high):
        super().__init__(position, stop_iteration)
        self.refresh_iteration = refresh_iteration
        self.low = low
        self.high = high
        self.iteration = 0

    def __next__(self):
        if self.iteration == self.stop_iteration:
            raise StopIteration
        else:
            if self.iteration % self.refresh_iteration == 0:
                self.position = np.random.uniform(self.low, self.high)
            self.iteration += 1
            return self.position


class OtherTarget(Target):
    def __init__(self, position, stop_iteration, other):
        super().__init__(position, stop_iteration)
        self.other = other
        self.iteration = 0

    def __next__(self):
        if self.iteration == self.stop_iteration:
            raise StopIteration
        else:
            self.iteration += 1
            self.position = self.other['target']
            return self.position


class AgentTarget(Target):
    def __init__(self, position, stop_iteration, other):
        super().__init__(position, stop_iteration)
        self.other = other
        self.iteration = 0

    def __next__(self):
        if self.iteration == self.stop_iteration:
            raise StopIteration
        else:
            self.iteration += 1
            self.position = self.other['position']
            return self.position


class LineTarget(Target):
    def __init__(self, position, stop_iteration, refresh_iteration, low, high):
        super().__init__(position, stop_iteration)
        self.refresh_iteration = refresh_iteration
        self.low = low
        self.high = high
        self.iteration = 0
        self.point_a = None
        self.point_b = None

    def __next__(self):
        if self.iteration == self.stop_iteration:
            raise StopIteration
        else:
            if self.iteration % self.refresh_iteration == 0:
                self.point_a = np.random.uniform(self.low, self.high)
                self.point_b = np.random.uniform(self.low, self.high)
            scale = (self.iteration % self.refresh_iteration) / self.refresh_iteration
            self.position = scale * self.point_a + (1 - scale) * self.point_b
            self.iteration += 1
            return self.position


class ProximityTarget(Target):
    def __init__(self, position, stop_iteration, refresh_iteration, low, high, proximity):
        super().__init__(position, stop_iteration)
        self.refresh_iteration = refresh_iteration
        self.low = low
        self.high = high
        self.proximity = proximity
        
    def __next__(self):
        raise NotImplementedError('sensor loss is not directly connected to proximity target')
