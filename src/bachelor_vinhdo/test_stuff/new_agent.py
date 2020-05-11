import bachelor_vinhdo.global_config as c
from bachelor_vinhdo.Net import Net
class Agent:
    def __init__(self, init_pos, target_pos, seek_proximity=False, input_type='all', obstacle=False, radius=0.06):
        self.obstacle = obstacle
        if not self.obstacle:
            self.input_type = input_type
            self.seek = seek_proximity
            self.target = target_pos
            if self.input_type == 'all':
                self.net = Net(c.INPUT_DIM, c.HIDDEN_DIM, c.OUTPUT_DIM)
            elif self.input_type == 'motor and sensor':
                self.net = Net(c.INPUT_DIM - 4, c.HIDDEN_DIM, c.OUTPUT_DIM)
            elif self.input_type == 'motor only':
                self.net = Net(4, c.HIDDEN_DIM, c.OUTPUT_DIM)

        self.position = init_pos
        self.radius = radius




