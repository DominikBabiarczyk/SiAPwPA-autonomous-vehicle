import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.vec1_net = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.vec2_net = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.vec3_net = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.pos_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        self.yaw_net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU()
        )
        
        self.combined_net = nn.Sequential(
            nn.Linear(32*3 + 16 + 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        

        self.velocity_head = nn.Linear(64, 11)
        self.steering_head = nn.Linear(64, 9)

        
    def forward(self, vec1, vec2, vec3, position, ywa):
        v1 = self.vec1_net(vec1)
        v2 = self.vec2_net(vec2)
        v3 = self.vec3_net(vec3)
        pos = self.pos_net(position)
        y = self.yaw_net(ywa)

        combined = torch.cat([v1, v2, v3, pos, y], dim=-1)
        features = self.combined_net(combined)

        velocity = self.velocity_head(features)
        steering = self.steering_head(features)

        return velocity, steering
