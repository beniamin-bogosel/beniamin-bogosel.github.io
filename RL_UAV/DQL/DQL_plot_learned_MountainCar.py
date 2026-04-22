import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import matplotlib.pyplot as plt

def plot_policy(model, device):
    # Define the range of the state space
    # Position: -1.2 to 0.6, Velocity: -0.07 to 0.07
    positions = np.linspace(-1.2, 0.6, 100)
    velocities = np.linspace(-0.07, 0.07, 100)
    
    # Create a grid
    px, vy = np.meshgrid(positions, velocities)
    results = np.zeros(px.shape)

    model.eval()
    with torch.no_grad():
        for i in range(len(positions)):
            for j in range(len(velocities)):
                state = np.array([positions[i], velocities[j]])
                s_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                action = model(s_t).argmax().item()
                results[j, i] = action  # j is velocity (y-axis), i is position (x-axis)

    # Plotting
    plt.figure(figsize=(10, 6))
    # 0: Left, 1: None, 2: Right
    contour = plt.contourf(px, vy, results, levels=[-0.5, 0.5, 1.5, 2.5], 
                           colors=['#e74c3c', '#bdc3c7', '#3498db'])
    
    cbar = plt.colorbar(contour, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Left (0)', 'None (1)', 'Right (2)'])
    
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Agent Policy Map (Decision Boundaries)')
    plt.grid(alpha=0.3)
    plt.show()

# To use it:
MODEL_PATH = "dqn_mountaincar_final.pt"
EPISODES = 500
GAMMA = 0.99
LR = 1e-3
EPS_DECAY = 0.995

# --- Buffer Hyperparameters ---
BATCH_SIZE = 64
MEMORY_SIZE = 10000

# --- Environment ---
render_mode = "human"
env = gym.make("MountainCar-v0", render_mode=render_mode)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))


plot_policy(model, device)
