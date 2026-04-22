import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# --- Config ---
TRAIN = False
MODEL_PATH = "dqn_mountaincar_pro.pt"
EPISODES = 500
GAMMA = 0.99
LR = 1e-3
EPS_DECAY = 0.995

# --- DQN Hyperparameters ---
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 5 # Sync every 5 episodes

# --- Environment ---
render_mode = "human" if not TRAIN else None
env = gym.make("MountainCar-v0", render_mode=render_mode)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    def forward(self, x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize Two Networks ---
policy_net = Net().to(device)
target_net = Net().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()
memory = deque(maxlen=MEMORY_SIZE)

if TRAIN:
    epsilon = 1.0
    for ep in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            # Action Selection (using Policy Net)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                    action = policy_net(s_t).argmax().item()

            next_state, reward, done, truncated, _ = env.step(action)
            
            # Reward Engineering
            mod_reward = reward + (abs(next_state[1]) * 10)
            memory.append((state, action, mod_reward, next_state, done))

            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Fixed Tensor Construction (Warning-free)
                states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
                actions_t = torch.as_tensor(actions, dtype=torch.long, device=device).view(-1, 1)
                rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(-1, 1)
                next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device)
                dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).view(-1, 1)

                # Current Q values (Policy Net)
                current_q = policy_net(states_t).gather(1, actions_t)

                # Target Q values (Target Net)
                with torch.no_grad():
                    max_next_q = target_net(next_states_t).max(1)[0].view(-1, 1)
                    y_target = rewards_t + (GAMMA * max_next_q * (1 - dones_t))

                loss = loss_fn(current_q, y_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            total_reward += reward
            if done or truncated: break

        epsilon = max(0.01, epsilon * EPS_DECAY)

        # Sync Networks
        if (ep + 1) % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1:3d} | Reward: {total_reward:4.0f} | Epsilon: {epsilon:.2f}")

    torch.save(policy_net.state_dict(), MODEL_PATH)
    print("Training complete!")

# --- Visualization & Policy Map ---
else:
    policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy_net.eval()
    
    # 1. Run a few visual episodes
    for _ in range(3):
        state, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                s_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                action = policy_net(s_t).argmax().item()
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()

    # 2. Plot the Policy Map
    positions = np.linspace(-1.2, 0.6, 100)
    velocities = np.linspace(-0.07, 0.07, 100)
    px, vy = np.meshgrid(positions, velocities)
    results = np.zeros(px.shape)

    for i in range(len(positions)):
        for j in range(len(velocities)):
            state = np.array([positions[i], velocities[j]])
            with torch.no_grad():
                s_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                results[j, i] = policy_net(s_t).argmax().item()

    plt.figure(figsize=(8, 5))
    contour = plt.contourf(px, vy, results, levels=[-0.5, 0.5, 1.5, 2.5], colors=['#e74c3c', '#bdc3c7', '#3498db'])
    plt.title("Policy Decision Map (Left/None/Right)")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    cbar = plt.colorbar(contour, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Left (0)', 'None (1)', 'Right (2)'])
    plt.show()
