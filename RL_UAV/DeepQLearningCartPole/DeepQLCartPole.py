import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- Config ---
TRAIN = False
MODEL_PATH = "dqn_cartpole.pt"
EPISODES = 300
GAMMA = 0.99
LR = 1e-3
EPS_DECAY = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE = 10  # How often (episodes) to sync target net

# --- Environment ---
render_mode = "human" if not TRAIN else None
env = gym.make("CartPole-v1", render_mode=render_mode)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# --- Network ---
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

# --- Setup Networks & Memory ---
policy_net = Net().to(device)
target_net = Net().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()
memory = deque(maxlen=BUFFER_SIZE)



if TRAIN:
    epsilon = 1.0
    for ep in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            # Action Selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                    action = policy_net(s_t).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            memory.append((state, action, reward, next_state, done))

            # Batch Training
            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Batch Tensors (Warning-free)
                b_s = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
                b_a = torch.as_tensor(actions, dtype=torch.long, device=device).view(-1, 1)
                b_r = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(-1, 1)
                b_ns = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device)
                b_d = torch.as_tensor(dones, dtype=torch.float32, device=device).view(-1, 1)

                # Current Q Values (Policy Net)
                current_q = policy_net(b_s).gather(1, b_a)

                # Target Q Values (Target Net)
                with torch.no_grad():
                    max_next_q = target_net(b_ns).max(1)[0].view(-1, 1)
                    target_q = b_r + (GAMMA * max_next_q * (1 - b_d))

                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            total_reward += reward
            if done: break

        # Decay epsilon
        epsilon = max(0.01, epsilon * EPS_DECAY)

        # Sync Target Network
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1:3d} | Reward: {total_reward:4.0f} | Epsilon: {epsilon:.2f}")

    torch.save(policy_net.state_dict(), MODEL_PATH)
    print("Training finished!")

else:
    # --- Visualization ---
    policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy_net.eval()
    for _ in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                s_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                action = policy_net(s_t).argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Visualized Reward: {total_reward}")
env.close()
