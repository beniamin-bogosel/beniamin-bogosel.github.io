import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- Config ---
TRAIN = True
MODEL_PATH = "dqn_mountaincar_final.pt"
EPISODES = 500
GAMMA = 0.99
LR = 1e-3
EPS_DECAY = 0.995

# --- Buffer Hyperparameters ---
BATCH_SIZE = 64
MEMORY_SIZE = 10000

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
            nn.Linear(64, n_actions)
        )
    def forward(self, x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# --- Replay Buffer ---
memory = deque(maxlen=MEMORY_SIZE)

if TRAIN:
    epsilon = 1.0
    for ep in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    # Fixed tensor construction
                    s_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                    action = model(s_t).argmax().item()

            next_state, reward, done, truncated, _ = env.step(action)
            
            # Modified reward for height/velocity
            mod_reward = reward + (abs(next_state[1]) * 10)

            memory.append((state, action, mod_reward, next_state, done))

            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Efficient tensor conversion
                states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
                actions_t = torch.as_tensor(actions, dtype=torch.long, device=device).view(-1, 1)
                rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(-1, 1)
                next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device)
                dones_t = torch.as_tensor(dones, dtype=torch.float32, device=device).view(-1, 1)

                # Current Q values
                current_q = model(states_t).gather(1, actions_t)

                # Target Q values (fixed target construction)
                with torch.no_grad():
                    max_next_q = model(next_states_t).max(1)[0].view(-1, 1)
                    target_q = rewards_t + (GAMMA * max_next_q * (1 - dones_t))

                # Update (using detach().clone() logic internally via the target_q calculation)
                loss = loss_fn(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            total_reward += reward
            if done or truncated: break

        epsilon = max(0.01, epsilon * EPS_DECAY)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1:3d} | Reward: {total_reward:4.0f} | Epsilon: {epsilon:.2f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Training complete!")

else:
    # --- Visualization ---
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    for _ in range(5):
        state, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                s_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                action = model(s_t).argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
env.close()
