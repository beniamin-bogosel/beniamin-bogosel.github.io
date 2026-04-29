import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- Config ---
TRAIN = False
MODEL_PATH = "dqn_lunar_engineered.pt"
EPISODES = 400
GAMMA = 0.99
LR = 5e-4
EPS_DECAY = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 50000
TARGET_UPDATE = 10

env = gym.make("LunarLander-v3", render_mode="human" if not TRAIN else None)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = Net().to(device)
target_net = Net().to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()
memory = deque(maxlen=BUFFER_SIZE)

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
                    action = policy_net(torch.as_tensor(state, dtype=torch.float32, device=device)).argmax().item()

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            # --- REWARD ENGINEERING ---
            # state: [pos_x, pos_y, vel_x, vel_y, angle, ang_vel, leg_l, leg_r]
            x, y, vx, vy, ang, ang_v, l_leg, r_leg = next_state

            # 1. Distance penalty (Stay near x=0, y=0)
            dist_penalty = - (abs(x) + abs(y))
            
            # 2. Velocity penalty (Don't fly too fast, especially near the ground)
            vel_penalty = - (abs(vx) + abs(vy))
            
            # 3. Orientation penalty (Stay upright)
            angle_penalty = - abs(ang)
            
            # 4. Anti-Flying Away: Heavy penalty if it goes too high (y > 1.0)
            height_penalty = -10 if y > 0.8 else 0

            # 5. Leg bonus: Encourage touching the ground
            leg_bonus = 10 if (l_leg or r_leg) else 0

            mod_reward = reward + dist_penalty + vel_penalty + angle_penalty + height_penalty + leg_bonus

            memory.append((state, action, mod_reward, next_state, done))

            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                s, a, r, ns, d = zip(*batch)
                
                b_s = torch.as_tensor(np.array(s), dtype=torch.float32, device=device)
                b_a = torch.as_tensor(a, dtype=torch.long, device=device).view(-1, 1)
                b_r = torch.as_tensor(r, dtype=torch.float32, device=device).view(-1, 1)
                b_ns = torch.as_tensor(np.array(ns), dtype=torch.float32, device=device)
                b_d = torch.as_tensor(d, dtype=torch.float32, device=device).view(-1, 1)

                current_q = policy_net(b_s).gather(1, b_a)
                with torch.no_grad():
                    max_next_q = target_net(b_ns).max(1)[0].view(-1, 1)
                    target_q = b_r + (GAMMA * max_next_q * (1 - b_d))

                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            state = next_state
            total_reward += reward 
            if done: break

        epsilon = max(0.01, epsilon * EPS_DECAY)
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1:3d} | Reward: {total_reward:4.0f} | Pos Y: {state[1]:.2f}")
            torch.save(policy_net.state_dict(), MODEL_PATH)
            print("Parameters saved!")
                
                
    torch.save(policy_net.state_dict(), MODEL_PATH)
else:
    policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy_net.eval()
    for _ in range(5):
        state, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = policy_net(torch.as_tensor(state, dtype=torch.float32, device=device)).argmax().item()
            state, _, term, trunc, _ = env.step(action)
            done = term or trunc
env.close()
