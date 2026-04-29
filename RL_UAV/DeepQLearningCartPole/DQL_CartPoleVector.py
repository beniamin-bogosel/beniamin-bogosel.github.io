import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- Config ---
TRAIN = True
MODEL_PATH = "dqn_cartpole_vector.pt"
NUM_ENVS = 8          # Number of parallel environments
EPISODES = 200        # Fewer episodes needed because each "step" is actually 8 steps
GAMMA = 0.99
LR = 1e-3
EPS_DECAY = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 20000
TARGET_UPDATE = 10 

# --- Environment Setup ---
if TRAIN:
    # SyncVectorEnv runs environments sequentially but handles them as a single batch
    envs = gym.vector.make("CartPole-v1", num_envs=NUM_ENVS)
else:
    envs = gym.make("CartPole-v1", render_mode="human")

n_states = envs.single_observation_space.shape[0]
n_actions = envs.single_action_space.n

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
policy_net = Net().to(device)
target_net = Net().to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()
memory = deque(maxlen=BUFFER_SIZE)

if TRAIN:
    epsilon = 1.0
    states, _ = envs.reset()
    
    # We track the scores of finished episodes in a list
    episode_rewards = deque(maxlen=20)
    current_rewards = np.zeros(NUM_ENVS)

    for step_count in range(10000): # Total steps across all envs
        # 1. Action Selection (Vectorized)
        if np.random.rand() < epsilon:
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                s_t = torch.as_tensor(states, dtype=torch.float32, device=device)
                actions = policy_net(s_t).argmax(dim=1).cpu().numpy()

        # 2. Step all environments
        next_states, rewards, terms, truncs, infos = envs.step(actions)
        dones = terms | truncs
        
        # 3. Store transitions for every environment
        for i in range(NUM_ENVS):
            # In vectorized envs, when an episode ends, next_states[i] is already 
            # the reset state of the NEW episode. 
            # The actual terminal state is stored in infos["final_observation"][i].
            if dones[i]:
                real_next_state = infos["final_observation"][i]
                episode_rewards.append(current_rewards[i])
                current_rewards[i] = 0
            else:
                real_next_state = next_states[i]
                current_rewards[i] += rewards[i]
            
            memory.append((states[i], actions[i], rewards[i], real_next_state, dones[i]))

        # 4. Training (same as before)
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

        states = next_states
        epsilon = max(0.01, epsilon * EPS_DECAY)

        if step_count % 100 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            avg_score = np.mean(episode_rewards) if episode_rewards else 0
            print(f"Steps: {step_count*NUM_ENVS:5d} | Avg Score (Last 20): {avg_score:.1f} | Epsilon: {epsilon:.2f}")
            if avg_score > 450: break # Consider it solved

    torch.save(policy_net.state_dict(), MODEL_PATH)
    print("Training complete!")

else:
    # --- Visualization (Single Env) ---
    policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy_net.eval()
    for _ in range(5):
        state, _ = envs.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = policy_net(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).argmax().item()
            state, _, term, trunc, _ = envs.step(action)
            done = term or trunc
