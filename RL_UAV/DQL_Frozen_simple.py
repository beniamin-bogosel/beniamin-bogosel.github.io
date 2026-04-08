import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time


custom_map = [
            "SFFFF",
            "FHFHF",
            "FFFHF",
            "HFFGF",
            "FFFFF"
        ]


# --- Config ---
TRAIN = False          # <-- switch to False for viewing
MODEL_PATH = "dqn_frozenlake.pt"

EPISODES = 2000
MAX_STEPS = 100
GAMMA = 0.99
LR = 1e-3

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

# --- Environment ---
render_mode = "human" if not TRAIN else None
env = gym.make("FrozenLake-v1",desc=custom_map, is_slippery=False, render_mode=render_mode)

n_states = env.observation_space.n
n_actions = env.action_space.n

# --- One-hot encoding ---
def one_hot(state):
    v = np.zeros(n_states)
    v[state] = 1.0
    return v

# --- Network ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# =========================
# TRAINING
# =========================
if TRAIN:
    epsilon = EPS_START

    for ep in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            # --- epsilon-greedy ---
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.FloatTensor(one_hot(state)).to(device)
                    q_vals = model(s)
                    action = torch.argmax(q_vals).item()

            next_state, reward, done, truncated, _ = env.step(action)

            # --- compute target ---
            with torch.no_grad():
                s_next = torch.FloatTensor(one_hot(next_state)).to(device)
                max_next_q = torch.max(model(s_next))
                target = reward + GAMMA * max_next_q * (0 if done else 1)

            # --- current Q ---
            s = torch.FloatTensor(one_hot(state)).to(device)
            q_vals = model(s)
            target_vec = q_vals.clone().detach()
            target_vec[action] = target

            # --- update ---
            loss = loss_fn(q_vals, target_vec)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}, reward={total_reward}, epsilon={epsilon:.3f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved!")

# =========================
# VIEWING (play learned policy)
# =========================
else:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    for ep in range(10):
        state, _ = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            s = torch.FloatTensor(one_hot(state)).to(device)

            with torch.no_grad():
                q_vals = model(s)
                action = torch.argmax(q_vals).item()

            next_state, reward, done, truncated, _ = env.step(action)
            time.sleep(0.3)

            state = next_state
            total_reward += reward

            if done or truncated:
                print(f"Episode {ep+1} reward: {total_reward}")
                break

env.close()
