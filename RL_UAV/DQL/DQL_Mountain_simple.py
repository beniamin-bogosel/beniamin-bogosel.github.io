import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- Config ---
TRAIN = False  # Switch to False to watch the trained agent
MODEL_PATH = "dqn_mountaincar.pt"
EPISODES = 500
GAMMA = 0.99
LR = 1e-3
EPS_DECAY = 0.995

# --- Environment ---
render_mode = "human" if not TRAIN else None
env = gym.make("MountainCar-v0", render_mode=render_mode)
n_states = env.observation_space.shape[0] # (position, velocity)
n_actions = env.action_space.n

# --- Simple Network ---
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

if TRAIN:
    epsilon = 1.0
    for ep in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            # Epsilon-greedy selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = model(torch.FloatTensor(state).to(device))
                    action = q_vals.argmax().item()

            next_state, reward, done, truncated, _ = env.step(action)
            
            # --- Reward Engineering (Essential for Mountain Car) ---
            # Standard reward is -1. We add a bonus for moving high up the hills.
            modified_reward = reward + (abs(next_state[1]) * 10)

            # --- Compute Target ---
            with torch.no_grad():
                next_q = model(torch.FloatTensor(next_state).to(device)).max()
                target = modified_reward + GAMMA * next_q * (0 if done else 1)

            # --- Update Model ---
            current_q = model(torch.FloatTensor(state).to(device))[action]
            loss = loss_fn(current_q, torch.tensor(target))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += modified_reward
            if done or truncated: break

        epsilon = max(0.01, epsilon * EPS_DECAY)
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}, Score: {total_reward}, Epsilon: {epsilon:.2f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Training complete and model saved!")

else:
    # --- Visualization ---
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    for _ in range(5):
        state, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = model(torch.FloatTensor(state).to(device)).argmax().item()
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
env.close()
