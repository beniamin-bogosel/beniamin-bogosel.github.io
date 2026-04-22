import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- Config ---
TRAIN = True
MODEL_PATH = "dqn_mountaincar_dual.pt"
EPISODES = 500
GAMMA = 0.99
LR = 1e-3
EPS_DECAY = 0.995
TARGET_UPDATE_FREQ = 10 

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

policy_net = Net().to(device)
target_net = Net().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()

if TRAIN:
    epsilon = 1.0
    for ep in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0  # Initialize total reward for the episode
        
        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(torch.FloatTensor(state).to(device)).argmax().item()

            next_state, reward, done, truncated, _ = env.step(action)
            
            # Simple reward boost for velocity to help learning
            mod_reward = reward + (abs(next_state[1]) * 10)

            with torch.no_grad():
                next_q = target_net(torch.FloatTensor(next_state).to(device)).max()
                y_target = mod_reward + GAMMA * next_q * (0 if done else 1)

            current_q = policy_net(torch.FloatTensor(state).to(device))[action]
            loss = loss_fn(current_q, torch.tensor(y_target))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward  # Accumulate the actual environment reward
            
            if done or truncated: break

        epsilon = max(0.01, epsilon * EPS_DECAY)

        if ep % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Updated print statement with total_reward
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1:3d} | Reward: {total_reward:4.0f} | Epsilon: {epsilon:.2f}")

    torch.save(policy_net.state_dict(), MODEL_PATH)
    print("Training complete and model saved!")
