# Dueling Deep Q-Network (Dueling DQN)

Dueling DQN is an architectural advancement in deep reinforcement learning, introduced by Wang et al. (2016), that enhances policy evaluation by decoupling the estimation of state value and action-specific advantage.

## The Core Concept
In standard Deep Q-Networks (DQN), the model directly estimates the state-action value $Q(s, a)$—the expected return for taking action $a$ in state $s$. However, in many environments, the specific action chosen in certain states may not significantly impact the outcome. 

Dueling DQN addresses this by splitting the neural network's final layers into two separate "streams":

*   **State Value Stream ($V(s)$):** This stream outputs a single scalar value representing the inherent value of being in state $s$, regardless of the action taken.
*   **Action Advantage Stream ($A(s, a)$):** This stream outputs a vector of values, one for each possible action, representing the relative "advantage" or extra benefit of taking a specific action over others in that state.

## The Aggregation Layer
To reconstruct the final $Q$-values used for action selection and training, these two streams must be combined. A simple addition ($Q = V + A$) leads to **unidentifiability**, meaning given $Q$, we cannot uniquely recover $V$ and $A$ because a constant could be added to one and subtracted from the other without changing the result.

To solve this, Dueling DQN uses a specialized aggregation layer that subtracts the mean of the advantages:

$$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)$$

where $|\mathcal{A}|$ is the number of available actions. This mean-subtraction ensures the advantage estimates are centered, making the value of $V(s)$ identifiable and stable during optimization.

## Key Benefits
*   **Faster Convergence:** The architecture generalizes more effectively across actions, especially in environments with many similar-valued or redundant actions.
*   **Improved Learning Efficiency:** By learning state values independently, the network can recognize valuable states without needing to learn the effect of every individual action in those states first.
*   **Robust Generalization:** It prevents spurious gradient propagation in redundant state-action regimes, allowing the model to focus on the underlying state distribution.

---

## Python Implementation (CartPole-v1)

This complete script implements the Dueling architecture alongside a Double Network setup (Target and Policy nets) and an Experience Replay buffer for maximum stability.
```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- Config ---
TRAIN = True
MODEL_PATH = "dueling_dqn_cartpole.pt"
EPISODES = 300
GAMMA = 0.99
LR = 1e-3
EPS_DECAY = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE = 10 

# --- Environment ---
render_mode = "human" if not TRAIN else None
env = gym.make("CartPole-v1", render_mode=render_mode)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# --- Dueling Network Architecture ---
class DuelingNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Common Feature Extractor
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU()
        )
        
        # Value Stream (V) - Estimates the value of the state itself
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream (A) - Estimates the relative advantage of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        # Handle 1D inputs (single states) by adding a batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Aggregation Layer: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Setup Networks, Optimizer, and Memory ---
policy_net = DuelingNet(n_states, n_actions).to(device)
target_net = DuelingNet(n_states, n_actions).to(device)
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

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            memory.append((state, action, reward, next_state, done))

            # Batch Training
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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            total_reward += reward
            if done: break

        epsilon = max(0.01, epsilon * EPS_DECAY)
        
        if (ep + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1:3d} | Reward: {total_reward:4.0f} | Epsilon: {epsilon:.2f}")

    torch.save(policy_net.state_dict(), MODEL_PATH)
    print("Training finished and model saved!")

else:
    # --- Visualization / Evaluation ---
    policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy_net.eval()
    for _ in range(5):
        state, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                s_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                action = policy_net(s_t).argmax().item()
            state, _, term, trunc, _ = env.step(action)
            done = term or trunc
env.close()
```