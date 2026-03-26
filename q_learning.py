import numpy as np
import pickle
import os
from env import SumoLaneChangeEnv
from torch.utils.tensorboard import SummaryWriter

def discretize_state(state, bins):
    # state: [lane, speed, dist_0, dist_1]
    # bins: list of bins for each dimension
    
    lane = int(state[0])
    
    speed_idx = np.digitize(state[1], bins[1]) - 1
    speed_idx = max(0, min(speed_idx, len(bins[1]) - 2))
    
    dist0_idx = np.digitize(state[2], bins[2]) - 1
    dist0_idx = max(0, min(dist0_idx, len(bins[2]) - 2))
    
    dist1_idx = np.digitize(state[3], bins[3]) - 1
    dist1_idx = max(0, min(dist1_idx, len(bins[3]) - 2))
    
    return (lane, speed_idx, dist0_idx, dist1_idx)

def train_q_learning(episodes=100, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    env = SumoLaneChangeEnv(gui=False)
    writer = SummaryWriter(log_dir="./log/q_learning")
    
    # Define bins for discretization
    # Speed: 0-50, 10 bins
    speed_bins = np.linspace(0, 50, 10)
    # Dist: 0-200, 10 bins
    dist_bins = np.linspace(0, 200, 10)
    
    bins = [
        None, # Lane is already discrete 0, 1
        speed_bins,
        dist_bins,
        dist_bins
    ]
    
    # Q-Table dimensions
    # Lane: 2
    # Speed: 9 (10 bins -> 9 intervals + outliers, but we clamped) -> len(bins)-1
    # Dist: 9
    # Dist: 9
    # Actions: 3
    
    q_table = np.zeros((2, 9, 9, 9, 3))
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretize_state(obs, bins)
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
                
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = discretize_state(next_obs, bins)
            
            # Update Q-Value
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            q_table[state][action] += alpha * (td_target - q_table[state][action])
            
            state = next_state
            total_reward += reward
            
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        writer.add_scalar("Reward/Episode", total_reward, episode)
        print(f"Episode {episode}: Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
        
    env.close()
    writer.close()
    
    # Save Q-Table
    np.save("q_table.npy", q_table)
    print("Q-Table saved to q_table.npy")

if __name__ == "__main__":
    train_q_learning(episodes=50)
