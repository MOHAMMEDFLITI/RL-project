import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from env import SumoLaneChangeEnv
import os

def train_deep_rl(timesteps=10000):
    # Create environment
    env = SumoLaneChangeEnv(gui=False)
    
    # Define model
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./log/deep_rl", learning_rate=1e-3, buffer_size=50000, learning_starts=1000, batch_size=32, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=1000, exploration_fraction=0.1, exploration_final_eps=0.02)
    
    # Train
    model.learn(total_timesteps=timesteps, log_interval=4)
    
    # Save
    model.save("dqn_lane_change")
    print("DQN model saved to dqn_lane_change.zip")
    
    env.close()

if __name__ == "__main__":
    train_deep_rl(timesteps=10000)
