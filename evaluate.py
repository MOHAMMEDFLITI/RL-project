import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from env import SumoLaneChangeEnv
from q_learning import discretize_state
import traci

def evaluate_model(model_type, model_path, episodes=10):
    env = SumoLaneChangeEnv(gui=False)
    
    # Load model
    if model_type == "q_learning":
        q_table = np.load(model_path)
        # Bins same as training
        speed_bins = np.linspace(0, 50, 10)
        dist_bins = np.linspace(0, 200, 10)
        bins = [None, speed_bins, dist_bins, dist_bins]
    elif model_type == "dqn":
        model = DQN.load(model_path)
    
    results = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated):
            if model_type == "q_learning":
                state = discretize_state(obs, bins)
                action = np.argmax(q_table[state])
            elif model_type == "dqn":
                action, _ = model.predict(obs, deterministic=True)
            elif model_type == "default":
                # For default, we don't control. But we need to step.
                # In our env, step(action) does control.
                traci.vehicle.setLaneChangeMode(env.ego_id, 1621) # Default mode
                action = 0 # Dummy action for step
                
            if model_type != "default":
                 # Ensure we control it
                 traci.vehicle.setLaneChangeMode(env.ego_id, 0)
            
            pass
            
            # Custom step for default to avoid env.step() overriding
            if model_type == "default":
                traci.simulationStep()
                env._freeze_obstacles() # We still need to freeze obstacles
                obs = env._get_obs()
                
                # Calculate reward same as env
                reward = 0
                if env.ego_id in traci.vehicle.getIDList():
                    speed = traci.vehicle.getSpeed(env.ego_id)
                    reward += speed * 0.1
                    
                    if env.ego_id in traci.simulation.getCollidingVehiclesIDList():
                        reward -= 50
                        done = True
                else:
                    arrived = traci.simulation.getArrivedIDList()
                    if env.ego_id in arrived:
                        reward += 100
                        done = True
                    else:
                        # Disappeared
                        pass
                
                env.step_count += 1
                if env.step_count >= env.max_steps:
                    truncated = True
            else:
                obs, reward, done, truncated, _ = env.step(action)
            
            total_reward += reward
            step += 1
            
        results.append({
            "Model": model_type,
            "Episode": episode,
            "Total Reward": total_reward,
            "Steps": step
        })
        
    env.close()
    return results

if __name__ == "__main__":
    all_results = []
    
    # Train models first

    
    print("Evaluating Q-Learning...")
    try:
        res_q = evaluate_model("q_learning", "q_table.npy", episodes=10)
        all_results.extend(res_q)
    except Exception as e:
        print(f"Q-Learning evaluation failed: {e}")

    print("Evaluating DQN...")
    try:
        res_dqn = evaluate_model("dqn", "dqn_lane_change", episodes=10)
        all_results.extend(res_dqn)
    except Exception as e:
        print(f"DQN evaluation failed: {e}")

    print("Evaluating Default SUMO...")
    try:
        res_def = evaluate_model("default", None, episodes=10)
        all_results.extend(res_def)
    except Exception as e:
        print(f"Default SUMO evaluation failed: {e}")
        
    df = pd.DataFrame(all_results)
    df.to_csv("evaluation_results.csv", index=False)
    print("Results saved to evaluation_results.csv")
    print(df.groupby("Model").mean())
