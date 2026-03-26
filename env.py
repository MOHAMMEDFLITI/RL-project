import gymnasium as gym
from gymnasium import spaces
import traci
import numpy as np
import os
import sys

# Check for SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

class SumoLaneChangeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gui=False, max_steps=1000):
        super(SumoLaneChangeEnv, self).__init__()
        
        self.gui = gui
        self.max_steps = max_steps
        self.step_count = 0
        self.ego_id = "vehAgent"
        
        # Actions: 0: Keep Lane, 1: Change Left, 2: Change Right
        self.action_space = spaces.Discrete(3)
        
        # Observation: [Ego Lane (0-1), Ego Speed (0-30), Dist Front Lane 0 (0-200), Dist Front Lane 1 (0-200)]
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        high = np.array([1, 50, 200, 200], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.sumo_binary = "sumo-gui" if self.gui else "sumo"
        self.sumo_cmd = [
            self.sumo_binary,
            "-c", "data/obstacles.sumocfg",
            "--start", "true",
            "--collision.action", "warn",
            "--xml-validation", "never",
            "--no-step-log", "true",
            "--no-warnings", "true"
        ]
        
        self.label = str(np.random.randint(10000)) # Unique label for traci connection

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        try:
            traci.close()
        except:
            pass
            
        try:
            traci.start(self.sumo_cmd, label=self.label)
        except Exception as e:
            pass

        self.step_count = 0
        
        # Initialize Ego Vehicle
        try:
            traci.vehicle.add(self.ego_id, "r_0", typeID="obstacle", depart=0)
            traci.vehicle.setLaneChangeMode(self.ego_id, 0) # Disable default LC
            traci.vehicle.setSpeedMode(self.ego_id, 31) # Enable all checks (collision etc)

            # "AV’s speed is controlled by SUMO" -> We only do LC.
        except traci.exceptions.TraCIException:
            # Vehicle might already exist if reset didn't fully clear
            pass

        # Setup obstacles as per run.py
        traci.simulationStep()
        
        # Freeze obstacles
        self._freeze_obstacles()

        return self._get_obs(), {}

    def _freeze_obstacles(self):
        vehicleIDs = traci.vehicle.getIDList()
        for veh in vehicleIDs:
            if veh != self.ego_id:
                traci.vehicle.setLaneChangeMode(veh, 0)
                traci.vehicle.setSpeed(veh, 0)

    def step(self, action):
        self.step_count += 1
        
        # Apply Action
        # 0: Keep, 1: Left, 2: Right
        # SUMO LC directions: 1 (left), -1 (right), 0 (stay)
        
        current_lane = traci.vehicle.getLaneIndex(self.ego_id)
        
        target_lane = current_lane
        if action == 1: # Left
            target_lane = min(current_lane + 1, 1) # Max lane 1
        elif action == 2: # Right
            target_lane = max(current_lane - 1, 0) # Min lane 0
            
        # using changeLane to request a change. 
        # duration=1s (or step length)
        traci.vehicle.changeLane(self.ego_id, target_lane, duration=1)
        
        # Step simulation
        traci.simulationStep()
        
        # Ensure obstacles stay frozen
        # The route file has trips with depart times. If they are all 0, they appear at start.
        # If some appear later, we need to freeze them.
        self._freeze_obstacles()

        # Get Observation
        obs = self._get_obs()
        
        
        terminated = False
        truncated = False
        reward = 0
        
        if self.ego_id not in traci.vehicle.getIDList():

            arrived = traci.simulation.getArrivedIDList()
            if self.ego_id in arrived:
                reward += 100 # Completion bonus
                terminated = True
            else:
                # Probably crashed or just disappeared
                pass
        
        colliding = traci.simulation.getCollidingVehiclesIDList()
        if self.ego_id in colliding:
            reward -= 50
            terminated = True # End episode on collision
            
        # Reward function
        speed = 0
        if self.ego_id in traci.vehicle.getIDList():
            speed = traci.vehicle.getSpeed(self.ego_id)
        
        reward += speed * 0.1 # Reward for moving
        
        # Penalty for lane change to encourage stability
        if action != 0:
            reward -= 0.1
            
        if self.step_count >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        if self.ego_id not in traci.vehicle.getIDList():
            return np.zeros(4, dtype=np.float32)
            
        lane = traci.vehicle.getLaneIndex(self.ego_id)
        speed = traci.vehicle.getSpeed(self.ego_id)
        pos = traci.vehicle.getLanePosition(self.ego_id)
        
        # Distance to obstacles
        # We need to find the closest obstacle in each lane ahead of us.
        
        def get_dist_in_lane(target_lane):
            min_dist = 200.0
            vehicleIDs = traci.vehicle.getIDList()
            for veh in vehicleIDs:
                if veh == self.ego_id:
                    continue
                v_lane = traci.vehicle.getLaneIndex(veh)
                v_pos = traci.vehicle.getLanePosition(veh)
                
                if v_lane == target_lane and v_pos > pos:
                    dist = v_pos - pos
                    if dist < min_dist:
                        min_dist = dist
            return min_dist

        dist_0 = get_dist_in_lane(0)
        dist_1 = get_dist_in_lane(1)
        
        return np.array([lane, speed, dist_0, dist_1], dtype=np.float32)

    def close(self):
        traci.close()
