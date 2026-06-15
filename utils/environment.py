import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from math import exp


class SteelLogistics(gym.Env):
    def __init__(self, config):
        super(SteelLogistics, self).__init__()
        
        # Unpack config variables to self
        self.time_frame = config.time_frame
        self.slab_mean_temp = config.slab_mean_temp
        self.slab_mean_std = config.slab_mean_std
        self.slab_temp_ubound = config.slab_temp_ubound
        self.slab_quantity_probability = config.slab_quantity_probability
        self.urgency_bounds = config.urgency_bounds
        self.temp_dict = config.temp_dict
        self.ncrane = config.ncrane
        self.nhotroom_fixed = config.nhotroom_fixed
        self.nhotroom_mobile = config.nhotroom_mobile
        self.crane_move_time = config.crane_move_time
        self.crane_cost_per_time_step = config.crane_cost_per_time_step
        self.celsius_utility_cost = config.celsius_utility_cost
        self.action_space_size = config.action_space_size
        self.state_space_size = config.state_space_size
        self.urgent_slabs_count = config.urgent_slabs_count
        self.temp_threshold = config.temp_threshold
        self.temp_treshold_utility_cost = config.temp_treshold_utility_cost

        # Seeding
        self.seed_value = getattr(config, 'seed', np.random.randint(0, 1000000))
        self.next_slab_id = 0
        
        # Action Space: 4 urgent slabs, each can be acted upon in 3 ways:
        # 0: Leave in Yard, 1: Pull to Fixed Room, 2: Pull to Mobile Cover
        self.action_space = spaces.MultiDiscrete(self.action_space_size)

        # Observation Space: 11 continuous/discrete features represented as a flat array
        low = np.array(
            [0.0] * 4 +                 # Wait times of the 4 most urgent yard slabs
            [0.0] * 4 +                 # Normalized temperatures (0 to 1)
            [0.0] +                     # Available slots in Fixed Hot Room
            [0.0] +                     # Available Mobile Covers
            [0.0],                      # Total Yard Backlog
            dtype=np.float32
        )

        high = np.array(
            [self.urgency_bounds[1]] * 4 +   # Wait times max out at upper bound
            [1.0] * 4 +                             # Normalized temperatures max out at 1.0
            [self.nhotroom_fixed] +          # Max Fixed Hot Room slots
            [self.nhotroom_mobile] +         # Max Mobile Covers
            [np.inf],                               # Total Yard Backlog
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.state = None
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.slab_history = []

        # 1. Location Parameters (from config)
        self.crane_costs = [
            self.crane_move_time[0] * self.crane_cost_per_time_step,
            self.crane_move_time[1] * self.crane_cost_per_time_step,
            self.crane_move_time[2] * self.crane_cost_per_time_step
        ]

    def _generate_new_slabs(self):
        """Generates new slabs based on probabilities and adds them to the yard."""
        if getattr(self, 'drain_mode', False):
            return
        num_new_slabs = self.np_random.choice(len(self.slab_quantity_probability), p=self.slab_quantity_probability)
        for _ in range(num_new_slabs):
            new_wait = self.np_random.integers(self.urgency_bounds[0], self.urgency_bounds[1] + 1)
            new_temp = min(self.np_random.normal(self.slab_mean_temp, self.slab_mean_std), self.slab_temp_ubound)
            self.yard_list.append({'id': self.next_slab_id, 'wait_time': new_wait, 'temp': new_temp})
            
            # Log arrival for LP optimization
            self.slab_history.append({
                'step': self.current_step,
                'id': self.next_slab_id,
                'initial_wait_time': new_wait,
                'initial_temp': new_temp
            })
            
            self.next_slab_id += 1

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        super().reset(seed=seed if seed is not None else self.seed_value)
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.next_slab_id = 0
        self.drain_mode = False
        
        self.yard_list = []
        self.fixed_room_list = []
        self.mobile_cover_list = []
        self.slab_history = []
        
        # Initial stochastic generation
        self._generate_new_slabs()
            
        self._update_state()
        
        return self.state, {}

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        self.current_step += 1
        step_reward = 0.0
        
        urgent_slabs = self.yard_list[:self.urgent_slabs_count]
        
        # 2. Parse Actions and Enforce Crane Constraint
        active_actions = action[:len(urgent_slabs)]
        requested_crane_moves = sum(self.crane_move_time[a] for a in active_actions)
        requested_crane_cost = sum(self.crane_costs[a] for a in active_actions)
        
        valid_action = []
        if requested_crane_moves > 1:
            step_reward -= (1000 + requested_crane_cost)
            tally = 0
            for a in active_actions:
                if tally + self.crane_move_time[a] > 1:
                    valid_action.append(0)
                else:
                    tally += self.crane_move_time[a]
                    valid_action.append(a)
        else:
            step_reward -= requested_crane_cost
            valid_action = list(active_actions)
            
        # 3. Execute Valid Movements
        slabs_to_remove_ids = set()
        
        for i, a in enumerate(valid_action):
            slab = urgent_slabs[i]
            if a == 1 and len(self.fixed_room_list) < self.nhotroom_fixed:
                slabs_to_remove_ids.add(slab['id'])
                self.fixed_room_list.append(slab)
            elif a == 2 and len(self.mobile_cover_list) < self.nhotroom_mobile:
                slabs_to_remove_ids.add(slab['id'])
                self.mobile_cover_list.append(slab)
                
        self.yard_list = [slab for slab in self.yard_list if slab['id'] not in slabs_to_remove_ids]
            
        # 4. Update Thermal Physics (End of Hour Decay)
        all_lists = [
            (self.yard_list, self.temp_dict['yard']),
            (self.fixed_room_list, self.temp_dict['fixed']),
            (self.mobile_cover_list, self.temp_dict['mobile'])
        ]
        
        for lst, params in all_lists:
            ambient, k = params
            for slab in lst:
                slab['wait_time'] -= 1
                slab['temp'] = self.temperature_decay(slab['temp'], ambient, k, time=1)
                
        # 5. Process Deliveries and Calculate Thermal Rewards
        def process_deliveries(lst):
            nonlocal step_reward
            kept = []
            for slab in lst:
                if slab['wait_time'] <= 0:
                    step_reward += slab['temp']
                    if slab['temp'] < self.temp_threshold:
                        step_reward -= self.temp_treshold_utility_cost
                else:
                    kept.append(slab)
            return kept
            
        self.yard_list = process_deliveries(self.yard_list)
        self.fixed_room_list = process_deliveries(self.fixed_room_list)
        self.mobile_cover_list = process_deliveries(self.mobile_cover_list)
        
        # 6. Stochastic Generation (New Slab Arrivals)
        self._generate_new_slabs()
            
        # 7. Build the Next Observation State
        self._update_state()

        # 8. Update cumulative reward
        self.cumulative_reward += step_reward
        
        # 9. Return
        terminated = False  # The factory logistics never reach a natural "game over"
        if getattr(self, 'drain_mode', False):
            # In drain mode, the episode is truncated only when all active slabs are delivered
            truncated = (len(self.yard_list) == 0 and 
                         len(self.fixed_room_list) == 0 and 
                         len(self.mobile_cover_list) == 0)
        else:
            truncated = self.current_step >= self.time_frame
            
        info = {
            "requested_crane_cost": requested_crane_cost,
            "invalid_crane_moves": True if requested_crane_moves > 1 else False,
            "active_slabs_fixed_room": len(self.fixed_room_list),
            "active_slabs_mobile_cover": len(self.mobile_cover_list),
            "backlog_count": len(self.yard_list),
            "step_reward": step_reward,
            "cumulative_reward": self.cumulative_reward,
            "slab_history": self.slab_history
        }
        
        return self.state, step_reward, terminated, truncated, info

    def _update_state(self):
        self.yard_list.sort(key=lambda x: x['wait_time'])
        obs = np.zeros(self.state_space_size, dtype=np.float32)
        
        urgent_slabs = self.yard_list[:self.urgent_slabs_count]
        for i in range(self.urgent_slabs_count):
            if i < len(urgent_slabs):
                obs[i] = urgent_slabs[i]['wait_time']
                obs[self.urgent_slabs_count + i] = urgent_slabs[i]['temp'] / 1000.0
            else:
                obs[i] = 0.0
                obs[self.urgent_slabs_count + i] = 0.0
                
        obs[self.urgent_slabs_count * 2] = self.nhotroom_fixed - len(self.fixed_room_list)
        obs[self.urgent_slabs_count * 2 + 1] = self.nhotroom_mobile - len(self.mobile_cover_list)
        obs[self.urgent_slabs_count * 2 + 2] = len(self.yard_list)
        
        self.state = obs

    @staticmethod
    def temperature_decay(initial_temperature, ambient_temperature, cooling_constant, time=1):
        return ambient_temperature + (initial_temperature - ambient_temperature) * exp(-cooling_constant * time)

    
    def get_slab_history_df(self):
        """
        Returns the slab arrival history as a pandas DataFrame.
        Useful for LP optimization after the episode terminates.
        """
        return pd.DataFrame(self.slab_history)

    def evaluate(self, model, test_time_frame=None, total_timesteps=None, seed=None):
        """
        Runs a full evaluation sequence using the trained model (predicting actions deterministically).
        Temporarily overrides the environment's time_frame if test_time_frame is provided.
        Can run over multiple sequential episodes/days if total_timesteps is provided.
        Keeps track of seeds used for each episode for traceability.
        Returns: (combined_df, list_of_episode_dfs)
        """
        # Save original time_frame
        original_time_frame = self.time_frame
        
        if test_time_frame is not None:
            self.time_frame = test_time_frame
            
        target_steps = total_timesteps if total_timesteps is not None else self.time_frame
        
        # Setup seeds for traceability
        start_seed = seed if seed is not None else np.random.randint(0, 1000000)
        current_seed = start_seed
        
        obs, _ = self.reset(seed=current_seed)
        
        terminated = False
        truncated = False
        steps_run = 0
        
        histories = []
        episode_dfs = []
        
        while steps_run < target_steps or self.drain_mode:
            action, _ = model.predict(obs, deterministic=True)
            
            # Count steps run during active generation phase
            if not self.drain_mode:
                steps_run += 1
                
            # If we reached the end of the active generation period for this episode, trigger drain mode
            if not self.drain_mode and self.current_step >= self.time_frame:
                self.drain_mode = True
                truncated = False
                
            obs, reward, terminated, truncated, _ = self.step(action)
            
            if terminated or truncated:
                # Capture history of the finished episode and attach its seed
                ep_history = [slab.copy() for slab in self.slab_history]
                for slab in ep_history:
                    slab['seed'] = current_seed
                histories.append(ep_history)
                episode_dfs.append(pd.DataFrame(ep_history))
                
                # Deactivate drain mode as the episode is fully finished
                self.drain_mode = False
                
                if steps_run < target_steps:
                    # Generate next sequential seed deterministically
                    current_seed = (current_seed + 1) % 1000000
                    obs, _ = self.reset(seed=current_seed)
                    terminated = False
                    truncated = False
        
        # Build combined DataFrame with episode indices
        combined_history = []
        for ep_idx, ep_history in enumerate(histories):
            for slab in ep_history:
                slab_copy = slab.copy()
                slab_copy['episode'] = ep_idx
                combined_history.append(slab_copy)
                
        combined_df = pd.DataFrame(combined_history) if combined_history else pd.DataFrame()
        
        # Restore original time_frame
        self.time_frame = original_time_frame
        
        return combined_df, episode_dfs

