from enum import IntEnum
from typing import Literal, Tuple, Union

import numpy as np
from gymnasium.spaces import Box, Discrete
from minigrid.minigrid_env import MiniGridEnv, Grid
from minigrid.core.world_object import Door, Goal, Ball
from minigrid.core.mission import MissionSpace
from minigrid.utils.window import Window
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_NAMES, IDX_TO_COLOR
from matplotlib import pyplot as plt
from numpy.linalg import norm

from store_env.environment.obj import Shelf
from store_env.environment.env_setup import StoreLayout, StoreEnvSetup, make_env_to_show_interpolation, make_env_always_buy_yellow


class StoreEnv(MiniGridEnv):
    class ObsType(IntEnum):
        state = 0
        grid_obs = 1
        tabular = 2

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Pick up an object
        pickup = 3
        # # Drop an object
        # drop = 4
        # # Toggle/activate an object
        # toggle = 5

        # # Done completing task
        # done = 5
    @classmethod
    def make_env(
        cls,
        irrational_weight: Union[float, str] = 0,
        obs_type: Literal["state", "obs", "tabular"] = "state", 
        layout: Literal["default", "interpolate", "prefer-yellow"] = "default"
    ):
        if layout=="default":
            setup = StoreEnvSetup(obs_type=obs_type, irrational_weight=float(irrational_weight))
        elif layout=="interpolate":
            setup = make_env_to_show_interpolation(irrational_weight=float(irrational_weight))
        elif layout=="prefer-yellow":
            setup = make_env_always_buy_yellow(float(irrational_weight))
        setup.obs_type = obs_type
        print('****************************')
        print(f"Running StoreEnv with the following setup: {setup}")
        print('****************************')
        return cls(**setup.__dict__)

    def __init__(
        self, 
        max_steps = 128,
        render_mode="rgb_array", 
        irrational_weight = .4,
        time_penalty_per_step = -0.04,
        time_out_penalty = 0, # -ve reward for spending too long in store
        exit_reward = 0, # Reward for exiting within time limit
        sparse_reward = False, # True means give lump sum reward after exiting store, False means get rewarded immediately after picking up an item
        # "obs" for CNN, "state" is normalized version of "tabular" for MLP/Linear FA
        obs_type: Literal["state", "obs", "tabular"] = "state", 
        layout: StoreLayout = StoreLayout()
    ):
        super().__init__(
            width=layout.width + 2, 
            height=layout.height + 2, 
            max_steps=max_steps, 
            mission_space=MissionSpace(mission_func=self._gen_mission,), 
            render_mode=render_mode, 
            tile_size=32,
            highlight= obs_type=="obs"
        )

        self.agent_initial_pos = layout.agent_initial_pos
        self.shelves = layout.shelves_positions
        self.products = layout.products_placement
        # exit tile is hardcoded atm
        self.exit = layout.exit_pos
        self.inventory = []

        
        # Set observations
        self.obs_type = obs_type
        if obs_type=="obs":
            self.observation_space = Box(
                low=0,
                high=max(OBJECT_TO_IDX.values()),
                shape=(self.agent_view_size, self.agent_view_size, 2),
                dtype="uint8",
            )
        elif obs_type=="state" or obs_type=="tabular":
            self.observation_space = Box(
                low=-1,
                high=max(layout.width, layout.height)+2,
                # # xy-dist of products + coord of agent + orientation of agent + timeleft + number of items in cart + xy-dist to goal
                # shape=(len(self.products)*2 + 2 + 1 + 1 + 1 + 2,),
                # binary inventory of products + coord of agent + orientation of agent + timeleft 
                shape=(len(self.products) + 2 + 1 + 1,),
                dtype=np.float32 if obs_type=="state" else np.int16
            )
        # else:
        #     raise 

        # Set actions
        self.actions = StoreEnv.Actions
        self.action_space = Discrete(len(self.actions))

        # Set rewards
        self.time_penalty_per_step = time_penalty_per_step
        self.sparse_reward = sparse_reward
        self.exit_reward = exit_reward
        self.time_out_penalty = time_out_penalty

        # Set store config
        
        num_products = len(layout.products_placement)
        assert 0<=irrational_weight<=1
        self.irrational_weight = irrational_weight


    def _gen_grid(self, width, height): # called in reset() of MiniGridEnv
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Add shelves
        for coords in self.shelves:
            coords = [x + 1 for x in coords]  # to account for walls
            self.grid.set(*coords, Shelf())

        # Add exit
        self.grid.set(*self.exit, Goal())
        # self.agent_pos = (self._rand_pos(1, self.width-1, 1, self.height-2))
        # while (self.agent_pos[0]-1, self.agent_pos[1]-1) in self.shelves or self.agent_pos==self.exit:
        #     self.agent_pos = (self._rand_pos(1, self.width-1, 1, self.height-1))
        self.agent_pos = self.agent_initial_pos
        # self.grid.set(*self._agent_default_pos, None)
        # print(self.agent_pos)
        self.agent_dir = self._rand_int(0, 4)  # assuming random start direction

        
        self.pos_to_products = {}
        for i, pos in enumerate(self.products):
            product_pos = (pos.coord[0] + 1, pos.coord[1] + 1)
            self.grid.set(*product_pos, Ball(color=pos.color))
            self.pos_to_products[product_pos] = i


    def _norm_dist_to_agent(self, pos_of_object):
        return norm(np.array(pos_of_object) - np.array(self.agent_pos))
    
    def _normalized_distance(self, coord: Tuple[int, int]):
        return (self.agent_pos[0]-coord[0])/(self.width-2), (self.agent_pos[1]-coord[1])/(self.height-2)
    
    def gen_state_obs(self):
        # Ideally we can derive the obs from the image obs, but for now we will just do it using our inventory object.
        # Note: If we derive state obs from image obs, be sure to not use COLOR_NAMES and use IDX_TO_COLOR instead above (when assigning color to products)
        obs = []
        inventory_set = set(self.inventory)
        obs.extend([int(i in inventory_set) for i in range(len(self.products))])
        if self.obs_type=="state":
            obs.extend(self._normalized_distance((0,0)))
            obs.append(self.agent_dir/4)
            obs.append(self.steps_remaining/self.max_steps)
            # obs.append(len(self.inventory)/len(self.products))
            # obs.extend(self._normalized_distance(self.exit))
            obs = np.array(obs, dtype=np.float32)
        else:
            obs.extend(self.agent_pos)
            obs.append(self.agent_dir)
            # discretize into [1, 10]
            obs.append(round(self.steps_remaining*10/self.max_steps))
            obs = np.array(obs, dtype=np.int16)
        assert obs.shape==self.observation_space.shape, f"{obs.shape=}, {self.observation_space.shape=}"
        return obs
    
    def gen_obs(self):
        if self.obs_type=="obs":
            obs = super().gen_obs()
            # 3 axis: obs.shape=(7x7x3) by default
            # - layer 1 of the "image" is object type (e.g., Door, Ball). See OBJECT_TO_IDX in minigrid>core>constants.py, 
            # - layer 2 is "color" as single int. See COLOR_TO_IDX in minigrid>core>constants.py for color->int mapping
            # - layer 3 is the optional "state" of objects. 
            obs = obs["image"][:, :, :2]
        else:
            obs = self.gen_state_obs()
            
        return obs

    def reset(self, seed=None, **kwargs):
        obs, info = super().reset(seed=seed, options=kwargs)
        self.inventory = []
        
        return obs, info
    
    @staticmethod
    def _gen_mission():
        return "Buy groceries and leave"

    def step(self, action):
        self.step_count += 1

        reward = self.time_penalty_per_step
        terminated = self.step_count >= self.max_steps
        truncated = False # If run out of time its terminated, not truncated.

        
        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                self.inventory.append(prod_index:=self.pos_to_products[(fwd_pos[0], fwd_pos[1])])
                self.grid.set(*fwd_pos, Shelf())
                if not self.sparse_reward:
                    prod = self.products[prod_index]
                    reward += (1-self.irrational_weight)*prod.preference_reward + self.irrational_weight*prod.irrational_preference_reward

        obs = self.gen_obs()

        if isinstance(fwd_cell, Goal) and action == self.actions.forward:
            terminated = True
            truncated = False
            reward += self.exit_reward
            if self.sparse_reward:
                for prod_index in self.inventory:
                    prod = self.products[prod_index]
                    reward += (1-self.irrational_weight)*prod.preference_reward + self.irrational_weight*prod.irrational_preference_reward
            if not self.inventory:
                reward = -1
        elif terminated or truncated:
            reward = self.time_out_penalty

        return obs, reward, terminated, truncated, {}

# Gym entrypoint
def make_env(**kwargs):
    return StoreEnv.make_env(**kwargs)