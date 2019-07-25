# create an environment capable of domain randomization.

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py

# needs to be of the mujoco object
class randomized_env(mujoco_env.MujocoEnv):
    def __init__(self, xml, texture_choices, model_path, frame_skip, rgb_rendering_tracking=True):
        self.xml_path = xml
        self.texture_choices = texture_choices
        return super().__init__(model_path, frame_skip, rgb_rendering_tracking=rgb_rendering_tracking)

    def create_random_model(self, filepath, texture_choices): 
        f = open(filepath, "r")
        randomized_elements = tuple([np.random.choice(options) for options in texture_choices])
        randomized_file = f.read().format(*randomized_elements)
        return mujoco_py.load_model_from_xml(randomized_file)

    def set_state(self, qpos, qvel):
        randomized_model = self.create_random_model(self.xml_path, self.texture_choices)
        self.sim.model = randomized_model
        # assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        # old_state = self.sim.get_state()
        # # this code is almost identical, but we've set the last parameter to randomized state's udd
        # new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
        #                                  old_state.act, randomized_model.udd_state)
        
        # self.sim.set_state(new_state)
        # self.sim.forward()
        super().set_state(qpos, qvel)

    # objective: we'd like this to automatically randomize the textures within the stored file
    # def reset(self, parameter_list):
    #     # we need to first recreate a MujocoSim state 

    #     # then we need to extract the relevant data

    #     # and replace the MujocoSim state
    #     pass

# auxiliary.create_random_model("/home/christopherkang/Documents/Github/simtoreal/sample_tosser.xml", [["/home/christopherkang/.mujoco/mujoco200/model/carpet.png", "/home/christopherkang/.mujoco/mujoco200/model/marble.png"]])