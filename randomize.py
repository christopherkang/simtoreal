import numpy as np
import cv2
import dm_control.mujoco as mujoco
from dm_control import suite

class Discritize_Wrapper():
    _TEXTUREPATH = [
        ["/home/christopherkang/.mujoco/mujoco200_linux/model/carpet.png", "/home/christopherkang/.mujoco/mujoco200_linux/model/marble.png"]
        ]
    
    def __init__(self, continuous_env, actions_per_control):
        self.actions_per_control=actions_per_control
        self.continuous_env=continuous_env
        self.number_actions=actions_per_control**self.continuous_env.action_spec().shape[0]
        self._action_set=list(range(self.number_actions))

    def create_random_model(self, filepath, texture_choices): 
        f = open(filepath, "r")
        randomized_elements = tuple([np.random.choice(options) for options in texture_choices])
        randomized_file = f.read().format(*randomized_elements)
        return randomized_file
        
    def reset(self):
        self.continuous_env.reset()
        self.continuous_env.physics.reload_from_xml_string(self.create_random_model("point_mass.xml", self._TEXTUREPATH))
        return self.continuous_env.physics.render(height=240, width=240, camera_id=0)
        
    def step(self, action):
        converted_actions=np.zeros(self.continuous_env.action_spec().shape)
        mod_amt=self.actions_per_control
        a=self.continuous_env.action_spec()
        for ind in range(self.continuous_env.action_spec().shape[0]):
            control_action=action%mod_amt
            action-=control_action
            mod_amt*=self.actions_per_control
            control_action/=(self.actions_per_control-1)
            control_action=control_action*(self.continuous_env.action_spec().maximum[ind]-self.continuous_env.action_spec().minimum[ind])+self.continuous_env.action_spec().minimum[ind]
            converted_actions[ind]=control_action
        time_step=self.continuous_env.step(converted_actions)
        return self.continuous_env.physics.render(height=240, width=240, camera_id=0), time_step.reward, time_step.last(), None
    
    def render(self):
        cv2.imshow('real_image', self.continuous_env.physics.render(height=240, width=240, camera_id=0))
        cv2.waitKey(20)

env=Discritize_Wrapper(suite.load(domain_name="point_mass", task_name="easy"), 3)

# Iterate over a task set:
# for domain_name, task_name in suite.BENCHMARKING:
#   env = suite.load(domain_name, task_name)

# Step through an episode and print out reward, discount and observation.
action_spec = env.continuous_env.action_spec()
time_step = env.reset()

# while not time_step.last():
for _ in range(100):
    env.render()
    env.reset()
    # action = np.random.uniform(action_spec.minimum,
    #                          action_spec.maximum,
    #                          size=action_spec.shape)
    action = 0
    time_step = env.step(action)
    # print(time_step.reward, time_step.discount, time_step.observation)