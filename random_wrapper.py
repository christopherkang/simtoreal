import numpy as np
import cv2
import dm_control.mujoco as mujoco
from dm_control import suite

class RandomWrapper():
    _TEXTUREPATH = [
        ["/home/christopherkang/.mujoco/mujoco200_linux/model/carpet.png", "/home/christopherkang/.mujoco/mujoco200_linux/model/marble.png"]
        ]
    
    def __init__(self, continuous_env):
        self.continuous_env=continuous_env

    def create_random_model(self, filepath, texture_choices): 
        f = open(filepath, "r")
        randomized_elements = tuple([np.random.choice(options) for options in texture_choices])
        randomized_file = f.read().format(*randomized_elements)
        return randomized_file
        
    def reset(self, render=False):
        reset_data = self.continuous_env.reset()
        self.continuous_env.physics.reload_from_xml_string(self.create_random_model("point_mass.xml", self._TEXTUREPATH))
        self.continuous_env._step_limit = 200 #FLAG forced
        if render:
            return self.continuous_env.physics.render(height=240, width=240, camera_id=0)
        return reset_data.observation
        
    def step(self, action, render=False):
        time_step=self.continuous_env.step(action)
        
        if render: 
            return self.continuous_env.physics.render(height=240, width=240, camera_id=0), time_step.reward, time_step.last(), None
        return time_step.observation, time_step.reward, time_step.last(), None
        

    def render(self, render_time=20):
        cv2.imshow('real_image', self.continuous_env.physics.render(height=240, width=240, camera_id=0))
        cv2.waitKey(render_time)




if __name__ == "__main__":
    env=RandomWrapper(suite.load(domain_name="point_mass", task_name="easy"))
    
    # Iterate over a task set:
    # for domain_name, task_name in suite.BENCHMARKING:
    #   env = suite.load(domain_name, task_name)

    # Step through an episode and print out reward, discount and observation.
    action_spec = env.continuous_env.action_spec()
    time_step = env.reset()

    for _ in range(1000):
        env.render(50)
        # if _ % 10 == 0:
        #     test = env.reset()
        action = np.random.uniform(action_spec.minimum,
                                action_spec.maximum,
                                size=action_spec.shape)
        action = (10, -10)
        time_step, test1, test2, test3 = env.step(action)
        print(test1, test2, test3)
        print(time_step)
        # print(time_step.reward, time_step.discount, time_step.observation)