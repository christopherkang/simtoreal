from collections import deque

import cv2
import dm_control.mujoco as mujoco
import matplotlib.pyplot as plt
import numpy as np
import torch
from dm_control import suite

import agent
from random_wrapper import RandomWrapper

env=RandomWrapper(suite.load(domain_name="point_mass", task_name="easy"))


action_size = 2 # there are x, y parameters
state_size = 4  # IDEK - see model.py self.batch_norm =
num_agents = 1

agent = agent.Agent(action_size, state_size, True, num_agents, random_seed = 0)

def d4pg(n_episodes=40):
    
    scores = []
    scores_deque = deque(maxlen=100)
    rolling_average_score = []
    
    max_score = 200
    current_score = 0
    
    for i_episode in range(1, n_episodes+1):
        # env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        # state = env_info.vector_observations                  # get the current state (for each agent)

        state = env.reset()
        # state = np.swapaxes(state, 2, 0)
        # state = np.expand_dims(state, axis=0)
        state = np.append(state['position'], state['velocity'])
        state = np.expand_dims(state, axis=0)

        agent.reset()
        score = np.zeros(num_agents)
        while True:
            action = agent.act(state, current_score, max_score)

            env.render()
            
            next_state, reward, done, empty = env.step(action)  # send all actions to the environment
            # next_state = env_info.vector_observations         # get next state (for each agent)
            # reward = env_info.rewards                         # get reward (for each agent)
            # done = env_info.local_done                        # to  see if episode finished

            # next_state = np.swapaxes(next_state, 2, 0) # make it channel, w, h
            next_state = np.append(next_state['position'], next_state['velocity'])
            next_state = np.expand_dims(next_state, axis=0)

            reward = np.asarray([reward])
            done = np.asarray([done])
            
            score += reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if np.any(done):                                  # see if any episode finished
                break             
        
        score = np.mean(score)
        scores_deque.append(score)
        scores.append(score)
        rolling_average_score.append(np.mean(scores_deque))
        
        current_score = np.mean(scores_deque)
        print("current score: ", current_score)
        
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        
        if i_episode % 100 == 0:                        
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
        if i_episode % 10 == 0:            
            torch.save(agent.local_actor.state_dict(), 'checkpoint_local_actor.pth')           # save local actor
            torch.save(agent.target_actor.state_dict(), 'checkpoint_target_actor.pth')         # save target actor 
            torch.save(agent.local_critic.state_dict(), 'checkpoint_local_critic.pth')         # save local critic
            torch.save(agent.target_critic.state_dict(), 'checkpoint_target_critic.pth')       # target critic
                        
        # if 30 < current_score and 99 < len(scores_deque):
        #     print('Target average reward achieved!')
        #     torch.save(agent.local_actor.state_dict(), 'checkpoint_local_actor.pth')           # save local actor
        #     torch.save(agent.target_actor.state_dict(), 'checkpoint_target_actor.pth')         # save target actor 
        #     torch.save(agent.local_critic.state_dict(), 'checkpoint_local_critic.pth')         # save local critic
        #     torch.save(agent.target_critic.state_dict(), 'checkpoint_target_critic.pth')       # target critic
        #     break
    return scores, rolling_average_score

scores, rolling_average_score = d4pg()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.plot(np.arange(1, len(rolling_average_score)+1), rolling_average_score)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()