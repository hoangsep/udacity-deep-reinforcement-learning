"""
load a save checkpoint and run the model times
"""

from unityagents import UnityEnvironment
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
checkpoint = 'checkpoint_s17.pth'
n_episodes = 5
max_t = 1000

if __name__ == "__main__":
    # load the weights from file
    agent = Agent(vector=True, action_size=4, seed=0, device=device)
    agent.qnetwork_local.load_state_dict(torch.load(checkpoint))
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    scores = []  # list containing scores from each episode

    for i_episode in range(n_episodes):
        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]  # get the next state
            next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            state = next_state
            score += reward
            if done:
                print('\rEpisode {}\tTime {}\tAverage Score: {:.2f}'.format(i_episode, t, score))
                break
        scores.append(score)  # save most recent score

    env.close()
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
