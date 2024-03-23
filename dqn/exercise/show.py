"""
load a save checkpoint and run the model 100 times
"""

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint2.pth'))
env = gym.make('LunarLander-v2', render_mode='human')
for i in range(5):
    state, info = env.reset()
    for j in range(200):
        action = agent.act(state)
        env.render()
        state, reward, done, truncated, info = env.step(action)
        if done:
            break

env.close()