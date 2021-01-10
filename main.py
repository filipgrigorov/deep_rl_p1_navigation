import matplotlib.pyplot as plt
import numpy as np
import torch

from collections import deque
from unityagents import UnityEnvironment

from agent import Agent

def run(env, brain_name):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]            # get the current state
    state_size = len(state)
    score = 0                                          # initialize the score
    nscores = 0

    eps = 1.0
    min_eps = 0.01
    eps_decay = 0.995
    lr = 5e-4
    gamma = 0.99
    batch_size = 64
    agent = Agent(lr, gamma, batch_size, state_size, action_size)

    scores = []
    scores_window = deque(maxlen=100)

    nepisodes = 2000
    for episode in range(1, nepisodes + 1):
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]

        score = 0; episodic_count = 0
        while True:
            action = agent.act(state, eps)

            env_info = env.step(action)[brain_name]        # send the action to the environment

            next_state = env_info.vector_observations[0]   # get the next state

            reward = env_info.rewards[0]                   # get the reward

            done = env_info.local_done[0]                  # see if episode has finished

            agent.train(state, action, reward, next_state, done)

            score += reward                                # update the score
            episodic_count += 1

            state = next_state                             # roll over the state to next time step

            if done:                                       # exit loop if episode finished
                print('\r\n\tFinal episodic average reward: {}'.format(score))
                break

        scores_window.append(score)
        scores.append(score)

        # Update eps at every episode and not on every step
        eps = max(min_eps, eps_decay * eps)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")

        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))

        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_window)))
            torch.save(agent.behavioral_model.state_dict(), 'p1_navigation_checkpoint.pth')
            break

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

if __name__ == '__main__':
    env = UnityEnvironment(file_name="Banana_Windows_x86/Banana.exe")

    brain_name = env.brain_names[0]
    print('brain_name: %s' % brain_name)

    run(env, brain_name)
    env.close()
