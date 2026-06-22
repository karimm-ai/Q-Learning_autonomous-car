from env import proj
import numpy as np
import random, time
import pickle
import matplotlib.pyplot as plt


def max_action(q_table, state, actions):
    values = np.array([q_table[state[0], state[1], a] for a in actions])
    action = np.argmax(values)
    return actions[action]


def display_results(q_table):
    env = proj(render_mode='human')
    observation = env.reset()
    done = False

    while not done:
        action = max_action(q_table, observation, env.action_space)

        print(
            f"State: {observation} | "
            f"Action: {action}"
        )

        observation, reward, done = env.step(action)
    
    time.sleep(3)
    env.close()
    print(f"Reward: {reward}")


if __name__ == '__main__':
    env = proj()
    # agent's hyperparameters
    ALPHA = 0.1 # Learning Rate
    GAMMA = 0.9 # Discount Factor
    EPS = 0.85 # For Exploration-Exploitation.

    Q = {}

    for row_state in range(env.observation_space.low[0], env.observation_space.high[0] + 1):
        for col_state in range(env.observation_space.low[0], env.observation_space.high[0] + 1):
            for action in env.action_space:
                Q[row_state, col_state, action] = 0

    episodes = 10000
    totalRewards = np.zeros(episodes)

    for i in range(episodes):
        if i % 10 == 0:
            print('starting game ', i)
        done = False
        episode_rewards = 0
        observation = env.reset()

        while not done:
            random_float = random.uniform(0, 1)
            if random_float >= EPS:
                action = random.choice(env.action_space)
            else:
                action = max_action(Q, observation, env.action_space) # Choose the action based on the q-table's values.
                
            new_observation, reward, done = env.step(action)
            episode_rewards += reward

            new_action = max_action(Q, new_observation, env.action_space)
            
            Q[observation[0], observation[1], action] = (1-ALPHA)*Q[observation[0], observation[1], action] + ALPHA*(reward + \
                        GAMMA*Q[new_observation[0], new_observation[1], new_action])
            observation = new_observation
            
            if env.steps > 50:
                done = True
        
        totalRewards[i] = episode_rewards
                
        print(f"REWARD for game:{i} = " + str(episode_rewards))
    
    print("\nTotal Rewards:\n")
    plt.plot(totalRewards)
    plt.savefig("total_rewards.png")

    with open("Q_Table.pkl", "wb") as f:
        pickle.dump(Q, f)

    display_results(Q)