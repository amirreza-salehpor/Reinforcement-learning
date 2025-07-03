# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam


#%%



class DQNAgent:
    def __init__(self, state_size, action_size, TRAIN):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99    # discount rate
        if TRAIN:
            self.epsilon = 1.0 # exploration rate
        else:
            self.epsilon = 0.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch, score):
        minibatch = random.sample(self.memory, batch)
        states, targets = [], []
    
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            states.append(state[0])
            targets.append(target)
    
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    TRAIN = False
    EPISODES = 10
    LOAD = False
    SAVE = False
    env = gym.make('MountainCar-v0', render_mode='human')
#    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, TRAIN)
    if LOAD:
        agent.load("model.weights.h5")
    done = False
    batch = 300
    scores = deque(maxlen=3000)
    total = 0
    mean = 0
    
    for e in range(EPISODES):
        state = env.reset()
        state = state[0]
        state = np.reshape(state, [1, state_size])
        score = 0
        for time in range(300):
            if not TRAIN:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)

            # Reward shaping: bonus for reaching the goal
            if next_state[0] >= 0.5:
                reward = 100
            
            # Additional shaping: encourage movement (momentum)
            if next_state[0] > state[0][0]:
                reward += 0.1  # moved right
            elif next_state[0] < state[0][0]:
                reward += 0.05  # moved left
                
                
            next_state = np.reshape(next_state, [1, state_size])            
            agent.remember(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if (done or time == 299):
                scores.append(score)
                mean = sum(scores)/len(scores)
                print("episode: {}/{}, score: {:.5}, e: {:.2}, average: {:.5}"
                      .format(e, EPISODES, score, agent.epsilon, mean))
                if len(agent.memory) > 300:
                    if TRAIN:
                        agent.replay(batch, score)
                break
if SAVE:
    agent.save("model.weights.h5")
#%%
env.close()
