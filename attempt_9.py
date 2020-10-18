import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten

import gym, time, cv2


# variables & CONSTANTS
env                 = gym.make("MsPacman-v0")
TOTAL_ITERATIONS    = 10_000
TRAINING_SIZE       = 15_000
MEMORY_SIZE         = 15_000
FRAME_STEP          = 5
EPSILON_DECAY       = .98
AVG_EXCEED          = .10

epsilon             = .4
X_train             = []
y_train             = []
X_memory            = []
y_memory            = []
score_list          = [250]

def basic_preprocess(obs):
    obs = obs[0:172]
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = cv2.resize(obs, (80, 80))
    obs = obs/255.0#(obs-127.5)/127.5
    return obs

def learned_preprocess(obs):
    #show = False if np.random.uniform()>.1 else True
    f_1 = obs[0]
    f_2 = obs[1]
    f_3 = obs[2]
    #if show:
    #    plt.imshow(f_3)
    #    plt.show()

    obs = (f_1 - f_2) + (f_2-f_3) + (f_1-f_3)
    obs[np.where(obs != 0)] = 1
    obs[np.where(obs == 0)] = 0
    #f_3  = obs * f_3
    #if show:
    #    plt.imshow(f_3)
    #    plt.show()
    #f_3  = f_3.flatten()
    return np.stack((obs, f_3), axis=-1)


# Neural Network
model = Sequential()
model.add(Conv2D(32, (3,3), activation="tanh", input_shape=(80,80,2)))
model.add(BatchNormalization())

model.add(Conv2D(16, (3,3), activation="tanh"))
model.add(BatchNormalization())

model.add(Conv2D(8, (5,5), activation="tanh"))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(64, activation="relu"))
model.add(Dropout(.2))

model.add(Dense(32, activation="relu"))
model.add(Dropout(.2))

model.add(Dense(32, activation="relu"))
model.add(Dropout(.2))

model.add(Dense(4, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


def create_next_data_set(use_model=True):
    global env, epsilon, model, TRAINING_SIZE, X_train, y_train, X_memory, y_memory
    #print(np.mean(score_list) * (1+AVG_EXCEED))
    #print(np.mean(score_list[-1_000:])*(1+AVG_EXCEED))
    required_score = np.max([np.mean(score_list) * (1+AVG_EXCEED), np.mean(score_list[-1_000:])*(1+AVG_EXCEED)])
    while len(y_train) < TRAINING_SIZE:
        env.reset()
        info                = {"ale.lives":3}
        observation_list    = []
        prev_observation    = []
        game_action         = []
        game_data           = []
        action              = 0
        score               = 0
        frame               = 0
        #if not len(y_train)%15:
        print(f"            -> {len(y_train)} / {TRAINING_SIZE}     required_score: {required_score}", end='\r')
        while info['ale.lives'] == 3:
            if frame%FRAME_STEP == 0:
                if np.random.uniform() <= epsilon or not use_model or len(prev_observation) == 0:
                    action = np.random.randint(0,4)
                else:
                    #print(model.predict(np.array([prev_observation])))
                    action = np.argmax(model.predict(np.array([prev_observation])))

            observation, reward, done, info = env.step(action+1)
            observation_list.append(basic_preprocess(observation))
            if len(observation_list) > 3:
                del observation_list[0]

            score += reward

            if frame%FRAME_STEP == 0 and len(prev_observation) != 0:
                game_data.append(prev_observation)
                game_action.append(action)

            if (frame+1)%FRAME_STEP==0 and len(observation_list)==3:
                prev_observation = learned_preprocess(observation_list)

            frame += 1

        score_list.append(score)
        if score > required_score or (use_model == False and score>250):
            # append to training data_set
            unique, counts  = np.unique(np.asarray(game_action), return_counts=True)
            min_count       = np.min(counts)
            action_allowed  = np.zeros(4)+min_count

            for g_data, g_action in zip(game_data, game_action):
                if action_allowed[g_action] > 0:
                    label = np.zeros(4)
                    label[g_action] = 1
                    X_train.append(g_data)
                    y_train.append(label)

                    if np.random.uniform() <= .075:
                        if len(y_memory) <= MEMORY_SIZE:
                            X_memory.append(g_data)
                            y_memory.append(label)
                        else:
                            rndm_pos = np.random.randint(0, len(y_memory))
                            X_memory[rndm_pos] = g_data
                            y_memory[rndm_pos] = label

                    action_allowed[g_action] = action_allowed[g_action] - 1


print("Generating initial data-set")
create_next_data_set(use_model=False)
c_iteration = 0
while 1:
    # train the model
    unique, counts  = np.unique(np.argmax(np.asarray(y_train), axis=1), return_counts=True)
    total           = np.sum(counts)
    u_a_dict        = dict(zip(unique, counts/total))
    print(f"--------------------------------- {c_iteration}   /{TOTAL_ITERATIONS} ---------------------------------")
    print("Score Analysis")
    print(f"    -> overall_avg_score: {np.mean(score_list)}\
          \n    -> past_50_avg_score: {np.mean(score_list[-50:])}\
          \n    -> past_1_000_avg_score: {np.mean(score_list[-1_000:])}\
          \n    -> max: {np.max(score_list)}")
    print(u_a_dict)
    print("Training Supervision")
    print(f"    -> epsilon:                 {epsilon}\
          \n    -> len train_data:          {len(y_train)}\
          \n    -> len memory_data:         {len(y_memory)}\
          \n    -> a_distribution_train:    {np.sum(y_train, axis=0).astype(int)}\
          \n    -> a_distribution_memory:   {np.sum(y_memory, axis=0).astype(int)}")
    # train the model
    model.fit(np.asarray(X_train+X_memory), np.asarray(y_train+y_memory), epochs=2)
    X_train = []
    y_train = []
    epsilon *= (EPSILON_DECAY)

    if not c_iteration%25:
        model.save(f"attempt_9_models/MsPacman-v0_{c_iteration}.model")
    c_iteration += 1

    if c_iteration>=TOTAL_ITERATIONS:
        break
    # generate the new data_set
    create_next_data_set()






















#
