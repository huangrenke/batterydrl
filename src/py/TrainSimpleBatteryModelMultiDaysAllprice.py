import os
import gym
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import time
from q_learning_bins import plot_running_avg
#from SimpleBatteryModelDef import SimpleBatterySimEnv
#from SimpleBatteryModelDefnoCappenalty import SimpleBatterySimEnv
from SimpleBatteryModelnoCapAllpriceDef import SimpleBatterySimEnv

from baselines import deepq
from baselines import logger
import baselines.common.tf_util as U

np.random.seed(19)

# create

Lmpfile = '../../TestData/2017_Zonal_LMP_LONGIL.csv'

storedData = "./storedData_nolastpenalty_multistartday_simu10days_fullprice"
if not os.path.exists(storedData):
    os.makedirs(storedData)

savedModel= "./previous_model"
if not os.path.exists(savedModel):
    os.makedirs(savedModel)
model_name = "simplebattery_model_" + storedData[13:]

def callback(lcl, glb):
    # stop training if reward exceeds -30
    #is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= -30.0
    #return is_solved
    episodes = 0
    if lcl['t'] > 0:
        step_rewards.append(lcl['rew'])
        step_actions.append(lcl['action'])
        step_observations.append(lcl['obs'])
        step_status.append(lcl['done'])
        step_starttime.append(lcl['starttime'])
        #step_durationtime.append(lcl['durationtime'])
        if lcl['t'] % 499 == 0:
            U.save_state(model_file)

def main(learning_rate):
   
    tf.reset_default_graph()    # to avoid the conflict with the existing parameters, but this is not suggested for reuse parameters
    simudays = 10
    env = SimpleBatterySimEnv(Lmpfile, 2, simudays)
    model = deepq.models.mlp([256,256])

    act = deepq.learn(
        env,
        q_func=model,
        lr=learning_rate,
        max_timesteps=1000000,
        buffer_size=50000,
        checkpoint_freq = 100,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving final model to simple_battery_model_lr_%s_100w.pkl" % (str(learning_rate)))
    act.save(savedModel + "/" + model_name + "_lr_%s_100w.pkl" % (str(learning_rate)))

#tf.reset_default_graph()    # to avoid the conflict the existnat parameters, but not suggested for reuse parameters
step_rewards = list()
step_actions = list()
step_observations = list()
step_status = list()
step_starttime = list()

check_pt_dir = "./SimpleBatteryModels"
if not os.path.exists(check_pt_dir):
    os.makedirs(check_pt_dir)

model_file = os.path.join(check_pt_dir, "simplebatterymodel")


import time
start = time.time()
dataname = storedData[13:]
for ll in [0.0001]:
    step_rewards = list()
    step_actions = list()
    step_observations = list()
    step_status = list()
    step_starttime = list()

    main(ll)

    np.save(os.path.join(storedData, "step_rewards_lr_%s_" % str(ll) + dataname), np.array(step_rewards))
    np.save(os.path.join(storedData, "step_actions_lr_%s_" % str(ll) + dataname), np.array(step_actions))
    np.save(os.path.join(storedData, "step_observations_lr_%s_" % str(ll) + dataname), np.array(step_observations))
    np.save(os.path.join(storedData, "step_status_lr_%s_" % str(ll) + dataname), np.array(step_status))
    np.save(os.path.join(storedData, "step_startday_lr_%s_" % str(ll) + dataname), np.array(step_starttime))

end = time.time()

print("total running time is %s" % (str(end - start)))


#np.save(os.path.join(storedData, "step_rewards_t"), np.array(step_rewards))
#np.save(os.path.join(storedData, "step_actions_t"), np.array(step_actions))
#np.save(os.path.join(storedData, "step_observations_t"), np.array(step_observations))
#np.save(os.path.join(storedData, "step_status_t"), np.array(step_status))
#np.save(os.path.join(storedData, "step_starttime_t"), np.array(step_starttime))
#np.save(os.path.join(storedData, "step_durationtime_t"), np.array(step_durationtime))

print("Finished!!")

'''
def test():
    act = deepq.load("power_model.pkl")
    done = False


    #for i in range(1):
    obs, done = env._validate(1,8,1.0,0.585), False
    episode_rew = 0
    actions = list()
    while not done:
        #env.render()
        action = act(obs[None])[0]
        #obs, rew, done, _ = env.step(act(obs[None])[0])
        obs, rew, done, _ = env.step(action)
        episode_rew += rew
    print("Episode reward", episode_rew)

    return actions
'''    
    

