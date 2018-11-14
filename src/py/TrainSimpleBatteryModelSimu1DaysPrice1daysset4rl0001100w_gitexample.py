import os
import gym
import sys
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import time
#from q_learning_bins import plot_running_avg
#from SimpleBatteryModelDef import SimpleBatterySimEnv
#from SimpleBatteryModelDefnoCappenaltyMultiDay import SimpleBatterySimEnv
from SimpleBatteryModelFinalDef import SimpleBatterySimEnv


from baselines import deepq
from baselines import logger
import baselines.common.tf_util as U

# this training file train a smaller day sets, each episode with 1 day simulation time, the input for AI includes forecasted 1 day price information

# this training will expand the train day sets by extending the second and third days of the original selected training days

np.random.seed(19)

# define parameters:
trainmaxsteps = 1000000
learning_rate = 0.0001
nsimudays = 1
npricedays = 1
dataset_interval = 4
dataset_start = 0
batteryEtini = 0.0

Lmpfile = '../../TestData/2017_Zonal_LMP_LONGIL.csv'

#ob_act_dim_ary = ipss_app.initStudyCase(case_files_array , dyn_config_file, rl_config_file)

storedData = "./storedData_nolastpenalty_BaEtIni%.1f_simu%dDays_price%dDays_extendsetdays_smalltrainset%d" %(batteryEtini,nsimudays, npricedays, dataset_interval)
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

def main(learning_rate, trainmaxsteps, nsimudays, npricedays):

    tf.reset_default_graph()    # to avoid the conflict with the existing parameters, but this is not suggested for reuse parameters

    basetraindaysets = [3,7,12,33,43,62,69,80,91,97,98,108,116,123,126,136,144,153,161,174,192,199,225,230,234,247,261,274,281,287,295,305,313,320,327,332,345,348,357,350,360]
    basedatasetlen = len(basetraindaysets)
    selectdays = basetraindaysets[dataset_start : basedatasetlen : dataset_interval]
    selectdaysfortrain = []
    for iday in selectdays:
        selectdaysfortrain.append(iday)
        selectdaysfortrain.append(iday+1)
        selectdaysfortrain.append(iday+2)
        
    startday = 3
    #nsimudays = 1
    #npricedays = 1
    print ('---------------selectdaysfortrain: ---------------')
    print (selectdaysfortrain)
    env = SimpleBatterySimEnv(Lmpfile, batteryEtini, startday, nsimudays, npricedays, selectdaysfortrain)
    model = deepq.models.mlp([256,256])
    
    act = deepq.learn(
        env,
        q_func=model,
        lr=learning_rate,
        max_timesteps=trainmaxsteps,
        buffer_size=50000,
        checkpoint_freq = 100,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    
    print("Saving final model to %s_lr_%s_%dw.pkl" % (model_name, str(learning_rate), int(trainmaxsteps/10000)))
    act.save( savedModel + "/" + model_name + "_lr_%s_%dw.pkl" % (str(learning_rate), int(trainmaxsteps/10000)) )

#aa._act_params


#tf.reset_default_graph()    # to avoid the conflict the existnat parameters, but not suggested for reuse parameters
step_rewards = list()
step_actions = list()
step_observations = list()
step_status = list()
step_starttime = list()

check_pt_dir = "./Checkpoint_" + storedData[13:] + "_lr%s_step%dw" %(str(learning_rate), int(trainmaxsteps/10000)) #"./SimpleBatteryModels"
if not os.path.exists(check_pt_dir):
    os.makedirs(check_pt_dir)

model_file = os.path.join(check_pt_dir, "lr%s_step%dw"%(str(learning_rate), int(trainmaxsteps/10000)))


import time
start = time.time()
dataname = "_step%dw" %(int(trainmaxsteps/10000))
for ll in [learning_rate]:
    step_rewards = list()
    step_actions = list()
    step_observations = list()
    step_status = list()
    step_starttime = list()

    main(ll, trainmaxsteps, nsimudays, npricedays)

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


