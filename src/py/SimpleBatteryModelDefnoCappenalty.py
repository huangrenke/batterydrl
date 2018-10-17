import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd

import random
import collections

# logging
logger = logging.getLogger(__name__)

class SimpleBatterySimEnv(gym.Env):
    metadata = {

    }

    #LMP_file =""
    step_time = 1
    action_type = 'discrete'

    def __init__(self, LMP_file, istartday):

        # internal states and variables for the battery model
        #print ('enter the __init__ function of SimpleBatterySimEnv')
        
        self.BatteryEt = 4*0.5          # SOC of the battery, MWh
        self.BatteryPt = 0.0            # charge/discharge rate of the battery, + for charge, - for discharge, MW
        self.BatteryCap = 4.0           # Capacity of the battery, MWh
        self.eta_p = 0.95               # Generation, discharge
        self.eta_n = 0.7955             # Pumping, charging
        self.maxPt = 1.0
        self.chargeBoundaryPenalty = 1000
        self.orgSOCbiaspenalty = 0.0
        self.simuhours = 0
        self.simudays = 3
        self.simustartday = istartday

        self._LMP_file = LMP_file

        # read the full year LMP price from the csv file
        self.LMP_df = pd.read_csv(self._LMP_file, header = 0)
        self.LMP_series = self.LMP_df['DAM Zonal LBMP']
        self.LMP_1dim = self.LMP_series.as_matrix()
        self.LMP_days = self.LMP_1dim.reshape(365,24)

        self.currentLMP = self.LMP_days[self.simustartday, 0]

        from gym import spaces

        self.observation_Et_length = 24
        self.observation_Pt_length =  24
        self.observation_LMP_length =  24
        action_space_dim = 21

        self.actionspace_to_Pt_mapper = np.linspace(-self.maxPt, self.maxPt, action_space_dim)

        self.observation_Et_queue = collections.deque(maxlen=self.observation_Et_length)
        self.observation_Pt_queue = collections.deque(maxlen=self.observation_Pt_length)
        self.observation_LMP_pastoneday_queue = collections.deque(maxlen=self.observation_LMP_length)
        self.observation_LMP_forecastoneday = self.LMP_days[self.simustartday, :]

        for i in range (0, self.observation_Et_length):
            self.observation_Et_queue.append(self.BatteryEt)
            self.observation_Pt_queue.append(self.BatteryPt)
            self.observation_LMP_pastoneday_queue.append(self.currentLMP)   # this may cause confuse for the AI????

        #define action and observation spaces
        self.action_space      = spaces.Discrete(action_space_dim)
        self.observation_space = spaces.Box(-999,999,shape=(self.observation_Et_length + self.observation_Pt_length + self.observation_LMP_length*2,))

        self._seed()

        #TOOD get the initial states
        self.state = np.array([self.observation_Et_queue, self.observation_Pt_queue, self.observation_LMP_pastoneday_queue, self.observation_LMP_forecastoneday])
        
        self.steps_beyond_done = None
        self.restart_simulation = True

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        #print(type(action))

        # mapping the action from AI to the charge/discharge rate
        self.BatteryPt = float(self.actionspace_to_Pt_mapper[action])
        #print ('self.BatteryPt value is %f, type is %s'%(self.BatteryPt, type(self.BatteryPt)))

        iday = int(self.simuhours / 24)
        ihour = int (self.simuhours % 24)
        
        #print ('self.simustartday is %d, iday is %d, and ihour is %d'%(self.simustartday, iday, ihour))
        self.currentLMP = self.LMP_days[self.simustartday + iday, ihour]

        # based on the charge/discharge rate, compute the SOC of the battery for the next hour
        BatteryEt_prev = self.BatteryEt
        if self.BatteryPt >= 0.0:
            self.BatteryEt = self.BatteryEt + self.BatteryPt * self.eta_n  # charging eta
        else :
            self.BatteryEt = self.BatteryEt + self.BatteryPt / self.eta_p ## check to make sure this is correct
        
        bchargepenlty = False
        
        
        if self.BatteryEt > self.BatteryCap:
            bchargepenlty = True
            reward_reduce = -self.currentLMP * (self.BatteryEt - self.BatteryCap)/self.eta_n
            self.BatteryEt = self.BatteryCap
            
        elif self.BatteryEt < 0.0:
            bchargepenlty = True
            reward_reduce = -self.currentLMP * (self.BatteryEt - 0.0)*self.eta_p
            self.BatteryEt = 0.0
            
        # compute reward:
        reward = -self.currentLMP*self.BatteryPt
        if bchargepenlty:  # if the charge/discharge makes the battery exceeds its capacity or less than 0 MWh
            #print ('time step is %d, reward is %f, reward_reduce is %f, BatteryEt_prev is %f, BatteryPt is %f, BatteryEt_now is %f'%(self.simuhours, reward, reward_reduce, BatteryEt_prev, self.BatteryPt, self.BatteryEt))
            reward -= reward_reduce              

        self.simuhours += 1
        
        #print ('before observation_Et_queue')
        #print(self.observation_Pt_queue)
        #print ('self.BatteryPt value is %f, type is %s'%(self.BatteryPt, type(self.BatteryPt)))
        self.observation_Et_queue.append(self.BatteryEt)
        self.observation_Pt_queue.append(self.BatteryPt)
        self.observation_LMP_pastoneday_queue.append(self.currentLMP)
        self.observation_LMP_forecastoneday = self.LMP_1dim[self.simustartday*24+self.simuhours : self.simustartday*24+self.simuhours+24]
        #print ('after observation_Et_queue')
        #print(self.observation_Pt_queue)

        # convert it from Java_collections array to native Python array
        self.state = np.array([self.observation_Et_queue, self.observation_Pt_queue, self.observation_LMP_pastoneday_queue, self.observation_LMP_forecastoneday])
        #print ('self.state:')
        #print(self.state)
        #teststate = self.state.ravel()
        
        #print ('self.state 2:')
        #print(teststate)

        if self.simuhours == self.simudays*24 - 1:
            done = True
            reward -= self.orgSOCbiaspenalty * abs(self.BatteryEt - 0.5*self.BatteryCap)
        else:
            done = False

        return self.state.ravel(), reward, done, {}

    def _reset(self):

        self.BatteryEt = 4*0.5  # SOC of the battery, MWh
        self.BatteryPt = 0.0   # charge/discharge rate of the battery, + for charge, - for discharge, MW
        #self.currentLMP = 1.0
        self.simuhours = 0

        # reset need to randomize the start day for the three-day simulation
        selectdays = [3,7,12,33,43,62,69,80,91,97,98,108,116,123,126,136,144,153,161,174,192,199,225,230,234,247,261,274,281,287,295,305,313,320,327,332,345,348,357,350,360]
        simustartday_idx = np.random.randint(0,41) # an integer, in be in the preset days?
        self.simustartday = selectdays[simustartday_idx]
        #print('--------- reset:  simustartday is %d -------------'%(self.simustartday))
        
        self.currentLMP = self.LMP_days[self.simustartday, 0]

        self.observation_Et_queue.clear()
        self.observation_Pt_queue.clear()
        self.observation_LMP_pastoneday_queue.clear()
        self.observation_LMP_forecastoneday = self.LMP_days[self.simustartday, :]

        for i in range (0, self.observation_Et_length):
            self.observation_Et_queue.append(self.BatteryEt)
            self.observation_Pt_queue.append(self.BatteryPt)
            self.observation_LMP_pastoneday_queue.append(self.currentLMP)   # this may cause confuse for the AI????

        self.state = np.array([self.observation_Et_queue, self.observation_Pt_queue, self.observation_LMP_pastoneday_queue, self.observation_LMP_forecastoneday])
        
        self.steps_beyond_done = None
        self.restart_simulation = True

        return self.state.ravel(), self.simustartday, self.simustartday

    # init the system with a specific simulation start day
    def _validate(self, istartday):

        self.BatteryEt = 4*0.5  # SOC of the battery, MWh
        self.BatteryPt = 0.0   # charge/discharge rate of the battery, + for charge, - for discharge, MW
        
        self.simuhours = 0
        self.simustartday = istartday
        
        self.currentLMP = self.LMP_days[self.simustartday, 0]

        self.observation_Et_queue.clear()
        self.observation_Pt_queue.clear()
        self.observation_LMP_pastoneday_queue.clear()
        self.observation_LMP_forecastoneday = self.LMP_days[self.simustartday, :]

        for i in range (0, self.observation_Et_length):
            self.observation_Et_queue.append(self.BatteryEt)
            self.observation_Pt_queue.append(self.BatteryPt)
            self.observation_LMP_pastoneday_queue.append(self.currentLMP)   # this may cause confuse for the AI????

        # convert it from Java_collections array to native Python array
        self.state = np.array([self.observation_Et_queue, self.observation_Pt_queue, self.observation_LMP_pastoneday_queue, self.observation_LMP_forecastoneday])
        
        self.steps_beyond_done = None
        self.restart_simulation = True

        return self.state.ravel()

    # def _render(self, mode='human', close=False):
