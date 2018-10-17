import logging
import math
import numpy as np
import pandas as pd

import random
from collections import deque

# logging
logger = logging.getLogger(__name__)

class SimpleBatterySimEnvwogym(object):
    metadata = {

    }

    LMP_file =""
    step_time = 1
    action_type = 'discrete'

    def __init__(self, LMP_file, istartday):
        
        # internal states and variables for the battery model
        
        print ('enter into the init of SimpleBatterySimEnvwogym')
        
        self.BatteryEt = 4*0.5          # SOC of the battery, MWh
        self.BatteryPt = 0.5            # charge/discharge rate of the battery, + for charge, - for discharge, MW
        self.BatteryCap = 4.0           # Capacity of the battery, MWh
        self.eta_p = 0.95               # Generation, discharge 
        self.eta_n = 0.7955             # Pumping, charging
        self.maxPt = 1.0
        self.chargeBoundaryPenalty = 1000
        self.orgSOCbiaspenalty = 1000
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

        observation_Et_length = 24
        observation_Pt_length =  24
        observation_LMP_length =  24
        action_space_dim = 21
        
        self.actionspace_to_Pt_mapper = np.linspace(-self.maxPt, self.maxPt, action_space_dim)
        
        self.observation_Et_queue = deque(maxlen=observation_Et_length)
        self.observation_Pt_queue = deque(maxlen=observation_Pt_length)
        self.observation_LMP_pastoneday_queue = deque(maxlen=observation_LMP_length)
        self.observation_LMP_forecastoneday = self.LMP_days[self.simustartday, :]

        for i in range (0, observation_Et_length):
            self.observation_Et_queue.append(self.BatteryEt)
            self.observation_Pt_queue.append(self.BatteryPt)
            self.observation_LMP_pastoneday_queue.append(self.currentLMP)   # this may cause confuse for the AI????       
        
        self.observations = np.array([self.observation_Et_queue, self.observation_Pt_queue, self.observation_LMP_pastoneday_queue, self.observation_LMP_forecastoneday])

        print ('end ini')
        
    
    '''
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        #print(type(action))
        
        # mapping the action from AI to the charge/discharge rate
        self.BatteryPt = actionspace_to_Pt_mapper[action]
        
        iday = self.simuhours / 24
        ihour = self.simuhours % 24
        self.currentLMP = self.LMP_days[self.simustartday + iday, ihour]
        
        # based on the charge/discharge rate, compute the SOC of the battery for the next hour
        if self.BatteryPt >= 0.0:
            self.BatteryEt = self.BatteryEt + self.BatteryPt * self.eta_n
        else :
            self.BatteryEt = self.BatteryEt + self.BatteryPt / self.eta_p ## check to make sure this is correct
            
        if self.BatteryEt > self.BatteryCap:
            self.BatteryEt = self.BatteryCap
            bchargepenlty = True
        elif self.BatteryEt < 0.0:  
            self.BatteryEt = 0.0
            bchargepenlty = True
            
        self.simuhours += 1
        
        self.observation_Et_queue.append(self.BatteryEt)
        self.observation_Pt_queue.append(self.BatteryPt)
        self.observation_LMP_pastoneday_queue.append(self.currentLMP)
        self.observation_LMP_forecastoneday = self.LMP_1dim[self.simustartday*24+self.simuhours : self.simustartday*24+self.simuhours+24].reshape(1,24)

        # convert it from Java_collections array to native Python array
        self.observations = np.array([self.observation_Et_queue, self.observation_Pt_queue, self.observation_LMP_pastoneday_queue, self.observation_LMP_forecastoneday])
        
        # compute reward:
        reward = -self.currentLMP*self.BatteryPt
        if bchargepenlty:  # if the charge/discharge makes the battery exceeds its capacity or less than 0 MWh
            reward -= self.chargeBoundaryPenalty
            
        if self.simuhours == self.simudays*24 - 1:
            done = True
            reward -= self.orgSOCbiaspenalty * abs(self.BatteryEt - 0.5*self.BatteryCap)
        else:
            done = False

        return self.observations.ravel(), reward, done, {}

    def _reset(self):

        self.BatteryEt = 4*0.5  # SOC of the battery, MWh
        self.BatteryPt = 0.5   # charge/discharge rate of the battery, + for charge, - for discharge, MW
        self.currentLMP = 1.0
        self.simuhours = 0
        
        # reset need to randomize the start day for the three-day simulation
        self.simustartday = np.random.randint(0,10) # an integer, in be in the preset days?
        
        self.observation_Et_queue.clear()
        self.observation_Pt_queue.clear()
        self.observation_LMP_pastoneday_queue.clear()
        self.observation_LMP_forecastoneday = self.LMP_days[self.simustartday, :]

        self.observations = self.observations = np.array([self.observation_Et_queue, self.observation_Pt_queue, self.observation_LMP_pastoneday_queue, self.observation_LMP_forecastoneday])

        return np.array(self.observations).ravel(), istartday

    # init the system with a specific simulation start day
    def _validate(self, istartday):

        self.BatteryEt = 4*0.5  # SOC of the battery, MWh
        self.BatteryPt = 0.5   # charge/discharge rate of the battery, + for charge, - for discharge, MW
        self.currentLMP = 1.0
        self.simuhours = 0
        self.simustartday = istartday
        
        self.observation_Et_queue.clear()
        self.observation_Pt_queue.clear()
        self.observation_LMP_pastoneday_queue.clear()
        self.observation_LMP_forecastoneday = self.LMP_days[self.simustartday, :]

        # convert it from Java_collections array to native Python array
        self.observations = np.array([self.observation_Et_queue, self.observation_Pt_queue, self.observation_LMP_pastoneday_queue, self.observation_LMP_forecastoneday])

        return np.array(self.observations).ravel()
        
        '''

    # def _render(self, mode='human', close=False):