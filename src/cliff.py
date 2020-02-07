# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:00:47 2020

@author: vyasraj
"""
import numpy as np
import random as random
import math as math

class QSAValue:
    def __init__ (self, name, alpha, gamma):
        self.name = name
        self.value = 0
        self.alpha = alpha
        self.gamma = gamma
        self.number_of_times_action = 0
            
    def UpdateQValue(self, reward, qsaprime):
        '''
        Q(S,A) <- Q(S,A) + alpha*(reward, discount*Q(s',A') - Q(S,A))
        '''
        self.value = self.value + self.alpha*(reward + self.gamma*qsaprime - self.value )
        self.number_of_times_action += 1
        return
    
    def GetValue(self):
        return self.value
    
    def GetName(self):
        return self.name
    
    def compare (self, compare_val):
        if (self.name == compare_val):
            return True
        else:
            return False 

    def GetActionValue(self):
        return self.GetValue()
    
    def GetNt(self):
        return self.number_of_times_action



class UCB:
    def __init__(self, c):
        self.c = c
    
    def GetNextAction(self, env):
        action, qvalue, qobj = env.GetUCBMaxQSA(self.c)
        #print("Max Q action is ..", action)
        
        if (action == "bug"):
            print("Screem....EpisonGreedy")
        # toss a random number
        return action
        

class EpisonGreedy:
    def __init__(self, e):
        self.e = e
    
    def GetNextAction(self, env):

        action, qvalue, qobj = env.GetMaxQSA()
        #print("Max Q action is ..", action)
        
        if (action == "bug"):
            print("Screem....EpisonGreedy")
        # toss a random number
        r2 = random.random()
        r2 = r2 + self.e
        if r2 > 1.0:
            valid_action = env.GetValidActions()
            l = len(valid_action)
            r= random.random()
            idx = int(np.floor(r*l))
            #print(idx)
            #print(".. episongreedy.. returning.. non-max action ", valid_action[idx])
            return valid_action[idx]
        else:
            return action
        

class Episode:
    def __init__(self, episode_num, algo):
        self.reward = 0
        self.episode = []
        self.episode_num = episode_num
        self.algo = algo
    
    def GenerateEpisode(self, env):
        s, r,c = env.InitStart()
        self.state = np.array([r,c])
        self.reward = np.array([])
        
        count =0

        while s != "end":
            count +=1
            #current state, get Next best action
            action = self.algo.GetNextAction(env)
            #print("Generate Episode: r=", r, ".c= ", c, ". s=", s , "..action= ", action)
            #env.PrintQSA()
            #if (count == 20):
            #    return
            qsa_action_current = env.GetQSAActionObj(r,c,action)
        
            #move to next state
            s, r, c, reward = env.GetNextStateReward(action)
            
            
            #Get next best action in next_state
            action2 = self.algo.GetNextAction(env)
            qsa_action_next = env.GetQSAActionObj(r,c,action2)
            action = action2
            
            #update current_q_sa
            qsa_action_current.UpdateQValue(reward,qsa_action_next.GetActionValue())
            env.UpdateStepCount()
            
            self.state = np.vstack((self.state,[r,c]))
            self.reward = np.append(self.reward, reward)
        
        
    def GetTotalReward(self):
        return sum(self.reward)
        

class CliffEnv:
    def __init__(self, alpha, discount):
        self.valid_action = np.array([
                                     ["RD" ,"LRD",  "LRD",  "LRD",  "LRD",  "LRD",  "LRD",  "LRD",  "LRD",  "LRD",  "LRD",  "LD"], 
                                     ["URD","LRUD", "LRUD", "LRUD", "LRUD", "LRUD", "LRUD", "LRUD", "LRUD", "LRUD", "LRUD", "LUD",],
                                     ["URD","LRUD", "LRUD", "LRUD", "LRUD", "LRUD", "LRUD", "LRUD", "LRUD", "LRUD", "LRUD", "LUD",],
                                     ["UR", "C",    "C",    "C",    "C",    "C",    "C",    "C",    "C",    "C",    "C",     "G"]])
        
        self.state_value = np.array([
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0]]
                )
        
        self.InitStart()
        self.alpha = alpha
        self.discount = discount
        self.row_count =4
        self.col_count =12
        self.qsa_list = [ ]
        for i in  np.arange(self.row_count):
            for j in  np.arange(self.col_count):
                valid_action = self.valid_action[i][j]
                l = len(valid_action)
                for a in np.arange(l):
                    s = str(i)+str(j) + valid_action[a] #name of Q(S,A)
                    q = QSAValue(s,alpha, discount)
                    self.qsa_list = np.append(self.qsa_list, q)
        
        self.total_steps = 0

    def UpdateStepCount(self):
        self.total_steps += 1
        
    def InitStart(self):
        self.current_row = 3
        self.current_col = 0
        return "start", self.current_row, self.current_col

    def GetValidActions(self):
        return self.GetValidActionsSpecific(self.current_row, self.current_col)
    
    def GetValidActionsSpecific(self, r, c):
        return self.valid_action[r][c]
    
    def GetNextRowCol(self, action):
        c = self.current_col
        r = self.current_row
        if (action == "U"):
            r = self.current_row - 1
        if (action == "R"):
            c = self.current_col +1
        if (action == "L"):
            c = self.current_col -1
        if (action == "D"):
            r = self.current_row + 1
        
        return r, c

    def GetQSAActionObj(self, row, col, action):
        s = str(row)+str(col)+action #Action name
        l = len(self.qsa_list)
        for i in np.arange(l):
            q = self.qsa_list[i]
            #print("len...", l, " i.. ", i, " .. name",q.GetName(), ".. s", s )
            if (q.compare(s) == True):
                return q
        print("GetQSAActionObj bug.. ", s)
        return "bug"
            
    def GetQValue(self, row, col, action):
        qsa_action_obj = self.GetQSAActionObj(row, col, action)
        return qsa_action_obj.GetActionValue(), qsa_action_obj
    

    def GetMaxQSASpecific(self, r, c):
        valid_action = self.GetValidActionsSpecific(r,c)
        l = len(valid_action)
        count = 0
        max_action = "bug"
        max_qsa = -100000
        max_qobj = "bug"
        while count < l:
            action = valid_action[count]
            #r,c = self.GetNextRowCol(action)
            q, qobj = self.GetQValue(r, c, action)
            #print(" .. GetMaxQSA..", valid_action, " ... ", action, "..max_qsa ", max_qsa, ".. q ", q)
            if (q >= max_qsa):
                max_action = action
                max_qsa = q
                max_qobj = qobj
            count += 1
            
        #print("returning ", max_action, "...", max_qobj.GetName())
        return max_action, max_qsa, max_qobj

    def GetMaxQSA(self):
        return self.GetMaxQSASpecific(self.current_row, self.current_col)

    def GetUCBMaxQSASpecific(self, qcb_constant, r, c):
        valid_action = self.GetValidActionsSpecific(r,c)
        l = len(valid_action)
        count = 0
        max_action = "bug"
        max_qsa = -1000000
        max_qobj = "bug"
        while count < l:
            action = valid_action[count]
            #r,c = self.GetNextRowCol(action)
            q, qobj = self.GetQValue(r, c, action)

            s = q
            if (qobj.GetNt() and self.total_steps):
                s = q + qcb_constant*math.sqrt(math.log(qobj.GetNt())/self.total_steps)

            if (s >= max_qsa):
                max_action = action
                max_qsa = q
                max_qobj = qobj
            count += 1
            
        #print("returning ", max_action, "...", max_qobj.GetName())
        return max_action, max_qsa, max_qobj
    
    def GetUCBMaxQSA(self, c):
        return self.GetUCBMaxQSASpecific(c, self.current_row, self.current_col)
        
                
    def GetNextStateReward(self, action):

        reward = -1
        if (self.valid_action[self.current_row][self.current_col] == "C"):
            self.InitStart()
            reward = -1
        
        if (self.valid_action[self.current_row][self.current_col] == "G"):
            reward = 0
            return "end", self.current_row, self.current_col, reward
        
        
        self.current_row, self.current_col = self.GetNextRowCol(action) 
        
        if (self.valid_action[self.current_row][self.current_col] == "C"):
            reward = -100
        #After all this compute we are in Goal state - terminate        
        return "continue", self.current_row, self.current_col, reward


    def PrintQSA(self):
        for i in  np.arange(self.row_count):
            for j in  np.arange(self.col_count):
                 action = "ddummy"
                 
                 if (action != "G"):
                     action, q, qobj = self.GetMaxQSASpecific(i, j)
                 else:
                     return
                 print("{0:.3f}".format(q), " ", end='')
                 #print(q, " ", end='')
            print("")
    
    def TestGetMaxQSASpecific(self, r, c):
        valid_action = self.GetValidActionsSpecific(r,c)
        l = len(valid_action)
        count = 0
        max_action = "bug"
        max_qsa = -100000
        max_qobj = "bug"
        while count < l:
            action = valid_action[count]
            #r,c = self.GetNextRowCol(action)
            q, qobj = self.GetQValue(r, c, action)
            #print(" .. GetMaxQSA..", valid_action, ".", action, " ... ", max_action, "..max_qsa ", max_qsa, ".. q ", q)
            if (q >= max_qsa):
                max_action = action
                max_qsa = q
                max_qobj = qobj
            count += 1
            
        #print("returning ", max_action, "...", max_qobj.GetName())
        return max_action, max_qsa, max_qobj
        
    def PrintPath(self):
        for i in  np.arange(self.row_count):
            for j in  np.arange(self.col_count):
                 action = "ddummy"
        
                 
                 if (action != "G"):
                     action, q, qobj = self.GetMaxQSASpecific(i, j)
                 else:
                     return
                 #print("{0:.3f}".format(q), " ", end='')
                 print(action, " ", end='')
            print("")   
            
if ("__main__" == __name__):
    #EpisonGreedy
    alpha = 0.1
    discount = 0.9

    total_episode = 500
    plot_epislon = True

    print("UCB : ", 10)
    c_10 = CliffEnv(alpha, discount)
    algo = UCB(10)
    ucb_reward_10=[]
    for i in np.arange(total_episode):
        e = Episode(i,algo)
        e.GenerateEpisode(c_10)
        #c.PrintQSA()
        #c.PrintPath()
        ucb_reward_10 = np.append(ucb_reward_10, e.GetTotalReward())

    print("UCB : ", 5)
    c_5 = CliffEnv(alpha, discount)
    algo = UCB(5)
    ucb_reward_5=[]
    for i in np.arange(total_episode):
        e = Episode(i,algo)
        e.GenerateEpisode(c_5)
        #c.PrintQSA()
        #c.PrintPath()
        ucb_reward_5 = np.append(ucb_reward_5, e.GetTotalReward())

    print("UCB : ", 1)
    c_1 = CliffEnv(alpha, discount)
    algo = UCB(1)
    ucb_reward_1=[]
    for i in np.arange(total_episode):
        e = Episode(i,algo)
        e.GenerateEpisode(c_1)
        #c.PrintQSA()
        #c.PrintPath()
        ucb_reward_1 = np.append(ucb_reward_1, e.GetTotalReward())
        
    print("UCB : ", 0.5)
    c_05 = CliffEnv(alpha, discount)
    algo = UCB(0.5)
    ucb_reward_05=[]
    for i in np.arange(total_episode):
        e = Episode(i,algo)
        e.GenerateEpisode(c_05)
        #c.PrintQSA()
        #c.PrintPath()
        ucb_reward_05 = np.append(ucb_reward_05, e.GetTotalReward())

    print("UCB : ", 0.1)
    c_01 = CliffEnv(alpha, discount)
    algo = UCB(0.1)
    ucb_reward_01=[]
    for i in np.arange(total_episode):
        e = Episode(i,algo)
        e.GenerateEpisode(c_01)
        #c.PrintQSA()
        #c.PrintPath()
        ucb_reward_01 = np.append(ucb_reward_01, e.GetTotalReward())

    print("Executing Epsilon greedy: ", 0.5)
    c_05 = CliffEnv(alpha, discount)
    algo = EpisonGreedy(0.5)
    reward_per_episode_05=[]
    for i in np.arange(total_episode):
        e = Episode(i,algo)
        e.GenerateEpisode(c_05)
        #c.PrintQSA()
        #c.PrintPath()
        reward_per_episode_05 = np.append(reward_per_episode_05, e.GetTotalReward())
    
    print("Executing Epsilon greedy: ", 0.2)
    c_02 = CliffEnv(alpha, discount)
    algo = EpisonGreedy(0.2)
    reward_per_episode_02=[]
    for i in np.arange(total_episode):
        e = Episode(i,algo)
        e.GenerateEpisode(c_02)
        #c.PrintQSA()
        #c.PrintPath()
        reward_per_episode_02 = np.append(reward_per_episode_02, e.GetTotalReward())

    print("Executing Epsilon greedy: ", 0.1)
    c_02 = CliffEnv(alpha, discount)
    algo = EpisonGreedy(0.1)
    reward_per_episode_01=[]
    for i in np.arange(total_episode):
        e = Episode(i,algo)
        e.GenerateEpisode(c_02)
        #c.PrintQSA()
        #c.PrintPath()
        reward_per_episode_01 = np.append(reward_per_episode_01, e.GetTotalReward())

    print("Executing Epsilon greedy: ", 0.05)
    c_005 = CliffEnv(alpha, discount)
    algo = EpisonGreedy(0.05)
    reward_per_episode_005=[]
    for i in np.arange(total_episode):
        e = Episode(i,algo)
        e.GenerateEpisode(c_005)
        #c.PrintQSA()
        #c.PrintPath()
        reward_per_episode_005 = np.append(reward_per_episode_005, e.GetTotalReward())
    
    
    
    import matplotlib.pyplot as plt
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    plt.plot(ucb_reward_5, label="c=5")
    plt.plot(reward_per_episode_01, label="e=0.1")
    plt.xlabel('total number of episodes')
    plt.ylabel('total rewards')
    plt.title('Cliff walking: comparision of UCB AND epsilon greedy - alpha = 0.1, discount = 0.9')
    plt.legend(loc="bottom right")
    plt.grid(True)
    plt.show()

"""
    if (plot_epislon):
        plt.plot(reward_per_episode_05, label="e=0.5")
        plt.plot(reward_per_episode_02, label="e=0.2")
        plt.plot(reward_per_episode_01, label="e=0.1")
        plt.plot(reward_per_episode_005, label="e=0.05")
        plt.xlabel('total number of episodes')
        plt.ylabel('total rewards')
        plt.title('Cliff walking: Epsilon greedy strategy - alpha = 0.1, discount = 0.9')
        plt.legend(loc="bottom right")
        plt.grid(True)
        plt.show()
    else:
        plt.plot(ucb_reward_05, label="c=0.5")
        plt.plot(ucb_reward_01, label="c=0.1")
        plt.plot(ucb_reward_5, label="c=5")
        plt.plot(ucb_reward_10, label="c=10")
        plt.plot(ucb_reward_1, label= "c=1")
        plt.ylabel('total rewards')
        plt.xlabel('total number of episodes')
        plt.title('Cliff walking: UCB greedy strategy, alpha = 0.1, discount = 0.9')
        plt.legend(loc="bottom right")
        plt.grid(True)
        plt.show()
"""
