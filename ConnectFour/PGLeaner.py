import sys

import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

from agent import Agent
from ConnectFour import ConnectFour

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class PGLearner:
    
    def __init__(self, gamma, learning_rate, rho, epsilon, load_network):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
    
        self.agent = Agent(load_network)
        self.testAgent = Agent(False)
        self.testPrediction_fn = self.testAgent.p_fn
        
        input_var = self.agent.input_var

        a_n = T.ivector()       # Vector of actions
        r_n = T.fvector()       # Vector of rewards
        
        N = input_var.shape[0]
        
        prediction = self.agent.p_prediction
        loss = -T.log(prediction[T.arange(N), a_n]).dot(r_n) / N
        
        params = lasagne.layers.get_all_params(self.agent.policy, 
                                                    trainable=True)
        
        updates = lasagne.updates.adadelta(loss, params, 
                                          learning_rate = self.learning_rate,
                                          rho = self.rho,
                                          epsilon = self.epsilon)

        self.prediction_fn = self.agent.p_fn
        
        self.x1 = np.zeros((1,4,7,7))
        self.x2 = np.zeros((1,4,7,7))
        
        self.game = ConnectFour(1,-1)
        
        print("Compiling Training Function...")
        self.train_fn = theano.function([input_var, a_n, r_n],
                                        [], 
                                        updates=updates,
                                        allow_input_downcast=True)
        print("Compiling Done!")
        
    def updateInputs(self, x):
        self.x1[0,3,:,:] = self.x1[0,2,:,:]
        self.x1[0,2,:,:] = self.x1[0,1,:,:]
        self.x1[0,1,:,:] = self.x1[0,0,:,:]
        self.x1[0,0,:,:] = x
        
        self.x2[0,3,:,:] = self.x2[0,2,:,:]
        self.x2[0,2,:,:] = self.x2[0,1,:,:]
        self.x2[0,1,:,:] = self.x2[0,0,:,:]
        self.x2[0,0,:,:] = -x
        
    def resetInputs(self):
        self.x1 = np.zeros((1,4,7,7),np.int8)
        self.x2 = np.zeros((1,4,7,7),np.int8)
        
    def checkGameEnd(self, inp):
        if inp == 1 or inp == 2 or inp == 3:
            return True
        else:
            return False
            
    def evaluate(self, evalSize, t, epoch, epochs):
        count = 0
        for epi in range(evalSize):
            self.resetInputs()
            while True:
                nextX = self.game.getGame()
                self.updateInputs(nextX)
                print self.x1
                
                action = self.agent.getDeterministicAction(self.x1)
                state = self.game.play(action, 1)
                
                if state == -2:
                    break
            
                game_end = self.checkGameEnd(state)
                
                if game_end == True:
                    break
                else:
                    nextX = self.game.getGame()
                    self.updateInputs(nextX)
                    print self.x2
                    action = self.agent.getDeterministicAction(self.x2)
                    state = self.game.play(action, 2)
                
                    game_end = self.checkGameEnd(state)
                    
                    if game_end == True:
                        break                                

            if state == 1:
                count += 1
                
            self.game.newGame()
            
        self.log(t, count)
        print("Game {} of {} took {:.3f}s and won {} out of {} games".format(
                            epoch + 1, epochs, t, count, evalSize))
            
        
    def train(self, epochs, episodes, batch_size, evalSize):
        print("Start Training the PGLearner.")

        for epo in range(epochs):
            start_time = time.time()
            x_n1, r_n1, a_n1 = [],[],[]
            x_n2, r_n2, a_n2 = [],[],[]
            game_end = False
            self.resetInputs()
            
            rd_s1 = np.empty((0,1))
            rd_s2 = np.empty((0,1)) 
           
            for epi in range(episodes):                
                while True:
                    # Get game state and update the input of both x
                    self.updateInputs(self.game.getGame())
                    # Get and perform action
                    state = -2
                    while state == -2:
                        action = self.agent.getNonDeterministicAction(self.x1)
                        state = self.game.play(action, self.game.getTurn())
                    
                        # Append state1 and his action to a list
                        x_n1.append(self.x1)
                        a_n1.append(action)
                        if state == -2:
                            r_n1.append(-0.3)
                        else:
                            r_n1.append(0)
                    
                    # Check if the game ended
                    game_end = self.checkGameEnd(state)      
                    
                    if game_end == True:
                        break
                    else:
                        # Get game state and update the input of both x
                        self.updateInputs(self.game.getGame())
                        # Get and perform action
                        state = -2
                        while state == -2:
                            action = self.agent.getNonDeterministicAction(self.x2)
                            state = self.game.play(action, self.game.getTurn())
                        
                            # Append state2 and his action to a list
                            x_n2.append(self.x2)
                            a_n2.append(action)
                            if state == -2:
                                r_n2.append(-0.3)
                            else:
                                r_n2.append(0)
                            
                        # Check if the game ended
                        game_end = self.checkGameEnd(state)
                        
                        if game_end == True:
                            break
                
                if state == 1:
                    r_n1[-1] = 1
                    r_n2[-1] = -1
                elif state == 2:
                    r_n1[-1] = -1
                    r_n2[-1] = 1

                rd_tmp = self.discount_rewards(np.vstack(r_n1))
                rd_s1 = np.concatenate((rd_s1, rd_tmp))
                    
                rd_tmp = self.discount_rewards(np.vstack(r_n2))
                rd_s2 = np.concatenate((rd_s2, rd_tmp))
                    
                r_n1, r_n2 = [], []

                    
                if np.mod(epi + 1, 5) == 0:
                    x_s1 = np.vstack(x_n1)
                    a_s1 = np.vstack(a_n1)
                    x_s2 = np.vstack(x_n2)
                    a_s2 = np.vstack(a_n2)
                
                    x_n1, a_n1 = [],[]
                    x_n2, a_n2 = [],[]
                    
                    x_s = np.concatenate((x_s1,x_s2))
                    a_s = np.concatenate((a_s1,a_s2))
                    rd_s = np.concatenate((rd_s1,rd_s2))
                    
                    rd_s1 = np.empty((0,1))
                    rd_s2 = np.empty((0,1)) 
                
                    a_s = a_s.reshape(a_s.shape[0],)
                    rd_s = rd_s.reshape(rd_s.shape[0],)
                
                    shuf = np.arange(x_s.shape[0])
                    np.random.shuffle(shuf)
                
                    for k in range(np.floor_divide(x_s.shape[0], batch_size)):
                        it1 = k*batch_size
                        it2 = (k+1)*batch_size
                        self.train_fn(x_s[shuf[it1:it2],:,:,:],
                                      a_s[it1:it2],rd_s[it1:it2])
                        
                    rd_s = np.empty((0,1))
                    
                self.game.newGame()
                        
            t = time.time() - start_time
            self.evaluate(evalSize, t, epo, epochs)
                    
        print("Saving Model to File...")
        self.agent.saveParams('trained_model1.npz', 'dummy1.npz')
        print("End Training Program!")
            
    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0.0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
            
        #normalize
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r
        
    def log(self, time, reward):
        with open("log_file.txt", "a") as myfile:
            myfile.writelines(str(time) + ', ' + str(reward) +'\n')
            
            
def main():
    PGL = PGLearner(0.99, 0.001, 0.9, 1e-6, False)
    PGL.train(1, 1, 20, 1000)
#==============================================================================
#     x = np.zeros((1,4,7,7),np.int8)
#     r = np.array([1])
#     a = np.array([3])
#     a = a.reshape(a.shape[0],)
#     r = r.reshape(r.shape[0],)
#     r.astype(np.float32)
#     print 'eta: ',PGL.learning_rate
#     print 'rho: ',PGL.rho
#     print 'eps: ',PGL.epsilon
#     print PGL.prediction_fn(x)
#     PGL.train_fn(x,a,r)
#     print PGL.prediction_fn(x)
#==============================================================================
    
    
if __name__ == '__main__':
    main()
        