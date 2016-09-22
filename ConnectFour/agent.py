import numpy as np
import theano
import theano.tensor as T

import lasagne

from network import build_policy_network
from network import build_value_network

class Agent:
    
    def __init__(self, pretrained):
        self.input_var = T.tensor4('input')
        self.policy = build_policy_network(7, self.input_var, (7,7))
        self.value = build_value_network(self.input_var, (7,7))
        
        self.p_prediction = lasagne.layers.get_output(self.policy, deterministic=True)
        self.q_prediction = lasagne.layers.get_output(self.value, deterministic=True)
        
        if pretrained:
            self.loadParams()
        
        print("Compiling networks...")
        self.p_fn = theano.function([self.input_var], self.p_prediction)
        self.q_fn = theano.function([self.input_var], self.q_prediction)
        print("Compiling done!")
            
    def getDeterministicAction(self, x):
        probabilities = self.p_fn(x)
        action = np.argmax(probabilities)
        return action 
        
    def getNonDeterministicAction(self, x):
        probabilities = self.p_fn(x)
        probabilities = np.asarray(probabilities)
        csprob_n = np.cumsum(probabilities)
        action = (csprob_n > np.random.rand()).argmax()
        return action
        
    def getQValue(self, x):
        return self.q_fn(x)
            
    def loadParams(self):
        print("Loading paramters...")
        with np.load('trained_policy.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.policy, param_values)
        with np.load('trained_value.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.value, param_values)
        print("Loading Done!")
            
    def saveParams(self, policy_string, value_string):
        print("Saving Model...")
        np.savez(policy_string, *lasagne.layers.get_all_param_values(self.policy))
        np.savez(value_string, *lasagne.layers.get_all_param_values(self.value))
        print("Saving Done!")

def main():
    agent = Agent(True)

if __name__ == '__main__':
    main()