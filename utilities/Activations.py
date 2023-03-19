import numpy as np

class Activation_Functions:
    
    '''
    
    supports activation functions: ['relu',sigmoid,'tanh','identity']
    
    '''
    def __init__(self,function):
        self.function=function
        
        
    def getActivations(self,x):
        ''' returns activations of passed array based'''
        if self.function=='relu':
            return np.maximum(0,x)
        elif self.function=='sigmoid':
            x=np.clip(x,-500,500) #changed
            return 1.0/(1+np.exp(-x))
        elif self.function=='tanh':
            return np.tanh(x)
        elif self.function=='identity':
            return x
        
    def getDerivatives(self,x):
        ''' returns derivatives of passed array'''
        if self.function=='relu':
             return 1*(x>0)
        elif self.function=='sigmoid':
            return self.getActivations(x)*(1-self.getActivations(x))
        elif self.function=='tanh':
            return (1 - (np.tanh(x)**2)) 
        elif self.function=='identity':
            return np.ones(x.shape)
