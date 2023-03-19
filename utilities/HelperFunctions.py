import numpy as np


class OneHotEncoder:
    '''
    OneHotEncoder custom implementation
    '''
    def __init__(self,max_):
        self.max_=max_
    def transform(self,x):
        out=np.zeros((len(x),self.max_))
        out[np.arange(len(x)),x]=1
        return out.T


def intialize_weights(method,rows,cols):
    '''
    intilaize weights based on method passed
    '''
    if method=='uniform':
        return np.random.uniform(-1,1,(rows,cols))
    elif method=='xavier':
        return np.random.randn(rows,cols)*np.sqrt(2/(rows+cols))
    else:
        return np.random.randn(rows,cols)*np.sqrt(2/cols)

def soft_max(x):
    '''
    softmax function
    
    subtracting max from each of the column to avoid large values
    '''
    max_=np.max(x,axis=0)
    x=x-max_
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def soft_max_prime(x):
    '''
    derivative of softmax
    '''
    return soft_max(x)*(1-soft_max(x))

def cross_entropy_loss(y_pre,y):
    '''
    computes avg cross entropy loss
    '''
    y_pre[y_pre<1e-19]=1e-19
    loss=-np.sum(y*np.log(y_pre))
    return loss/float(y_pre.shape[1])

def mean_squared_loss(y_pre,y):
    '''
    computes avg squared loss
    '''
    return np.sum((y_pre-y)**2)/y_pre.shape[1]

def compute_confusion_matrix(target,pred):
    matrix = np.zeros((len(labels), len(labels)),dtype=int)
    for actual, predicted in zip(target, pred):
        matrix[actual][predicted]+= 1
    return matrix

def compute_accuracy_score(target,pred):
    return np.sum(target==pred)/len(target)

