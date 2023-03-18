import numpy as np


class OneHotEncoder:
    def __init__(self,max_):
        self.max_=max_
    def transform(self,x):
        out=np.zeros((len(x),self.max_))
        out[np.arange(len(x)),x]=1
        return out.T


def intialize_weights(method,rows,cols):
    if method=='uniform':
        return np.random.uniform(-1,1,(rows,cols))
    elif method=='xavier':
        return np.random.randn(rows,cols)*np.sqrt(2/(rows+cols))
    else:
        return np.random.randn(rows,cols)*np.sqrt(2/cols)

def soft_max(x):
    max_=np.max(x,axis=0)
    x=x-max_
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def soft_max_prime(x):
    return soft_max(x)*(1-soft_max(x))

def cross_entropy_loss(y_pre,y):
    y_pre[y_pre<1e-15]=1e-15
    loss=-np.sum(y*np.log(y_pre))
    return loss/float(y_pre.shape[1])

def mean_squared_loss(y_pre,y):
    return np.mean((y_pre-y)**2)

