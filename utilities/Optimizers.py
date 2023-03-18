import numpy as np
import copy
from .config import *
from .HelperFunctions import *

class Optimizers:
    def __init__(self,method):
        self.method=method
    def update(self,obj,x_batch=None,y_batch=None):
        
        lr=obj.params.learning_rate
        momentum=0.9
        adam_beta1=0.9
        adam_beta2=0.99
        rms_beta=0.9
        weight_decay=obj.params.weight_decay
        epsilon=1e-9
        
        if self.method=='sgd':
            for i in range(len(obj.layers)):
                obj.weights[i]=obj.weights[i]-lr*(2*weight_decay*obj.weights[i]+obj.gradients_w[i])
                obj.biases[i]=obj.biases[i]-lr*obj.gradients_b[i]
        elif self.method=='momentum':
            init=0
            new_gradients_w=[None]*len(obj.gradients_w)
            new_gradients_b=[None]*len(obj.gradients_b)
            for i in range(len(obj.layers)):
                if obj.prev_gradients_w!=None:
                    new_gradients_w[i]=(1-momentum)*obj.gradients_w[i]+obj.prev_gradients_w[i]*momentum
                    new_gradients_b[i]=(1-momentum)*obj.gradients_b[i]+obj.prev_gradients_b[i]*momentum
                    
                else:
                    new_gradients_w[i]=lr*obj.gradients_w[i]
                    new_gradients_b[i]=lr*obj.gradients_b[i]
                    
                obj.weights[i]=obj.weights[i]-lr*(new_gradients_w[i]+2*weight_decay*obj.weights[i])
                obj.biases[i]=obj.biases[i]-lr*(new_gradients_b[i])
                
            obj.prev_gradients_w=copy.deepcopy(new_gradients_w)
            obj.prev_gradients_b=copy.deepcopy(new_gradients_b)
            pass
        elif self.method=='rmsprop':
            for i in range(len(obj.layers)):
                obj.rms_v[i]=rms_beta*obj.rms_v[i]+(1-rms_beta)*(obj.gradients_w[i]**2)
                
                obj.weights[i]=obj.weights[i]-(lr/(np.sqrt(obj.rms_v[i])+epsilon))*(obj.gradients_w[i])
                 
                obj.biases[i]=obj.biases[i]-lr*obj.gradients_b[i] 
                
        elif self.method=='adam':
            for i in range(len(obj.layers)):
                obj.adam_m[i]=adam_beta1*obj.adam_m[i]+(1-adam_beta1)*obj.gradients_w[i]
                m_hat=obj.adam_m[i]/(1-np.power(adam_beta1,obj.counter))
                
                obj.adam_v[i]=adam_beta2*obj.adam_v[i]+(1-adam_beta2)*obj.gradients_w[i]*obj.gradients_w[i]
                v_hat=obj.adam_v[i]/(1-np.power(adam_beta2,obj.counter))
                obj.weights[i]=obj.weights[i]-(lr/(np.sqrt(v_hat)+epsilon))*m_hat
                obj.biases[i]=obj.biases[i]-lr*obj.gradients_b[i]
        
        elif self.method=='nadam':
             for i in range(len(obj.layers)):
                obj.adam_m[i]=adam_beta1*obj.adam_m[i]+(1-adam_beta1)*obj.gradients_w[i]
                m_hat=obj.adam_m[i]/(1-adam_beta1)
                
                obj.adam_v[i]=adam_beta2*obj.adam_v[i]+(1-adam_beta2)*obj.gradients_w[i]*obj.gradients_w[i]
                v_hat=obj.adam_v[i]/(1-adam_beta2)
                obj.weights[i]=obj.weights[i]-(lr/(np.sqrt(v_hat)+epsilon))*m_hat
                
                obj.weights[i]=obj.weights[i]-(lr*(np.divide((adam_beta1*m_hat)+((1-adam_beta1)/(1-np.power(adam_beta1,obj.counter)))*obj.gradients_w[i],np.sqrt(v_hat)+1e-9)))
                obj.biases[i]=obj.biases[i]-lr*obj.gradients_b[i]
        elif self.method=='nag':
            original_weights=copy.deepcopy(obj.weights)
            original_biases=copy.deepcopy(obj.biases)
            original_graidents_w=copy.deepcopy(obj.gradients_w)
            original_graidents_b=copy.deepcopy(obj.gradients_b)
            
            if obj.prev_gradients_w!=None:
                 for i in range(len(obj.layers)):
                        obj.weights[i]=obj.weights[i]-momentum*obj.gradients_w[i]
                        obj.biases[i]=obj.biases[i]-momentum*obj.gradients_b[i]
            
            layer_outs,inter_values=obj.forward(x_batch)
            loss=0
            if(obj.loss_function==entropy_loss):
                loss=cross_entropy_loss(layer_outs[-1],y_batch)
            elif(obj.loss_function==squared_loss):
                loss=cross_entropy_loss(layer_outs[-1],y_batch)
            obj.compute_deltas(layer_outs,inter_values,y_batch)
            obj.find_gradients(x_batch,layer_outs)
            
            for i in range(len(obj.gradients_w)):
                if obj.prev_gradients_w!=None:
                    obj.gradients_w[i]=original_graidents_w[i]*momentum+obj.gradients_w[i]
                    obj.gradients_b[i]=original_graidents_b[i]*momentum+obj.gradients_b[i]
                    
                obj.weights[i]=original_weights[i]-lr*(obj.gradients_w[i]+2*weight_decay*obj.weights[i])
                obj.biases[i]=original_biases[i]-lr*(obj.gradients_b[i])
            obj.prev_gradients_w=True
            return loss

