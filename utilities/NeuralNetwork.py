from .Activations import Activation_Functions
from .Optimizers import Optimizers
from .HelperFunctions import *
import copy
from .config import *
import wandb

class NN:
    def __init__(self,input_size,layers,params):
        self.input_size=input_size
        self.layers=layers
        self.nlayers=len(layers)
        self.activations=Activation_Functions(params.activation)
        self.optimizer=Optimizers(params.optimizer)
        self.weights=[]
        self.weights_init=params.weight_init
        self.weights.append(intialize_weights(self.weights_init,layers[0],input_size))
        self.deltas=[None]*self.nlayers
        self.gradients_w=[None]*self.nlayers
        self.gradients_b=[None]*self.nlayers
        self.prev_gradients_w=None
        self.pre_gradients_b=None
        self.loss_function=params.loss
        self.counter=1
        self.biases=[]
        self.batch_size=params.batch_size
        for i in range(1,len(layers)):
            self.weights.append(intialize_weights(self.weights_init,layers[i],layers[i-1]))
        for i in range(len(layers)):
            self.biases.append(np.random.randn(layers[i],1))   
        
        if self.optimizer.method in ['adam','nadam']:
            self.adam_m=copy.deepcopy(self.weights)
            self.adam_v=copy.deepcopy(self.adam_m)
            _=[each.fill(0) for each in self.adam_m]
            _=[each.fill(0) for each in self.adam_v]
        if self.optimizer.method=='rmsprop':
            self.rms_v=copy.deepcopy(self.weights)
            _=[each.fill(0) for each in self.rms_v]
        self.params=params
        
        
    def forward(self,x):
        layer_outputs=[]
        inter_values=[]
        
        for i in range(len(self.weights)):
            inter_values.append(self.weights[i].dot(x)+self.biases[i])
            if i!= len(self.weights)-1:
                layer_outputs.append(self.activations.getActivations(inter_values[-1]))
                x=layer_outputs[-1]
        layer_outputs.append(soft_max(inter_values[-1]))
        return layer_outputs,inter_values
    def compute_deltas(self,layer_outputs,inter_values,y_one_hot):
        if(self.loss_function==entropy_loss):
            self.deltas[self.nlayers-1]=(layer_outputs[self.nlayers-1]-y_one_hot)
        elif(self.loss_function==squared_loss):
            self.deltas[self.nlayers-1] = (layer_outputs[self.nlayers-1]-y_one_hot) * soft_max_prime(inter_values[self.nlayers-1])
            
        for i in range(self.nlayers-2,-1,-1):
            self.deltas[i]=np.matmul(self.weights[i+1].T,self.deltas[i+1])*self.activations.getDerivatives(inter_values[i])
        
    def find_gradients(self,x,layer_outs):
        self.gradients_w[0]=(np.matmul(self.deltas[0],x.T)+2*self.params.weight_decay*self.weights[0])/self.batch_size
        for i in range(1,self.nlayers):
            self.gradients_w[i]=(np.dot(self.deltas[i],layer_outs[i-1].T)+2*self.params.weight_decay*self.weights[i])/self.batch_size
        for i in range(self.nlayers):
            self.gradients_b[i]=np.sum(self.deltas[i],axis=1,keepdims=True)/self.batch_size
             
    def update_weights(self,lr):
        for each in range(self.nlayers):
            self.optimizer.update(self)
    
            
    def train(self,x,y,x_valid,y_valid):
        lr=self.params.learning_rate
        batch_size=self.params.batch_size
        train_losses=[]
        valid_losses=[] 
        epochs=self.params.epochs
        for epoch in range(epochs):                
            i=0
            batch_count=0
            loss=0.0
            while i+batch_size < x.shape[1]:
                batch_count+=1
                x_batch=x[:,i:i+batch_size]
                y_batch=y[:,i:i+batch_size]
                i+=batch_size
                if self.optimizer.method!='nag':
                    layer_outs,inter_values=self.forward(x_batch)
                    if(self.loss_function==entropy_loss):
                        loss+=cross_entropy_loss(layer_outs[-1],y_batch)
                    elif(self.loss_function==squared_loss):
                        loss+=mean_squared_loss(layer_outs[-1],y_batch)
                    self.compute_deltas(layer_outs,inter_values,y_batch)
                    self.find_gradients(x_batch,layer_outs)
                    self.optimizer.update(self)
                else:
                    loss+=self.optimizer.update(self,x_batch=x_batch,y_batch=y_batch)
                self.counter+=1
                    
            train_losses.append(loss/batch_count)
            layer_outs,inter_values=self.forward(x_valid)
            if(self.loss_function==entropy_loss):
                valid_loss=cross_entropy_loss(layer_outs[-1],y_valid)
            else:
                valid_loss=mean_squared_loss(layer_outs[-1],y_valid)
                
            valid_losses.append(valid_loss)
            tr_ac=self.calc_accuracy(x,y)
            val_ac=self.calc_accuracy(x_valid,y_valid)
            print(f"epoch {epoch+1} : train loss = {train_losses[-1]:.2f} valid loss = {valid_loss:.2f} train accuracy = {tr_ac:.2f} valid accuracy = {val_ac:.2f}")
            wandb.log({'train loss':train_losses[-1],'train accuracy':tr_ac,'valid loss':valid_loss,'valid accuracy':val_ac})
        return train_losses,tr_ac,valid_losses,val_ac
    
    def predict_probas(self,x):
        layer_outs,inter_values=self.forward(x)
        return layer_outs[-1]
        
    def calc_accuracy(self,x,y):
        layer_outs,inter_values=self.forward(x)
        pred=np.argmax(layer_outs[-1],axis=0)
        expected=np.argmax(y,axis=0)
        return (np.sum(pred==expected)/y.shape[1])*100