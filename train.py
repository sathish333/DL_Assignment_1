from keras.datasets import fashion_mnist,mnist
import wandb
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import copy
import argparse
from types import SimpleNamespace
np.random.seed(42)


from utilities.NeuralNetwork import NN
from utilities.HelperFunctions import OneHotEncoder
from utilities.config import * # reading global variables 


def pre_process(x):
    '''
    reshape and normalized the data to bring to 0-1 scale.
    '''
    x=x.reshape(-1,784)
    x=x/255
    return x

def load_data(dataset=fmnist_dataset,split_size=1-valid_split_size):
    
    '''
    loads and returns data after doing train-valid split.
    '''
    if dataset==fmnist_dataset: # load based on dataset passed
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif dataset==mnist_dataset:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    
    # split the data into train and valid
    x_train,x_valid=x_train[:int(len(x_train)*split_size)],x_train[int(len(x_train)*split_size):] #splitting train into train and valid
    y_train,y_valid=y_train[:int(len(y_train)*split_size)],y_train[int(len(y_train)*split_size):]

    x_train=pre_process(x_train) # preprocess data
    x_valid=pre_process(x_valid)
    x_test=pre_process(x_test) 
    
    one_hot=OneHotEncoder(10) # custome implementation of onehotencoder
    y_train=one_hot.transform(y_train)
    y_valid=one_hot.transform(y_valid)
    y_test=one_hot.transform(y_test)
    
    
    return x_train,y_train,x_valid,y_valid,x_test,y_test

    

def model_fit(params):
    '''
    fits the model with provided parameters
    '''
    print("Building the model with provided hyper parameters...\n",params)
    wandb.init(project=params['wandb_project'],entity=params['wandb_entity'],config=params)
    params=SimpleNamespace(**params)
    layers=[params.hidden_size]*params.num_layers # list of layer sizes 
    layers.append(output_layer_size) # append output layer size
    obj=NN(input_layer_size,layers,params)
    x_train,y_train,x_valid,y_valid,x_test,y_test=load_data(params.dataset,1-valid_split_size)
    obj.train(x_train.T,y_train,x_valid.T,y_valid)
    print(": Done")
    return obj



if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Set Hyper Parameters')
    parser.add_argument('-wp'   , '--wandb_project'  , type = str  , default='CS22M080', help = 'WandB Project Name (Non-Empty String) ')
    parser.add_argument('-we'   , '--wandb_entity'   , type = str  , default='CS22M080', help = 'WandB Entity Name (Non-Empty String)')
    parser.add_argument('-d'    , '--dataset'        , type = str  , default='fashion_mnist', help = 'Dataset : ["fashion_mnist", "mnist"]')
    parser.add_argument('-e'    , '--epochs'         , type = int  , default=10, help = 'Number of Epochs (Positive Integer)')
    parser.add_argument('-b'    , '--batch_size'     , type = int  , default=64, help = 'Batch Size (Positive Integer)')
    parser.add_argument('-l'    , '--loss'           , type = str  , default='cross_entropy', help = 'Loss function : ["mean_squared_error", "cross_entropy"]',choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument('-o'    , '--optimizer'      , type = str  , default='nadam', help = 'Optimizer : ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]',choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument('-lr'   , '--learning_rate'  , type = float, default=0.0001, help = 'Learning Rate (Positive Float)')
    parser.add_argument('-m'    , '--momentum'       , type = float, default=0.9, help = 'Momentum (Positive Float)')
    parser.add_argument('-beta' , '--beta'           , type = float, default=0.9, help = 'Beta (Positive Float) ')
    parser.add_argument('-beta1', '--beta1'          , type = float, default=0.9, help = 'Beta1 (Positive Float)')
    parser.add_argument('-beta2', '--beta2'          , type = float, default=0.99, help = 'Beta2 (Positive Float)')
    parser.add_argument('-eps'  , '--epsilon'        , type = float, default=1e-9, help = 'Epsilon (Positive Float)')
    parser.add_argument('-w_d'  , '--weight_decay'   , type = float, default=0, help = 'Weight Decay (Positive Float)')
    parser.add_argument('-w_i'  , '--weight_init'    , type = str.lower  , default='xavier', help = 'Weight Init : ["random", "normal","Xavier"]',choices=['random','normal','Xavier'])
    parser.add_argument('-nhl'  , '--num_layers'     , type = int  , default=3, help = 'Number of HIDDEN Layers (Positive Integer)')
    parser.add_argument('-sz'   , '--hidden_size'    , type = int  , default=64, help = 'Number of Neurons in Hidden Layers (Positive Integer)')
    parser.add_argument('-a'    , '--activation'     , type = str.lower  , default='tanh', help = 'Activation Function : ["identity", "sigmoid", "tanh", "relu"]   ',choices=["identity", "sigmoid", "tanh", "ReLU"] )
    
    
    # Parse the Input Args
    params = vars(parser.parse_args())
    obj=model_fit(params)