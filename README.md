# Fashion MNIST classification using Feed Forward neural network.




## Dataset Overview:
Fashion Mnist dataset consists of 60k images as train and 10k images as test dataset. Each image has dimension of (28,28) and belongs to exactly one of the ten target labels.


10% of the training data is kept aside as validation dataset which will be used in picking the best hyperparameter configuration.
## Objective:
Goal is to build a neural network using from scratch using **Numpy** operations which can classify a given image to its target label.




## Folder structure


* *utilities*
   - *NeuralNetwork.py*: Contains class **NN** which acts as backbone for all the operations and holds the information required.
   - *Activations.py*: Supports functionality for different activation functions i.e **sigmoid**,**tanh**,etc.
   - *Optimizers.py*: Supports functionality for different optimizers i.e **nadam**,**sgd**,etc.
   - *HelperFunctions.py*: Contains generic helper functions.
   - *config.py*: Contains global varibales which are used through out the program.
* *FMNIST_WandB.ipynb*: Optimal choice of hyperparameters are explored by using **WandB**.
* *FMNIST_Best_Model.ipynb: Final model after hyper parameter search.
* *FMNIST_Entropy_VS_MSE.ipynb*: Comparision between cross entropy vs MSE
* *MNIST_top3.ipynb*: Manually choosing the 3 best combinations from FMNIST and trying on MNIST.


## How to use:


### Training and Validation:


```Python train.py ```


Above file contains hyperparameters which are set to the best possible configuration after extensive search perfomed in *FMNIST_WandB.ipynb*.


However,one can overwrite the values by passing them as command line arguments.


**Arguments supported:**
<br>


| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | CS22M080 | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | CS22M080 | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 10 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 16 | Batch size used to train neural network. |
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | nadam | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] |
| `-lr`, `--learning_rate` | 0.0001 | Learning rate used to optimize model parameters |
| `-m`, `--momentum` | 0.9 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.9 | Beta used by rmsprop optimizer |
| `-beta1`, `--beta1` | 0.9 | Beta1 used by adam and nadam optimizers. |
| `-beta2`, `--beta2` | 0.99 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | 0.0005 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | xavier | choices:  ["random", "Xavier"] |
| `-nhl`, `--num_layers` | 3 | Number of hidden layers used in feedforward neural network. |
| `-sz`, `--hidden_size` | 64 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | tanh | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |


<br>


## Results


Using the best hyperparameter configuration we are able to achieve test accuracy of **87.3%** on the test data.


* Optimizer - Nadam
* Number of Hidden layers - 3
* Hidden layer size - 64
* Weight Decay- 0.0005
* Activation - Tanh
* Learning Rate - 1e-4
* Batch size - 16
* Weight Initialization - Xavier





