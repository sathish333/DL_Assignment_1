import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-wp','--wandb_project',default='myprojectbame',help='Project name used to track experiments in Weights & Biases dashboard')

parser.add_argument('-we','--wandb_entity',default='myname',help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')

parser.add_argument('-d','--dataset',default='fashion_mnist',choices=['fashion_mnist','mnist'],help='')

parser.add_argument('-e','--epochs',default=1,type=int,help='Number of epochs to train neural network')

parser.add_argument('-b','--batch_size',default=4,type=int,help='Batch size used to train neural network')







args = parser.parse_args()

print(args.dataset)