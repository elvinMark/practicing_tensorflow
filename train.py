import tensorflow as tf
import argparse
import os
import numpy as np

from models import create_model
from datasets import create_datasets
from optimizers import create_optimizer
from schedulers import create_scheduler
from utils import train

parser =argparse.ArgumentParser(description="train helper")
parser.add_argument("--dataset",type=str,default="mnist",choices=["mnist","cifar10","cifar100"],help="choose the dataset to train")
parser.add_argument("--model",type=str,default="mlp",choices=["mlp","cnn","resnet"],help="choose what type of model to be used")
parser.add_argument("--batch-size",type=int,default=128,help="batch size used for training")
parser.add_argument("--project",type=str,default="project",help="name of the project")
parser.add_argument("--experiment",type=str,default="experiment",help="name of the experiment")
parser.add_argument("--lr",type=float,default=0.1,help="learning rate")
parser.add_argument("--epochs",type=int,default=50,help="number of epochs used for the training")
parser.add_argument("--optim",type=str,default="sgd",help="specify the optimizer to be used in training")
parser.add_argument("--sched",type=str,default="step",help="specify the type of scheduler to be used")
parser.add_argument("--step-size",type=int,default=50,help="step size used in the StepLR scheduler")
parser.add_argument("--gamma",type=float,default=0.2,help="gamma factor used in the StepLR scheduler")
parser.add_argument("--T_max",type=float,default=200,help="T_max factor used in CosineAnnealingLR scheduler")
parser.add_argument("--eta_min",type=float,default=0.,help="eta_min factor used in CosineAnnealingLR scheduler")
parser.add_argument("--checkpoint",type=int, default=-1,help="specified how frequent to save the models")
parser.add_argument("--path",type=str,default="./saved_models",help="specified the path where the checkpoint models are going to be saved")


args = parser.parse_args()
args.T_max = args.epochs
args.path = os.path.join(args.path,args.experiment)

model = create_model(args)
train_ds, test_ds = create_datasets(args)

optim = create_optimizer(args)
crit = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
sched_callback = tf.keras.callbacks.LearningRateScheduler(create_scheduler(args))
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=args.path,
    monitor="accuracy",
    save_weights_only=True,
    save_best_only=True,
    mode="max")

# model.compile(optimizer=optim,loss=crit,metrics=["accuracy"])

# model.fit(train_ds,epochs=args.epochs,callbacks=[sched_callback,model_checkpoint_callback])
# model.evaluate(test_ds)
 
train(model,train_ds,test_ds,crit,optim,None,args.epochs)
