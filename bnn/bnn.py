import sys
sys.path.append("..")

import time
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pyro
from pyro.infer.mcmc import MCMC
import pyro.distributions as dist

import seaborn as sns
import pandas as pd

from kernel.sghmc import SGHMC
from kernel.sgld import SGLD
from kernel.sgd import SGD
from kernel.sgnuts import NUTS as SGNUTS

import argparse

BATCH_SIZE = 500
NUM_EPOCHS = 800
WARMUP_EPOCHS = 50
HIDDEN_SIZE = 100
UPDATER = "SGHMC"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

PyroLinear = pyro.nn.PyroModule[torch.nn.Linear]
    
class BNN(pyro.nn.PyroModule):
    
    def __init__(self, input_size, hidden_size, output_size, prec=1.):
        super().__init__()
        # prec is a kwarg that should only used by SGD to set the regularization strength 
        # recall that a Guassian prior over the weights is equivalent to L2 norm regularization in the non-Bayes setting
        
        # TODO add gamma priors to precision terms
        self.fc1 = PyroLinear(input_size, hidden_size)
        self.fc1.weight = pyro.nn.PyroSample(dist.Normal(0., prec).expand([hidden_size, input_size]).to_event(2))
        self.fc1.bias   = pyro.nn.PyroSample(dist.Normal(0., prec).expand([hidden_size]).to_event(1))
        
        self.fc2 = PyroLinear(hidden_size, output_size)
        self.fc2.weight = pyro.nn.PyroSample(dist.Normal(0., prec).expand([output_size, hidden_size]).to_event(2))
        self.fc2.bias   = pyro.nn.PyroSample(dist.Normal(0., prec).expand([output_size]).to_event(1))
        
        self.relu = torch.nn.ReLU()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, y=None):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)# output (log) softmax probabilities of each class
        
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=x), obs=y)

def run_inference(sampler):
    test_errs = []

    # full posterior predictive 
    full_predictive = torch.FloatTensor(10000, 10)
    full_predictive.zero_()

    for epoch in range(1, 1+NUM_EPOCHS + WARMUP_EPOCHS):
        sampler.run(X_train, Y_train)
        
        if epoch >= WARMUP_EPOCHS:
            
            samples = sampler.get_samples()
            predictive = pyro.infer.Predictive(bnn, posterior_samples=samples)
            start = time.time()
            
            with torch.no_grad():
                epoch_predictive = None
                for x, y in val_loader:
                    if epoch_predictive is None:
                        epoch_predictive = predictive(x)['obs'].to(torch.int64)
                    else:
                        epoch_predictive = torch.cat((epoch_predictive, predictive(x)['obs'].to(torch.int64)), dim=1)
                        
                for sample in epoch_predictive:
                    predictive_one_hot = F.one_hot(sample, num_classes=10)
                    full_predictive = full_predictive + predictive_one_hot
                    
                full_y_hat = torch.argmax(full_predictive, dim=1)
                total = Y_val.shape[0]
                correct = int((full_y_hat == Y_val).sum())
                
            end = time.time()

            test_errs.append(1.0 - correct/total)

            print("Epoch [{}/{}] test accuracy: {:.4f} time: {:.2f}".format(epoch-WARMUP_EPOCHS, NUM_EPOCHS, correct/total, end - start))

    return test_errs

def run_optim(optimizer):
    test_errs = []

    for epoch in range(1, 1+NUM_EPOCHS+WARMUP_EPOCHS):
        optimizer.run(X_train, Y_train)
            
        if epoch >= WARMUP_EPOCHS:
            
            samples = optimizer.get_samples()
            point_estimate = {site : samples[site][-1, :].unsqueeze(0) for site in samples.keys()}
            predictive = pyro.infer.Predictive(bnn, posterior_samples=point_estimate)
            start = time.time()
            
            with torch.no_grad():
                total = 0
                correct = 0
                for x, y in val_loader:
                    batch_predictive = predictive(x)['obs']
                    batch_y_hat = batch_predictive.mode(0)[0]
                    total += y.shape[0]
                    correct += int((batch_y_hat == y).sum())
                
            end = time.time()

            test_errs.append(1.0 - correct/total)

            print("Epoch [{}/{}] test accuracy: {:.4f} time: {:.2f}".format(epoch-WARMUP_EPOCHS, NUM_EPOCHS, correct/total, end - start))

    return test_errs

def plot(err_arr):
    sns.set_style("dark")
    y = np.array(err_arr)
    x = np.arange(1, NUM_EPOCHS+1)
    df = pd.DataFrame(list(zip(x, y)), columns=['iterations', 'test error'])
    sns.lineplot(data=df, x='iterations', y='test error').set(title='{} best test error: {:.4f}'.format(UPDATER, np.min(y)))

    if UPDATER == 'SGHMC':
        PATH = './imgs/bnn_{}_lr={}_alpha={}_resample_n={}.png'.format(UPDATER, LR, MOMENTUM_DECAY, RESAMPLE_EVERY_N)
    elif UPDATER == 'SGLD':
        PATH = './imgs/bnn_{}_lr={}.png'.format(UPDATER, LR)
    elif UPDATER == 'SGD':
        PATH = './imgs/bnn_{}_lr={}_reg={}_wd={}.png'.format(UPDATER, LR, REGULARIZATION_TERM, WEIGHT_DECAY)
    elif UPDATER == 'SGDMOM':
        PATH = './imgs/bnn_{}_lr={}_reg={}_wd={}.png'.format(UPDATER, LR, REGULARIZATION_TERM, WEIGHT_DECAY)

    plt.savefig(PATH) #dpi=300

if __name__ == "__main__":
    LR = 2e-6
    RESAMPLE_EVERY_N = 0
    NUM_STEPS = 1
    WEIGHT_DECAY=0.0
    MOMENTUM_DECAY=0.01
    REGULARIZATION_TERM=1.

    parser = argparse.ArgumentParser()
    parser.add_argument("--updater", default=UPDATER)
    parser.add_argument("--n-warmup", default=WARMUP_EPOCHS)
    parser.add_argument("--n-epochs", default=NUM_EPOCHS)
    parser.add_argument("--batch-size", default=BATCH_SIZE)
    parser.add_argument("--hidden-size", default=HIDDEN_SIZE)
    parser.add_argument("--lr", default=LR)
    parser.add_argument("--alpha", default=MOMENTUM_DECAY)
    parser.add_argument("--resample-n", default=RESAMPLE_EVERY_N)
    parser.add_argument("--wd", default=WEIGHT_DECAY)
    parser.add_argument("--reg", default=REGULARIZATION_TERM)
    args = parser.parse_args()

    UPDATER = args.updater
    WARMUP_EPOCHS = int(args.n_warmup)
    NUM_EPOCHS = int(args.n_epochs)
    BATCH_SIZE = int(args.batch_size)
    HIDDEN_SIZE = int(args.hidden_size)
    LR = float(args.lr)
    MOMENTUM_DECAY = float(args.alpha)
    RESAMPLE_EVERY_N = int(args.resample_n)
    WEIGHT_DECAY = float(args.wd)
    REGULARIZATION_TERM = float(args.reg)

    train_dataset = datasets.MNIST('./data', train=True, download=True)
    test_dataset = datasets.MNIST('./data', train=False, download=True)

    nvalid = 10000

    perm = torch.arange(len(train_dataset))
    train_idx = perm[nvalid:]
    val_idx = perm[:nvalid]
        
    mean = 0.1307
    std = 0.3081

    # scale and normalise the datasets
    X_train = train_dataset.data[train_idx] / 255.0
    Y_train = train_dataset.targets[train_idx]

    X_val = train_dataset.data[val_idx] / 255.0 
    Y_val = train_dataset.targets[val_idx]

    X_test = (test_dataset.data / 255.0 - mean) / std
    Y_test = test_dataset.targets

    # redefine the datasets
    train_dataset = Dataset(X_train, Y_train)
    val_dataset = Dataset(X_val, Y_val)
    test_dataset = Dataset(X_test, Y_test)

    # setup the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    bnn = BNN(28*28, HIDDEN_SIZE, 10, prec=REGULARIZATION_TERM)

    pyro.clear_param_store()

    kernel = None

    if UPDATER == 'SGHMC':
        kernel = SGHMC(bnn,
                       subsample_positions=[0, 1],
                       batch_size=BATCH_SIZE,
                       learning_rate=LR,
                       momentum_decay=MOMENTUM_DECAY,
                       num_steps=NUM_STEPS,
                       resample_every_n=RESAMPLE_EVERY_N)
    elif UPDATER == 'SGLD':
        kernel = SGLD(bnn,
                      subsample_positions=[0, 1],
                      batch_size=BATCH_SIZE,
                      learning_rate=LR,
                      noise_rate=2*LR,
                      num_steps=NUM_STEPS)
    elif UPDATER == 'SGD':
        kernel = SGD(bnn,
                     subsample_positions=[0, 1],
                     batch_size=BATCH_SIZE,
                     learning_rate=LR,
                     weight_decay=WEIGHT_DECAY,
                     with_momentum=False,
                     momentum_decay=0.0)
    elif UPDATER == 'SGDMOM':
        kernel = SGD(bnn,
                     subsample_positions=[0, 1],
                     batch_size=BATCH_SIZE,
                     learning_rate=LR,
                     weight_decay=WEIGHT_DECAY,
                     with_momentum=True,
                     momentum_decay=MOMENTUM_DECAY)
    else:
        raise RuntimeError('Invalid argument given for UPDATER {}, require one of "SGHMC", "SGLD", "SGD", "SGDMOM"'.format(UPDATER))

    mcmc_sampler = MCMC(kernel, num_samples=len(train_dataset)//BATCH_SIZE, warmup_steps=0)

    if UPDATER in ['SGHMC', 'SGLD']:
        test_errs = run_inference(mcmc_sampler)
    else:
        test_errs = run_optim(mcmc_sampler)

    plot(test_errs)