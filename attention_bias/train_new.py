import os
import sys
import time
from datetime import datetime

from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fairchem.core.modules.scheduler import CosineLRLambda

from utils.datasets import *
from utils.point_clouds import collate_tokenize

from models.transformer import Transformer
from models.bias_maps import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bias_map_name, E, H, D, *args = sys.argv[1:]

E, H, D = int(E), int(H), int(D)
BiasMap = bias_maps[bias_map_name]
kwargs = dict(map(lambda x: x.split('='), args))
if kwargs: 
    bias_map_name += list(kwargs.values())[0]

#############
# Directories
#############

logdir = f'./logs/{bias_map_name}/E{E}_H{H}_D{D}/'
weightsdir = f'./weights/{bias_map_name}/E{E}_H{H}_D{D}/'

run_fname = datetime.now().strftime("%m_%d_%H_%M_%S")
run_fname += '_'
run_fname += str(os.environ.get("SLURM_ARRAY_TASK_ID"))

###################
# Logging directory
###################

os.makedirs(os.path.dirname(logdir + run_fname + '.csv'), exist_ok=True)
with open(logdir + run_fname + '.csv', 'w') as f:
    f.write('train_mse,validation_mse,validation_r2\n')

######
# Data
######

datadir = '/scratch/midway3/jshe/data/qm9/scaffolded/'
fnames = [
    'atoms.npy',
    'coordinates.npy',
    'y.npy', 
]

# Datasets

train_dataset = NPYDataset(*[
    datadir + f'train/{fname}' for fname in fnames
])
validation_dataset = NPYDataset(*[
    datadir + f'validation/{fname}' for fname in fnames
])
n_properties = train_dataset.n_properties

# Dataloaders

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=64, collate_fn=collate_tokenize, shuffle=True, 
    pin_memory=True,
)
validation_dataloader = DataLoader(
    validation_dataset, 
    batch_size=4096, collate_fn=collate_tokenize, shuffle=True, 
    pin_memory=True,
)

#######
# Model
#######

model = Transformer(
    n_tokens=6, 
    out_features=n_properties, 
    E=E, H=H, D=D, 
    BiasMap=BiasMap, 
    dropout=0.1, 
    **kwargs
).to(device)

#######
# Train
#######

optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)

#lambda_fn = CosineLRLambda(warmup_epochs=2, warmup_factor=0.1, epochs=64, lr_min_factor=0.01)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)

mse = nn.MSELoss()

for epoch in range(64):
    for tokens, padding, coordinates, y_true in train_dataloader:
        model.train()
        optimizer.zero_grad()

        # Forward pass

        y_pred = model(tokens, coordinates.float(), padding)
        loss = mse(y_pred, y_true.float())
        loss.backward()
        optimizer.step()

        print(float(loss))

    # Log train statistics

    model.eval()
    with torch.no_grad():

        tokens, padding, coordinates, y_true = next(iter(validation_dataloader))
        y_pred = model(tokens, coordinates.float(), padding)
        validation_loss = float(mse(y_pred, y_true.float()))
        validation_score = float(r2_score(y_true.cpu(), y_pred.cpu()))

    # Write to log

    with open(logdir + run_fname + '.csv', 'a') as f:
        f.write(f'{float(loss)},{validation_loss},{validation_score}\n')

    # Step scheduler

    #scheduler.step()

# Save model

os.makedirs(os.path.dirname(weightsdir + run_fname + '.pt'), exist_ok=True)
torch.save(model.state_dict(), weightsdir + run_fname + '.pt')
