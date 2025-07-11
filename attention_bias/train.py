import os
import sys
import time
from datetime import datetime

from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.datasets import *
from utils.point_clouds import collate_featurize, collate_tokenize
from utils.smiles import *

from models.transformer import Transformer
from models.bias_maps import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bias_map_name, E, H, D, *_ = sys.argv[1:]

E, H, D = int(E), int(H), int(D)
BiasMap = bias_maps[bias_map_name]

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
collate_fn = collate_tokenize

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
    batch_size=64, collate_fn=collate_fn, shuffle=True
)
validation_dataloader = DataLoader(
    validation_dataset, 
    batch_size=2048, collate_fn=collate_fn, shuffle=True
)

#######
# Model
#######

model = Transformer(
    in_features=6, 
    out_features=n_properties, 
    E=E, H=H, D=D, 
    BiasMap=BiasMap, 
    dropout=0.1, 
).to(device)

#######
# Train
#######

optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
mse = nn.MSELoss()

for epoch in range(64):
    for features, padding, coordinates, y_true in train_dataloader:
        model.train()
        optimizer.zero_grad()

        # Move to device

        features = features.to(device)
        padding = padding.to(device)
        coordinates = coordinates.float().to(device)
        y_true = y_true.float().to(device)

        # Forward pass

        y_pred = model(
            features, coordinates, padding, 
        )
        loss = mse(y_pred, y_true)
        loss.backward()
        optimizer.step()

        #print(float(loss))

    # Log train statistics

    model.eval()
    with torch.no_grad():

        features, padding, coordinates, y_true = next(iter(validation_dataloader))
        y_pred = model(
            features.to(device), 
            coordinates.float().to(device), 
            padding.to(device), 
        )
        validation_loss = float(mse(y_pred, y_true.float().to(device)))
        validation_score = float(r2_score(y_true.cpu(), y_pred.cpu()))

    # Write to log

    with open(logdir + run_fname + '.csv', 'a') as f:
        f.write(f'{float(loss)},{validation_loss},{validation_score}\n')

# Save model

os.makedirs(os.path.dirname(weightsdir + run_fname + '.pt'), exist_ok=True)
torch.save(model.state_dict(), weightsdir + run_fname + '.pt')
