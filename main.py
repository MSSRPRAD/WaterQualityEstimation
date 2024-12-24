import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from torch import nn
from torch import flatten
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn.functional as F
import torcheval
from torcheval.metrics.functional import r2_score
import time
import itertools
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import sys

args = sys.argv

if len(args) < 3:
    print("Insufficient Command Line Arguments Passed. Pls pass the target attribute and season!")
    exit(0)

if args[1] not in ["salinity", "turbidity", "temperature", "chlorophyll_a_fluorescence", "chlorophyll"]:
    print("Wrong Command Line Argument Passed. Pls pass the correct, valid target attribute!")
    exit(0)

if args[2] not in ["fall", "spring"]:
    print("Wrong Season passed. Pls pass the correct season.")
    exit(0)

target = args[1]
season = args[2]

if season == "spring":
    x = np.load("./WQ_Spring/data_array_olci_new.npy")
    y = np.load(f"./WQ_Spring/y_{target}.npy")
elif season == "fall":
    # print("Fall not supported currently!")
    x = np.load("./WQ_Project_Fall/Satellite_Data/data_array_olci_new.npy")
    y = np.load(f"./WQ_Project_Fall/Satellite_Data/y_{target}.npy")

# Initialize max dictionary
max_values = {i: -float('inf') for i in range(x.shape[1])}

# Iterate through samples with tqdm
for sample in tqdm(x, desc="Samples"):
    l = 0
    # Iterate through layers
    for layer in sample:
        # Iterate through rows
        for row in layer:
            # Iterate through columns
            for val in row:
                if val > max_values[l]:
                    max_values[l] = val
        l += 1

# print("\n\nMax Values:\n\n", max_values)p

# Normalize each element in x by the maximum value in its layer
for i in tqdm(range(x.shape[0]), desc="Samples"):
    for layer_idx in range(x.shape[1]):
        max_val = max_values[layer_idx]

        # Check if max_val is not zero to avoid division by zero
        if max_val != 0:
            for a in range(x.shape[2]):
                for b in range(x.shape[3]):
                    x[i][layer_idx][a][b] = x[i][layer_idx][a][b] / max_val
        else:
            # Handle division by zero, e.g., set to zero or another appropriate value
            x[i][layer_idx] = 0.0

print("Normalized x:", x)

if season == "spring":
    nan_indices = [6, 90]
    valid_indices = [i for i in range(x.shape[0]) if i not in nan_indices]
    x = x[valid_indices]
    y = y[valid_indices]

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = torch.Tensor(x_train).float()
y_train = torch.Tensor(y_train).float()
x_test = torch.Tensor(x_test).float()
y_test = torch.Tensor(y_test).float()
dataset = data_utils.TensorDataset(x_train, y_train)
batch_size = 10
data_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_conv_blocks=2, initial_channels=32, dropout_rate=0.2):
        super(ConvNet, self).__init__()
        
        self.conv_blocks = nn.ModuleList()
        in_channels = 21  # Initial input channels
        
        # Create conv blocks
        for i in range(num_conv_blocks):
            out_channels = initial_channels * (2 ** i)  # Double channels in each block
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_rate)
            )
            self.conv_blocks.append(block)
            in_channels = out_channels
        
        # Calculate size after conv blocks
        with torch.no_grad():
            x = torch.randn(1, 21, 68, 57)
            for block in self.conv_blocks:
                x = block(x)
            flattened_size = x.view(1, -1).size(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Pass through conv blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

grid_params = {
    'num_conv_blocks': [2, 3, 4],
    'initial_channels': [16, 32, 64],
    'dropout_rate': [0.33],
    'learning_rate': [0.001, 0.0001]
}

grid_search_logs = []

# Create all combinations of parameters
from itertools import product
param_combinations = list(product(
    grid_params['num_conv_blocks'],
    grid_params['initial_channels'],
    grid_params['dropout_rate'],
    grid_params['learning_rate']
))

import os
import json
import matplotlib.pyplot as plt
import pandas as pd

for blocks, channels, dropout, lr in tqdm(param_combinations):
    filename_base = f"{season}_{target}_{blocks}_{channels}_{dropout}_{lr}"
    results_csv_path_train = os.path.join(season, f"train_{filename_base}.csv")
    results_csv_path_test = os.path.join(season, f"test_{filename_base}.csv")

    # Check if CSV files already exist
    if os.path.exists(results_csv_path_train) and os.path.exists(results_csv_path_test):
        print(f"Skipping configuration: {filename_base} as results already exist.")
        continue
    
    # Create the model
    model = ConvNet(
        num_conv_blocks=blocks,
        initial_channels=channels,
        dropout_rate=dropout
    )
    model = model.to(device)

    # Define the optimizer with L2 regularization
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-5)

    # Define the loss function
    criterion = nn.MSELoss()

    # Initialize variables for tracking best models
    best_test_loss = float('inf')
    best_test_r2 = float('-inf')
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    # Variable learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    # Number of epochs
    num_epochs = 1000

    # Lists to track metrics
    train_losses = []
    test_losses = []
    train_r2_scores = []
    test_r2_scores = []

    # Training loop
    progress_bar = tqdm(range(num_epochs), 
                       desc=f"Blocks:{blocks} Chan:{channels} Drop:{dropout} LR:{lr}",
                       leave=True)
    
    for epoch in progress_bar:
        # Training phase
        model.train()
        total_train_loss = 0.0
        train_y_true = []
        train_y_pred = []

        for inputs, labels in data_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_y_true.extend(labels.detach().cpu().numpy())
            train_y_pred.extend(outputs.detach().cpu().numpy())

        avg_train_loss = total_train_loss / len(data_loader)
        train_r2 = r2_score(torch.tensor(train_y_pred), torch.tensor(train_y_true))
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test.to(device))
            test_loss = criterion(test_outputs, y_test.to(device)).item()
            test_r2 = r2_score(test_outputs.cpu(), y_test)
            
        # Update best model if test loss improves
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_test_loss = test_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

        # Step the learning rate scheduler
        scheduler.step(test_loss)

        # Store metrics
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        train_r2_scores.append(train_r2)
        test_r2_scores.append(test_r2)

        progress_bar.set_postfix({
            'Train_R2': f'{train_r2:.4f}',
            'Test_R2': f'{test_r2:.4f}',
            'Best_Test_R2': f'{best_test_r2:.4f}',
            'Train_Loss': f'{avg_train_loss:.4f}',
            'Test_Loss': f'{test_loss:.4f}'
        })

    # Load the best model for final predictions
    model.load_state_dict(best_model_state)
    model.eval()
    
    # Generate final predictions
    with torch.no_grad():
        y_pred_train = model(x_train.to(device)).cpu().numpy()
        y_pred_test = model(x_test.to(device)).cpu().numpy()

    # Save predictions and actuals
    results_df_train = pd.DataFrame({
        'Actual_Train': y_train.numpy().flatten(),
        'Predicted_Train': y_pred_train.flatten()
    })
    results_df_test = pd.DataFrame({
        'Actual_Test': y_test.numpy().flatten(),
        'Predicted_Test': y_pred_test.flatten()
    })

    # Save results
    results_csv_path_train = os.path.join(season, f"train_{filename_base}.csv")
    results_csv_path_test = os.path.join(season, f"test_{filename_base}.csv")
    results_df_train.to_csv(results_csv_path_train, index=False)
    results_df_test.to_csv(results_csv_path_test, index=False)

    # Create and save plots
    def create_plots(y_train, y_pred_train, y_test, y_pred_test, filename):
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Plot 1: Test Data Scatter Plot
        ax1 = axes[0, 0]
        ax1.scatter(y_test, y_pred_test, color='blue', alpha=0.5)
        ax1.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Test Data: Actual vs Predicted')
        ax1.grid(True)

        # Plot 2: Train Data Scatter Plot
        ax2 = axes[0, 1]
        ax2.scatter(y_train, y_pred_train, color='blue', alpha=0.5)
        ax2.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Train Data: Actual vs Predicted')
        ax2.grid(True)

        # Plot 3: Test Data Time Series
        ax3 = axes[1, 0]
        ax3.plot(range(len(y_test)), y_test, 'r-', label='Actual')
        ax3.plot(range(len(y_pred_test)), y_pred_test, 'b-', label='Predicted')
        ax3.set_xlabel('Data Point')
        ax3.set_ylabel('Value')
        ax3.set_title('Test Data: Actual vs Predicted')
        ax3.legend()
        ax3.grid(True)

        # Plot 4: Train Data Time Series
        ax4 = axes[1, 1]
        ax4.plot(range(len(y_train)), y_train, 'r-', label='Actual')
        ax4.plot(range(len(y_pred_train)), y_pred_train, 'b-', label='Predicted')
        ax4.set_xlabel('Data Point')
        ax4.set_ylabel('Value')
        ax4.set_title('Train Data: Actual vs Predicted')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_path = os.path.join(season, f"{filename_base}.png")
    create_plots(
        y_train.numpy().flatten(),
        y_pred_train.flatten(),
        y_test.numpy().flatten(),
        y_pred_test.flatten(),
        plot_path
    )

    # Save hyperparameters and metrics
    hyperparams = {
        'num_conv_blocks': blocks,
        'initial_channels': channels,
        'dropout_rate': dropout,
        'learning_rate': lr,
        'best_test_r2': best_test_r2.tolist(),
        'best_test_loss': best_test_loss,
        'final_train_r2': train_r2_scores[-1].tolist(),
        'train_r2_history': [train_r2_scores[i].tolist() for i in range(len(train_r2_scores))],
        'test_r2_history': [test_r2_scores[i].tolist() for i in range(len(test_r2_scores))],
        'train_loss_history': train_losses,
        'test_loss_history': test_losses
    }
    grid_search_logs.append(hyperparams)
    hyperparams_path = os.path.join(season, f"{filename_base}.json")
    print(hyperparams)
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=4)

# Print best model results
print("\nGrid Search Results:")
best_model = max(grid_search_logs, key=lambda x: x['best_test_r2'])
print("\nBest Model Configuration:")
print(f"Number of Conv Blocks: {best_model['num_conv_blocks']}")
print(f"Initial Channels: {best_model['initial_channels']}")
print(f"Dropout Rate: {best_model['dropout_rate']}")
print(f"Learning Rate: {best_model['learning_rate']}")
print(f"Best Test R2: {best_model['best_test_r2']}")
print(f"Best Test Loss: {best_model['best_test_loss']}")