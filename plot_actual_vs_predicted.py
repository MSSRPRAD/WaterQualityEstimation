import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from tqdm import tqdm
from torcheval.metrics.functional import r2_score
import torch
from torcheval.metrics.functional import mean_squared_error

def plot_actual_vs_predicted(df, filename):
    plt.figure(figsize=(6, 6))

    split = "Train" if "train" in filename else "Test"
    feature = filename.split('_')[3]
    season = "fall" if "fall" in filename else "spring"

    # Scatter plot with diamond markers
    plt.scatter(df[f"Actual_{split}"], df[f"Predicted_{split}"], marker="D", facecolors="none", edgecolors="black", label=feature)

    # x=y line
    min_val = min(df[f"Actual_{split}"].min(), df[f"Predicted_{split}"].min())
    max_val = max(df[f"Actual_{split}"].max(), df[f"Predicted_{split}"].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label="x = y")

    # Calculate R^2 score
    predicted = torch.tensor(df[f"Predicted_{split}"].values)
    real = torch.tensor(df[f"Actual_{split}"].values)
    r2 = r2_score(predicted, real)
    mse = mean_squared_error(predicted, real)

    # Add R² text in upper left
    plt.text(0.05, 0.95, f"$R^2$ = {r2:.3f}", 
             transform=plt.gca().transAxes,
             fontsize=10)
    plt.text(0.05, 0.90, f"$MSE$ = {mse:.3f}",
    	     transform=plt.gca().transAxes,
    	     fontsize=10)

    # Labels and Title
    plt.xlabel(f"Measured {split} {feature} in {season}", fontsize=12, fontweight="bold")
    plt.ylabel(f"Predicted {split} {feature} in {season}", fontsize=12, fontweight="bold")
    plt.title(f"Model prediction for {split} {feature} in {season}", fontsize=12, fontweight="bold")

    # Custom legend
    plt.legend(loc="upper right", frameon=True, edgecolor="black")

    # Grid
    plt.grid(True, linestyle="--", linewidth=0.5)

    # Adjust axes to start at the same number and have equal ratio
    min_val -= 2
    max_val += 2
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Save and close the plot
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    for season in ["fall", "spring"]:
        for root, dirs, files in tqdm(os.walk(f'./{season}/')):
            for file in tqdm(files):
                if file.endswith('.csv'):
                    # Read CSV file
                    csv_path = os.path.join(root, file)
                    # if not (("turbidity" in csv_path) and ("spring" in csv_path)):
                    #     continue
                    print(f"Processing file {csv_path}")
                    df = pd.read_csv(csv_path)
                    
                    # Generate output filename
                    output_filename = os.path.join(root, 'plot_' + file.replace('.csv', '.png'))
                    
                    # Generate plot
                    plot_actual_vs_predicted(df, output_filename)
