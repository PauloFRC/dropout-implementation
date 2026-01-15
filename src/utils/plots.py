import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_dropout_histories(results, metric="f1"):
    sns.set_theme(style="darkgrid")
    
    model_names = list(results.keys())
    dropout_rates = sorted(next(iter(results.values())).keys())
    
    n_rows = len(model_names)
    n_cols = len(dropout_rates)
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        sharex=True
    )
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # compute y-axis limits
    y_limits = {}
    for model_name in model_names:
        values = []
        for p in dropout_rates:
            history = results[model_name][p][metric]
            values.extend(history)
        
        ymin, ymax = np.min(values), np.max(values)
        margin = 0.05 * (ymax - ymin + 1e-8)
        y_limits[model_name] = (ymin - margin, ymax + margin)
    
    for row, model_name in enumerate(model_names):
        for col, p in enumerate(dropout_rates):
            history = results[model_name][p]
            
            df = pd.DataFrame({
                "epoch": range(1, len(history[metric]) + 1),
                metric: history[metric]
            })
            
            ax = axes[row, col]
            sns.lineplot(data=df, x="epoch", y=metric, ax=ax)

            ax.set_ylim(*y_limits[model_name])
            
            if row == 0:
                ax.set_title(f"p = {p}", fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{model_name}\n{metric.upper()}")
            else:
                ax.set_ylabel("")
            
            ax.set_xlabel("Epoch")
    
    plt.tight_layout()
    plt.show()
