import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch

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
            history = results[model_name][p]
            if metric == "loss":
                values.extend(history["test_loss"])
                values.extend(history["train_loss"])
            else:
                values.extend(history[metric])

        ymin, ymax = np.min(values), np.max(values)
        margin = 0.05 * (ymax - ymin + 1e-8)
        y_limits[model_name] = (ymin - margin, ymax + margin)
    
    for row, model_name in enumerate(model_names):
        for col, p in enumerate(dropout_rates):
            history = results[model_name][p]
            ax = axes[row, col]

            if metric == "loss":
                epochs = range(1, len(history["train_loss"]) + 1)
                df_test = pd.DataFrame({
                    "epoch": epochs,
                    "value": history["test_loss"],
                    "type": "Test loss"
                })
                df_train = pd.DataFrame({
                    "epoch": epochs,
                    "value": history["train_loss"],
                    "type": "Train loss"
                })
                df = pd.concat([df_test, df_train])
                sns.lineplot(
                    data=df,
                    x="epoch",
                    y="value",
                    hue="type",
                    ax=ax
                )
                ax.set_ylim(*y_limits[model_name])
                if row == 0:
                    ax.set_title(f"p = {p}", fontsize=12)
                if col == 0:
                    ax.set_ylabel(f"{model_name}\n{metric.upper()}")
                else:
                    ax.set_ylabel("")
                
                ax.set_xlabel("Epoch")
            else:
                df = pd.DataFrame({
                    "epoch": range(1, len(history[metric]) + 1),
                    metric: history[metric]
                })

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

def summarize_metrics(results):
    summary_data = []

    for model_name, rates_dict in results.items():
        for p, history in rates_dict.items():
            final_acc = history['acc'][-1]
            final_f1 = history['f1'][-1]
            final_loss = history['test_loss'][-1]
            
            summary_data.append({
                "Model": model_name,
                "Dropout Rate": p,
                "Test Acc (%)": final_acc,
                "F1 Score": final_f1,
                "Test Loss": final_loss
            })

    df = pd.DataFrame(summary_data)
    
    df = df.sort_values(by=["Model", "Dropout Rate"])
    pd.options.display.float_format = '{:.4f}'.format
    
    return df

def get_layer_activations(model, dataloader, layer_name='dropout1', device='cpu', max_samples=2000):
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(input[0].detach().cpu())

    layer = getattr(model, layer_name, None)
    if layer is None:
        return None

    handle = layer.register_forward_hook(hook_fn)
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            model(inputs)
            
            if sum(a.shape[0] for a in activations) >= max_samples:
                break

    handle.remove()
    return torch.cat(activations, dim=0).numpy()

def plot_neuron_correlations(models_dict, dataloader, layer_name='dropout1', device='cpu'):
    sns.set_theme(style="white")
    
    if isinstance(next(iter(models_dict.values())), dict):
        flattened = {}
        for model_name, dropout_dict in models_dict.items():
            for dropout_rate, model in dropout_dict.items():
                flattened[(model_name, dropout_rate)] = model
        models_dict = flattened
    
    # Sort by model name, then dropout rate
    sorted_keys = sorted(models_dict.keys(), key=lambda x: (x[0], x[1]) if isinstance(x, tuple) else x)
    
    n_plots = len(sorted_keys)
    n_cols = min(5, n_plots) # Max 5 columns
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, key in enumerate(sorted_keys):
        model = models_dict[key]
        
        if isinstance(key, tuple):
            model_name, dropout_rate = key
            title = f"{model_name}\nDropout p={dropout_rate}"
        else:
            title = f"Dropout p={key}"
        
        # Get activations and compute correlations
        acts = get_layer_activations(model, dataloader, layer_name, device)
        if acts is None:
            continue
        corr_matrix = np.corrcoef(acts, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Calculate average off-diagonal absolute correlation
        mask = np.eye(len(corr_matrix), dtype=bool)
        avg_corr = np.abs(corr_matrix[~mask]).mean()
        
        sns.heatmap(
            corr_matrix, 
            ax=axes[idx], 
            cmap="vlag",
            center=0, 
            vmin=-1, 
            vmax=1,
            square=True,
            cbar=(idx % n_cols == n_cols - 1),
            xticklabels=False,
            yticklabels=False
        )
        
        axes[idx].set_title(f"{title}\nAvg |Corr|: {avg_corr:.3f}")
        axes[idx].set_xlabel("Neuron Index")
        if idx % n_cols == 0:
            axes[idx].set_ylabel("Neuron Index")
    
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f"Neuron Co-adaptation (Layer: {layer_name})", fontsize=16)
    plt.tight_layout()
    plt.show()