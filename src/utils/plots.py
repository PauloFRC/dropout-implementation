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
