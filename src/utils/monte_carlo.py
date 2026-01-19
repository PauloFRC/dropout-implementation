import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def monte_carlo_predict(model, x, n_samples=100):
    model.train()
    
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(x)
            probs = torch.softmax(output, dim=1)
            predictions.append(probs.cpu())
    
    predictions = torch.stack(predictions)  # (n_samples, batch_size, n_classes)
    
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    
    return mean_pred, std_pred, predictions


def evaluate_mc_dropout(model, test_loader, n_samples=100, device='cuda'):
    model.to(device)
    
    all_mean_preds = []
    all_std_preds = []
    all_uncertainties = []
    all_labels = []
    all_predictions = []
    
    for x, y in test_loader:
        x = x.to(device)
        
        mean_pred, std_pred, _ = monte_carlo_predict(model, x, n_samples)
        
        uncertainty = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=1)
        
        all_mean_preds.append(mean_pred)
        all_std_preds.append(std_pred)
        all_uncertainties.append(uncertainty)
        all_labels.append(y.cpu())
        all_predictions.append(mean_pred.argmax(dim=1))
    
    all_mean_preds = torch.cat(all_mean_preds)
    all_std_preds = torch.cat(all_std_preds)
    all_uncertainties = torch.cat(all_uncertainties)
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)
    
    accuracy = (all_predictions == all_labels).float().mean().item()
    
    return {
        'accuracy': accuracy,
        'mean_predictions': all_mean_preds,
        'std_predictions': all_std_preds,
        'labels': all_labels,
        'predictions': all_predictions,
        'uncertainties': all_uncertainties
    }


def plot_mc_uncertainty_analysis(mc_results, model_name, p, class_names):
    results = mc_results[model_name][p]
    uncertainties = results['uncertainties'].numpy()
    predictions = results['predictions'].numpy()
    labels = results['labels'].numpy()
    correct = (predictions == labels)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} | p={p} | MC Dropout Averaging', 
                 fontsize=16, fontweight='bold')
    
    axes[0, 0].hist(uncertainties[correct], bins=50, alpha=0.6, 
                    label=f'Correct (n={correct.sum()})', color='green', edgecolor='black')
    axes[0, 0].hist(uncertainties[~correct], bins=50, alpha=0.6, 
                    label=f'Incorrect (n={(~correct).sum()})', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Uncertainty', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Certainty Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    mean_unc_per_class = []
    std_unc_per_class = []
    accuracy_per_class = []
    
    for class_idx in range(10):
        class_mask = labels == class_idx
        mean_unc_per_class.append(uncertainties[class_mask].mean())
        std_unc_per_class.append(uncertainties[class_mask].std())
        accuracy_per_class.append(correct[class_mask].mean())
    
    x_pos = np.arange(10)
    bars = axes[0, 1].bar(x_pos, mean_unc_per_class, yerr=std_unc_per_class,
                          capsize=5, alpha=0.7, edgecolor='black',
                          color=plt.cm.viridis(np.array(accuracy_per_class)))
    axes[0, 1].set_xlabel('Class', fontsize=11)
    axes[0, 1].set_ylabel('Average uncertainty', fontsize=11)
    axes[0, 1].set_title('Uncertainty by class', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    bins = np.percentile(uncertainties, np.linspace(0, 100, 16))
    bin_indices = np.digitize(uncertainties, bins)
    
    bin_accuracies = []
    bin_centers = []
    bin_counts = []
    
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() >= 10:
            bin_acc = correct[mask].mean()
            bin_accuracies.append(bin_acc)
            bin_centers.append((bins[i-1] + bins[i]) / 2)
            bin_counts.append(mask.sum())
    
    axes[1, 0].plot(bin_centers, bin_accuracies, marker='o', 
                    linestyle='-', linewidth=2, markersize=8, color='darkgreen')
    axes[1, 0].axhline(y=results['accuracy'], color='red', linestyle='--', 
                       label=f'Average accuracy: {results["accuracy"]:.3f}')
    axes[1, 0].set_xlabel('Uncertainty', fontsize=11)
    axes[1, 0].set_ylabel('Accuracy', fontsize=11)
    axes[1, 0].set_title('Accuracy vs Uncertainty', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend(fontsize=10)
    
    mean_std = results['std_predictions'].mean(dim=1).numpy()
    axes[1, 1].hist(mean_std[correct], bins=50, alpha=0.6, 
                    label='Correct', color='green', edgecolor='black')
    axes[1, 1].hist(mean_std[~correct], bins=50, alpha=0.6, 
                    label='Incorrect', color='red', edgecolor='black')
    axes[1, 1].set_xlabel('Average std (MC Samples)', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Variance distribution', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"STATISTICS - {model_name} | p={p}")
    print(f"{'='*60}")
    print(f"Accuracy MC Dropout: {results['accuracy']:.4f}")
    print(f"Correct predictions: {correct.sum()}/{len(correct)} ({100*correct.mean():.2f}%)")
    print(f"\nUncertainty (Entropy):")
    print(f"  Average (correct): {uncertainties[correct].mean():.4f}")
    print(f"  Average (incorrect): {uncertainties[~correct].mean():.4f}")
    print(f"  Difference: {uncertainties[~correct].mean() - uncertainties[correct].mean():.4f}")
    print(f"\nDesvio Padrão MC:")
    print(f"  Average (correct): {mean_std[correct].mean():.4f}")
    print(f"  Average (incorrect): {mean_std[~correct].mean():.4f}")


def compare_dropout_rates_mc(mc_results, model_name, dropout_rates):
    dropout_ps = [p for p in dropout_rates if p > 0 and p in mc_results[model_name]]
    
    if not dropout_ps:
        print(f"No results for model {model_name}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} - Dropout rates comparison', 
                 fontsize=16, fontweight='bold')
    
    mean_uncertainties = []
    std_uncertainties = []
    accuracies = []
    mean_stds = []
    
    for p in dropout_ps:
        results = mc_results[model_name][p]
        mean_uncertainties.append(results['uncertainties'].mean().item())
        std_uncertainties.append(results['uncertainties'].std().item())
        accuracies.append(results['accuracy'])
        mean_stds.append(results['std_predictions'].mean().item())
    
    axes[0, 0].errorbar(dropout_ps, mean_uncertainties, yerr=std_uncertainties,
                        marker='o', linestyle='-', linewidth=2, markersize=10,
                        capsize=5, color='blue', label='Incerteza (Entropia)')
    axes[0, 0].set_xlabel('Dropout rate', fontsize=11)
    axes[0, 0].set_ylabel('Uncertainty average', fontsize=11)
    axes[0, 0].set_title('Uncertainty vs Dropout rate', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=10)
    
    axes[0, 1].plot(dropout_ps, accuracies, marker='s', linestyle='-',
                    linewidth=2, markersize=10, color='green')
    axes[0, 1].set_xlabel('Dropout rate', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy MC Dropout', fontsize=11)
    axes[0, 1].set_title('Accuracy vs Dropout rate', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    axes[1, 0].plot(dropout_ps, mean_stds, marker='^', linestyle='-',
                    linewidth=2, markersize=10, color='orange')
    axes[1, 0].set_xlabel('Dropout rate', fontsize=11)
    axes[1, 0].set_ylabel('Average std (MC)', fontsize=11)
    axes[1, 0].set_title('Variance MC vs Dropout rate', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    for p in dropout_ps:
        results = mc_results[model_name][p]
        axes[1, 1].hist(results['uncertainties'].numpy(), bins=30, 
                       alpha=0.4, label=f'p={p}', edgecolor='black')
    axes[1, 1].set_xlabel('Uncertainty', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Uncertainty distribution', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_uncertainty_examples(mc_results, model_name, p, test_dataset, class_names, n_examples=5):
    results = mc_results[model_name][p]
    uncertainties = results['uncertainties'].numpy()
    predictions = results['predictions'].numpy()
    labels = results['labels'].numpy()
    mean_preds = results['mean_predictions'].numpy()
    
    high_unc_idx = uncertainties.argsort()[-n_examples:][::-1]
    low_unc_idx = uncertainties.argsort()[:n_examples]
    
    fig, axes = plt.subplots(2, n_examples, figsize=(16, 10))
    fig.suptitle(f'{model_name} (p={p}) - Uncertainty examples', 
                 fontsize=14, fontweight='bold')
    
    for i, idx in enumerate(high_unc_idx):
        img, true_label = test_dataset[idx]
        axes[0, i].imshow(img.squeeze(), cmap='gray')
        pred_label = predictions[idx]
        is_correct = '✓' if pred_label == true_label else '✗'
        
        top3_probs = mean_preds[idx].argsort()[-3:][::-1]
        prob_text = '\n'.join([f'{class_names[j]}: {mean_preds[idx][j]:.2f}' 
                               for j in top3_probs])
        
        axes[0, i].set_title(
            f'HIGH UNCERTAINTY {is_correct}\n'
            f'True: {class_names[true_label]}\n'
            f'Pred: {class_names[pred_label]}\n'
            f'Unc: {uncertainties[idx]:.3f}\n'
            f'---\n{prob_text}',
            fontsize=8
        )
        axes[0, i].axis('off')
    
    for i, idx in enumerate(low_unc_idx):
        img, true_label = test_dataset[idx]
        axes[1, i].imshow(img.squeeze(), cmap='gray')
        pred_label = predictions[idx]
        is_correct = '✓' if pred_label == true_label else '✗'
        
        # Top 3 probabilidades
        top3_probs = mean_preds[idx].argsort()[-3:][::-1]
        prob_text = '\n'.join([f'{class_names[j]}: {mean_preds[idx][j]:.2f}' 
                               for j in top3_probs])
        
        axes[1, i].set_title(
            f'LOW UNCERTAINTY {is_correct}\n'
            f'True: {class_names[true_label]}\n'
            f'Pred: {class_names[pred_label]}\n'
            f'Unc: {uncertainties[idx]:.3f}\n'
            f'---\n{prob_text}',
            fontsize=8
        )
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_mc_summary_table(mc_results):
    rows = []
    for model_name in mc_results.keys():
        for p in sorted([k for k in mc_results[model_name].keys()]):
            results = mc_results[model_name][p]
            uncertainties = results['uncertainties'].numpy()
            predictions = results['predictions'].numpy()
            labels = results['labels'].numpy()
            correct = (predictions == labels)
            rows.append({
                'Model': model_name,
                'Dropout Rate': p,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Mean Uncertainty': f"{uncertainties.mean():.4f}",
                'Std Uncertainty': f"{uncertainties.std():.4f}",
                'Unc (Correct)': f"{uncertainties[correct].mean():.4f}",
                'Unc (Incorrect)': f"{uncertainties[~correct].mean():.4f}",
                'Unc Delta': f"{uncertainties[~correct].mean() - uncertainties[correct].mean():.4f}"
            })
    
    df = pd.DataFrame(rows)
    return df

def compare_all_models_mc(mc_results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('General comparison - Monte Carlo Dropout Averaging', 
                 fontsize=16, fontweight='bold')
    
    dropout_ps = [0.2, 0.4, 0.6, 0.8]
    
    for model_name in mc_results.keys():
        accs = []
        ps = []
        for p in dropout_ps:
            if p in mc_results[model_name]:
                accs.append(mc_results[model_name][p]['accuracy'])
                ps.append(p)
        axes[0].plot(ps, accs, marker='o', linestyle='-', linewidth=2, 
                    markersize=8, label=model_name)
    
    axes[0].set_xlabel('Dropout rate', fontsize=12)
    axes[0].set_ylabel('Accuracy MC Dropout', fontsize=12)
    axes[0].set_title('Accuracy by Model', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    for model_name in mc_results.keys():
        uncs = []
        ps = []
        for p in dropout_ps:
            if p in mc_results[model_name]:
                uncs.append(mc_results[model_name][p]['uncertainties'].mean().item())
                ps.append(p)
        axes[1].plot(ps, uncs, marker='s', linestyle='-', linewidth=2, 
                    markersize=8, label=model_name)
    
    axes[1].set_xlabel('Dropout rate', fontsize=12)
    axes[1].set_ylabel('Average uncertainty', fontsize=12)
    axes[1].set_title('Model uncertainty', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    