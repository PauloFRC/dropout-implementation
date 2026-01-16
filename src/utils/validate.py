import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.metrics import f1_score

def enforce_max_norm(model, max_val=3.0, norm_type=2):
    # Applies max-norm regularization to the weights of the model.
    for name, param in model.named_parameters():
        # Constraint only weights
        if 'weight' in name and param.dim() >= 2:
            param.data.renorm_(p=norm_type, dim=0, maxnorm=max_val)

def train_and_validate(model, train_loader, test_loader, epochs=50, max_norm=3.0):
    name = model.name

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    history = {'train_loss': [], 'test_loss': [], 'acc': [], 'f1': []}
    
    start_time = time.time()
    
    for epoch in tqdm(range(epochs), desc=f"Training model '{name}'"):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # use max-norm only along dropout
            if max_norm != None and model.dropout_rate > 0: 
                enforce_max_norm(model, max_val=max_norm)
            running_loss += loss.item()
            
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        f1 = f1_score(all_labels, all_preds, average="macro")
        
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['acc'].append(acc)
        history['f1'].append(f1)
        
    print(f"Finished in {time.time() - start_time:.2f}s")
    return history
