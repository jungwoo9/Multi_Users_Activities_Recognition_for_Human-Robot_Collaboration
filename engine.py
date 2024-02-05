import torch.optim as optim
import torch
import torch.nn as nn

import data_setup, models

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train_lstm(model, trainloader, num_epochs=10, lr=0.00001, device='cpu'):
    model.train()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_rst_dict = {'loss':[], 'acc':[]}
    
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            y -= 1
            # Forward pass
            outputs = model(x)
        
            # Compute loss
            loss = criterion(outputs, y)
        
            # Backward and optimize
            optimizer.zero_grad()
        
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
    
            outputs = torch.argmax(outputs, axis=1)
            train_acc += torch.sum(outputs==y).item() / len(y)
        
        train_loss /= len(trainloader)
        train_acc /= len(trainloader)
        train_rst_dict['loss'].append(train_loss)
        train_rst_dict['acc'].append(train_acc)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
      
    return train_rst_dict

def test_lstm(model, testloader, device='cpu'):
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    test_rst_dict = {'loss':[], 'acc':[]}
    
    with torch.no_grad():
        test_loss = 0
        test_acc = 0
        y_pred = []
        y_true = []
        for x, y in testloader:
            y_true.extend(y.tolist())
            x = x.view(x.shape[0], x.shape[1], -1).float().to(device)
            y = y.to(device)
            y -= 1
        
            # Forward pass
            outputs = model(x)
            
            # Compute loss
            loss = criterion(outputs, y)
            
            test_loss += loss.item()
            
            outputs = torch.argmax(outputs, axis=1)
            y_pred.extend((outputs+1).tolist())
            test_acc += torch.sum(outputs==y).item() / len(y)
    
    test_loss /= len(testloader)
    test_acc /= len(testloader)
    
    test_rst_dict['loss'].append(test_loss)
    test_rst_dict['acc'].append(test_acc)
    
    return test_rst_dict, y_true, y_pred
    
def train_test_lstm(task='grouped-grouped', device='cpu'):
    if task=="grouped-grouped":
        train_rst_dicts = {}
        test_rst_dicts = {}
        y_true_dicts = {}
        y_pred_dicts = {}
        
        for ptcp_id in ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11']:
            print(f"grouped-grouped experiment with {ptcp_id}")
            grouped_trainloader = data_setup.get_grouped_dataloader(ptcp_id=ptcp_id)
            grouped_testloader = data_setup.get_grouped_dataloader(ptcp_id=ptcp_id, train=False)

            lstm = models.get_lstm(device=device)

            train_rst_dict = train_lstm(model=lstm, trainloader=grouped_trainloader, device=device)
            
            test_rst_dict, y_true, y_pred = test_lstm(model=lstm, testloader=grouped_testloader, device=device)
            
            print(f"test loss: {test_rst_dict['loss'][0]:.2f} | test acc: {test_rst_dict['acc'][0]:.2f}")
            print()
            
            train_rst_dicts[ptcp_id] = train_rst_dict
            test_rst_dicts[ptcp_id] = test_rst_dict
            y_true_dicts[ptcp_id] = y_true
            y_pred_dicts[ptcp_id] = y_pred
            
    elif task=="paired-paired":
        train_rst_dicts = {}
        test_rst_dicts = {}
        y_true_dicts = {}
        y_pred_dicts = {}
        
        for ptcp_id in ['s1', 's2', 's3']:
            print(f"paired-paired experiment with {ptcp_id}")
            paired_trainloader = data_setup.get_paired_dataloader(ptcp_id=ptcp_id)
            paired_testloader = data_setup.get_paired_dataloader(ptcp_id=ptcp_id, train=False)

            lstm = models.get_lstm(device=device)

            train_rst_dict = train_lstm(model=lstm, trainloader=paired_trainloader, device=device)
            
            test_rst_dict, y_true, y_pred = test_lstm(model=lstm, testloader=paired_testloader, device=device)
            
            print(f"test loss: {test_rst_dict['loss'][0]} | test acc: {test_rst_dict['acc'][0]}")
            print()
            
            train_rst_dicts[ptcp_id] = train_rst_dict
            test_rst_dicts[ptcp_id] = test_rst_dict
            y_true_dicts[ptcp_id] = y_true
            y_pred_dicts[ptcp_id] = y_pred
    
    elif task=="grouped-paired":
        print("grouped-paired experiment")
        
        train_rst_dicts = {}
        test_rst_dicts = {}
        y_true_dicts = {}
        y_pred_dicts = {}
        
        grouped_trainloader = data_setup.get_grouped_dataloader(ptcp_id='all')
        paired_testloader = data_setup.get_paired_dataloader(ptcp_id='all')
        
        lstm = models.get_lstm(device=device)

        train_rst_dict = train_lstm(model=lstm, trainloader=grouped_trainloader, device=device)
        
        test_rst_dict, y_true, y_pred = test_lstm(model=lstm, testloader=paired_testloader, device=device)
        
        train_rst_dicts['all'] = train_rst_dict
        test_rst_dicts['all'] = test_rst_dict
        y_true_dicts['all'] = y_true
        y_pred_dicts['all'] = y_pred
        
        print(f"test loss: {test_rst_dict['loss'][0]} | test acc: {test_rst_dict['acc'][0]}")
        print()
        
    return train_rst_dicts, test_rst_dicts, y_true_dicts, y_pred_dicts

def train_vae_stgcn(model, trainloader, num_epochs=1, lr=0.0001, device='cpu'):
    model.train()
    
    # Loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_rst_dict = {'loss':[]}
    
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        for x, _ in trainloader:
            x = x.to(device)
            
            # Forward pass
            recon_x, mean, log_var = model(x)
        
            # Compute loss
            loss = models.vae_loss(x.to(torch.float32), recon_x, mean, log_var)
        
            # Backward and optimize
            optimizer.zero_grad()
        
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
        
        train_loss /= len(trainloader)
        train_rst_dict['loss'].append(train_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')
      
    return train_rst_dict
    
def train_predictor(model, trainloader, num_epochs=5, lr=0.0001, device='cpu'):
    model.train()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_rst_dict = {'loss':[], 'acc':[]}
    
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            y -= 1
            # Forward pass
            outputs = model(x)
        
            # Compute loss
            loss = criterion(outputs, y)
        
            # Backward and optimize
            optimizer.zero_grad()
        
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
    
            outputs = torch.argmax(outputs, axis=1)
            train_acc += torch.sum(outputs==y).item() / len(y)
        
        train_loss /= len(trainloader)
        train_acc /= len(trainloader)
        train_rst_dict['loss'].append(train_loss)
        train_rst_dict['acc'].append(train_acc)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
      
    return train_rst_dict
    
def test_predictor(model, testloader, device):
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    test_rst_dict = {'loss':[], 'acc':[]}
    
    with torch.no_grad():
        test_loss = 0
        test_acc = 0
        y_pred = []
        y_true = []
        for x, y in testloader:
            y_true.extend(y.tolist())
            x = x.to(device)
            y = y.to(device)
            y -= 1
        
            # Forward pass
            outputs = model(x)
            
            # Compute loss
            loss = criterion(outputs, y)
            
            test_loss += loss.item()
            
            outputs = torch.argmax(outputs, axis=1)
            y_pred.extend((outputs+1).tolist())
            test_acc += torch.sum(outputs==y).item() / len(y)
    
    test_loss /= len(testloader)
    test_acc /= len(testloader)
    
    test_rst_dict['loss'].append(test_loss)
    test_rst_dict['acc'].append(test_acc)
    
    return test_rst_dict, y_true, y_pred

def train_test_predictor(task='grouped-grouped', device='cpu'):
    if task=="grouped-grouped":
        vae_stgcn_train_rst_dicts = {}
        predictor_train_rst_dicts = {}
        test_rst_dicts = {}
        y_true_dicts = {}
        y_pred_dicts = {}
        
        for ptcp_id in ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11']:
            print(f"grouped-grouped experiment with {ptcp_id}")
            grouped_trainloader = data_setup.get_grouped_dataloader(ptcp_id=ptcp_id)
            grouped_testloader = data_setup.get_grouped_dataloader(ptcp_id=ptcp_id, train=False)

            vae_stgcn = models.get_vae_stgcn(device=device)

            vae_stgcn_train_rst_dict = train_vae_stgcn(model=vae_stgcn, trainloader=grouped_trainloader, device=device)
            
            predictor = models.get_predictor(vae_stgcn=vae_stgcn, device=device)
            
            predictor_train_rst_dict = train_predictor(model=predictor, trainloader=grouped_trainloader, device=device)
            
            test_rst_dict, y_true, y_pred = test_predictor(model=predictor, testloader=grouped_testloader, device=device)
            
            print(f"test loss: {test_rst_dict['loss'][0]:.2f} | test acc: {test_rst_dict['acc'][0]:.2f}")
            print()
            
            vae_stgcn_train_rst_dicts[ptcp_id] = vae_stgcn_train_rst_dict
            predictor_train_rst_dicts[ptcp_id] = predictor_train_rst_dict
            test_rst_dicts[ptcp_id] = test_rst_dict
            y_true_dicts[ptcp_id] = y_true
            y_pred_dicts[ptcp_id] = y_pred
            
    elif task=="paired-paired":
        vae_stgcn_train_rst_dicts = {}
        predictor_train_rst_dicts = {}
        test_rst_dicts = {}
        y_true_dicts = {}
        y_pred_dicts = {}
        
        for ptcp_id in ['s1', 's2', 's3']:
            print(f"paired-paired experiment with {ptcp_id}")
            paired_trainloader = data_setup.get_paired_dataloader(ptcp_id=ptcp_id)
            paired_testloader = data_setup.get_paired_dataloader(ptcp_id=ptcp_id, train=False)

            vae_stgcn = models.get_vae_stgcn(device=device)

            vae_stgcn_train_rst_dict = train_vae_stgcn(model=vae_stgcn, trainloader=paired_trainloader, device=device)
            
            predictor = models.get_predictor(vae_stgcn=vae_stgcn, device=device)
            
            predictor_train_rst_dict = train_predictor(model=predictor, trainloader=paired_trainloader, device=device)
            
            test_rst_dict, y_true, y_pred = test_predictor(model=predictor, testloader=paired_testloader, device=device)
            
            print(f"test loss: {test_rst_dict['loss'][0]:.2f} | test acc: {test_rst_dict['acc'][0]:.2f}")
            print()
            
            vae_stgcn_train_rst_dicts[ptcp_id] = vae_stgcn_train_rst_dict
            predictor_train_rst_dicts[ptcp_id] = predictor_train_rst_dict
            test_rst_dicts[ptcp_id] = test_rst_dict
            y_true_dicts[ptcp_id] = y_true
            y_pred_dicts[ptcp_id] = y_pred
    
    elif task=="grouped-paired":
        print("grouped-paired experiment")
        
        vae_stgcn_train_rst_dicts = {}
        predictor_train_rst_dicts = {}
        test_rst_dicts = {}
        y_true_dicts = {}
        y_pred_dicts = {}
        
        grouped_trainloader = data_setup.get_grouped_dataloader(ptcp_id='all')
        paired_testloader = data_setup.get_paired_dataloader(ptcp_id='all')
        
        vae_stgcn = models.get_vae_stgcn(device=device)

        vae_stgcn_train_rst_dict = train_vae_stgcn(model=vae_stgcn, trainloader=grouped_trainloader, device=device)
        
        predictor = models.get_predictor(vae_stgcn=vae_stgcn, device=device)
        
        predictor_train_rst_dict = train_predictor(model=predictor, trainloader=grouped_trainloader, device=device)
        
        test_rst_dict, y_true, y_pred = test_predictor(model=predictor, testloader=paired_testloader, device=device)
        
        print(f"test loss: {test_rst_dict['loss'][0]:.2f} | test acc: {test_rst_dict['acc'][0]:.2f}")
        print()
        
        vae_stgcn_train_rst_dicts['all'] = vae_stgcn_train_rst_dict
        predictor_train_rst_dicts['all'] = predictor_train_rst_dict
        test_rst_dicts['all'] = test_rst_dict
        y_true_dicts['all'] = y_true
        y_pred_dicts['all'] = y_pred
        
        print(f"test loss: {test_rst_dict['loss']} | test acc: {test_rst_dict['acc']}")
        print()
        
    return vae_stgcn_train_rst_dict, predictor_train_rst_dict, test_rst_dicts, y_true_dicts, y_pred_dicts

def plot_cfm(y_true, y_pred):
    cfm = confusion_matrix(y_true, y_pred)
    
    ConfusionMatrixDisplay(np.round(cfm / cfm.astype(np.float).sum(axis=1), 4)).plot()