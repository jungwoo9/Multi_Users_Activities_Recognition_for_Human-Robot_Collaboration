import torch.optim as optim
import torch
import torch.nn as nn

import data_setup, models

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import pickle
import json

import matplotlib.pyplot as plt

def load_configs(experiment_name, model, csf3):
    if csf3:
        path = f"../project/config/{model}/{experiment_name}.json"
    else:
        # path = f'./config/lstm/experiment_name.json'
        path = f"/content/drive/MyDrive/3rd_year_project/config/{model}/{experiment_name}.json"

    with open(path) as f: 
        data = f.read() 

    js = json.loads(data)

    return js

class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < 0.1:
            return True
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_lstm(model, trainloader, testloader, config, device='cpu'):
    # Loss and optimizer
    cross_entropy_weights = torch.tensor(config["cross_entropy_weights"]).to(torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(cross_entropy_weights)
    if config["optimizer"]=="Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    
    train_rst_dict = {'loss':[], 'acc':[]}
    
    early_stopper = EarlyStopping(patience=config["patience"], min_delta=config["min_delta"])
    
    for epoch in range(config["num_epochs"]):
        # train
        model.train()
        
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
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        
        # test
        test_rst_dict, y_true, y_pred = test_lstm(model, testloader, device)
        print(f"test loss: {test_rst_dict['loss'][0]:.2f} | test acc: {test_rst_dict['acc'][0]:.2f}")
        
        # early stop
        if early_stopper.early_stop(test_rst_dict['loss'][0]):          
            print('early stopped')
            break
      
    return train_rst_dict, test_rst_dict, y_true, y_pred

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
    
def train_test_lstm(task='grouped-grouped', experiment_name="example", device='cpu', csf3=True):
    lstm_configs = load_configs(experiment_name, model="lstm", csf3=csf3)

    if task=="grouped-grouped":
        train_rst_dicts = {}
        test_rst_dicts = {}
        y_true_dicts = {}
        y_pred_dicts = {}
        
        for ptcp_id in ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11']:
            print(f"grouped-grouped experiment with {ptcp_id}")
            grouped_trainloader = data_setup.get_grouped_dataloader(ptcp_id=ptcp_id, batch_size=lstm_configs["batch_size"],  csf3=csf3)
            grouped_testloader = data_setup.get_grouped_dataloader(ptcp_id=ptcp_id, train=False, batch_size=lstm_configs["batch_size"], csf3=csf3)

            lstm = models.get_lstm(hidden_size=lstm_configs["number_of_hidden_layers"], batch=lstm_configs["batch_normalization"], device=device)

            train_rst_dict, test_rst_dict, y_true, y_pred = train_lstm(model=lstm, trainloader=grouped_trainloader, testloader=grouped_testloader, config=lstm_configs, device=device)
            
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
            paired_trainloader = data_setup.get_paired_dataloader(ptcp_id=ptcp_id, batch_size=lstm_configs["batch_size"], csf3=csf3)
            paired_testloader = data_setup.get_paired_dataloader(ptcp_id=ptcp_id, train=False, batch_size=lstm_configs["batch_size"], csf3=csf3)

            lstm = models.get_lstm(hidden_size=lstm_configs["number_of_hidden_layers"], batch=lstm_configs["batch_normalization"], device=device)

            train_rst_dict, test_rst_dict, y_true, y_pred = train_lstm(model=lstm, trainloader=paired_trainloader, testloader=paired_testloader, config=lstm_configs, device=device)
            
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
        
        grouped_trainloader = data_setup.get_grouped_dataloader(ptcp_id='all', batch_size=lstm_configs["batch_size"], csf3=csf3)
        paired_testloader = data_setup.get_paired_dataloader(ptcp_id='all', batch_size=lstm_configs["batch_size"], csf3=csf3)
        
        lstm = models.get_lstm(hidden_size=lstm_configs["number_of_hidden_layers"], batch=lstm_configs["batch_normalization"], device=device)

        train_rst_dict, test_rst_dict, y_true, y_pred = train_lstm(model=lstm, trainloader=grouped_trainloader, testloader=paired_testloader, config=lstm_configs, device=device)
        
        train_rst_dicts['all'] = train_rst_dict
        test_rst_dicts['all'] = test_rst_dict
        y_true_dicts['all'] = y_true
        y_pred_dicts['all'] = y_pred

    elif task=="grouped_by_ptcp-paired":
        train_rst_dicts = {}
        test_rst_dicts = {}
        y_true_dicts = {}
        y_pred_dicts = {}
        
        for ptcp_id in ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11']:
            print(f"grouped_by_ptcp-paired experiment with {ptcp_id}")
            grouped_by_ptcp_trainloader = data_setup.get_grouped_by_ptcp_dataloader(ptcp_id=ptcp_id, batch_size=lstm_configs["batch_size"], csf3=csf3)
            paired_testloader = data_setup.get_paired_dataloader(ptcp_id='all', batch_size=lstm_configs["batch_size"], csf3=csf3)

            lstm = models.get_lstm(hidden_size=lstm_configs["number_of_hidden_layers"], batch=lstm_configs["batch_normalization"], device=device)

            train_rst_dict, test_rst_dict, y_true, y_pred = train_lstm(model=lstm, trainloader=grouped_by_ptcp_trainloader, testloader=paired_testloader, config=lstm_configs, device=device)
            
            train_rst_dicts[ptcp_id] = train_rst_dict
            test_rst_dicts[ptcp_id] = test_rst_dict
            y_true_dicts[ptcp_id] = y_true
            y_pred_dicts[ptcp_id] = y_pred
        
    return train_rst_dicts, test_rst_dicts, y_true_dicts, y_pred_dicts

def train_vae_stgcn(model, trainloader, config, device='cpu'):
    model.train()
    
    # Loss and optimizer
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])

    train_rst_dict = {'loss':[]}
    
    for epoch in range(config['num_epochs_vae']):
        train_loss = 0
        for x, _ in trainloader:
            x = x.to(device)
            
            # Forward pass
            x_mean, x_var, z, mean, log_var = model(x)
        
            # Compute loss
            loss = models.vae_loss(x.to(torch.float32), x_mean, x_var, z, mean, log_var, device)
        
            # Backward and optimize
            optimizer.zero_grad()
        
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
        
        train_loss /= len(trainloader)
        train_rst_dict['loss'].append(train_loss)
        print(f'Epoch [{epoch+1}/{config["num_epochs_vae"]}], Loss: {train_loss:.4f}')
      
    return train_rst_dict
    
def train_predictor(model, trainloader, testloader, config, device='cpu'):
    # Loss and optimizer
    cross_entropy_weights = torch.tensor(config["cross_entropy_weights"]).to(torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(cross_entropy_weights)
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    
    train_rst_dict = {'loss':[], 'acc':[]}
    
    early_stopper = EarlyStopping(patience=config["patience"], min_delta=config["min_delta"])
    
    for epoch in range(config["num_epochs_predictor"]):
        model.train()
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
        print(f'Epoch [{epoch+1}/{config["num_epochs_predictor"]}], Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        
        # test
        test_rst_dict, y_true, y_pred = test_predictor(model, testloader, device)
        print(f"test loss: {test_rst_dict['loss'][0]:.2f} | test acc: {test_rst_dict['acc'][0]:.2f}")
        
        if early_stopper.early_stop(test_rst_dict['loss'][0]):          
            print('early stopped')
            break
        
    return train_rst_dict, test_rst_dict, y_true, y_pred
    
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

def train_test_predictor(task='grouped-grouped', experiment_name="example", device='cpu', csf3=True):
    stgcn_configs = load_configs(experiment_name, model="stgcn", csf3=csf3)

    if task=="grouped-grouped":
        vae_stgcn_train_rst_dicts = {}
        predictor_train_rst_dicts = {}
        test_rst_dicts = {}
        y_true_dicts = {}
        y_pred_dicts = {}
        
        for ptcp_id in ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11']:
            print(f"grouped-grouped experiment with {ptcp_id}")
            grouped_trainloader = data_setup.get_grouped_dataloader(ptcp_id=ptcp_id, batch_size=stgcn_configs["batch_size"], csf3=csf3)
            grouped_testloader = data_setup.get_grouped_dataloader(ptcp_id=ptcp_id, batch_size=stgcn_configs["batch_size"], train=False, csf3=csf3)

            vae_stgcn = models.get_vae_stgcn(args=stgcn_configs, device=device)

            vae_stgcn_train_rst_dict = train_vae_stgcn(model=vae_stgcn, trainloader=grouped_trainloader, config=stgcn_configs, device=device)
            
            predictor = models.get_predictor(vae_stgcn=vae_stgcn, args=stgcn_configs, device=device)
            
            predictor_train_rst_dict, test_rst_dict, y_true, y_pred = train_predictor(model=predictor, trainloader=grouped_trainloader, testloader=grouped_testloader, config=stgcn_configs, device=device)

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
            paired_trainloader = data_setup.get_paired_dataloader(ptcp_id=ptcp_id, batch_size=stgcn_configs["batch_size"], csf3=csf3)
            paired_testloader = data_setup.get_paired_dataloader(ptcp_id=ptcp_id, batch_size=stgcn_configs["batch_size"], train=False, csf3=csf3)

            vae_stgcn = models.get_vae_stgcn(args=stgcn_configs, device=device)

            vae_stgcn_train_rst_dict = train_vae_stgcn(model=vae_stgcn, trainloader=paired_trainloader, config=stgcn_configs, device=device)
            
            predictor = models.get_predictor(vae_stgcn=vae_stgcn, args=stgcn_configs, device=device)
            
            predictor_train_rst_dict, test_rst_dict, y_true, y_pred = train_predictor(model=predictor, trainloader=paired_trainloader, testloader=paired_testloader, config=stgcn_configs, device=device)
            
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
        
        grouped_trainloader = data_setup.get_grouped_dataloader(ptcp_id='all', batch_size=stgcn_configs["batch_size"], csf3=csf3)
        paired_testloader = data_setup.get_paired_dataloader(ptcp_id='all', batch_size=stgcn_configs["batch_size"], csf3=csf3)
        
        vae_stgcn = models.get_vae_stgcn(args=stgcn_configs, device=device)

        vae_stgcn_train_rst_dict = train_vae_stgcn(model=vae_stgcn, trainloader=grouped_trainloader, config=stgcn_configs, device=device)
        
        predictor = models.get_predictor(vae_stgcn=vae_stgcn, args=stgcn_configs, device=device)
        
        predictor_train_rst_dict, test_rst_dict, y_true, y_pred = train_predictor(model=predictor, trainloader=grouped_trainloader, testloader=paired_testloader, config=stgcn_configs, device=device)
        
        vae_stgcn_train_rst_dicts['all'] = vae_stgcn_train_rst_dict
        predictor_train_rst_dicts['all'] = predictor_train_rst_dict
        test_rst_dicts['all'] = test_rst_dict
        y_true_dicts['all'] = y_true
        y_pred_dicts['all'] = y_pred

    elif task=="grouped_by_ptcp-paired":
        vae_stgcn_train_rst_dicts = {}
        predictor_train_rst_dicts = {}
        test_rst_dicts = {}
        y_true_dicts = {}
        y_pred_dicts = {}
        
        for ptcp_id in ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11']:
            print(f"grouped_by_ptcp-paired experiment with {ptcp_id}")
            grouped_by_ptcp_trainloader = data_setup.get_grouped_by_ptcp_dataloader(ptcp_id=ptcp_id, batch_size=stgcn_configs["batch_size"], csf3=csf3)
            paired_testloader = data_setup.get_paired_dataloader(ptcp_id='all', batch_size=stgcn_configs["batch_size"], csf3=csf3)

            vae_stgcn = models.get_vae_stgcn(args=stgcn_configs, device=device)

            vae_stgcn_train_rst_dict = train_vae_stgcn(model=vae_stgcn, trainloader=grouped_by_ptcp_trainloader, config=stgcn_configs, device=device)
            
            predictor = models.get_predictor(vae_stgcn=vae_stgcn, args=stgcn_configs, device=device)
            
            predictor_train_rst_dict, test_rst_dict, y_true, y_pred = train_predictor(model=predictor, trainloader=grouped_by_ptcp_trainloader, testloader=paired_testloader, config=stgcn_configs, device=device)
            
            vae_stgcn_train_rst_dicts[ptcp_id] = vae_stgcn_train_rst_dict
            predictor_train_rst_dicts[ptcp_id] = predictor_train_rst_dict
            test_rst_dicts[ptcp_id] = test_rst_dict
            y_true_dicts[ptcp_id] = y_true
            y_pred_dicts[ptcp_id] = y_pred
        
    return vae_stgcn_train_rst_dicts, predictor_train_rst_dicts, test_rst_dicts, y_true_dicts, y_pred_dicts

def save_result(dict_keys, dict_items, dict_name):
    # path_to_save = "/content/drive/MyDrive/3rd_year_project/result/output_dict/" + dict_name.split("_")[0] + "/" + dict_name + ".pkl"
    path_to_save = "./result/output_dict/" + dict_name.split("_")[0] + "/" + dict_name + ".pkl"

    dict_ = {k:i for (k, i) in zip(dict_keys, dict_items)}

    with open(path_to_save, 'wb') as f:
        pickle.dump(dict_, f)

def plot_cfm(y_true, y_pred, title="Confusion Matrix"):
    cfm = confusion_matrix(y_true, y_pred)
    cfmn = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]

    cfmn = np.round(cfmn, 3)
    
    # ConfusionMatrixDisplay(confusion_matrix=cfmn, display_labels=['WW', "WR", "WP", "RW", "RR", "RP", "PW", "PR", "PP"]).plot()
    disp = ConfusionMatrixDisplay(confusion_matrix=cfmn, display_labels=['WW', "WR", "WP", "RW", "RR", "RP", "PW", "PR", "PP"])
    disp.plot(ax=None, xticks_rotation='horizontal')
    plt.title(title)
    plt.show()