import data_setup, models, engine
import torch
import pickle

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("model", choices=["lstm", "stgcn"], help="choose model to experiment")
parser.add_argument("task", choices=["grouped-grouped", "paired-paired", "grouped-paired", "grouped_by_ptcp-paired", "aug_grouped_by_ptcp-paired"], help="choose task to do in the experiment")
parser.add_argument("directory_name", help="give directory to save results.")
parser.add_argument("etc", help="specific configuration")

args = parser.parse_args()

device = torch.device("cuda:0")

print(f"{args.model} in {args.task}")
print(f"{args.etc}")
#print("test with no additional layer in predictor")

if args.model == "lstm":
    train_rst_dicts, test_rst_dicts, y_true_dicts, y_pred_dicts = engine.train_test_lstm(task=args.task, experiment_name=args.directory_name, device=device)
    
    save_dict = {
    	'train_rst_dicts':train_rst_dicts,
    	'test_rst_dicts':test_rst_dicts,
    	'y_true_dicts':y_true_dicts,
    	'y_pred_dicts':y_pred_dicts
    }

    #with open(f'./project/result/dict_outputs/lstm/{args.task}/lstm_{args.etc}.pickle', 'wb') as f:
    with open(f'/mnt/iusers01/fse-ugpgt01/compsci01/y44694jk/project/result/dict_outputs/lstm/{args.directory_name}/lstm_{args.task}_{args.etc}.pickle', 'wb') as f:
	    pickle.dump(save_dict, f)

elif args.model == "stgcn":
    vae_stgcn_train_rst_dict, predictor_train_rst_dict, test_rst_dicts, y_true_dicts, y_pred_dicts = engine.train_test_predictor(task=args.task, experiment_name=args.directory_name, device=device)
    
    save_dict = {
    	'vae_stgcn_train_rst_dict':vae_stgcn_train_rst_dict,
    	'predictor_train_rst_dict':predictor_train_rst_dict,
    	'test_rst_dicts':test_rst_dicts,
    	'y_true_dicts':y_true_dicts,
    	'y_pred_dicts':y_pred_dicts
    }

    #with open(f'./project/result/dict_outputs/stgcn/{args.task}/stgcn_{args.etc}.pickle', 'wb') as f:
    with open(f'/mnt/iusers01/fse-ugpgt01/compsci01/y44694jk/project/result/dict_outputs/stgcn/{args.directory_name}/stgcn_{args.task}_{args.etc}.pickle', 'wb') as f:
        pickle.dump(save_dict, f)


