import json

import numpy as np

import torch

from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, config):
        self.config = config
        self.training_losses = []
        self.best_training_loss = float("inf")

        self.valid_rmse = []
        self.best_valid_rmse = float("inf")

        self.TBWriter_path = self.get_TBwriter()

        self.exp_directory = config["exp_path"]
        self.exp_name = config["exp_name"]
        self.exp_log = self.exp_directory + "logs/" + self.config["city"] + "_" + self.exp_name + "_" + str(self.config["random_seed"]) + ".json"
       
        self.model_directory = config["exp_path"] + "weights/"
        self.model_name = "model_" + self.config["city"]
        self.model_path = self.model_directory + self.model_name + "_" + self.exp_name + "_" + str(self.config["random_seed"]) + '.pkl'

        self.optimizer_directory = config["exp_path"] + "optimizers/"
        self.optimizer_name = "optm_" + self.config["city"]
        self.optimizer_path = self.optimizer_directory + self.optimizer_name + "_" + self.exp_name + "_" + str(self.config["random_seed"]) + '.pkl'

        self.prediction_directory = self.exp_directory + "results/"
        self.prediction_name = "prediction_" + self.config["city"]
        self.groundtruth_name = "groundtruth_" + self.config["city"]
        self.prediction_path = self.prediction_directory + self.prediction_name + "_" + self.exp_name + "_" + str(self.config["random_seed"]) + '.npy'
        self.groundtruth_path = self.prediction_directory + self.groundtruth_name + "_" + self.exp_name + "_" + str(self.config["random_seed"]) + '.npy'
        self.pair_idx_path = self.prediction_directory + self.prediction_name + "_" + self.exp_name + "_" + str(self.config["random_seed"]) + '_idx.npy'

        self.lowest_trainingloss_flags = []

        self.lowest_validloss_flags = []

        self.exp_content = {
                         "config" : self.config,

                         "training_loss" : self.training_losses,
                         "valid_rmse" : self.valid_rmse,
                        
                         "eval_results" : {
                             "RMSE" : float("inf"),
                             "NRMSE" : float("inf"),
                             "MAE" : float("inf"),
                             "MAPE" : float("inf"),
                             "SMAPE" : float("inf"),
                             "CPC" : 0
                          }
                         }

    def get_TBwriter(self):
        writer = SummaryWriter(log_dir=self.config["exp_path"]+"runs/"+self.config["exp_name"], flush_secs=10)
        self.writer = writer

    def log_training_loss(self, current_loss):
        # print("loss = ", current_loss)
        self.training_losses.append(float(current_loss))
        if current_loss < self.best_training_loss:
            self.best_training_loss = current_loss
            self.lowest_trainingloss_flags = []
        else:
            self.lowest_trainingloss_flags.append(1)
        

    def log_valid_rmse(self, rmse):
        # print("valid_rmse = ", rmse)
        self.valid_rmse.append(float(rmse))

    def log_results(self, rmse = float("inf"), 
                          nrmse = float("inf"), 
                          mae = float("inf"), 
                          mape = float("inf"), 
                          smape = float("inf"), 
                          cpc = 0):
        self.exp_content["eval_results"]["RMSE"] = float(rmse)
        self.exp_content["eval_results"]["NRMSE"] = float(nrmse)
        self.exp_content["eval_results"]["MAE"] = float(mae)
        self.exp_content["eval_results"]["MAPE"] = float(mape)
        self.exp_content["eval_results"]["SMAPE"] = float(smape)
        self.exp_content["eval_results"]["CPC"] = float(cpc)

    def log_test_prediction(self, prediction):
        prediction = prediction.cpu().numpy()
        np.save(self.prediction_path, prediction)

    def log_test_groundtruth(self, groundtruth):
        groundtruth = groundtruth.cpu().numpy()
        np.save(self.groundtruth_path, groundtruth)

    def log_pred_odpair_idx(self, od_pair_idx):
        idx_save = []
        for one_item in od_pair_idx:
            one_item = one_item.cpu().numpy().reshape([1, -1])
            idx_save.append(one_item)
        od_pair_idx = np.concatenate(idx_save, axis=0)
        np.save(self.pair_idx_path, od_pair_idx)

    def check_save_model(self, rmse, model, optimizer):
        if rmse < self.best_valid_rmse:
            self.best_valid_rmse = rmse
            self.lowest_validloss_flags = []
            torch.save(model.state_dict(), self.model_path)
            torch.save(optimizer.state_dict(), self.optimizer_path)
            print("Best valid RMSE ever and save the model.")
        else:
            self.lowest_validloss_flags.append(1)

    def check_overfitting(self, current_valid_rmse):
        self.log_valid_rmse(current_valid_rmse)

        if len(self.lowest_validloss_flags) > self.config["overfit_EPOCH"]:
            print("Overfitting!")
            return True
        else:
            print("Overfitting : " + int(len(self.lowest_validloss_flags) * (50 / self.config["overfit_EPOCH"])) * "*" + (50 - int(len(self.lowest_validloss_flags) * (50 / self.config["overfit_EPOCH"]))) * "-")
            return False

    def check_converge(self):
        if len(self.lowest_trainingloss_flags) > self.config["converge_EPOCH"]:
            print("Converged!!!")
            return True
        else:
            print("Convergence : " + int(len(self.lowest_trainingloss_flags) * (50 / self.config["converge_EPOCH"]))  * "*" + (50 - int(len(self.lowest_trainingloss_flags) * (50 / self.config["overfit_EPOCH"])) ) * "-")
            return False

    def clear_check(self):
        self.lowest_trainingloss_flags = []
        self.lowest_validloss_flags = []
        


    def save_exp_log(self):
        json.dump(self.exp_content, open(self.exp_log, "w"), indent=4)
