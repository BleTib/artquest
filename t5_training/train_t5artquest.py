import argparse
import copy
import json
import os
import time

import torch
from helpers import (average_accs, average_losses, init_accs_dict,
                     init_losses_dict, init_optimizer, init_train_str,
                     losses_output, make_log_dict, output_training_end,
                     save_model_and_logs)
from helpers_vis import *
from tqdm import tqdm

import wandb

seed = 0
device = ""


def paths_exist(cfg):
    if not os.path.exists(cfg["output_path"]):
        print("Output path does not exist. EXIT.")
        return False
    if cfg.get("image_path") is not None and not os.path.exists(cfg["image_path"]):
        print("Image path does not exist. EXIT.")
        return False
    if not os.path.exists(cfg["traindata"]):
        print("Traindata does not exist. EXIT.")
        return False
    if not os.path.exists(cfg["valdata"]):
        print("Valdata does not exist. EXIT.")
        return False
    return True


def train(cfg={}):
    cfg["gpu_id"] = 0

    wandb.init(config=cfg, project=cfg.get("project"))
    cfg.update(wandb.config)

    if paths_exist(cfg) != True:
        return
    # Models are in models dict
    models = init_models(cfg)
    # Losses in dict and optimizer
    optimizer = init_optimizer(models, cfg)
    # Dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(cfg)
    wandb.config.update(cfg)
    train_str = init_train_str(cfg)

    # Training
    start_time = time.time()
    best_em = 0
    epochs_no_improvement = 0
    model_cp = {}
    for epoch in tqdm(range(cfg["epochs"])):
        losses_dict = init_losses_dict()
        accs_dict = init_accs_dict()
        train_str += f"Epoch {str(epoch)}" + "\n"
        print("Epoch", epoch)
        train_time = time.time()
        models["model"].train(not ast.literal_eval(cfg["eval_mode"]))
        for batch in tqdm(train_dataloader, ascii=" #"):
            optimizer.zero_grad()
            loss = calc_losses(batch, models,
                               losses_dict, accs_dict, cfg, "train")
            loss.backward()
            optimizer.step()

        average_losses(losses_dict, cfg, "train")
        train_str += losses_output(losses_dict, accs_dict, train_time, "train")

        train_str += "Inference\n"
        print("Running Inference")
        inference_time = time.time()
        models["model"].eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader, ascii=" >="):
                calc_losses(batch, models,
                            losses_dict, accs_dict, cfg, "val")
        average_accs(accs_dict, cfg)
        average_losses(losses_dict, cfg, "val")
        train_str += losses_output(losses_dict,
                                   accs_dict, inference_time, "val")

        # check if val loss did improve
        if accs_dict["em_val"] > best_em or accs_dict["em_val"] == 0:
            model_cp["model_state"] = copy.deepcopy(
                models["model"].state_dict())
            model_cp["optimizer_state"] = copy.deepcopy(optimizer.state_dict())
            model_cp["epoch"] = epoch
            model_cp["loss"] = losses_dict["ar_loss"]
            best_em = accs_dict["em_val"]
            accs_dict.update({"best_em_val": best_em})
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
        # Log to wandb
        wandb.log(losses_dict, step=epoch)
        wandb.log(accs_dict, step=epoch)
        # Check early stopping
        if epochs_no_improvement == cfg["early_stopping_epochs"]:
            print(
                f'Early stopping! For {str(cfg["early_stopping_epochs"])} epochs the val loss did not improve!')
            break

    # End of Training
    train_str += output_training_end(losses_dict, start_time)
    # load best checkpoint
    models["model"].load_state_dict(model_cp["model_state"])
    # calc metrics and logging
    dfs = {}
    print("Evaluation: val set")
    dfs["val"] = get_pred_df_from_dataloader(val_dataloader, models, cfg)
    string, metrics = all_metrics_to_string(dfs["val"])
    print("em_val:", metrics["all"]["em"], "bleu_val:", metrics["all"]["bleu"])
    metrics_logging = make_log_dict(metrics, "val")
    wandb.log(metrics_logging)
    train_str += "\nval set:\n\n"+string
    print("Evaluation: test set")
    dfs["test"] = get_pred_df_from_dataloader(test_dataloader, models, cfg)
    string, metrics = all_metrics_to_string(dfs["test"])
    print("em_test:", metrics["all"]["em"],
          "bleu_test:", metrics["all"]["bleu"])
    metrics_logging = make_log_dict(metrics, "test")
    wandb.log(metrics_logging)
    train_str += "test set:\n\n" + string
    wandb.log(get_type_accuracy(dfs["test"]))
    save_model_and_logs(cfg["output_path"], train_str,
                        dfs, model_cp, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config', help='config.json with parameters', required=True)
    args = vars(parser.parse_args())

    with open(args["config"]) as json_file:
        cfg = (json.load(json_file))
        train(cfg)
