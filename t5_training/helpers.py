import ast
import os
import time
from datetime import datetime
import numpy as np
import shutil
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from metrics import eval_df
import wandb

device = ""


def set_device(dev):
    global device
    device = dev


def load_model(model_path, model):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])


def init_optimizer(models, cfg):
    # Optimizer
    optimizer = AdamW(models["model"].parameters(), lr=cfg["learning_rate"],
                      betas=ast.literal_eval(cfg["betas"]), eps=cfg["eps"], weight_decay=cfg["weight_decay"])
    return  optimizer


def init_losses_dict():
    losses = ["ar"]
    losses_dict = {}
    for loss in losses:
        losses_dict[loss+"_loss"] = []
        losses_dict[loss+"_loss_val"] = []
    return losses_dict


def init_accs_dict():
    accs = ["em"]
    accs_dict = {}
    for acc in accs:
        accs_dict[acc+"_val"] = []
    return accs_dict


def make_inputs(questions, contexts, answers, models, cfg):
    appendix = torch.tensor([[1948,    35, 50264,     2]])
    inputs = []
    groundtruth = []
    for q, c, a in zip(questions, contexts, answers):
        input = f"question: {q}? context: {c}"

        q_c_ids = models["tokenizer"](
            input, padding=True, return_tensors="pt")["input_ids"]

        answer_ids = models["tokenizer"](f" answer: {a}", padding=True,
                                         return_tensors="pt")["input_ids"][:,1:]

        cut_length = 1024-answer_ids.shape[1]
        if q_c_ids.shape[1] < cut_length:
            ids_question = torch.cat((q_c_ids[:, :-1], appendix), dim=1)
            ids_label = torch.cat((q_c_ids[:, :-1], answer_ids), dim=1)
        else:
            ids_question = torch.cat((q_c_ids[:, :cut_length], appendix), dim=1)
            ids_label = torch.cat(
                (q_c_ids[:, :cut_length], answer_ids), dim=1)
        inputs.append(ids_question.squeeze())
        groundtruth.append(ids_label.squeeze())

    input_ids = pad_sequence(inputs, batch_first=True,
                             padding_value=-100).to(device)
    labels = pad_sequence(groundtruth, batch_first=True,
                          padding_value=-100).to(device)

    if input_ids.shape[1] > labels.shape[1]:
        cat_tensor = torch.full((labels.shape[0], input_ids.shape[1] -
                                 labels.shape[1]), models["tokenizer"].pad_token_id).to(device)
        labels = torch.cat((labels, cat_tensor), dim=1)

    elif labels.shape[1] > input_ids.shape[1]:
        cat_tensor = torch.full((input_ids.shape[0], labels.shape[1] -
                                 input_ids.shape[1]), models["tokenizer"].pad_token_id).to(device)
        input_ids = torch.cat((input_ids, cat_tensor), dim=1)
    labels[labels == models["tokenizer"].pad_token_id] = -100
    if len(input_ids)>1024:
        print("input_ids",input_ids.shape)
    if len(labels)>1024:
        print("labels",input_ids.shape)
    return input_ids, labels


def sum_losses_per_batch(losses_dict, loss, batchsize, train_val):
    for k, v in loss.items():
        if train_val == "train":
            losses_dict[f'{k}_loss'].append(v.item()*batchsize)
        if train_val == "val":
            losses_dict[f'{k}_loss_val'].append(v.item()*batchsize)


def average_losses(losses_dict, cfg, train_val):
    for k, v in losses_dict.items():
        if train_val == "train":
            if k[-3:] != "val":
                losses_dict.update({k: np.sum(v)/cfg["train_size"]})
        if train_val == "val":
            if k[-3:] == "val" and k[:4] != "best":
                losses_dict.update({k: np.sum(v)/cfg["val_size"]})


def average_accs(accs_dict, cfg):
    for k, v in accs_dict.items():
        accs_dict.update({k: np.sum(v)/cfg["val_size"]})


def init_train_str(cfg):
    return "".join(k+": "+str(v)+"\n" for k, v in cfg.items())


def _losses_to_string(losses_dict, train_val):
    string = ""
    for k, v in losses_dict.items():
        if train_val == "train" and k[-3:] != "val":
            string += f'{k}: {str(v)}, '
        if train_val == "val" and k[-3:] == "val":
            string += f'{k}: {str(v)}, '

    return string


def _accs_to_string(accs_dict):
    string = ""
    for k, v in accs_dict.items():
        string += f'{k}: {str(v)}, '
    return string


def metrics_to_string(metrics_dict):
    string = ""
    for k, v in metrics_dict.items():
        string += f'{k}: {str(v)}\n'
    return string


def losses_output(losses_dict, accs_dict, start_time, train_val):
    output_losses = _losses_to_string(losses_dict, train_val)
    if train_val == "val":
        output_accs = _accs_to_string(accs_dict)
    else:
        output_accs = ""
    print(output_losses, output_accs)
    return output_losses + output_accs+"\n"+"--- %s minutes ---\n" % (
        round(time.time() - start_time)/60)


def all_metrics_to_string(df, cfg):
    string = ""
    metrics = {}
    metrics["all"] = eval_df(df)
    if len(df[df["external_knowledge"] == False]) > 0:
        metrics["visual"] = eval_df(df[df["external_knowledge"] == False])
    if len(df[df["external_knowledge"] == True]) > 0:
        metrics["knowledge"] = eval_df(df[df["external_knowledge"] == True])
    for k, v in metrics.items():
        string += k + "\n"+metrics_to_string(v)+"\n"
    return string, metrics


def make_log_dict(metrics, split):
    dic = {}
    for s, d in metrics.items():
        for k, v in d.items():
            if s != "all":
                k += "_"+s
            k += "_"+split
            dic[k] = v
    return dic


def save_model(model_cp, path, name):
    torch.save({
        'epoch': model_cp["epoch"],
        'model_state_dict': model_cp["model_state"],
        'optimizer_state_dict': model_cp["optimizer_state"],
        'loss': model_cp["loss"],
    }, path+name+".pt")


def false_preds_to_string(df):
    string = ""
    for i, d in df.iterrows():
        if d.answer != d.pred:
            string += "image: "+d.image+"\n"
            string += "question: "+d.question+"\n"
            string += "gt  : "+d.answer+"\n"
            string += "pred: "+d.pred+"\n\n"
    return string


def save_model_and_logs(output_path, train_str, dfs, model_cp, cfg):
    dirname="t5_artquest_"+cfg["book_variant"]
    if cfg["text_only"]=="True":
        dirname+="_textonly"
    savedir = os.path.join(
        output_path, dirname)
    if os.path.exists(savedir):
        shutil.rmtree(savedir)
    os.makedirs(savedir, )
    if cfg["save_model"] == "True":
        print("Saving model to: " + savedir)
        save_model(model_cp, savedir+"/", "model")
    with open(savedir+"/trainlog.txt", "w+") as f:
        f.writelines(train_str)
    dfs["val"].to_csv(savedir+"/val_df.csv")
    dfs["test"].to_csv(savedir+"/test_df.csv")
    wandb.save(savedir+"/*", base_path=savedir, policy="now")

def output_training_end(losses_dict, start_time):
    train_str = "--- Total time: %s minutes ---\n\n" % (
        round(time.time() - start_time)/60)
    train_str += "".join(k+": "+str(v)+"\n" for k,
                         v in losses_dict.items())
    return train_str