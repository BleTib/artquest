import argparse
import ast
import copy
import json
import os
import time

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb

import os
os.environ["WANDB_MODE"]="offline"
torch.manual_seed(42)


class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_txt, preprocess):

        self.image_path = list_image_path
        # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.title = clip.tokenize(list_txt, truncate=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Image from PIL module
        image = self.preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image, title

# https://github.com/openai/CLIP/issues/57


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def save_model(epoch, model_state, optimizer_state, total_loss, path, name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'loss': total_loss,
    }, path+name+".pt")


def load_model(model_path, model):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])


def save_model_and_logs(output_path, train_str, epoch, model, optimizer, total_loss):
    savedir = os.path.join(
        output_path, "clip_finetuned")
    os.makedirs(savedir, exist_ok=True)
    with open(savedir+"/losses.txt", "w+") as f:
        f.writelines(train_str)
    print("Saving model to: " + savedir)
    save_model(epoch, model, optimizer, total_loss,
               savedir+"/", "clip_finetuned")
    wandb.save(savedir+"/*", base_path=savedir, policy="now")


def paths_exist(cfg):
    if not os.path.exists(cfg["output_path"]):
        print("Output path does not exist. EXIT.")
        return False
    if not os.path.exists(cfg["image_path"]):
        print("Image path does not exist. EXIT.")
        return False
    if not os.path.exists(cfg["traindata"]):
        print("Traindata does not exist. EXIT.")
        return False
    if not os.path.exists(cfg["valdata"]):
        print("Valdata does not exist. EXIT.")
        return False
    return True


def init_model(cfg):
    # If using GPU then use mixed precision training.
    device = f"cuda:{cfg['gpu_id']}" if torch.cuda.is_available() else "cpu"
    print("Running on "+device)
    # Must set jit=False for training
    model, preprocess = clip.load(cfg["backbone"], device=device, jit=False)

    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)

    return device, model, preprocess


def read_data(data_path, image_path, preprocess, batch_size, shuffle):
    data = pd.read_csv(data_path, encoding='latin1',sep="\t")[:10]
    list_image_path = [image_path+"/" +
                       path for path in data["IMAGE_FILE"].tolist()]
    list_txt = data["DESCRIPTION"].tolist()
    dataset = image_title_dataset(list_image_path, list_txt, preprocess)
    return DataLoader(dataset, batch_size=batch_size, shuffle=ast.literal_eval(shuffle))


def calc_losses(batch, device, model, loss_img, loss_txt, losses):
    images, texts = batch

    images = images.to(device)
    texts = texts.to(device)

    logits_per_image, logits_per_text = model(images, texts)

    ground_truth = torch.arange(
        len(images), dtype=torch.long, device=device)

    image_loss = loss_img(logits_per_image, ground_truth)
    text_loss = loss_txt(logits_per_text, ground_truth)
    total_loss = (image_loss + text_loss)/2

    losses["total_loss"].append(total_loss.item()*len(batch[0]))
    losses["image_loss"].append(image_loss.item()*len(batch[0]))
    losses["text_loss"].append(text_loss.item()*len(batch[0]))

    return total_loss


def losses_to_string(losses, batchsize):
    return "total_loss: {}, image_loss: {}, text_loss: {}".format(
        str(np.array(losses["total_loss"]).sum()/batchsize), str(np.array(losses["image_loss"]).sum()/batchsize), str(np.array(losses["text_loss"]).sum()/batchsize))


def finetune(cfg={}):
    cfg["gpu_id"] = 0

    wandb.init(config=cfg, project=cfg.get("project"))
    cfg.update(wandb.config)
    if paths_exist(cfg) != True:
        return

    device, model, preprocess = init_model(cfg)

    train_dataloader = read_data(
        cfg["traindata"], cfg["image_path"], preprocess, cfg["batch_size"], cfg["train_batch_shuffle"])
    cfg.update({"train_size": len(train_dataloader.dataset)})
    val_dataloader = read_data(
        cfg["valdata"], cfg["image_path"], preprocess, cfg["batch_size_val"], "False")
    cfg.update({"val_size": len(val_dataloader.dataset)})

    # losses and optimizer
    loss_img = CrossEntropyLoss()
    loss_txt = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"],
                      betas=ast.literal_eval(cfg["betas"]), eps=cfg["eps"], weight_decay=cfg["weight_decay"])

    train_str = "".join(k+": "+str(v)+"\n" for k,
                        v in cfg.items())  # log string
    # training
    start_time = time.time()
    best_total_loss = 999999
    epochs_no_improvement = 0

    for epoch in tqdm(range(cfg["epochs"])):
        model.train(not ast.literal_eval(cfg["eval_mode"]))
        losses_per_batch = {"total_loss": [],
                            "image_loss": [],
                            "text_loss": [],
                            }
        train_str += f"Epoch {str(epoch)}" + "\n"
        print("Epoch", epoch)
        epoch_time = time.time()
        for batch in tqdm(train_dataloader, ascii=' #'):
            optimizer.zero_grad()
            total_loss = calc_losses(
                batch, device, model, loss_img, loss_txt, losses_per_batch)
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        losses = {"total_loss_train": np.sum(losses_per_batch["total_loss"])/cfg["train_size"],
                  "image_loss_train": np.sum(losses_per_batch["image_loss"])/cfg["train_size"],
                  "text_loss_train": np.sum(losses_per_batch["text_loss"])/cfg["train_size"]}

        output_losses = losses_to_string(losses_per_batch, cfg["train_size"])
        print(output_losses)
        train_str += output_losses + "\n"+"--- %s minutes ---\n" % (
            round(time.time() - epoch_time)/60)

        # INFERENCE
        if (epoch+1) % cfg["inference_interval"] == 0 or (epoch+1) == cfg["epochs"]:
            val_losses_per_batch = {"total_loss": [],
                                    "image_loss": [],
                                    "text_loss": [],
                                    }
            train_str += "Inference\n"
            print("Running Inference")
            inference_time = time.time()
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_dataloader, ascii=' >='):
                    _ = calc_losses(
                        batch, device, model, loss_img, loss_txt, val_losses_per_batch)

            losses.update({"total_loss_val": np.sum(val_losses_per_batch["total_loss"])/cfg["val_size"],
                           "image_loss_val": np.sum(val_losses_per_batch["image_loss"])/cfg["val_size"],
                           "text_loss_val": np.sum(val_losses_per_batch["text_loss"])/cfg["val_size"]})

            output_losses = losses_to_string(
                val_losses_per_batch, cfg["val_size"])
            print(output_losses)
            train_str += output_losses + "\n"+"--- %s minutes ---\n" % (
                round(time.time() - inference_time)/60)

            if losses["total_loss_val"] < best_total_loss:
                model_state_cp = copy.deepcopy(model.state_dict().copy())
                optimizer_state_cp = copy.deepcopy(optimizer.state_dict())
                best_total_loss = losses["total_loss_val"]
                losses.update({"best_total_loss_val": best_total_loss})
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1

        wandb.log(losses)
        if epochs_no_improvement == cfg["early_stopping_epochs"]:
            print(
                f'Early stopping! For {str(cfg["early_stopping_epochs"])} epochs the val loss did not improve!')
            break

    train_str += "--- Total time: %s minutes ---\n\n" % (
        round(time.time() - start_time)/60)
    train_str += "".join(k+": "+str(v)+"\n" for k, v in losses.items())
    train_str += "\n Best total_loss_val: " + str(best_total_loss)

    # saves to disk and wandb
    save_model_and_logs(cfg["output_path"], train_str, epoch,
                        model_state_cp, optimizer_state_cp, losses["total_loss_train"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config', help='config.json with parameters', required=True)
    args = vars(parser.parse_args())

    with open(args["config"]) as json_file:
        cfg = (json.load(json_file))
    finetune(cfg)
