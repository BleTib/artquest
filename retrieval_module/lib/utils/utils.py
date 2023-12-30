import logging
import time
import os
import numpy as np
import torch

def create_logger(cfg):
    dataset = cfg.DATASET.DATASET
    log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = f"{dataset}_{time_str}.log"
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file



def get_optimizer(cfg, model):
    base_lr = cfg.TRAIN.OPTIMIZER.BASE_LR
    params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append({"params": p})

    if cfg.TRAIN.OPTIMIZER.TYPE == "SGD":
        optimizer = SGD_GC(
            params,
            lr=base_lr,
            momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
            nesterov=True,
        )
    elif cfg.TRAIN.OPTIMIZER.TYPE == "ADAM":
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        )
    return optimizer


def cosine_scheduler(optimizer, cfg, total_steps):
    base_lr = cfg.TRAIN.OPTIMIZER.BASE_LR

    num_warmup_steps = cfg.TRAIN.OPTIMIZER.NUM_WARMUP_STEPS
    def _scheduler(current_step):
        if(current_step < num_warmup_steps):
            lr = base_lr * (current_step + 1) / num_warmup_steps
        else:
            n = current_step - num_warmup_steps
            d = total_steps - num_warmup_steps
            lr = 0.5 * (1 + np.cos(np.pi * n / d)) * base_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _scheduler
