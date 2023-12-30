#!/usr/bin/env python

import _init_paths
import pickle 
import os 
from config import cfg, update_config
import clip
from utils.utils import create_logger
import numpy as np
import pandas as pd
import parse_args
from dataset import *
import torch
from torch.utils.data import DataLoader
import faiss
from sklearn.metrics import recall_score, precision_score


def run_image_text(model, cfg, data_loader, data_size, device):
    logger, _ = create_logger(cfg)
    
    model = model.float().to(device)
    
    encoder_type = cfg.TRAIN.VISION_ENCODER

    logger.info(f"-------Loading {encoder_type} with backend vision encoder {cfg.TRAIN.VISION_ENCODER} -------")
    logger.info(f"-------KNN-Retrieval started: Image to Text-------")

    total_iteration = len(data_loader)

    ids_of_predicted_captions = []

    total_score = 0
    total_accuracy = 0
    batch_size = cfg.TEST.BATCH_SIZE
    best_sims = [0 for i in range(data_size)]
    ids_of_candidate_texts = [0 for i in range(data_size)]
    image_names_of_candidate_texts = ["" for i in range(data_size)]
    all_reference_image_names = []
    reference_img_ids = []

    res = faiss.StandardGpuResources()
    
    for j, (img_ids, reference_image_names, imgs, _) in enumerate(data_loader):

        images = torch.stack([img for img in imgs], dim=0).to(device)
        v_embs = model.encode_image(images)
        v_embs = v_embs / v_embs.norm(dim=-1, keepdim=True)
        v_embs = v_embs.cpu().detach().numpy()

        all_reference_image_names += reference_image_names
        reference_img_ids += img_ids

        img_inds = img_ids.numpy()
        
        # since cosine sim is the dot product of the normalized vectors:
        faiss.normalize_L2(v_embs)
        
        for i, (txt_ids, image_names, _, texts) in enumerate(data_loader):
            prompted_texts = texts
            captions = clip.tokenize(prompted_texts, context_length=cfg.TRAIN.MAX_SEQ_LENGTH, truncate=True).to(device)
            t_embs = model.encode_text(captions)
            t_embs = t_embs / t_embs.norm(dim=-1, keepdim=True)
            t_embs = t_embs.cpu().detach().numpy()

            faiss.normalize_L2(t_embs)
            index = faiss.IndexFlatIP(cfg.TRAIN.DIM)  # IP: inner product
            index = faiss.IndexIDMap(index)

            index_ids = txt_ids.numpy()
            index.add_with_ids(t_embs, index_ids)
            index = faiss.index_cpu_to_gpu(res, 0, index)
            similarities, similarities_ids = index.search(v_embs, k=cfg.KNN.K)

            offset = j * batch_size
            for p, sim in enumerate(similarities):  # TODO: This only works for k=1. Should be fixed!
                iinndd = offset + p
                if sim > best_sims[iinndd]:
                    best_sims[iinndd] = sim
                    loc = torch.nonzero(txt_ids == similarities_ids[p].item())
                    ids_of_candidate_texts[iinndd] = similarities_ids[p].item()
                    image_names_of_candidate_texts[iinndd] = image_names[loc]
            pbar_str = f"---Inner Iteration {i}/{len(data_loader)} in Outer Iteration {j}"
            logger.info(pbar_str)
        pbar_str = f"---Outer Iteration {j}/{len(data_loader)} starts"
        logger.info(pbar_str)
    
    image_names_of_candidate_texts = np.array(image_names_of_candidate_texts)
    all_reference_image_names = np.array(all_reference_image_names)
    corrects_cnt = (image_names_of_candidate_texts == all_reference_image_names).sum()
    total_accuracy = 100 * corrects_cnt / data_size 
    recall = recall_score(all_reference_image_names, image_names_of_candidate_texts, average='macro')
    precision = precision_score(all_reference_image_names, image_names_of_candidate_texts, average='macro')

    pbar_str = f"---Total accuracy: {total_accuracy}  Recall: {recall}   Precision:{precision}"
    logger.info(pbar_str)

    reference_candidate_dict = {
            'reference_image_names': all_reference_image_names,
            'image_names_of_candidate_texts': image_names_of_candidate_texts,
            'similarities': best_sims,
            'acc': total_accuracy,
            'recall': recall,
            'precision': precision
            }

    results_path = os.path.join(cfg.RESULTS_DIR, f'{cfg.NAME}_reference_candidate.pickle')
    with open(results_path, 'wb') as f:
        pickle.dump(reference_candidate_dict, f)
    # TODOs: support k>1
    

if __name__ == '__main__':
    data = cfg.DATASET.DATA_DIR
    args = parse_args.parse()
    args.data_dir = data
    # set GPU device
    device = torch.device("cuda" if args.num_gpus >= 0 else "cpu")
    update_config(cfg, args)
    # Fixed random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    model, preprocess = clip.load(cfg.TRAIN.VISION_ENCODER, device=device, jit=False)

    if not cfg.TEST.CLIP_ORG:
        checkpoint = torch.load(cfg.TEST.CLIP_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])

    test_dataset = ImageTextDataset("test", cfg, preprocess=preprocess)
    test_loader = DataLoader(test_dataset, cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=cfg.TEST.NUM_WORKERS, drop_last=False, pin_memory=True)


    run_image_text(model, cfg, test_loader, len(test_dataset), device)
