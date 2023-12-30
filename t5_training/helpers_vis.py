import ast

import numpy as np
import pandas as pd
import torch
from helpers import metrics_to_string, sum_losses_per_batch
from metrics import eval_df
from t5_model.t5vis import T5VisForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = ""


def set_device(dev):
    global device
    device = dev


def load_model(model_path, model):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])


class image_title_dataset(Dataset):
    def __init__(self, list_image_emb, images, questions, answers, context, question_classes, question_types):

        self.list_image_emb = list_image_emb
        self.images = images
        self.questions = [q.lower() for q in questions]
        self.answers = [a.lower() for a in answers]
        self.contexts = [c.lower() for c in context]
        self.question_classes = question_classes  # only used for AQUA visual/knowledge
        self.question_types = question_types

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        image_emb = self.list_image_emb[idx]
        question = self.questions[idx]
        answer = self.answers[idx]
        context = self.contexts[idx]
        question_class = self.question_classes[idx]
        question_type = self.question_types[idx]
        return image_emb, question, answer, context, question_class


def read_data(data_path, cache_path, batch_size, question_class, shuffle):
    data = pd.read_csv(data_path)
    data = data.join(pd.read_csv(cache_path).set_index("image"), on="image")
    data["question_class"] = ["additional" for _ in range(len(data))]
    if question_class != '':
        data = data[data["question_class"] == question_class]
    list_image_emb = [[ast.literal_eval(emb)]
                      for emb in data["img_emb"].tolist()]
    images = data["image"].tolist()
    questions = data["question"].tolist()
    answers = data["answer"].tolist()
    question_class = data["question_class"].tolist()
    question_type = data["question_type"].tolist()
    contexts = data["context"].tolist()
    dataset = image_title_dataset(
        list_image_emb, images, questions, answers, contexts, question_class, question_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=ast.literal_eval(shuffle))


def get_dataloaders(cfg):
    train_dataloader = read_data(
        cfg["traindata"], cfg["semart_cache"], cfg["batch_size"], cfg["question_class"], cfg["train_batch_shuffle"])
    val_dataloader = read_data(
        cfg["valdata"],  cfg["semart_cache"], cfg["batch_size"], cfg["question_class"], "False")
    test_dataloader = read_data(
        cfg["testdata"], cfg["semart_cache"],  cfg["batch_size"], "", "False")
    cfg.update({"train_size": len(train_dataloader.dataset)})
    cfg.update({"val_size": len(val_dataloader.dataset)})
    cfg.update({"test_size": len(test_dataloader.dataset)})
    return train_dataloader, val_dataloader, test_dataloader


def init_models(cfg):
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_device(device)
    print("Running on "+device)
    models = {}

    # load model
    models["tokenizer"] = T5Tokenizer.from_pretrained(cfg["t5_model"])
    if cfg["text_only"] == "True":
        models["model"] = T5ForConditionalGeneration.from_pretrained(
            cfg["t5_model"]).to(device)
    else:
        models["model"] = T5VisForConditionalGeneration.from_pretrained(
            cfg["t5_model"]).to(device)
    if cfg["pretrained_model_path"] not in ["", "dummy"]:
        print("using pretrained model:", cfg["pretrained_model_path"])
        load_model(cfg["pretrained_model_path"], models["model"])
    return models


def get_preds(input_ids, img_embs, models):
    models["model"].to(device)
    if models["model"].__class__.__name__ == 'T5VisForConditionalGeneration':
        output_ids = models["model"].generate(
            input_ids.to(device), clip_img_emb=img_embs, max_new_tokens=100)
    else:
        output_ids = models["model"].generate(
            input_ids.to(device), max_new_tokens=100)
    outputs = models["tokenizer"].batch_decode(
        output_ids, skip_special_tokens=True)
    return outputs


def calc_em(input_ids, img_embs, answers, models, accs, val_test):
    preds = get_preds(input_ids, img_embs, models)
    correct_preds = 0
    for a1, a2 in zip(preds, answers):
        if a1 == a2:
            correct_preds += 1
    if val_test == "val":
        accs["em_val"].append(correct_preds)
    if val_test == "test":
        accs["em_test"].append(correct_preds)


def make_inputs(questions, answers, contexts, q_classes, models, book_variant):
    if book_variant == "openbook":
        input = []
        for q, c, q_class in zip(questions, contexts, q_classes):
            # if q_class != "knowledge":
            #     c = ""
            input.append("question: "+q+" context: "+c)
    elif book_variant == "closedbook":
        input = questions
    else:
        raise Exception("book_variant not supported")
    encoding = models["tokenizer"](
        input,
        padding="longest",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    # encode the targets
    target_encoding = models["tokenizer"](
        answers,
        padding="longest",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
    labels = target_encoding.input_ids
    # replace padding token id's of the labels by -100 so it's ignored by the loss
    labels[labels == models["tokenizer"].pad_token_id] = -100

    return input_ids, attention_mask, labels


def calc_losses(batch, models, losses_dict, accs_dict, cfg, train_val):
    img_embs, questions, answers, contexts, q_class = batch
    img_embs = torch.stack(img_embs[0], dim=1).to(torch.float32).to(device)
    batchsize = len(questions)
    input_ids, attention_mask, labels = make_inputs(
        questions, answers, contexts, q_class, models, cfg["book_variant"])
    loss = {}
    if models["model"].__class__.__name__ == 'T5VisForConditionalGeneration':
        loss["ar"] = models["model"](input_ids=input_ids.to(device), clip_img_emb=img_embs,
                                     attention_mask=attention_mask.to(device), labels=labels.to(device)).loss
    else:
        loss["ar"] = models["model"](input_ids=input_ids.to(
            device), labels=labels.to(device)).loss
    if train_val == "val":
        calc_em(input_ids, img_embs, answers, models, accs_dict, "val")
    sum_losses_per_batch(losses_dict, loss, batchsize, train_val)
    return loss["ar"]


def get_pred_df(batch, models, cfg):
    columns = ["question", "answer",
               "pred"]
    img_embs, questions, answers, contexts, q_class = batch
    img_embs = torch.stack(img_embs[0], dim=1).to(torch.float32).to(device)
    input_ids, _, _ = make_inputs(
        questions, answers, contexts, q_class, models, cfg["book_variant"])
    preds = get_preds(input_ids, img_embs, models)
    return pd.DataFrame(np.array([list(questions), list(answers), preds]).T, columns=columns)


def get_pred_df_from_dataloader(dataloader, models, cfg):
    columns = ["question", "answer",
               "pred"]
    df = pd.DataFrame(columns=columns)
    with torch.no_grad():
        for batch in tqdm(dataloader, ascii=" >-"):
            df = pd.concat([df, get_pred_df(batch, models, cfg)])
    df["image"] = list(dataloader.dataset.images)
    df["question_class"] = list(dataloader.dataset.question_classes)
    df["question_type"] = list(dataloader.dataset.question_types)
    return df


def all_metrics_to_string(df):
    string = ""
    metrics = {}
    metrics["all"] = eval_df(df)
    if len(df[df["question_class"] == "visual"]) > 0:
        metrics["visual"] = eval_df(df[df["question_class"] == "visual"])
    if len(df[df["question_class"] == "knowledge"]) > 0:
        metrics["knowledge"] = eval_df(df[df["question_class"] == "knowledge"])
    if len(df[df["question_class"] == "visual"]) > 0 and len(df[df["question_class"] == "knowledge"]) > 0:
        metrics["vis+know"] = eval_df(df[df["question_class"] != "additional"])
    if len(df[df["question_class"] == "additional"]) > 0:
        metrics["additional"] = eval_df(
            df[df["question_class"] == "additional"])
    for k, v in metrics.items():
        string += k + "\n"+metrics_to_string(v)+"\n"
    return string, metrics


def get_type_accuracy(df):
    types = ["artist", "school", "timeframe", "type", "technique", "title"]
    df["answer"] = df["answer"].apply(lambda x: x.lower())
    df["pred"] = df["pred"].apply(lambda x: x.lower())
    type_accs = {}
    for t in types:
        tmp = df[df["question_type"] == t]
        type_accs["accuracy_test_" +
                  t] = (tmp["answer"] == tmp["pred"]).sum()/len(tmp)
    return type_accs
