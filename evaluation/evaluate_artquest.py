import ast
import os

import pandas as pd
import torch
from t5_model.t5vis import T5VisForConditionalGeneration
from t5_training.metrics import calc_bleu_sentence, calc_em
from tqdm import tqdm
from transformers import T5Tokenizer

torch.manual_seed(2)

device = "cuda" if torch.cuda.is_available() else "cpu"

# read artquest test data an join with semart cache
artquest_test = pd.read_csv("../data/artquest/artquest_test.csv")
semart_cache = pd.read_csv("../data/semart_cache.csv")
artquest_test = artquest_test.join(semart_cache.set_index("image"), on="image")


# Read retrieved context candidates and make a dataframe from the matching results mapping image to texts with image names
file = "../retrieval_module/output/SEMARTCLIP.200BS.IMAGE_TO_TEXT_reference_candidate.pickle"
ref_candidates_unpickled = pd.read_pickle(file)
ref_candidates = pd.DataFrame(
    ref_candidates_unpickled["reference_image_names"], columns=["reference_image_names"])
ref_candidates["image_names_of_candidate_texts"] = ref_candidates_unpickled["image_names_of_candidate_texts"]
print("Accuracy of retrieved contexts: ",
      (ref_candidates["image_names_of_candidate_texts"] == ref_candidates["reference_image_names"]).sum()/len(ref_candidates))

# Map candidate texts to the artquest test data
candiate_contexts = ref_candidates.join(semart_cache.set_index(
    "image"), on="image_names_of_candidate_texts")[["reference_image_names", "context"]]
candiate_contexts = candiate_contexts.rename(
    columns={"reference_image_names": "image", "context": "candidate_context"})
artquest_test = artquest_test.join(
    candiate_contexts.set_index("image"), on="image")


# Load T5 model
model = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model)
t5openbook = T5VisForConditionalGeneration.from_pretrained(
    model).to(device)
t5openbook.load_state_dict(torch.load(
    "../model_checkpoints/t5_artquest/t5_openbook/model.pt")["model_state_dict"], strict=True)

# Make predictions


def get_pred(q, c, img_emb, model, tokenizer):
    input_ids = tokenizer(
        f"question: {q} context: {c}",
        padding="longest",
        max_length=512,
        return_tensors="pt",
    ).input_ids
    output_ids = model.generate(
        input_ids.to(device), clip_img_emb=img_emb, max_new_tokens=100)
    return tokenizer.batch_decode(
        output_ids, skip_special_tokens=True)[0]


preds = []
preds_ret = []
img_emb_cache = {}
print("Making predictions... running on " + device)
for row in tqdm(artquest_test.itertuples(), total=len(artquest_test)):
    with torch.no_grad():
        if row.image not in img_emb_cache:
            img_emb_cache[row.image] = torch.tensor(
                [[ast.literal_eval(row.img_emb)]]).to(device)
        img_emb = img_emb_cache[row.image]

        q = row.question.lower()
        c = row.context.lower()
        c_r = row.candidate_context.lower()

        preds.append(get_pred(q, c, img_emb, t5openbook, tokenizer))
        preds_ret.append(get_pred(q, c_r, img_emb, t5openbook, tokenizer))
artquest_test["pred"] = preds
artquest_test["pred_retrieved_context"] = preds_ret

# Calculate metrics
artquest_test["answer"] = artquest_test["answer"].apply(lambda x: x.lower())
print("Metrics for the original context")
print("EM", calc_em(artquest_test["answer"].tolist(),
      artquest_test["pred"].tolist()))
print("BLEU", calc_bleu_sentence(
    artquest_test["answer"].tolist(), artquest_test["pred"].tolist()))
print("Metrics for the retrieved context")
print("EM", calc_em(artquest_test["answer"].tolist(),
      artquest_test["pred_retrieved_context"].tolist()))
print("BLEU", calc_bleu_sentence(artquest_test["answer"].tolist(
), artquest_test["pred_retrieved_context"].tolist()))
