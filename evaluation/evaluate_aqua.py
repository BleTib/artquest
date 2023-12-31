import ast
import os

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from t5_model.t5vis import T5VisForConditionalGeneration
from t5_training.metrics import calc_bleu_sentence, calc_em
from tqdm import tqdm
from transformers import T5Tokenizer

torch.manual_seed(2)

device = "cuda" if torch.cuda.is_available() else "cpu"

# read artquest test data an join with semart cache
aqua_test = pd.read_json("../data/AQUA/test.json")
semart_cache = pd.read_csv("../data/semart_cache.csv")
aqua_test = aqua_test.join(semart_cache.set_index("image"), on="image")
aqua_test["question_class"] = aqua_test["need_external_knowledge"].apply(
    lambda x: "knowledge" if x else "visual")
contexts_retrieved_with_bert = pd.read_csv(
    "../data/AQUA/test_retrieved_contexts.csv")
aqua_test = pd.merge(aqua_test, contexts_retrieved_with_bert, on=[
                     'image', 'question'], how='left')
# fill retrieved context with context for vision questions
aqua_test['retrieved_context'].fillna(aqua_test['context'], inplace=True)
all_comments = semart_cache["context"].drop_duplicates().tolist()


# get candidate contexts with tfidf
vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")

# Fit and transform the descriptions
X = vectorizer.fit_transform(all_comments)

context_top10 = []
counts = {"visual": 0, "knowledge": 0, "both": 0,
          "visual_top10": 0, "knowledge_top10": 0, "both_top10": 0}
correct = 0
correct_top10 = 0
print("Calculating top contexts with tfidf...")
for row in tqdm(aqua_test.itertuples(), total=len(aqua_test)):
    question = row.question

    # Transform the query using the fitted vectorizer
    query_vec = vectorizer.transform([question])
    # Calculate cosine similarity between the query and each description
    similarity_scores = cosine_similarity(X, query_vec)
    # Rank the descriptions by similarity score
    ranked_descriptions = np.argsort(similarity_scores, axis=0)[::-1][:10]
    context_top10.append([all_comments[i[0]] for i in ranked_descriptions])

    if row.context == all_comments[ranked_descriptions[0][0]]:
        if row.question_class == "knowledge":
            counts["knowledge"] += 1
        elif row.question_class == "visual":
            counts["visual"] += 1
        counts["both"] += 1
    if row.context in [all_comments[i[0]] for i in ranked_descriptions]:
        if row.question_class == "knowledge":
            counts["knowledge_top10"] += 1
        elif row.question_class == "visual":
            counts["visual_top10"] += 1
        counts["both_top10"] += 1

print("tfidf context retrieval accuracys")
print()
print("correct knowledge", counts["knowledge"] /
      len(aqua_test[aqua_test["question_class"] == "knowledge"]))
print("correct knowledge top10", counts["knowledge_top10"] /
      len(aqua_test[aqua_test["question_class"] == "knowledge"]))
print()
print("correct visual", counts["visual"] /
      len(aqua_test[aqua_test["question_class"] == "visual"]))
print("correct visual top10", counts["visual_top10"] /
      len(aqua_test[aqua_test["question_class"] == "visual"]))
print()
print("correct", counts["both"]/len(aqua_test))
print("correct top10", counts["both_top10"]/len(aqua_test))


print("correct", counts["both"]/len(aqua_test))
print("correct top10", counts["both_top10"]/len(aqua_test))

print("Correct retrieved contexts for knowledge questions with BERT reranking:",
      (aqua_test["retrieved_context"] == aqua_test["context"]).sum()/len(aqua_test))

# Load T5 model
model = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model)
t5openbook = T5VisForConditionalGeneration.from_pretrained(
    model).to(device)
t5openbook.load_state_dict(torch.load(
    "../model_checkpoints/t5_aqua/t5_openbook/model.pt")["model_state_dict"], strict=True)

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
for row in tqdm(aqua_test.itertuples(), total=len(aqua_test)):
    with torch.no_grad():
        if row.image not in img_emb_cache:
            img_emb_cache[row.image] = torch.tensor(
                [[ast.literal_eval(row.img_emb)]]).to(device)
        img_emb = img_emb_cache[row.image]

        # model is trained with no context for visual questions
        q = row.question.lower()
        c = ""
        c_r = ""
        # we assume we know the question class. A classifier with over 99.9% accuracy can be built with the BERT vision model, which is provided in the model directory
        if row.question_class == "knowledge":
            c = row.context.lower()
            c_r = row.retrieved_context.lower()

        preds.append(get_pred(q, c, img_emb, t5openbook, tokenizer))
        preds_ret.append(get_pred(q, c_r, img_emb, t5openbook, tokenizer))
aqua_test["pred"] = preds
aqua_test["pred_retrieved_context"] = preds_ret

# Calculate metrics


def print_metrics(df):
    print("with original context")
    print("EM", calc_em(df["answer"].tolist(),
                        df["pred"].tolist()))
    print("BLEU", calc_bleu_sentence(
        df["answer"].tolist(), df["pred"].tolist()))
    print("with retrieved context")
    print("EM", calc_em(df["answer"].tolist(),
                        df["pred_retrieved_context"].tolist()))
    print("BLEU", calc_bleu_sentence(df["answer"].tolist(
    ), df["pred_retrieved_context"].tolist()))


aqua_test["answer"] = aqua_test["answer"].apply(lambda x: x.lower())
print("Metrics for visual questions")
print_metrics(aqua_test[aqua_test["question_class"] == "visual"])
print("\nMetrics for knowledge questions")
print_metrics(aqua_test[aqua_test["question_class"] == "knowledge"])
print("\nMetrics for all questions")
print_metrics(aqua_test)
