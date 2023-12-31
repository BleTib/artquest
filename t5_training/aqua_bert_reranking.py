import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from datasets import Dataset

semart_cache = pd.read_csv("../data/semart_cache.csv")
aqua_train = pd.read_json(
    "../data/AQUA/train.json").join(semart_cache[["image", "context"]].set_index("image"), on="image")
aqua_val = pd.read_json(
    "../data/AQUA/val.json").join(semart_cache[["image", "context"]].set_index("image"), on="image")
aqua_test = pd.read_json(
    "../data/AQUA/test.json").join(semart_cache[["image", "context"]].set_index("image"), on="image")
aqua_train["question_class"] = aqua_train["need_external_knowledge"].apply(
    lambda x: "knowledge" if x else "visual")
aqua_val["question_class"] = aqua_val["need_external_knowledge"].apply(
    lambda x: "knowledge" if x else "visual")
aqua_test["question_class"] = aqua_test["need_external_knowledge"].apply(
    lambda x: "knowledge" if x else "visual")
aqua_train = aqua_train[aqua_train["question_class"] == "knowledge"]
aqua_val = aqua_val[aqua_val["question_class"] == "knowledge"]
aqua_test = aqua_test[aqua_test["question_class"] == "knowledge"]
all_comments = semart_cache["context"].drop_duplicates().tolist()

# get candidate contexts with tfidf
vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
X = vectorizer.fit_transform(all_comments)


def get_top10_contexts(df):
    context_top10 = []
    for row in tqdm(df.itertuples(), total=len(df)):
        question = row.question

        # Transform the query using the fitted vectorizer
        query_vec = vectorizer.transform([question])
        # Calculate cosine similarity between the query and each description
        similarity_scores = cosine_similarity(X, query_vec)
        # Rank the descriptions by similarity score
        ranked_descriptions = np.argsort(similarity_scores, axis=0)[::-1][:10]
        context_top10.append([all_comments[i[0]] for i in ranked_descriptions])
    return context_top10


print("Getting top 10 contexts with tfidf...")
aqua_train["context_top10"] = get_top10_contexts(aqua_train)
aqua_val["context_top10"] = get_top10_contexts(aqua_val)
aqua_test["context_top10"] = get_top10_contexts(aqua_test)


def extend_df(df, testset=False):
    labels = [1] + [0]*9
    labels *= len(df)
    texts = []

    for row in df.itertuples():
        q = row.question.lower() + "?"
        top10_contexts = row.context_top10
        if testset:
            # for the testset we know the correct context and only take the top 9
            top9_contexts = [
                c for c in row.context_top10 if not row.context == c]
            top10_contexts = [row.context]+top9_contexts[:9]
        texts.extend(
            [f"question: {q} context: {c.lower()}" for c in top10_contexts])

    return pd.DataFrame({"text": texts, "label": labels})


def make_input_df(df, tokenizer):
    data = tokenizer(df['text'].tolist(
    ), padding='max_length', truncation=True, max_length=512)
    # df_out = pd.DataFrame({k: v for k, v in data.items()})
    # df_out['labels'] = df['label'].tolist()
    data['labels'] = df['label'].tolist()
    return data


def make_dataset(df, tokenizer, testset=False):
    df = extend_df(df)
    df = make_input_df(df, tokenizer)
    return Dataset.from_dict(df)


device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
training_args = TrainingArguments(
    "../model_checkpoints/aqua_bert_stage2_reranking", num_train_epochs=10, logging_strategy="epoch", evaluation_strategy="epoch", save_strategy="epoch", save_total_limit=2, per_device_train_batch_size=16, per_device_eval_batch_size=16, learning_rate=1e-5, weight_decay=0.01, load_best_model_at_end=True, report_to="wandb", metric_for_best_model="top1 accuracy", greater_is_better=True)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2).to(device)


traindata = make_dataset(aqua_train, tokenizer)
valdata = make_dataset(aqua_val, tokenizer)
testdata = make_dataset(aqua_test, tokenizer, testset=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {'top1 accuracy': (logits.reshape(int(len(logits)/10), 10, 2)[:, :, 1].argmax(axis=1) == 0).sum().item()/int(len(logits)/10)}


print("Training BERT reranking... on", device)
trainer = Trainer(
    model=model, args=training_args, compute_metrics=compute_metrics, train_dataset=traindata, eval_dataset=valdata)

trainer.train()
output = trainer.predict(testdata)


def get_top1_preds(test_df, preds):
    top1_ids = preds.reshape(
        int(len(testdata)/10), 10, 2)[:, :, 1].argmax(axis=1)
    return [c[i] for i, c in zip(
        top1_ids, test_df.context_top10.tolist())]


def get_top1_accuracy(test_df, preds):
    top1_preds = get_top1_preds(test_df, preds)
    correct = 0
    for gt, pred in zip(test_df.context.tolist(), top1_preds):
        if gt == pred:
            correct += 1
    return correct/len(test_df)


top1_preds = get_top1_preds(aqua_test, output.predictions)
print("Top1 accuracy with BERT reranking",
      get_top1_accuracy(aqua_test, output.predictions))


# save results
aqua_test["retrieved_context"] = top1_preds
aqua_test[["image", "question", "retrieved_context"]].to_csv(
    "../data/AQUA/test_retrieved_contexts.csv", index=False)
