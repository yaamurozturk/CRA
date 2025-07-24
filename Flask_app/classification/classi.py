from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

citation_column = "Citation_context"        # Column with citation sentences
output_jsonl = "citations_Fred.jsonl" # Output file for MULTICITE
output_csv = "with_predictions.csv"
model_name = "allenai/multicite-multilabel-scibert"

def classif(df):
    df = df.dropna(subset=[citation_column])
    df = df[df[citation_column].str.strip() != ""]

    # === CONVERSION TO JSONL ===
    with open(output_jsonl, "w") as f:
        for i, row in df.iterrows():
            context = row[citation_column]
            example = {
                "cite_context": context,
                "citing_paper_id": f"paper_{i}",
                "cited_paper_id": f"cited_{i}"
            }
            f.write(json.dumps(example) + "\n")
    print(df)
    print(f"Saved  cleaned examples to {output_jsonl}")

    #df = df.head(10)
    df = df.dropna(subset=[citation_column])
    df = df[df[citation_column].str.strip() != ""]

    # === LOAD MODEL AND TOKENIZER ===
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # === CREATE PIPELINE ===
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True)

    # === RUN PREDICTIONS AND TOKENIZATION ===
    texts = df[citation_column].tolist()
    predictions = classifier(texts, batch_size=8, truncation=True, max_length=512)
    #predictions = classifier(texts, batch_size=8, truncation=True)

    # === ADD RESULTS TO DATAFRAME ===
    df["predicted_label"] = [pred["label"] for pred in predictions]
    #df["confidence"] = [pred["score"] for pred in predictions]
    #df["tokens"] = [tokenizer.tokenize(text) for text in texts]

    # === SAVE TO CSV ===
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions and tokenized sentences to {output_csv}")
    #print(df)
    df_expanded = df.assign(predicted_label=df["predicted_label"].str.split(";")).explode("predicted_label")
    print(df)
    """df = pd.read_csv("with_predictions.csv")
    df["predicted_label"] = df["predicted_label"].astype(str)
    label_totals = df_expanded["predicted_label"].value_counts(); label_totals"""
    #print(df_expanded)#['confidence','tokens'].drop)
    return(df)