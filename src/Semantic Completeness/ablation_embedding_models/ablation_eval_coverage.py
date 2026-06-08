# -*- coding: utf-8 -*-
"""
Created on Fri May 15 19:31:54 2026

@author: Deperias Kerre
"""
#importing libraries
import json
import time
import gc
import torch
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT

# input/output
INPUT_FILE = "dataset_sample_10k.json"
OUTPUT_FILE = "coverage_results_rich.json"

TOP_K = 5
THRESHOLDS = [0.6, 0.7, 0.8]
BATCH_SIZE = 64

# MODELS
MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"
]


# KEYBERT SETUP

kw_extractor = KeyBERT()


# HELPER FUNCTIONS

def normalize(text):
    return text.lower().strip()


def normalize_keywords(keywords):
    if isinstance(keywords, list):
        return [normalize(k) for k in keywords if isinstance(k, str)]
    elif isinstance(keywords, str):
        return [normalize(keywords)]
    return []


def extract_keywords(text):

    if not text or not text.strip():
        return []

    try:
        kws = kw_extractor.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=TOP_K
        )

        return [normalize(k[0]) for k in kws]

    except Exception:
        return []


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# PREPROCESS

def preprocess_data(data):
    print("Preprocessing keywords and concept extraction...")

    prepared = []

    all_keywords = []
    all_title_concepts = []
    all_description_concepts = []

    for i, d in enumerate(data, 1):
        keywords = normalize_keywords(d.get("keywords", []))
        title_concepts = extract_keywords(d.get("title", ""))
        description_concepts = extract_keywords(d.get("description", ""))

        prepared.append({
            "dataset_id": d.get("dataset_id"),
            "keywords": keywords,
            "title_concepts": title_concepts,
            "description_concepts": description_concepts
        })

        all_keywords.extend(keywords)
        all_title_concepts.extend(title_concepts)
        all_description_concepts.extend(description_concepts)

        if i % 100 == 0:
            print(f"Preprocessed {i}/{len(data)}")

    return prepared, all_keywords, all_title_concepts, all_description_concepts


# EMBEDDING LOOKUP

def build_embedding_lookup(model, texts, label):
    unique_texts = list(set(texts))

    print(f"\nEncoding {len(unique_texts)} unique {label} phrases...")

    embeddings = model.encode(
        unique_texts,
        convert_to_tensor=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )

    lookup = {
        text: emb for text, emb in zip(unique_texts, embeddings)
    }

    return lookup


# COVERAGE EVALUATION

def run_coverage_evaluation(model_name,
                            prepared_data,
                            keyword_lookup,
                            title_lookup,
                            desc_lookup):

    print(f"\nRunning coverage evaluation for {model_name}")

    results = []

    for i, item in enumerate(prepared_data, 1):
        keywords = item["keywords"]
        title_concepts = item["title_concepts"]
        description_concepts = item["description_concepts"]

        title_coverage_dict = {
            f"title_coverage_{t}": 0.0 for t in THRESHOLDS
        }

        description_coverage_dict = {
            f"description_coverage_{t}": 0.0 for t in THRESHOLDS
        }

        avg_title_max_similarity = 0.0
        avg_description_max_similarity = 0.0

        keyword_embs = None

        if len(keywords) > 0:
            keyword_embs = torch.stack(
                [keyword_lookup[k] for k in keywords]
            )

        # TITLE
        if len(title_concepts) > 0 and keyword_embs is not None:
            title_embs = torch.stack(
                [title_lookup[t] for t in title_concepts]
            )

            sims = util.cos_sim(title_embs, keyword_embs)

            for t in THRESHOLDS:
                matches = (sims >= t).any(dim=1)

                title_coverage_dict[f"title_coverage_{t}"] = float(
                    matches.sum().item() / len(title_concepts)
                )

            avg_title_max_similarity = float(
                sims.max(dim=1).values.mean().item()
            )

        # DESCRIPTION
        if len(description_concepts) > 0 and keyword_embs is not None:
            desc_embs = torch.stack(
                [desc_lookup[d] for d in description_concepts]
            )

            sims = util.cos_sim(desc_embs, keyword_embs)

            for t in THRESHOLDS:
                matches = (sims >= t).any(dim=1)

                description_coverage_dict[f"description_coverage_{t}"] = float(
                    matches.sum().item() / len(description_concepts)
                )

            avg_description_max_similarity = float(
                sims.max(dim=1).values.mean().item()
            )

        results.append({
            "dataset_id": item["dataset_id"],
            "num_keywords": len(keywords),
            "num_title_concepts": len(title_concepts),
            "num_description_concepts": len(description_concepts),
            "avg_title_max_similarity": avg_title_max_similarity,
            "avg_description_max_similarity": avg_description_max_similarity,
            **title_coverage_dict,
            **description_coverage_dict
        })

        if i % 100 == 0:
            print(f"[{model_name}] Processed {i}/{len(prepared_data)}")

    return results


# MAIN

if __name__ == "__main__":
    start_time = time.time()

    print("Loading dataset...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} datasets")

    prepared_data, all_keywords, all_title_concepts, all_description_concepts = preprocess_data(data)

    all_results = {}

    for model_name in MODELS:
        print(f"\n{'=' * 80}")
        print(f"PROCESSING MODEL: {model_name}")
        print(f"{'=' * 80}")

        model = SentenceTransformer(model_name, device="cpu")

        keyword_lookup = build_embedding_lookup(model, all_keywords, "keyword")
        title_lookup = build_embedding_lookup(model, all_title_concepts, "title")
        desc_lookup = build_embedding_lookup(model, all_description_concepts, "description")

        results = run_coverage_evaluation(
            model_name,
            prepared_data,
            keyword_lookup,
            title_lookup,
            desc_lookup
        )

        all_results[model_name] = results

        # immediate save
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        # cleanup model memory
        del model
        del keyword_lookup
        del title_lookup
        del desc_lookup
        del results

        cleanup()

        print(f"\nSaved partial results for {model_name}")

    end_time = time.time()

    print(f"\nFinal results saved to {OUTPUT_FILE}")
    print(f"Total time: {end_time - start_time:.2f}s")
