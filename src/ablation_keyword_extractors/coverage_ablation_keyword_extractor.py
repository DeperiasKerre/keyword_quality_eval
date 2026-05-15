# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:54:21 2026

@author: Deperias Kerre
"""
# -*- coding: utf-8 -*-
"""
Coverage robustness across concept extraction methods
YAKE vs KeyBERT
"""
#import sys
#!{sys.executable} -m pip install keybert==0.8.5
import json
import time
import gc
import os
import torch
from sentence_transformers import SentenceTransformer, util
import yake
from keybert import KeyBERT

# input/output
INPUT_FILE = "dataset_sample_10k.json"
OUTPUT_FILE = "coverage_extractor_results.json"

TOP_K = 5
THRESHOLDS = [0.6, 0.7, 0.8]
BATCH_SIZE = 64

# fixed embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EXTRACTORS = ["YAKE", "KeyBERT"]


# ---------------------------
# EXTRACTOR SETUP
# ---------------------------
yake_extractor = yake.KeywordExtractor(
    lan="en",
    n=2,
    dedupLim=0.9,
    top=TOP_K
)


# ---------------------------
# TEXT NORMALIZATION
# ---------------------------
def normalize(text):
    return text.lower().strip()


def normalize_keywords(keywords):
    if isinstance(keywords, list):
        return [normalize(k) for k in keywords if isinstance(k, str)]
    elif isinstance(keywords, str):
        return [normalize(keywords)]
    return []


def extract_yake(text):
    kws = yake_extractor.extract_keywords(text)
    return [normalize(k[0]) for k in kws]


def extract_keybert(text, kw_model):
    kws = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=TOP_K
    )
    return [normalize(k[0]) for k in kws]


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------
# PREPROCESS
# ---------------------------
def preprocess_data(data, kw_model):
    print("Preprocessing dataset with YAKE + KeyBERT...")

    prepared = []

    all_keywords = []
    all_yake_title = []
    all_yake_desc = []
    all_keybert_title = []
    all_keybert_desc = []

    for i, d in enumerate(data, 1):
        keywords = normalize_keywords(d.get("keywords", []))

        title = d.get("title", "")
        desc = d.get("description", "")

        yake_title = extract_yake(title)
        yake_desc = extract_yake(desc)

        keybert_title = extract_keybert(title, kw_model)
        keybert_desc = extract_keybert(desc, kw_model)

        prepared.append({
            "dataset_id": d.get("dataset_id"),
            "keywords": keywords,
            "YAKE_title": yake_title,
            "YAKE_desc": yake_desc,
            "KeyBERT_title": keybert_title,
            "KeyBERT_desc": keybert_desc
        })

        all_keywords.extend(keywords)
        all_yake_title.extend(yake_title)
        all_yake_desc.extend(yake_desc)
        all_keybert_title.extend(keybert_title)
        all_keybert_desc.extend(keybert_desc)

        if i % 500 == 0:
            print(f"Preprocessed {i}/{len(data)}")

    all_concepts = (
        all_keywords +
        all_yake_title +
        all_yake_desc +
        all_keybert_title +
        all_keybert_desc
    )

    return prepared, all_concepts


# ---------------------------
# EMBEDDING LOOKUP
# ---------------------------
def build_embedding_lookup(model, texts):
    unique_texts = list(set(texts))

    print(f"\nEncoding {len(unique_texts)} unique phrases...")

    embeddings = model.encode(
        unique_texts,
        convert_to_tensor=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )

    return {
        text: emb for text, emb in zip(unique_texts, embeddings)
    }


# ---------------------------
# COVERAGE EVALUATION
# ---------------------------
def run_coverage(prepared_data, embedding_lookup, extractor_name):
    print(f"\nRunning coverage evaluation for {extractor_name}")

    results = []

    for i, item in enumerate(prepared_data, 1):
        keywords = item["keywords"]

        if extractor_name == "YAKE":
            title_concepts = item["YAKE_title"]
            desc_concepts = item["YAKE_desc"]
        else:
            title_concepts = item["KeyBERT_title"]
            desc_concepts = item["KeyBERT_desc"]

        title_coverage_dict = {
            f"title_coverage_{t}": 0.0 for t in THRESHOLDS
        }

        description_coverage_dict = {
            f"description_coverage_{t}": 0.0 for t in THRESHOLDS
        }

        avg_title_max_similarity = 0.0
        avg_description_max_similarity = 0.0

        if len(keywords) > 0:
            keyword_embs = torch.stack(
                [embedding_lookup[k] for k in keywords]
            )
        else:
            keyword_embs = None

        # TITLE
        if len(title_concepts) > 0 and keyword_embs is not None:
            title_embs = torch.stack(
                [embedding_lookup[t] for t in title_concepts]
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
        if len(desc_concepts) > 0 and keyword_embs is not None:
            desc_embs = torch.stack(
                [embedding_lookup[d] for d in desc_concepts]
            )

            sims = util.cos_sim(desc_embs, keyword_embs)

            for t in THRESHOLDS:
                matches = (sims >= t).any(dim=1)

                description_coverage_dict[f"description_coverage_{t}"] = float(
                    matches.sum().item() / len(desc_concepts)
                )

            avg_description_max_similarity = float(
                sims.max(dim=1).values.mean().item()
            )

        results.append({
            "dataset_id": item["dataset_id"],
            "num_keywords": len(keywords),
            "num_title_concepts": len(title_concepts),
            "num_description_concepts": len(desc_concepts),
            "avg_title_max_similarity": avg_title_max_similarity,
            "avg_description_max_similarity": avg_description_max_similarity,
            **title_coverage_dict,
            **description_coverage_dict
        })

        if i % 500 == 0:
            print(f"[{extractor_name}] Processed {i}/{len(prepared_data)}")

    return results


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    start_time = time.time()

    print("Loading dataset...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} datasets")

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    kw_model = KeyBERT(model=model)

    prepared_data, all_concepts = preprocess_data(data, kw_model)

    embedding_lookup = build_embedding_lookup(model, all_concepts)

    all_results = {}

    for extractor in EXTRACTORS:
        results = run_coverage(
            prepared_data,
            embedding_lookup,
            extractor
        )

        all_results[extractor] = results

        # SAFE ATOMIC SAVE
        tmp_output = OUTPUT_FILE + ".tmp"

        with open(tmp_output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        os.replace(tmp_output, OUTPUT_FILE)

        print(f"Saved partial results for {extractor}")

    cleanup()

    end_time = time.time()

    print(f"\nFinal results saved to {OUTPUT_FILE}")
    print(f"Total time: {end_time - start_time:.2f}s")
