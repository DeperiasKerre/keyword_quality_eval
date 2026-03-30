# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 09:04:16 2026

@author: Deperias Kerre
"""
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import time

# Endpoint
ENDPOINT = "https://data.europa.eu/sparql"
sparql = SPARQLWrapper(ENDPOINT)

# -----------------------------
# SPARQL Query to extract metadata: dataset, title, description
# -----------------------------
query = """
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dct:  <http://purl.org/dc/terms/>

SELECT ?dataset ?title ?description ?keyword
WHERE {
  ?dataset a dcat:Dataset ;
           dct:title ?title ;
           dct:description ?description .

  OPTIONAL { ?dataset dcat:keyword ?keyword }

  FILTER (lang(?title) = "en")
  FILTER (lang(?description) = "en")
}
LIMIT 2500
"""

sparql.setQuery(query)
sparql.setReturnFormat(JSON)

results = sparql.query().convert()

# -----------------------------
# STEP 2: Aggregating the datasets
# -----------------------------
datasets = {}

for result in results["results"]["bindings"]:
    dataset_uri = result["dataset"]["value"]

    if dataset_uri not in datasets:
        datasets[dataset_uri] = {
            "id": dataset_uri,
            "title": result["title"]["value"],
            "description": result["description"]["value"],
            "keywords": set(),
            "license": None
        }

    if "keyword" in result:
        kw = result["keyword"]["value"].strip()
        if kw:
            datasets[dataset_uri]["keywords"].add(kw)

# -----------------------------
# STEP 3: Fetching the data license 
# -----------------------------
for dataset_uri in datasets:

    license_query = f"""
    PREFIX dcat: <http://www.w3.org/ns/dcat#>
    PREFIX dct:  <http://purl.org/dc/terms/>

    SELECT ?license
    WHERE {{
      <{dataset_uri}> dcat:distribution ?dist .
      ?dist dct:license ?license .
    }}
    LIMIT 1
    """

    sparql.setQuery(license_query)
    sparql.setReturnFormat(JSON)

    try:
        res = sparql.query().convert()
        bindings = res["results"]["bindings"]

        if bindings:
            datasets[dataset_uri]["license"] = bindings[0]["license"]["value"]

    except Exception as e:
        print(f"License fetch error for {dataset_uri}: {e}")

    time.sleep(0.3)  # avoid endpoint overload

# -----------------------------
# STEP 4: Convertion to JSON
# -----------------------------
final_datasets = []

for d in datasets.values():
    final_datasets.append({
        "id": d["id"],
        "title": d["title"],
        "description": d["description"],
        "keywords": sorted(list(d["keywords"])),
        "license": d["license"]
    })

# -----------------------------
# Save
# -----------------------------
with open("eu_datasets_clean.json", "w", encoding="utf-8") as f:
    json.dump(final_datasets, f, indent=2, ensure_ascii=False)

print(f"Saved {len(final_datasets)} datasets.")