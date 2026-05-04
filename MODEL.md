# MODEL

Use a retrieval model to compare a user-provided project idea with historical EU-funded project records and return the most similar examples. Programme, fund, category, objective, and budget suggestions come from the top retrieved records.

The model is hybrid, with optional geographic ranking adjustments. It represents the user's project description and each historical project record as searchable text, compares them, ranks the closest records, and adjusts the ranking with location information when available.

The model comparison uses BM25 as the sparse baseline, dense embedding models for semantic retrieval, and a hybrid model that combines dense similarity, sparse keyword evidence, and optional geography.

The model receives two possible inputs:

| Input                    | Role                                                                        |
| ------------------------ | --------------------------------------------------------------------------- |
| Project description text | Main input used for matching.                                               |
| Optional location string | Used to geocode the user location and apply geographic ranking adjustments. |

The project description can come from raw text entered by the user or from text extracted from an uploaded PDF.

## 1. Preprocessing

1. Load all country CSV files from `data/raw`.
2. Validate that each file follows the expected schema.
3. Clean text fields by removing empty values, repeated whitespace, and formatting noise.
4. Build combined text fields from `Operation_Name_English`, `Operation_Summary_English`, `Operation_Name_Programme_Language`, and `Operation_Summary_Programme_Language` so each record has searchable English and programme-language descriptions.
5. Create a lightweight lexical preprocessing variant for BM25: tokenized lowercase text, stop-word removal, and a combined title-summary text field.
6. Keep an unstemmed, readable text field for dense embeddings, result display, and RAG explanation evidence. This field is used to show project titles, summaries, matched snippets, and the grounded context passed to the local LLM.
7. Preserve multilingual input compatibility for EU project descriptions. Query cleaning keeps the original wording as much as possible and avoids transformations that would reduce the quality of multilingual embeddings.
8. Convert `Total_Eligible_Expenditure_amount` and `Project_EU_Budget` into numeric values so the system can calculate budget benchmarks from matched projects.
9. Split `Location_Indicator_latitude_longitude` into numeric latitude and longitude values so the system can compare project locations.
10. Standardize country codes and geographic labels by trimming whitespace, using consistent casing, and handling missing values.
11. Store the processed dataset as a `.parquet` file in `data/processed/` with a clear `dataset_version`.
12. Build one searchable index for each implemented candidate model and store it in `data/indexes/`, using the `{candidate_name}_{dataset_version}` naming format, such as `bm25_v1`, `sbert_mpnet_v1`, `multilingual_e5_v1`, `bge_m3_v1`, or `hybrid_bm25_bge_m3_geo_v1`. Each index stores the model-specific search structure created from the processed dataset, such as a BM25 index, embedding matrix, vector index, or hybrid score configuration. This lets the system retrieve similar projects efficiently without scanning the raw CSV files directly.
13. Create benchmark queries from a selected subset of historical project records and store them in `data/benchmarks/` with the relevance fields needed for evaluation. More detail is provided in the [evaluation plan](#4-evaluation-plan).

## 2. Candidate Models

Compare several retrieval approaches before choosing the final one.

Compare candidate ranking performance using the benchmark queries and metrics described in the [evaluation plan](#4-evaluation-plan):

| Candidate                       | Description                                                                                                                               | Base                                                                                                                                      | Why                                                                                                                       |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| BM25 sparse baseline            | Ranks records using a search-engine-style relevance score based on term frequency, document frequency, and document length normalization. | No neural base. The BM25 index is built from the tokenized project corpus.                                                                | Serves as the sparse baseline for exact terms, project sectors, policy labels, locations, and domain-specific words.      |
| SBERT / MPNet dense retriever   | Converts the user text and project records into dense embeddings and ranks records by cosine similarity.                                  | Pretrained base: `sentence-transformers/all-mpnet-base-v2`.                                                                               | Provides a semantic retrieval reference and tests whether contextual embeddings improve over sparse keyword matching.     |
| Multilingual E5 dense retriever | Uses a multilingual embedding model to compare user queries and project records across languages.                                         | Pretrained base: `intfloat/multilingual-e5-large`, using the recommended query/document text prefixes.                                    | Tests whether the system can handle input written in another language.                                                    |
| BGE-M3 multilingual retriever   | Uses another multilingual embedding model to produce dense vectors for the same retrieval task.                                           | Pretrained base: `BAAI/bge-m3`, used in dense retrieval mode for this comparison.                                                         | Benchmarks a second multilingual dense retriever against multilingual E5.                                                 |
| Hybrid BM25 + dense + geography | Combines sparse BM25 scores, dense semantic similarity from the best embedding model, and optional geographic ranking signals.            | Uses the best dense retriever from the comparison plus a fitted BM25 index. Hybrid weights are tuned separately from the embedding model. | Tests whether exact keyword evidence plus multilingual semantic similarity gives the most useful and explainable results. |

The final model combines sparse BM25, dense semantic similarity, and optional geography. BM25 preserves exact domain terms. Dense embeddings find similar meaning across different wording and languages. Geography adjusts the ranking only when the user provides a location and the dataset contains matching country, regional, local, or coordinate information.

All candidate models return similarity scores. For the hybrid model, this score combines semantic similarity, keyword evidence, and geography. More detail is provided in the [confidence score](#5-confidence-score) section.

After ranking, a local LLM connected to the [RAG retrieval pipeline](RAG.md) explains each match. It receives the matched project document, relevant snippets from the user input, scores, and geographic evidence, then explains why the ranking model selected the record.

## 3. Tuning Strategy

Tuning is intentionally lightweight. It tests a small set of retrieval settings and hybrid weights to find the configuration that returns the most useful historical matches.

Tuning is done for the implemented candidate models:

| Candidate                       | Tuning choices                                                                                               |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| BM25 sparse baseline            | BM25 `k1` and `b`, score normalization, and top-k value.                                                     |
| SBERT / MPNet dense retriever   | Embedding normalization and top-k value.                                                                     |
| Multilingual E5 dense retriever | Query/document prefix format, embedding normalization, and top-k value.                                      |
| BGE-M3 multilingual retriever   | Embedding normalization and top-k value, using the same dense retrieval setup as multilingual E5.            |
| Hybrid BM25 + dense + geography | Dense score weight, BM25 score weight, geographic score weight, score normalization method, and top-k value. |

Tuning uses randomized search or a small manual grid over a constrained parameter space. Each sampled configuration is evaluated on the benchmark queries. The best configuration is selected mainly by `NDCG@k`, with `Precision@k` and `Mean Reciprocal Rank` as supporting metrics. More detail is provided in the [evaluation plan](#4-evaluation-plan).

For the hybrid model, tuning the ranking weights also tunes the [confidence score](#5-confidence-score), because the confidence score is the weighted combination used to rank results.

After the best configuration is selected for each candidate model, compare the tuned candidates against each other. Select the final model based on retrieval quality, multilingual robustness, runtime manageability, and ranking signals that support clear explanations. If two tuned models perform similarly, prefer the simpler and more transparent one.

## 4. Evaluation Plan

Because this is a retrieval system rather than a supervised classifier, the historical project records are not split into train, validation, and test sets. The full processed dataset is used as the retrieval corpus: the collection of past EU-funded projects the model searches over.

Evaluation uses the benchmark queries generated during preprocessing. These queries test whether the retrieval model can find relevant historical projects for user-like inputs. Each benchmark query is derived from a selected historical record using its English or programme-language name and summary fields, plus location fields when available. The source record already contains known output fields, such as `Programme_Name`, `Fund_Name`, `Category_Label`, `Specific_Objective_Label`, `Policy_Objective_Label`, `Total_Eligible_Expenditure_amount`, and `Project_EU_Budget`, which can be used as relevance signals.

During evaluation, the source record used to generate each benchmark query is removed from the retrieved results, so the model is evaluated on its ability to find other similar projects.

Relevance is estimated only for benchmark queries, where the source project already has known labels. Each retrieved project is compared with the source project and assigned a grade:

| Grade | Match condition                                                           | Meaning                                                                                                        |
| ----- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `3`   | Same `Specific_Objective_Label`                                           | Strong match: both projects target the same specific objective.                                                |
| `2`   | Same `Category_Label` or same `Policy_Objective_Label`                    | Partial match: both projects share the same intervention category or high-level policy objective.              |
| `1`   | Same `Fund_Name` or same `Programme_Name`, with no grade `2` or `3` match | Weak match: both projects are connected to the same fund or programme, but not to the same objective/category. |
| `0`   | None of the fields above match                                            | Not relevant for the benchmark query.                                                                          |

`NDCG@k` uses this graded relevance score, while `Precision@k` and `Mean Reciprocal Rank` treat grades `2` and `3` as relevant. For this POC, the same benchmark query set is used to compare candidate models and tune configurations.

| Metric                 | What it measures                                                               | Why it matters                                                                    |
| ---------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `Precision@k`          | The share of the top-k retrieved projects that are relevant.                   | It checks whether the user sees useful matches in the first results.              |
| `Mean Reciprocal Rank` | How high the first relevant result appears in the ranking.                     | It rewards models that place a good match near the top.                           |
| `NDCG@k`               | Whether highly relevant projects are ranked above partially relevant projects. | It checks the quality of the ordering, not just whether relevant projects appear. |

To check multilingual performance, use records that include both English and programme-language text. Treat the two versions as separate queries with the same relevance labels, then compare whether multilingual E5 and BGE-M3 retrieve similar projects.

## 5. Confidence Score

The confidence score is the user-facing similarity score produced by the hybrid model. It shows how close a retrieved match is.

The initial confidence formula is:

`confidence = 0.60 * semantic_score + 0.25 * keyword_score + 0.15 * geographic_score`

When no user location is provided, or when a matched project has no usable geographic data, omit the geography score and re-normalize the remaining weights:

`confidence = (0.60 * semantic_score + 0.25 * keyword_score) / 0.85`

Each score is scaled between 0 and 1 before being combined. `semantic_score` measures meaning similarity between the user input and the historical project text using the selected dense retriever. `keyword_score` measures exact or near-exact term overlap using BM25 or a normalized sparse score. `geographic_score` measures whether the matched project is geographically relevant to the user-provided location.

The semantic score receives the highest initial weight because project meaning is the main retrieval signal, especially with multilingual input. The keyword score receives a smaller weight because exact terms preserve important domain words, but should not dominate semantic similarity. The geographic score receives the smallest weight because location adjusts the ranking only when the user provides location information and matching geographic data exists. The final weights are selected during hybrid model tuning.
