# MODEL

The model is a retrieval model. Its goal is to compare a user-provided project idea with historical EU-funded project records and return a ranked list of the most similar examples. The programme, fund, category, objective, and budget information shown to the user will be inferred from the top retrieved records.

It is a semantic retrieval model with optional geographic ranking adjustments. It will represent the user's project description and each historical project record as searchable text, compare them, rank the most similar records, and optionally adjust the ranking using location information. The model is not trained from scratch: the main work is to [prepare the dataset and build a searchable index](#1-preprocessing), [choose a retrieval method](#2-candidate-models), [tune the ranking logic](#3-tuning-strategy), and [evaluate which retrieval approach gives the most useful matches](#4-evaluation-plan).

The model receives two possible inputs:

| Input                    | Role                                                                        |
| ------------------------ | --------------------------------------------------------------------------- |
| Project description text | Main input used for semantic matching.                                      |
| Optional location string | Used to geocode the user location and apply geographic ranking adjustments. |

The project description can come from raw text entered by the user or from text extracted from an uploaded PDF. After extraction, the text should be cleaned into plain text before being passed to the retrieval model.

## 1. Preprocessing

The preprocessing step should make the historical project records consistent and searchable:

1. Load all country CSV files from `data/raw`.
2. Validate that each file follows the expected schema.
3. Clean text fields by removing empty values, repeated whitespace, and other formatting noise.
4. Build a combined text field from `Operation_Name_English` and `Operation_Summary_English` to give the retrieval model one searchable description for each historical project.
5. Convert `Total_Eligible_Expenditure_amount` and `Project_EU_Budget` into numeric values so the system can calculate budget benchmarks from matched projects.
6. Split `Location_Indicator_latitude_longitude` into numeric latitude and longitude values so the system can compare project locations.
7. Standardize country codes and geographic labels by trimming whitespace, using consistent casing, and handling missing values.
8. Store the processed dataset as a `.parquet` file in `data/processed/` with a clear `dataset_version`.
9. Build one searchable index for each candidate retrieval model and store it in `data/indexes/`, using the `{candidate_name}_{dataset_version}` naming format, such as `tfidf_v1`, `bm25_v1`, or `embeddings_v1`. A searchable index is the model-specific search structure created from the processed dataset, such as a TF-IDF matrix, a BM25 index, or an embedding vector index, so the system can retrieve similar projects efficiently without scanning the raw CSV files directly.
10. Create benchmark queries from a selected subset of historical project records and store them in `data/benchmarks/` with the relevance fields needed for evaluation. More detail is provided in the [evaluation plan](#4-evaluation-plan).

## 2. Candidate Models

The project should compare multiple retrieval approaches before choosing the final one. The ranking performance of these candidates will be compared using the benchmark queries and metrics described in the [evaluation plan](#4-evaluation-plan):

| Candidate                | Description                                                                             | Why it is included                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Keyword baseline         | Uses lexical overlap between the user text and historical project text.                 | Provides the simplest reference point and shows whether exact word overlap is enough.                      |
| TF-IDF cosine similarity | Represents documents using weighted word importance.                                    | Provides a reproducible classical retrieval baseline that gives more weight to distinctive terms.          |
| BM25                     | Ranks records using a search-engine-style relevance score.                              | Provides a stronger keyword-based search baseline commonly used for document retrieval.                    |
| Sentence embedding model | Converts the user text and project records into semantic vectors.                       | Tests whether semantic similarity improves matches when the same idea is expressed with different wording. |
| Hybrid retrieval         | Combines semantic similarity with keyword retrieval and geographic ranking adjustments. | Tests whether combining text meaning, exact terms, and location gives the most useful results.             |

The likely final model is a hybrid retrieval model built by stacking three components: a semantic component, a keyword component, and a geographic component. The semantic component retrieves projects with similar meaning even when the wording differs. The keyword component preserves exact matches for important terms such as "school", "energy efficiency", "wastewater", "digitalization", or "SME". The geographic component adjusts the ranking when the user provides a location and the dataset contains matching country, regional, local, or coordinate information.

All candidate models return similarity scores. For the hybrid model, this score is calculated from the semantic, keyword, and geographic components. More detail is provided in the [confidence score](#5-confidence-score) section.

The ranking should remain transparent enough to explain. During retrieval, the system should store the main score components for each returned result, such as semantic similarity, keyword overlap, country match, geographic distance, and final combined score. After ranking, these score components will be used to generate a short template-based explanation stating whether a result was selected mainly because of text similarity, geographic proximity, or both.

## 3. Tuning Strategy

Tuning is a lightweight hyperparameter process. It tests different retrieval and ranking configurations to understand which one returns the most useful historical project matches.

Tuning is done for each candidate model because each model has different settings:

| Candidate                | Tuning choices                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Keyword baseline         | Text cleaning rules, token matching rules, and top-k value.                                                                    |
| TF-IDF cosine similarity | Word range, minimum document frequency, maximum document frequency, title vs summary weighting, and top-k value.               |
| BM25                     | Tokenization rules, BM25 parameters, title vs summary weighting, and top-k value.                                              |
| Sentence embedding model | Embedding model, text field weighting, similarity metric, and top-k value.                                                     |
| Hybrid retrieval         | Semantic score weight, keyword score weight, country adjustment, distance adjustment, NUTS3 / LAU adjustment, and top-k value. |

The tuning process should use randomized search over a constrained parameter space. The valid parameters and ranges are defined based on the model design, then the algorithm samples configurations from those ranges. Each sampled configuration is evaluated on the benchmark queries, and the best configuration is selected mainly by `NDCG@k`, with `Precision@k` and `Mean Reciprocal Rank` as supporting metrics. More detail is provided in the [evaluation plan](#4-evaluation-plan).

For the hybrid model, tuning the ranking weights also tunes the [confidence score](#5-confidence-score), because the confidence score is the weighted combination used to rank results.

After the best configuration is selected for each candidate model, the tuned candidates should be compared against each other. The final model should be the one with the strongest overall retrieval quality and ranking signals that can support clear explanations. If two tuned models perform similarly, the simpler and more transparent one should be preferred.

## 4. Evaluation Plan

Because this is a retrieval system rather than a supervised classifier, the historical project records are not split into train, validation, and test sets. The full processed dataset is used as the retrieval corpus: it is the collection of past EU-funded projects that the model searches over.

Evaluation uses the benchmark queries generated during preprocessing. These queries test whether the retrieval model can find relevant historical projects for realistic inputs. Each benchmark query is derived from a selected historical record by using its `Operation_Name_English`, `Operation_Summary_English`, and location fields. This approach is useful because the source record already contains known output fields, such as `Programme_Name`, `Fund_Name`, `Category_Label`, `Specific_Objective_Label`, `Policy_Objective_Label`, `Total_Eligible_Expenditure_amount`, and `Project_EU_Budget`, which can be used as relevance signals.

During evaluation, the source record used to generate each benchmark query is removed from the retrieved results, so the model is evaluated on its ability to find other similar projects.

Relevance is estimated by checking whether the retrieved projects share meaningful output fields with the source record. For simplicity, the same benchmark query set is used to compare candidate models and tune configurations, and the main metrics measure whether relevant projects appear in the top results and whether the strongest matches are ranked first.

| Metric                 | What it measures                                                                      | Why it matters                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `Precision@k`          | The share of the top-k retrieved projects that are relevant.                          | It checks whether the user sees useful matches in the first results.              |
| `Mean Reciprocal Rank` | How high the first relevant result appears in the ranking.                            | It rewards models that place a good match near the top.                           |
| `NDCG@k`               | Whether highly relevant projects are ranked above weaker but still relevant projects. | It checks the quality of the ordering, not just whether relevant projects appear. |

## 5. Confidence Score

The confidence score is the user-facing similarity score produced by the hybrid approach. It should communicate how strong a retrieved match is.

`confidence = 0.60 * semantic_score + 0.25 * keyword_score + 0.15 * geographic_score`

Each component should be scaled between 0 and 1 before being combined. `semantic_score` measures meaning similarity between the user input and the historical project text. `keyword_score` measures exact or near-exact term overlap. `geographic_score` measures whether the matched project is geographically relevant to the user-provided location.

The semantic score receives the highest weight because project meaning should be the main driver of retrieval. The keyword score receives a smaller weight because exact terms are useful for preserving important domain words, but should not dominate semantic similarity. The geographic score receives the smallest weight because location should adjust the ranking only when the user provides location information and matching geographic data exists.
