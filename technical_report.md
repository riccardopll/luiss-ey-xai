---
papersize: a4
geometry: margin=0.75in
fontsize: 11pt
mainfont: Helvetica
sansfont: Helvetica
monofont: Menlo
colorlinks: true
linkcolor: blue
urlcolor: blue
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
---

\begin{titlepage}
\centering
\vspace{2.8cm}
{\Huge Funding Compass\par}
\vspace{1.2cm}
{\large Technical report\par}
{\large Artificial Intelligence Techniques\par}
{\large Department of AI, Data and Decision Sciences\par}
{\large Prof. Martino Alessio\par}
{\large Company: EY\par}
{\large May 2026\par}
\vspace{3.6cm}
{\large Team members\par}
\vspace{0.4cm}
{\large
\renewcommand{\arraystretch}{1.35}
\begin{tabular}{@{}llll@{}}
Simone Prezioso & simone.prezioso@studenti.luiss.it & 324111 & Group Delegate \\
Lapo Chiaselotti & lapo.chiaselotti@studenti.luiss.it & 308291 & \\
Silvia Monteleone & silvia.monteleone@studenti.luiss.it & 315221 & \\
Riccardo Palleschi & riccardo.palleschi@studenti.luiss.it & 319401 & \\
\end{tabular}\par}
\vfill
\end{titlepage}

# 1. Introduction

Funding Compass is a prototype for early EU funding research. It helps
municipalities, consultants, and public-sector advisors compare a new local project
idea with past EU-funded projects. The user enters a project description and, when
available, a location. The system retrieves similar historical projects and reports
the programme, fund, category, objective, geography, and budget fields found in those
records.

The system is designed as explainable and traceable AI. Its outputs can be checked,
explained, and traced back to evidence. We handle that by separating the pipeline
into three parts: retrieval selects the past projects, a local language model explains
why each match was returned, and an append-only audit log records the input, selected
evidence, model configuration, and generated output.

The data source is the public corpus of EU 2021-2027 funded project records stored in
`data/raw`. The processed dataset contains 124,988 projects from 26 countries. Each
record includes project names, summaries, programme and fund labels, policy and
specific objectives, intervention categories, geography fields, budget values, and
source URLs. The project uses past projects as evidence for comparison and budget
benchmarking.

The implementation is contained in `src/main.ipynb`. The notebook runs the full
workflow: schema validation, corpus creation, BM25 text preparation, dense index
construction, geocoding, hybrid retrieval, benchmarking, RAG explanations, audit
logging, and an end-to-end user example.

# 2. Methods

## 2.1 Data collection and corpus construction

The notebook loads country-level CSV files from `data/raw`. Before any index is
built, it checks that every file has the columns needed for this project: project
names and summaries for matching, geography fields for location scoring, funding
labels for suggestions, and budget fields for benchmarks. After validation, the
corpus builder cleans text fields, standardizes labels and country codes, parses
budget values, extracts latitude and longitude from location strings, and creates
stable record identifiers.

The corpus separates ranking signals from output fields. Ranking uses project names,
summaries, country codes, coordinates, NUTS3 labels, and LAU labels. The fields shown
to the user are copied from the retrieved projects: programme, fund, category, policy
objective, specific objective, eligible expenditure, EU budget, place labels, and
source URL. This prevents the system from inventing funding information. It can only
suggest fields that appear in similar historical records.

The pipeline builds two text views. The readable `search_text` keeps English and
programme-language names and summaries for dense retrieval, display, and RAG evidence.
A second view removes stop words and tokenizes text for BM25. The BM25 text is shorter
than the readable text on average, so it gives the sparse model a cleaner set of
terms without removing the original text needed for explanations.

![Retrieval, explanation, and audit pipeline.](assets/pipeline.png)

## 2.2 Retrieval models

This is a retrieval problem, not a supervised classification problem. The system
searches a historical corpus and ranks the closest examples to the input idea. The
implemented candidates are:

| Candidate                       | Role in the comparison                                                                         |
| ------------------------------- | ---------------------------------------------------------------------------------------------- |
| BM25 sparse retrieval           | Keyword baseline for exact terms, locations, sectors, policy labels, and domain words.         |
| SBERT MPNet dense retrieval     | Semantic baseline using `sentence-transformers/all-mpnet-base-v2`.                             |
| Multilingual E5 dense retrieval | Multilingual dense retrieval using `intfloat/multilingual-e5-large`.                           |
| Hybrid retrieval                | Final ranking model combining dense similarity, BM25 keyword evidence, and optional geography. |

The hybrid confidence score combines three normalized components:

`confidence = semantic_weight * semantic_score + keyword_weight * keyword_score + geographic_weight * geographic_score`

When no usable location is available, the geography component is left out and the
remaining weights are renormalized. Geography can improve a match only when the user
provides a location and the retrieved records contain usable location data.

The selected configuration uses multilingual E5 as the dense component. It weights
semantic similarity at 0.50, keyword evidence at 0.35, and geography at 0.15. This
keeps meaning as the strongest signal, while BM25 preserves exact public-policy terms
and geography adds a small local adjustment.

## 2.3 Benchmarking and tuning

The benchmark uses 200 historical project records sampled from the processed corpus.
Each sampled record becomes a query. Label names are removed from the query text so
the model cannot win by matching the answer directly. During evaluation, the source
record is also excluded from the retrieval corpus, so the model cannot receive credit
for returning the same project.

Retrieved records are scored with graded relevance:

| Grade | Match condition                        | Interpretation |
| ----- | -------------------------------------- | -------------- |
| 3     | Same specific objective                | Strong match   |
| 2     | Same category or same policy objective | Partial match  |
| 1     | Same fund or programme only            | Weak match     |
| 0     | None of the selected labels match      | Not relevant   |

The benchmark reports Precision@10, Mean Reciprocal Rank, and NDCG@10. NDCG@10 is
the main metric because it rewards the model for placing stronger matches above
weaker ones. Precision@10 checks how many relevant records appear in the first ten
results. Mean Reciprocal Rank checks how quickly the first relevant record appears.

## 2.4 Explanation and traceability

Retrieval controls the ranking. The local LLM receives one selected project at a
time, together with the input idea, match metadata, and score components. Its job is
limited: explain why that project was selected. It cannot change the rank, confidence
score, programme, fund, category, objective, budget, or location metadata. The
implemented local model is `gemma4:e4b`, run through Ollama, with a strict JSON output
contract.

Traceability is implemented as a local append-only audit log instead of a deployed
blockchain. Each run writes a JSON block under `data/audit`. The block stores hashes
of the input, location metadata, LLM input, explanations, suggestions, and retrieved
project IDs. It also stores the previous and current block hashes. If the input,
result, or explanation changes later, the hash changes too, so the run can be checked
again.

# 3. Results and Discussion

## 3.1 Corpus and label structure

The processed dataset contains 124,988 records, 48 columns, 26 country codes, 7 fund
values, 620 category labels, and 66 specific objective labels. The labels have very
different levels of detail. Fund and policy labels are broad: many projects share the
same value. Category and specific objective labels are narrower, so they are better
signals when judging whether two projects are close.

| Field              | Unique values | Top value share | Interpretation                                      |
| ------------------ | ------------: | --------------: | --------------------------------------------------- |
| Specific objective |            65 |           18.8% | Strong signal, though still shared by many records. |
| Category           |           619 |           13.5% | Most detailed relevance signal used here.           |
| Policy objective   |            10 |           45.1% | Broad policy context.                               |
| Fund               |             7 |           50.1% | Funding context, weak on its own.                   |
| Programme          |           173 |           10.3% | Programme context and relevance support.            |

## 3.2 Retrieval quality

Multilingual E5 performs best in the benchmark. The selected configuration has the
highest NDCG@10, while its Precision@10 and Mean Reciprocal Rank stay close to the
best values in the table.

| Rank | Candidate       | Weights S/K/G      | Precision@10 |    MRR | NDCG@10 |
| ---: | --------------- | ------------------ | -----------: | -----: | ------: |
|    1 | multilingual E5 | 0.50 / 0.35 / 0.15 |       0.8225 | 0.9185 |  0.9408 |
|    2 | multilingual E5 | 0.60 / 0.30 / 0.10 |       0.8185 | 0.9166 |  0.9401 |
|    3 | multilingual E5 | 0.50 / 0.30 / 0.20 |       0.8220 | 0.9215 |  0.9395 |
|    4 | multilingual E5 | 0.60 / 0.25 / 0.15 |       0.8195 | 0.9189 |  0.9388 |
|    5 | SBERT MPNet     | 0.50 / 0.35 / 0.15 |       0.8210 | 0.9297 |  0.9373 |

The ranking results fit the structure of the corpus. Many records contain both
English and programme-language descriptions, so multilingual embeddings help. BM25
still matters because project summaries contain exact sector terms, policy terms,
and location names that should not be lost inside a dense embedding. The geography
score works best as a small adjustment rather than a dominant ranking factor.

## 3.3 Example run

The final notebook example searches for projects similar to renovating public school
buildings in Italy with insulation, smart energy monitoring, and renewable heating
systems. The top ten results are all energy-efficiency projects for public or school
buildings. They mention insulation, heating replacement, renewable energy, and energy
performance improvements. The top matches come from Romania, Poland, France, Czechia,
and Slovakia. All ten are funded by the European Regional Development Fund.

The confidence scores range from 0.7544 for the top match to 0.6842 for the tenth
match. The median eligible expenditure benchmark is about EUR 573,675. The
explanations focus on the shared public-building energy-efficiency theme and on
similar technical measures. Geography is not used in this example because the
retrieved records are not Italian records, even though the input location is Italy.

The notebook also includes a station-reuse example near Trento. In that run, the top
result is an Italian project in Rovereto with confidence close to 1.0. Here the
geography score behaves as intended: it strengthens the match when both the project
theme and the local context line up.

## 3.4 Business value and limitations

Funding Compass is most helpful at the early framing stage of a project. A user can
start with a plain idea and quickly see comparable past projects, funding context,
budget benchmarks, source links, and short explanations. The scope is limited on
purpose: the system suggests evidence from similar projects, but it does not claim
that the new project is eligible or guaranteed to receive funding.

The explanation and audit layers make the output easier to review. Users can see why
each result was selected, which fields came from retrieved evidence, and whether
geography changed the score. The audit log keeps hashes of the input, evidence,
explanations, suggestions, and previous run, which makes the process traceable after
the fact.

The main limitation is the benchmark itself. Relevance labels are approximated from
project metadata. Sharing a category or objective is a reasonable signal, but it is
not the same as a human reviewer judging that two projects are truly similar. The
benchmark evaluates retrieval quality, not the legal or business correctness of a
funding recommendation. The local LLM also has a narrow role: it explains retrieved
evidence, but it does not validate call rules, legal eligibility, or future funding
availability.

# 4. Conclusions

Funding Compass shows that hybrid retrieval can give municipalities a practical way
to compare a new idea with previous EU-funded projects. The best benchmark result
uses multilingual E5, BM25, and a limited geography adjustment, reaching NDCG@10 of
0.9408 on the 200-query benchmark. The architecture is also easy to inspect:
retrieval selects the evidence, the local LLM explains that evidence, and the audit
log records the run in a verifiable chain.

The next version should add human-labeled relevance judgments, a more modular code
layout, stronger multilingual tests, and a user interface. It should also keep project
similarity separate from funding eligibility. Similarity can be estimated from
historical records. Eligibility must be checked against current calls, rules, budgets,
and administrative constraints before anyone makes a real funding decision.

# Bibliography

[1] Sentence Transformers model card for `sentence-transformers/all-mpnet-base-v2`.

[2] Multilingual E5 model card for `intfloat/multilingual-e5-large`.

[3] Ollama local model reference for `gemma4:e4b`.

\newpage

# Appendix A: Code Description

The implementation is currently contained in `src/main.ipynb`. The notebook is
organized as an executable pipeline:

| Notebook section   | Role                                                                                               |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| Setup              | Loads libraries and defines paths, versions, `TOP_K`, and random seed.                             |
| Schema Validation  | Checks that all raw CSV files contain the required columns.                                        |
| Project Corpus     | Cleans fields, parses budgets and coordinates, and writes `data/processed/eu_projects_v1.parquet`. |
| BM25 Lexical Text  | Builds tokenized text for sparse retrieval.                                                        |
| BM25 Index         | Builds or loads the cached BM25 index from `data/indexes`.                                         |
| Dense Indexes      | Builds or loads SBERT MPNet and multilingual E5 embeddings.                                        |
| Geocoding          | Uses Nominatim and country-code shortcuts to produce location metadata and geography scores.       |
| Hybrid Model       | Combines semantic, keyword, and geography scores and returns ranked matches.                       |
| Benchmark Queries  | Builds the 200-query benchmark set.                                                                |
| Label Distribution | Checks whether relevance labels are broad or specific.                                             |
| Benchmark Results  | Evaluates candidates and weight combinations.                                                      |
| RAG                | Creates suggestions and match explanations with the local LLM.                                     |
| Audit Block        | Writes chained audit records.                                                                      |
| User Example       | Runs the full POC end to end.                                                                      |

Pseudocode:

```text
load raw country CSV files
validate required schema
clean names, summaries, labels, budgets, and coordinates
build readable search text and BM25 token text
cache processed dataset and indexes
for a user query:
    clean text and geocode optional location
    compute dense, BM25, and optional geography scores
    combine scores into confidence
    retrieve top-k historical projects
    copy programme, fund, objective, category, budget, and source fields
    ask local LLM to explain each selected match using only retrieved evidence
    write a chained audit block
return ranked table, explanations, suggestions, and audit metadata
```

# Appendix B: Author Contribution and Generative AI Statement

Author contribution statement: Simone Prezioso, Lapo Chiaselotti, Silvia
Monteleone, and Riccardo Palleschi contributed to conceptualization, methodology,
data curation, software development, validation, analysis, visualization, and
writing. Simone Prezioso is the group delegate.

Generative AI statement: The group used Generative AI tools for brainstorming,
drafting, editing, and clarifying parts of the technical report and source README.
The group reviewed generated text and code suggestions against the project files,
notebook outputs, benchmark results, and generated artifacts. The group takes
responsibility for understanding, validating, and explaining all submitted materials.
