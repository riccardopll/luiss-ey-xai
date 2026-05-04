# PLAN

## 1. Project Goal

Build a working proof of concept that helps municipalities, consultants, or public-sector advisors compare a new local project idea with past EU-funded projects. The system uses historical projects as reference examples to suggest how the new idea could be positioned.

The POC should answer one core question:

> Given a municipal project idea, which past EU-funded projects are most similar, and what programme, fund, category, objective, and budget information can be suggested from those matches?

Use retrieval to find similar EU-funded projects, then use a local LLM to explain each match. Retrieval controls the ranking; the LLM only explains it. The approach is detailed in [RAG.md](RAG.md). Each run is recorded in an append-only, hash-linked audit log that covers the user's input, model configuration, generated outputs, retrieved evidence, and explanations.

## 2. Context

The EY brief focuses on XAI, blockchain traceability, and EU funding support. XAI makes AI outputs understandable to users and evaluators. Traceability preserves evidence of the process in an auditable record. The funding support layer helps reduce information fragmentation and identify relevant funding opportunities.

The final submission must include a working prototype with clean, reproducible source code, a technical report, and a short business-oriented presentation.

The POC should support this flow:

1. The user provides a project idea as raw text or PDF, such as a municipal project description, concept note, or planning document.
2. The user can add an optional location string, which the system geocodes into place metadata.
3. The system extracts and cleans the project description into plain text that can be used by the retrieval model.
4. The model retrieves the most similar historical project records.
5. The system returns ranked matches, recommended positioning information, and a short grounded explanation of why the matches were selected.
6. The system stores an audit record for the run.

## 3. Signals vs Outputs

Keep a clear distinction between fields used to rank similarity (signals) and fields returned to help the user (outputs).

**Signals** influence retrieval and ranking:

| Signal                                                               | Use                                                                                                               |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `Operation_Name_English` / `Operation_Name_Programme_Language`       | Short semantic signal from the project title, using both English and programme-language text when available.      |
| `Operation_Summary_English` / `Operation_Summary_Programme_Language` | Main semantic signal from the project description, using both English and programme-language text when available. |
| `Country` / `CountryCode`                                            | Optional ranking adjustment when the project country matches the geocoded location.                               |
| `Location_Indicator_latitude_longitude`                              | Optional ranking adjustment for projects geographically close to the user location.                               |
| `NUTS3_Label`                                                        | Optional ranking adjustment for projects in the same NUTS3 area as the geocoded location.                         |
| `LAU_Labels`                                                         | Optional ranking adjustment for projects in the same local administrative area as the geocoded location.          |

**Outputs** help the user position the project:

| Output                                 | Use                                                                              | Example                                                  |
| -------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `Programme_Name`                       | Shows which programmes funded similar projects.                                  | `Employment - AT - ESF+/JTF`                             |
| `Fund_Name`                            | Shows which EU funds supported similar projects.                                 | `European Regional Development Fund`                     |
| `Category_Label`                       | Shows the intervention categories used for similar projects.                     | `Support for adult education (excluding infrastructure)` |
| `Specific_Objective_Label`             | Shows the detailed programme objective code or label linked to similar projects. | `ESO4.7`                                                 |
| `Policy_Objective_Label`               | Shows the high-level EU policy priority linked to similar projects.              | `Social Europe`                                          |
| `Total_Eligible_Expenditure_amount`    | Provides a benchmark for total eligible project cost.                            | `EUR 144,920`                                            |
| `Project_EU_Budget`                    | Provides a benchmark for the EU contribution amount.                             | `EUR 57,968`                                             |
| `Country`, `NUTS3_Label`, `LAU_Labels` | Shows where the matched projects are located.                                    | `Austria`, `Salzburg und Umgebung`, `Salzburg`           |
| `InfoRegio_URL`                        | Links back to the original project source when available.                        | `https://...`                                            |

## 4. [Model](MODEL.md)

## 5. Explainability and Traceability

Each result should explain the match in non-technical language: what was similar, which fields contributed, whether geography affected the score, which fields were copied from similar projects, and what confidence score was assigned.

The explanation layer should use the RAG pipeline described in [RAG.md](RAG.md).

Example explanation structure:

> This project is similar because the matched record's summary and the user's project description both discuss energy efficiency in public infrastructure. The score was supported by semantic similarity and overlapping terms in the summaries. Geography also increased the score because the project is in the same country. The suggested fund, category, objective, and budget are copied from similar historical projects and should be treated as benchmarks.

For the POC, the blockchain layer can be a local append-only audit log. Each run creates one immutable block that records what was submitted, which model configuration was used, which results were returned, and how the block connects to the previous run.

| Field                    | Purpose                                                              |
| ------------------------ | -------------------------------------------------------------------- |
| `run_id`                 | Unique identifier for the inference run.                             |
| `timestamp`              | Records when the run was created.                                    |
| `input_text_hash`        | Verifies the cleaned text used by the model.                         |
| `uploaded_file_hash`     | Verifies the uploaded PDF when the input is a file.                  |
| `location_metadata_hash` | Verifies the geocoded location metadata when a location is provided. |
| `model_version`          | Records which model was used.                                        |
| `dataset_version`        | Records which processed dataset was used.                            |
| `top_k_result_ids`       | Records the project IDs returned by the retrieval model.             |
| `llm_input_hash`         | Verifies the snippets and scores given to the local LLM.             |
| `prompt_template_version` | Records which explanation prompt contract was used.                  |
| `llm_model_version`      | Records which local LLM generated the explanation.                   |
| `explanation_hash`       | Verifies the generated explanation.                                  |
| `previous_block_hash`    | Links the block to the previous audit record.                        |
| `current_block_hash`     | Verifies the integrity of the current block.                         |
