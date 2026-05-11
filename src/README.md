## Purpose

This [notebook](main.ipynb) starts from a municipal project idea and searches past
EU-funded projects for similar cases. It returns the closest matches with
programme and fund suggestions, budget benchmarks, short explanations, and linked
audit records. It also builds the corpus, indexes, benchmark queries, and
evaluation files used by the prototype. More information is available in the
[technical report](../technical_report.md).

## Environment

Create a Python (3.12.9) virtual environment from the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a root `.env` file for local settings:

```bash
touch .env
printf 'NOMINATIM_EMAIL=your-email@example.com\n' >> .env
printf 'HF_TOKEN=your-hugging-face-token\n' >> .env
```

`NOMINATIM_EMAIL` is optional and is only used for live geocoding. It does not require registration; any valid e-mail address is enough.

`HF_TOKEN` is optional. It is used only to make public model downloads faster.

The explanation layer expects Ollama to be running locally with `gemma4:e4b`:

```bash
ollama pull gemma4:e4b
ollama serve
```

Then run `main.ipynb` from top to bottom.

## Generated artifacts

The notebook writes reproducible artifacts outside `src`:

| Path                                                      | Contents                        |
| --------------------------------------------------------- | ------------------------------- |
| `data/processed/eu_projects_*.parquet`                    | Processed corpus                |
| `data/indexes/bm25s_*.pkl`                                | Sparse BM25 index               |
| `data/indexes/sbert_mpnet_*.pkl`                          | SBERT embedding index           |
| `data/indexes/multilingual_e5_*.pkl`                      | Multilingual E5 embedding index |
| `data/benchmarks/benchmark_queries_*.csv`                 | Benchmark query set             |
| `data/benchmarks/benchmark_queries_evaluation_*_top*.csv` | Retrieval evaluation            |
| `data/audit/*.json`                                       | Linked audit records            |

If cached artifacts already exist, the notebook loads them.
