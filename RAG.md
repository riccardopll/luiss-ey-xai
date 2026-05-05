# RAG

The ranking model finds and orders the best historical EU-funded project matches. The local LLM receives the user project text, one ranked match, and the match metadata, then explains why that record was selected.

The local LLM is `gemma4:e4b-instruct`, run through Ollama. It is small enough to run locally for the proof of concept, supports multilingual project text reasonably well, and can follow a strict JSON output format. An overview of the selected model is available on [Gemma 4 E4B non-reasoning](https://artificialanalysis.ai/models/gemma-4-e4b-non-reasoning).

For simplicity, the prototype uses the model's non-reasoning mode.

## Input

The local LLM receives one matched project at a time:

```json
{
  "user_project": {
    "text": "User project description.",
    "location": "Optional user location."
  },
  "matched_project": {
    "project_id": "record_001",
    "rank": 1,
    "title": "Historical project title.",
    "summary": "Historical project summary.",
    "country": "Italy",
    "nuts3_label": "Roma",
    "lau_labels": "Roma"
  },
  "matched_project_metadata": {
    "programme_name": "Programme copied from the matched project.",
    "fund_name": "Fund copied from the matched project.",
    "category_label": "Category copied from the matched project.",
    "specific_objective_label": "Specific objective copied from the matched project.",
    "policy_objective_label": "Policy objective copied from the matched project.",
    "total_eligible_expenditure_amount": "Budget benchmark copied from the matched project.",
    "project_eu_budget": "EU budget benchmark copied from the matched project."
  },
  "scores": {
    "semantic_score": 0.82,
    "keyword_score": 0.67,
    "geographic_score": 0.4,
    "confidence_score": 0.74
  }
}
```

## Output

The local LLM returns valid JSON only:

```json
{
  "project_id": "record_001",
  "rank": 1,
  "confidence_score": 0.74,
  "explanation": "The project is similar because...",
  "geography_note": "Geography affected the score because..."
}
```

## Prompt

```text
You are an explanation assistant for an EU funding retrieval system.

The ranking model has already selected this historical project. Only explain why it was matched.

Rules:
- Use only the provided JSON input.
- Do not change the project rank or confidence score.
- Do not invent programme, fund, category, objective, budget, or location information.
- Do not claim that the user project is eligible for funding.
- Explain the match in plain English for a non-technical user.
- Do not include internal reasoning steps or notes.
- If geography did not contribute to the score, say that geography was not used.
- Return valid JSON only, with no markdown.

Input JSON:
{{llm_input_json}}

Return this JSON structure:
{
  "project_id": "...",
  "rank": 1,
  "confidence_score": 0.0,
  "explanation": "...",
  "geography_note": "..."
}
```
