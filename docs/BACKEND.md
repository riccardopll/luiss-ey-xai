# Backend

## Goal

The backend should own the full application logic:

- input handling;
- PDF text extraction;
- model loading;
- funding call matching;
- explanation generation;
- audit record creation;
- response rendering or API responses.

## Recommended Stack

- `Python 3.11+`
- `FastAPI`
- `pydantic`
- `pandas`
- `scikit-learn`
- `pypdf` or `pdfplumber`

## Role Of FastAPI

`FastAPI` should be the single backend entry point.

It can be used for:

- request handling;
- HTML page rendering;
- `htmx` partial responses;
- JSON endpoints if needed later.

This keeps the system simple and avoids splitting the app too early.

## Core Responsibilities

The backend should:

- accept pasted text and uploaded PDFs;
- normalize extracted text;
- load the saved model artifacts;
- compute top matching funding calls;
- generate explanations for each recommendation;
- persist a mock audit trail;
- return the results to the frontend.

## Suggested Internal Modules

- `api`
- `services`
- `models`
- `explainability`
- `traceability`
- `data_access`

The exact module names can change, but the separation should remain clear.

## Persistence

For v1, the backend should keep persistence simple:

- funding dataset in CSV or similar flat files;
- trained model artifacts saved by the notebook;
- audit trail stored as JSON files.

There is no need for a database at the beginning unless it becomes clearly useful.

## Engineering Principle

The backend should optimize for clarity and determinism.
This is an academic POC, so readable logic is more valuable than clever abstractions.
