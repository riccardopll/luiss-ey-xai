# Frontend

## Goal

The frontend should be lightweight, credible, and easy to demo.

It should present the product as a decision-support tool for municipalities and consultants, not as a complicated technical dashboard.

## Recommended Stack

- `HTML`
- `Tailwind CSS`
- server-rendered HTML fragments from the backend

## Build Strategy

The simplest solution is:

- use `FastAPI` to serve pages and HTML partials;
- use `Tailwind` for styling.

## Product Experience

The interface should support one clear flow:

1. the user uploads a PDF or pastes text;
2. the system returns ranked funding calls;
3. the user inspects explanations;
4. the user sees the audit trail.

## Main UI Areas

### Input area

- PDF upload
- text paste field
- submit action

### Results area

- ranked funding calls
- score
- short recommendation summary

### Explainability area

- overlapping keywords
- matched passage
- category fit
- explanation of why the call was selected

### Audit area

- record ID
- timestamp
- input hash
- model version
- output hash

## UI Principle

The frontend should feel polished, but the implementation should stay simple.

The goal is strong demo quality with minimal moving parts.
