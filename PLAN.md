# PLAN

## Project Overview

This project is a proof of concept for an AI system that helps municipalities and consultants identify relevant EU funding opportunities more quickly and with greater transparency.

The core idea is:

- a user provides a municipal need, project brief, or planning document;
- the system analyzes the content;
- the system recommends the most relevant EU funding calls;
- the system explains why those calls were selected;
- the system stores a traceability record for accountability.

Its purpose is to support users with:

- better opportunity discovery;
- clearer reasoning behind recommendations;
- a trustworthy and auditable recommendation flow.

## Business Goal

Municipalities often face a fragmented funding landscape, complex documentation, and limited visibility into which calls are relevant for their local priorities.

This project aims to reduce that friction by offering a decision-support tool that:

- improves discovery of funding opportunities;
- reduces information gaps;
- supports more transparent recommendations;
- strengthens trust through explainability and auditability.

## Target Users

The main users are:

- municipal officers looking for relevant EU funding calls;
- consultants supporting municipalities in funding strategy and proposal preparation.

## Scope Of The POC

The first version will:

- support `English-only` input;
- accept `PDF upload` and `pasted text`;
- focus on `3-4 municipality-relevant funding categories`;
- return ranked funding recommendations;
- provide a short explanation for each recommendation;
- show a mock audit trail inspired by the blockchain idea from the brief.

## Supporting Documents

Detailed planning is split into separate files:

- [Model Strategy](./docs/MODEL.md)
- [Backend Plan](./docs/BACKEND.md)
- [Frontend Plan](./docs/FRONTEND.md)
