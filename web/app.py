import hashlib
import html
import json
import pickle
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import stopwordsiso as stopwords
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import minmax_scale

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_PATH = DATA_DIR / "processed" / "eu_projects_v1.parquet"
BM25_INDEX_PATH = DATA_DIR / "indexes" / "bm25s_v1.pkl"
DENSE_INDEX_PATH = DATA_DIR / "indexes" / "multilingual_e5_v1.pkl"
AUDIT_DIR = DATA_DIR / "audit"
REQUIRED_ARTIFACTS = [PROCESSED_PATH, BM25_INDEX_PATH, DENSE_INDEX_PATH]

DATASET_VERSION = "v1"
MODEL_VERSION = "hybrid_bm25_dense_geo_v1"
LOCAL_LLM_MODEL_VERSION = "gemma4:e4b"
PROMPT_TEMPLATE_VERSION = "xai_v1"
EXPLANATION_PROMPT_TEMPLATE = """
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
""".strip()
LOCATION_PIN_DATA_URL = (
    "data:image/svg+xml,%3Csvg viewBox='0 0 24 24' fill='none' "
    "xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M12 21C15.5 17.4 "
    "19 14.1764 19 10.2C19 6.22355 15.866 3 12 3C8.13401 3 5 6.22355 "
    "5 10.2C5 14.1764 8.5 17.4 12 21Z' stroke='%23087d82' stroke-width='2' "
    "stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M12 13"
    "C13.6569 13 15 11.6569 15 10C15 8.34315 13.6569 7 12 7C10.3431 7 "
    "9 8.34315 9 10C9 11.6569 10.3431 13 12 13Z' stroke='%23087d82' "
    "stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E"
)

DEFAULT_QUERY = (
    "Renovate public school buildings in France with better insulation, smart energy "
    "monitoring, and renewable heating systems."
)
DEFAULT_LOCATION = "France"
DEFAULT_TOP_K = 10
RESULT_COLUMNS = [
    "rank",
    "confidence",
    "semantic_score",
    "keyword_score",
    "geographic_score",
    "Operation_Unique_Identifier",
    "Programme_Name",
    "Fund_Name",
    "Category_Label",
    "Specific_Objective_Label",
    "Policy_Objective_Label",
    "Total_Eligible_Expenditure_amount",
    "Project_EU_Budget",
    "Country",
    "CountryCode",
    "NUTS3_Label",
    "LAU_Labels",
    "search_text",
]
POSITIONING_SUGGESTION_FIELDS = {
    "Programme_Name": "programme_name",
    "Fund_Name": "fund_name",
    "Category_Label": "category_label",
    "Specific_Objective_Label": "specific_objective_label",
    "Policy_Objective_Label": "policy_objective_label",
}
BUDGET_SUGGESTION_FIELDS = {
    "Total_Eligible_Expenditure_amount": "total_eligible_expenditure_amount",
    "Project_EU_Budget": "project_eu_budget",
}
MATCH_METADATA_FIELDS = POSITIONING_SUGGESTION_FIELDS | BUDGET_SUGGESTION_FIELDS
EU_LANGUAGE_CODES = [
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hr",
    "hu",
    "it",
    "lt",
    "lv",
    "mt",
    "nl",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sv",
]


st.set_page_config(
    page_title="Funding Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def clean_text(value):
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def safe(value):
    return html.escape(clean_text(value))


def json_ready_value(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        text = clean_text(value)
        return text or None
    return value


def icon_svg(name):
    icons = {
        "compass": """
            <svg viewBox="0 0 24 24" fill="none" role="img" aria-hidden="true">
                <path d="M14.1214 14.1213L16.5165 8.13347C16.6798 7.72532
                    16.2747 7.32028 15.8666 7.48354L9.87872 9.87868M14.1214
                    14.1213L8.13351 16.5165C7.72536 16.6797 7.32032 16.2747
                    7.48358 15.8665L9.87872 9.87868M14.1214 14.1213L9.87872
                    9.87868"
                    stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"
                    stroke-width="1.45" />
                <circle cx="12" cy="12" r="9" stroke="currentColor"
                    stroke-linecap="round" stroke-linejoin="round" stroke-width="1.45" />
            </svg>
        """,
        "search": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="10.5" cy="10.5" r="6.2" fill="none" stroke="currentColor"
                    stroke-width="2.2" />
                <path d="m15.2 15.2 5 5" fill="none" stroke="currentColor"
                    stroke-linecap="round" stroke-width="2.2" />
            </svg>
        """,
        "pin": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M12 21s6-5.7 6-11a6 6 0 0 0-12 0c0 5.3 6 11 6 11Z"
                    fill="none" stroke="currentColor" stroke-width="2.1" />
                <circle cx="12" cy="10" r="2.1" fill="none" stroke="currentColor"
                    stroke-width="2.1" />
            </svg>
        """,
        "sliders": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M4 7h9M17 7h3M4 17h3M11 17h9M13 5v4M9 15v4"
                    fill="none" stroke="currentColor" stroke-linecap="round"
                    stroke-width="2.1" />
            </svg>
        """,
        "brain": """
            <svg viewBox="0 0 512 512" aria-hidden="true">
                <g fill="currentColor">
                    <path
                    d="M410.34,153.043c-1.278-1.293-2.712-2.477-4.288-3.527l-1.615-0.337
                    c-0.846-0.368-1.52-0.972-1.96-1.702
                    c-2.524-1.191-5.166-1.959-7.854-2.343c-2.321-0.322-3.951-2.477-3.614-4.814c0.313-2.336,2.484-3.966,4.805-3.645
                    c2.658,0.392,5.322,1.067,7.878,2.07c0.415-1.866,0.65-3.771,0.65-5.691c-0.008-4.202-0.995-8.443-3.096-12.433
                    c-2.399-4.562-5.918-8.121-10.034-10.55c-3.606-2.132-7.682-3.402-11.868-3.7c-0.305,3.817-1.214,7.619-2.798,11.303
                    c-0.941,2.163-3.441,3.159-5.612,2.226c-2.164-0.925-3.167-3.433-2.234-5.597c1.52-3.55,2.234-7.203,2.25-10.825l-0.055-1.544
                    c-0.29-5.017-1.952-9.861-4.782-13.992c-2.83-4.124-6.796-7.502-11.734-9.642c-3.543-1.513-7.212-2.226-10.826-2.226
                    c-6.922,0-13.624,2.666-18.687,7.368c0.948,3.222,1.489,6.616,1.489,10.144c0,2.352-1.928,4.264-4.28,4.264
                    s-4.272-1.912-4.272-4.264c0-3.457-0.635-6.734-1.795-9.783h0.008c-1.968-5.134-5.448-9.555-9.908-12.675
                    c-4.445-3.119-9.83-4.946-15.693-4.946c-5.087,0-9.83,1.388-13.906,3.786c-3.261,1.928-6.075,4.531-8.285,7.588
                    c4.319,5.918,6.874,13.232,6.874,21.118c0,2.351-1.92,4.256-4.28,4.256c-2.352,0-4.272-1.905-4.272-4.256
                    c0-6.929-2.548-13.232-6.788-18.053c-5.032-5.746-12.377-9.343-20.608-9.343c-5.714,0-10.974,1.732-15.372,4.711
                    c-4.382,2.979-7.87,7.196-9.932,12.15c-1.332,3.237-2.093,6.804-2.093,10.535c0,2.351-1.905,4.256-4.264,4.256
                    c-2.36,0-4.264-1.905-4.264-4.256c0-3.465,0.502-6.804,1.411-9.987c-3.214-1.301-6.694-2.022-10.268-2.022
                    c-2.03,0-4.068,0.235-6.138,0.698c-6.083,1.403-11.17,4.68-14.894,9.116c-2.681,3.199-4.594,7-5.589,11.084
                    c8.819,0.156,17.629,3.566,24.434,10.229c1.678,1.654,1.709,4.351,0.063,6.028c-1.654,1.693-4.351,1.709-6.036,0.07
                    c-5.346-5.212-12.236-7.823-19.166-7.831c-1.238,0-2.477,0.086-3.716,0.243h-0.015h-0.008
                    c-5.808,0.807-11.405,3.456-15.85,7.988c-5.221,5.33-7.808,12.213-7.815,19.158c0,3.959,0.862,7.948,2.586,11.617
                    c1.34,2.892,3.222,5.589,5.652,7.956v0.007c2.744,2.697,5.911,4.688,9.29,5.989c2.186,0.855,3.292,3.324,2.43,5.518
                    c-0.854,2.211-3.316,3.309-5.518,2.446c-4.413-1.709-8.583-4.326-12.19-7.862c-1.575-1.552-2.979-3.206-4.224-4.961
                    c-3.222,4.617-4.97,10.142-4.97,15.771c0,4.359,1.043,8.756,3.245,12.887c2.477,4.625,6.06,8.231,10.277,10.692
                    c3.331,1.952,7.055,3.183,10.896,3.614l-0.008-0.259c0-2.258,0.212-4.468,0.596-6.616c0.439-2.32,2.673-3.834,4.978-3.418
                    c2.32,0.431,3.849,2.657,3.433,4.977c-0.306,1.638-0.47,3.339-0.47,5.056l0.07,1.984l0.227,1.999v0.023l0.008,0.039
                    c1.96,13.389,13.53,23.398,27.075,23.398l1.968-0.078h0.008c6.004-0.431,11.397-2.752,15.677-6.365
                    c3.002-2.532,5.463-5.706,7.149-9.282c0.995-2.14,3.558-3.048,5.683-2.038c2.132,1.004,3.042,3.543,2.038,5.676
                    c-1.826,3.849-4.319,7.321-7.313,10.276c1.677,5.158,4.828,9.587,8.944,12.879c4.704,3.771,10.637,5.997,17.049,5.997
                    l1.976-0.078h0.008c4.06-0.29,7.815-1.458,11.147-3.277c-0.518-2.453-0.815-4.961-0.815-7.501
                    c0-5.237,1.168-10.575,3.582-15.592c1.027-2.124,3.574-3.01,5.691-1.983c2.124,1.011,3.026,3.567,1.999,5.691
                    c-1.858,3.849-2.728,7.894-2.728,11.884c0,2.79,0.431,5.573,1.262,8.214h0.008c2.187,6.969,7.126,13.06,14.235,16.477
                    c3.841,1.858,7.886,2.728,11.868,2.728c6.326,0,12.511-2.211,17.418-6.216l0.016-0.008c2.979-2.454,5.472-5.558,7.266-9.266
                    l0.008-0.016c1.67-3.464,2.548-7.093,2.696-10.715c0.11-2.352,2.093-4.186,4.46-4.076c2.352,0.102,4.17,2.092,4.068,4.444
                    c-0.204,4.75-1.356,9.547-3.543,14.07v-0.008c-1.756,3.661-4.068,6.883-6.78,9.634c5.032,7.227,13.404,11.782,22.458,11.774
                    c1.677,0,3.386-0.156,5.103-0.486c6.592-1.238,12.15-4.727,16.101-9.524c3.943-4.806,6.24-10.912,6.24-17.356l-0.008-0.579
                    v-0.032c-0.031-1.45-0.18-2.97-0.462-4.5c-0.682-3.629-2.07-6.922-3.974-9.83c-2.759-2.43-5.824-5.604-8.716-9.132
                    c-3.081-3.762-5.918-7.878-7.823-12.024c-1.254-2.775-2.148-5.581-2.164-8.536c0-1.309,0.18-2.665,0.674-3.966
                    c0.478-1.325,1.262-2.579,2.312-3.614c1.662-1.654,4.358-1.654,6.028,0c1.662,1.677,1.662,4.389,0,6.044l-0.322,0.51
                    l-0.164,1.027c-0.016,1.074,0.431,2.893,1.395,4.985c0.956,2.086,2.375,4.429,4.045,6.741
                    c3.308,4.633,7.596,9.18,10.825,11.954l0.439,0.377l0.322,0.478c2.642,3.935,4.578,8.466,5.51,13.412v0.008l0.306,2.116
                    l1.278,0.055c5.409-0.008,10.856-1.607,15.646-4.97c3.81-2.681,6.71-6.122,8.67-9.987c1.967-3.872,2.994-8.152,2.994-12.463
                    c0-4.037-0.91-8.09-2.728-11.86c-1.686,0.541-3.395,0.972-5.127,1.27c-2.32,0.415-4.523-1.161-4.938-3.48
                    c-0.393-2.328,1.176-4.523,3.496-4.931c7.556-1.301,14.478-5.706,18.758-12.777c2.712-4.444,3.99-9.336,3.99-14.172
                    c0-7.126-2.752-14.094-7.878-19.26L410.34,153.043z
                    M214.788,120.481c4.076-4.821,9.485-8.247,15.301-10.489
                    c5.84-2.242,12.095-3.316,18.1-3.316c5.84,0.008,11.452,0.996,16.218,3.207c2.132,0.971,3.065,3.519,2.077,5.659
                    c-0.98,2.133-3.512,3.073-5.667,2.086c-3.292-1.544-7.8-2.43-12.628-2.43c-4.97,0-10.268,0.916-15.027,2.767
                    c-4.758,1.826-8.936,4.554-11.876,8.019c-1.505,1.803-4.209,2.038-6.012,0.502C213.479,124.965,213.252,122.276,214.788,120.481z
                    M311.925,221.396c-4.962,0-9.462-0.564-13.569-1.512c0.11,1.175,0.188,2.484,0.188,3.872c0,2.783-0.306,5.964-1.176,9.304
                    c-0.878,3.331-2.359,6.835-4.695,10.198c-1.34,1.944-3.998,2.422-5.934,1.081c-1.944-1.341-2.414-3.99-1.09-5.942
                    c1.724-2.485,2.798-5.025,3.457-7.525c0.659-2.5,0.902-4.931,0.902-7.117c0.008-3.018-0.455-5.503-0.784-6.922
                    c-2.085-0.941-4.006-1.999-5.8-3.183c-7.149-4.625-11.93-10.731-14.933-16.54c-2.014-3.896-3.238-7.658-3.896-10.927
                    c-0.376-1.866-0.564-3.558-0.564-5.096c0-2.367,1.913-4.28,4.272-4.28c2.359,0,4.272,1.913,4.272,4.28
                    c-0.008,1.238,0.259,3.316,0.933,5.676c0.674,2.367,1.748,5.056,3.3,7.76c3.112,5.44,8.058,10.888,15.693,14.424
                    c5.103,2.367,11.452,3.903,19.424,3.903c2.359,0,4.272,1.913,4.272,4.272C316.197,219.484,314.285,221.396,311.925,221.396z
                    M341.979,190.229c-0.878,2.18-3.363,3.246-5.558,2.375c-2.187-0.878-3.237-3.347-2.375-5.534v-0.008l0.024-0.063l0.126-0.376
                    l0.439-1.536c0.322-1.348,0.651-3.253,0.651-5.346l-0.008-0.298c-5.863,1.991-11.303,2.884-16.312,2.884
                    c-7.572,0.016-14.125-2.061-19.456-5.212c-5.338-3.15-9.492-7.321-12.597-11.562c-3.527-4.805-8.152-9.116-13.451-12.165
                    c-5.314-3.058-11.256-4.876-17.566-4.892c-5.526,0.016-11.405,1.395-17.543,4.766c-5.824,3.198-9.571,7.454-11.993,12.126
                    c-2.406,4.68-3.425,9.79-3.418,14.447c-0.016,4.311,0.894,8.214,2.077,10.676c1.011,2.132,0.094,4.672-2.03,5.683
                    c-2.124,1.012-4.672,0.094-5.683-2.03c-1.858-3.936-2.892-8.89-2.908-14.329c0.008-5.409,1.082-11.32,3.724-16.971
                    c-5.298,1.2-10.221,1.733-14.705,1.733c-2.422,0-4.704-0.157-6.875-0.439l-0.792,1.599l-0.008,0.015l-0.063,0.064l-0.227,0.29
                    l-0.91,1.238c-0.776,1.114-1.85,2.736-3.018,4.805c-2.328,4.1-4.993,9.885-6.287,16.382c-0.462,2.313-2.704,3.826-5.017,3.355
                    c-2.312-0.462-3.818-2.704-3.339-5.032c1.536-7.706,4.571-14.259,7.212-18.923c1.379-2.414,2.642-4.326,3.566-5.628
                    c-3.48-1.042-6.49-2.32-8.951-3.622c-5.62-2.987-8.545-5.981-8.795-6.224c-1.631-1.701-1.584-4.398,0.109-6.036
                    c1.701-1.63,4.39-1.583,6.028,0.094l0.048,0.047l0.25,0.243l1.223,1.026c1.112,0.878,2.83,2.086,5.126,3.308
                    c4.609,2.43,11.492,4.892,20.718,4.908c6.482,0,14.125-1.238,23.006-4.656c2.25-2.093,4.844-4.005,7.823-5.636
                    c7.306-4.005,14.666-5.808,21.65-5.808c7.98,0,15.403,2.321,21.831,6.02c6.428,3.716,11.876,8.803,16.069,14.525
                    c2.548,3.456,5.895,6.804,10.042,9.241c4.17,2.446,9.1,4.038,15.128,4.045c4.131,0,8.787-0.768,14.055-2.618
                    c-1.285-2.062-3.198-4.053-6.349-5.856c-2.054-1.168-2.783-3.77-1.623-5.816c1.168-2.062,3.763-2.768,5.816-1.623
                    c5.048,2.846,8.427,6.616,10.386,10.559c1.983,3.927,2.571,7.909,2.571,11.296C343.797,185.652,342.096,189.924,341.979,190.229z
                    M340.38,137.938c-9.971,0-18.695-2.689-25.75-6.757c-0.995,1.38-2.328,3.112-4.045,4.993
                    c-3.864,4.241-9.54,9.219-17.135,11.946c-2.234,0.784-4.664-0.36-5.456-2.586c-0.792-2.227,0.376-4.664,2.579-5.448
                    c5.605-1.999,10.292-5.926,13.538-9.477c1.505-1.623,2.673-3.159,3.52-4.335c-4.21-3.558-7.533-7.611-9.814-11.781
                    c-2.163-3.998-3.418-8.121-3.433-12.127c0-2.351,1.913-4.264,4.272-4.264c2.352,0,4.264,1.913,4.264,4.264
                    c-0.016,2.156,0.737,5.033,2.398,8.051c1.638,3.025,4.131,6.176,7.392,8.999c6.522,5.652,15.936,9.978,27.671,9.978
                    c2.359,0,4.264,1.921,4.264,4.28C344.644,136.025,342.739,137.938,340.38,137.938z
                    M374.023,198.037
                    c-1.497,1.803-4.194,2.069-6.012,0.556c-1.818-1.497-2.069-4.193-0.58-6.004c4.656-5.652,6.373-11.343,6.388-16.79
                    c0.016-4.962-1.505-9.752-3.888-13.788c-2.352-4.03-5.604-7.259-8.732-9.054c-2.101-1.207-4.076-1.748-5.644-1.748
                    c-2.36,0-4.264-1.904-4.264-4.264c0-2.359,1.905-4.272,4.264-4.272c2.594,0,5.134,0.635,7.549,1.701
                    c2.406,1.051,4.68,2.548,6.796,4.375c4.225,3.652,7.815,8.622,10.073,14.447c1.489,3.887,2.398,8.152,2.398,12.604
                    C382.379,183.12,379.879,190.958,374.023,198.037z" />
                    <path d="M172.459,411.86l0.291,0.259c-0.008-0.008-0.008-0.008-0.016-0.016
                    L172.459,411.86z" />
                    <path
                    d="M428.699,66.574C396.943,29.442,349.183,4.155,287.986,0.462
                    C283.008,0.156,278.054,0,273.147,0
                    c-51.469,0-98.815,16.712-133.524,47.393c-34.733,30.602-56.36,75.408-56.282,129.072c0,1.215,0.008,2.43,0.031,3.653
                    l-0.054,6.474l-44.179,78.018c-2.375,4.209-3.575,8.905-3.575,13.601c0,3.918,0.832,7.854,2.516,11.522
                    c3.567,7.784,10.582,13.381,18.891,15.309l-0.016,0.07l20.318,5.597l8.027,77.423l0.063-0.008
                    c0.565,11.28,5.597,21.87,14.142,29.301c7.596,6.616,17.292,10.206,27.263,10.206c1.01,0,2.006-0.094,3.01-0.164
                    l23.022,1.826l1.153-0.125l0.368-0.016c0.885,0,1.709,0.314,2.344,0.886l0.038,0.04c0.73,0.658,1.161,1.622,1.161,2.61V512
                    h214.96v-12.04c0,0,0-32.616,0-45.628c-0.024-5.237,1.019-15.591,3.073-25.578c1.019-5.008,2.28-10.002,3.708-14.32
                    c1.411-4.311,3.042-7.98,4.405-10.12c14.298-23.046,34.514-41.952,53.028-68.69
                    c18.492-26.714,34.255-61.369,38.574-113.951c0.549-6.592,0.824-13.185,0.824-19.745
                    C476.436,152.079,460.508,103.674,428.699,66.574z
                    M451.611,219.664c-4.178,48.875-17.849,78.12-34.405,102.303
                    c-16.531,24.159-36.974,43.152-53.616,69.577c-2.9,4.672-5.025,9.846-6.851,15.348
                    c-2.712,8.239-4.648,17.222-5.973,25.586c-1.317,8.387-2.014,15.991-2.022,21.854c0,8.356,0,20.992,0,33.588H181.944
                    v-55.231c0-7.847-3.339-15.317-9.188-20.56c-5.103-4.586-11.687-7.063-18.436-7.063c-0.494,0-0.972,0.054-1.458,0.086
                    l-22.998-1.834l-1.129,0.125l-1.967,0.11c-4.186,0-8.262-1.513-11.453-4.288c-3.684-3.206-5.864-7.815-6.005-12.714
                    l-0.007-0.454l-9.767-94.221l-36.607-10.096l-0.439-0.086c-1.113-0.22-2.045-0.957-2.524-1.992l-0.322-1.489l0.455-1.74
                    l47.251-83.443l0.102-13.036v-0.173c-0.023-1.128-0.031-2.242-0.031-3.354
                    c0.079-47.251,18.444-84.658,48.153-111.035c29.74-26.315,71.27-41.35,117.573-41.35c4.413,0,8.866,0.141,13.35,0.416
                    l0.744-12.017l-0.737,12.017c55.443,3.488,96.252,25.609,123.914,57.755c27.616,32.17,41.938,74.898,41.938,119.674
                    C452.356,207.812,452.113,213.73,451.611,219.664z" />
                </g>
            </svg>
        """,
        "trophy": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M8 4h8v4.8a4 4 0 0 1-8 0V4Z" fill="currentColor"
                    opacity="0.18" />
                <path d="M8 4h8v4.8a4 4 0 0 1-8 0V4ZM8 6H4v1.5
                    A3.5 3.5 0 0 0 7.5 11M16 6h4v1.5a3.5 3.5 0 0 1-3.5
                    3.5M12 13v4M8.5 20h7"
                    fill="none" stroke="currentColor" stroke-linecap="round"
                    stroke-linejoin="round" stroke-width="2" />
            </svg>
        """,
        "bulb": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M9 18h6M10 21h4M8 14.5a6 6 0 1 1 8 0
                    c-.9.8-1.3 1.6-1.4 2.5H9.4c-.1-.9-.5-1.7-1.4-2.5Z"
                    fill="none" stroke="currentColor" stroke-linecap="round"
                    stroke-linejoin="round" stroke-width="2" />
            </svg>
        """,
        "shield": """
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M9 12L11 14L15 9.99999M20 12C20 16.4611 14.54
                    19.6937 12.6414 20.683C12.4361 20.79 12.3334 20.8435
                    12.191 20.8712C12.08 20.8928 11.92 20.8928 11.809
                    20.8712C11.6666 20.8435 11.5639 20.79 11.3586 20.683C9.45996
                    19.6937 4 16.4611 4 12V8.21759C4 7.41808 4 7.01833
                    4.13076 6.6747C4.24627 6.37113 4.43398 6.10027
                    4.67766 5.88552C4.9535 5.64243 5.3278 5.50207 6.0764
                    5.22134L11.4382 3.21067C11.6461 3.13271 11.75 3.09373
                    11.857 3.07827C11.9518 3.06457 12.0482 3.06457 12.143
                    3.07827C12.25 3.09373 12.3539 3.13271 12.5618 3.21067L17.9236
                    5.22134C18.6722 5.50207 19.0465 5.64243 19.3223
                    5.88552C19.566 6.10027 19.7537 6.37113 19.8692 6.6747C20
                    7.01833 20 7.41808 20 8.21759V12Z"
                    stroke="currentColor" stroke-linecap="round"
                    stroke-linejoin="round" stroke-width="2" />
            </svg>
        """,
        "database": """
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M4 18V6" stroke="currentColor" stroke-linecap="round"
                    stroke-width="1.5" />
                <path d="M20 6V18" stroke="currentColor" stroke-linecap="round"
                    stroke-width="1.5" />
                <path d="M12 10C16.4183 10 20 8.20914 20 6C20 3.79086
                    16.4183 2 12 2C7.58172 2 4 3.79086 4 6C4 8.20914
                    7.58172 10 12 10Z"
                    stroke="currentColor" stroke-width="1.5" />
                <path d="M20 12C20 14.2091 16.4183 16 12 16C7.58172 16 4
                    14.2091 4 12"
                    stroke="currentColor" stroke-width="1.5" />
                <path d="M20 18C20 20.2091 16.4183 22 12 22C7.58172 22 4
                    20.2091 4 18"
                    stroke="currentColor" stroke-width="1.5" />
            </svg>
        """,
        "file": """
            <svg viewBox="0 0 24 24" aria-hidden="true">
                <path fill="currentColor" fill-rule="evenodd" clip-rule="evenodd"
                    d="M9.29289 1.29289C9.48043 1.10536 9.73478 1 10 1H18C19.6569
                    1 21 2.34315 21 4V20C21 21.6569 19.6569 23 18 23H6C4.34315 23
                    3 21.6569 3 20V8C3 7.73478 3.10536 7.48043 3.29289
                    7.29289L9.29289 1.29289ZM18 3H11V8C11 8.55228 10.5523 9 10
                    9H5V20C5 20.5523 5.44772 21 6 21H18C18.5523 21 19 20.5523
                    19 20V4C19 3.44772 18.5523 3 18 3ZM6.41421 7H9V4.41421L6.41421
                    7ZM7 13C7 12.4477 7.44772 12 8 12H16C16.5523 12 17 12.4477
                    17 13C17 13.5523 16.5523 14 16 14H8C7.44772 14 7 13.5523 7
                    13ZM7 17C7 16.4477 7.44772 16 8 16H16C16.5523 16 17 16.4477
                    17 17C17 17.5523 16.5523 18 16 18H8C7.44772 18 7 17.5523 7 17Z" />
            </svg>
        """,
    }
    return re.sub(r"\s+", " ", icons[name]).strip()


def stable_hash(value):
    payload = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False)
def missing_artifacts():
    return [path for path in REQUIRED_ARTIFACTS if not path.exists()]


@st.cache_data(show_spinner=False)
def load_projects():
    return pd.read_parquet(PROCESSED_PATH)


@st.cache_resource(show_spinner=False)
def load_bm25_index():
    with open(BM25_INDEX_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_resource(show_spinner=False)
def load_dense_index():
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError:
        st.error("Missing sentence-transformers. Install the project requirements first.")
        st.stop()

    with open(DENSE_INDEX_PATH, "rb") as file:
        dense_index = pickle.load(file)
    dense_index["model"] = SentenceTransformer(dense_index["model_name"])
    return dense_index


@st.cache_data(show_spinner=False)
def stop_words():
    words = set()
    for language_code in EU_LANGUAGE_CODES:
        words.update(stopwords.stopwords(language_code))
    return words


def tokenize(text):
    words = stop_words()
    tokens = re.findall(r"[\wÀ-ÿ]+", clean_text(text).lower())
    return [token for token in tokens if token not in words and len(token) > 1]


def scale(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    return np.nan_to_num(minmax_scale(values))


def detect_country_code(location_text, projects):
    query = clean_text(location_text).lower()
    if not query or projects.empty:
        return None
    if re.fullmatch(r"[a-z]{2}", query):
        return query.upper()

    countries = (
        projects[["Country", "CountryCode"]]
        .dropna()
        .drop_duplicates()
        .sort_values("Country", key=lambda column: column.str.len(), ascending=False)
    )
    for _, row in countries.iterrows():
        country = clean_text(row["Country"]).lower()
        country_code = clean_text(row["CountryCode"]).upper()
        if country and re.search(rf"\b{re.escape(country)}\b", query):
            return country_code
    return None


def keyword_scores(query, projects):
    tokens = tokenize(query)
    if not tokens or projects.empty:
        return np.zeros(len(projects))

    index = load_bm25_index()
    if index is not None:
        return index.get_scores(tokens).astype(float, copy=False)

    token_pattern = "|".join(re.escape(token) for token in tokens)
    return projects["search_text"].str.lower().str.count(token_pattern).to_numpy(dtype=float)


def dense_scores(query, projects):
    if projects.empty:
        return None
    dense_index = load_dense_index()
    if dense_index is None:
        return None
    model = dense_index["model"]
    query_embedding = model.encode(
        [f"query: {clean_text(query)}"],
        normalize_embeddings=True,
    )
    return linear_kernel(dense_index["embeddings"], query_embedding).ravel()


def geography_scores(country_code, projects):
    if not country_code or projects.empty:
        return None
    return (projects["CountryCode"].fillna("").str.upper() == country_code).astype(float).to_numpy()


def confidence_scores(keyword_score, semantic_score, geographic_score):
    keyword_score = scale(keyword_score)
    semantic_score = scale(semantic_score) if semantic_score is not None else None

    if geographic_score is not None:
        geographic_score = scale(geographic_score)
    else:
        geographic_score = np.zeros_like(keyword_score)

    if semantic_score is None:
        return (keyword_score * 0.75) + (geographic_score * 0.25)

    return (semantic_score * 0.50) + (keyword_score * 0.35) + (geographic_score * 0.15)


def retrieve_projects(query, location_text, top_k):
    projects = load_projects()
    if projects.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS), None

    country_code = detect_country_code(location_text, projects)
    raw_keyword_score = keyword_scores(query, projects)
    raw_semantic_score = dense_scores(query, projects)
    raw_geographic_score = geography_scores(country_code, projects)
    confidence = confidence_scores(raw_keyword_score, raw_semantic_score, raw_geographic_score)

    ranked_indices = np.argsort(confidence)[::-1][:top_k]
    results = projects.iloc[ranked_indices].copy()
    results["rank"] = range(1, len(results) + 1)
    results["confidence"] = confidence[ranked_indices]
    results["keyword_score"] = scale(raw_keyword_score)[ranked_indices]
    results["semantic_score"] = (
        scale(raw_semantic_score)[ranked_indices] if raw_semantic_score is not None else np.nan
    )
    results["geographic_score"] = (
        scale(raw_geographic_score)[ranked_indices] if raw_geographic_score is not None else np.nan
    )

    available_columns = [column for column in RESULT_COLUMNS if column in results.columns]
    return results[available_columns], country_code


def split_title_summary(text):
    search_text = clean_text(text)
    parts = search_text.split(". ", 1)
    title = parts[0]
    summary = parts[1] if len(parts) > 1 else search_text
    return title, summary


def format_money(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"EUR {float(value):,.0f}"


def short_fund(value):
    text = clean_text(value)
    replacements = {
        "European Regional Development Fund": "ERDF",
        "European Social Fund Plus": "ESF+",
        "Just Transition Fund": "JTF",
        "Cohesion Fund": "CF",
    }
    return replacements.get(text, text or "n/a")


def short_hash(value):
    text = clean_text(value)
    return text[:8] if text else "n/a"


def summarize_suggestions(results):
    if results.empty:
        return {}

    suggestions = {}
    for source_name, output_name in [
        ("Programme_Name", "programme_name"),
        ("Fund_Name", "fund_name"),
        ("Category_Label", "category_label"),
        ("Specific_Objective_Label", "specific_objective_label"),
        ("Policy_Objective_Label", "policy_objective_label"),
    ]:
        values = results[source_name].dropna().map(clean_text)
        values = values[values.str.len() > 0]
        suggestions[output_name] = [
            {"value": value, "match_count": int(count)}
            for value, count in values.value_counts().head(3).items()
        ]

    budget_values = pd.to_numeric(
        results["Total_Eligible_Expenditure_amount"],
        errors="coerce",
    ).dropna()
    suggestions["budget_median"] = float(budget_values.median()) if len(budget_values) else None
    return suggestions


def overlap_terms(query, text, limit=5):
    query_tokens = set(tokenize(query))
    text_tokens = tokenize(text)
    seen = []
    for token in text_tokens:
        if token in query_tokens and token not in seen:
            seen.append(token)
        if len(seen) == limit:
            break
    return seen


def explain_match(query, location_text, row):
    terms = overlap_terms(query, row.get("search_text", ""))
    title, _ = split_title_summary(row.get("search_text", ""))
    country = clean_text(row.get("Country"))
    fund = clean_text(row.get("Fund_Name"))
    programme = clean_text(row.get("Programme_Name"))
    rank = int(row.get("rank", 0) or 0)
    rank_phrase = "ranked first" if rank == 1 else f"ranked #{rank}"

    reasons = []
    if terms:
        reasons.append(f"shared terms around {', '.join(terms)}")
    if (
        country
        and clean_text(location_text)
        and country.lower() in clean_text(location_text).lower()
    ):
        reasons.append(f"a country match for {country}")
    if fund:
        reasons.append(f"the {fund} funding context")
    if not reasons:
        reasons.append("the closest available text match in the corpus")

    return (
        f"{title} {rank_phrase} because it has {', '.join(reasons)}. "
        f"The programme context is {programme or 'not available'}, and the score combines "
        "semantic similarity, keyword evidence, and geography when location is available."
    )


def score_breakdown(row):
    components = [
        ("Semantic", row.get("semantic_score"), 0.50),
        ("Keyword", row.get("keyword_score"), 0.35),
        ("Geography", row.get("geographic_score"), 0.15),
    ]
    breakdown = []
    for label, value, weight in components:
        numeric_value = json_ready_value(value)
        score = float(numeric_value) if isinstance(numeric_value, int | float) else 0.0
        breakdown.append(
            {
                "label": label,
                "points": score * weight * 1000,
                "width": min(max(score, 0), 1) * 100,
            }
        )
    return breakdown


def build_llm_input(input_text, location_text, result_row):
    search_text = clean_text(result_row.get("search_text", ""))
    title, summary = split_title_summary(search_text)

    return {
        "user_project": {
            "text": clean_text(input_text),
            "location": json_ready_value(location_text),
        },
        "matched_project": {
            "project_id": json_ready_value(result_row.get("Operation_Unique_Identifier")),
            "rank": int(result_row["rank"]),
            "title": title,
            "summary": summary,
            "country": json_ready_value(result_row.get("Country")),
            "nuts3_label": json_ready_value(result_row.get("NUTS3_Label")),
            "lau_labels": json_ready_value(result_row.get("LAU_Labels")),
        },
        "matched_project_metadata": {
            output_name: json_ready_value(result_row.get(source_name))
            for source_name, output_name in MATCH_METADATA_FIELDS.items()
        },
        "scores": {
            "semantic_score": json_ready_value(result_row.get("semantic_score")),
            "keyword_score": json_ready_value(result_row.get("keyword_score")),
            "geographic_score": json_ready_value(result_row.get("geographic_score")),
            "confidence_score": json_ready_value(result_row.get("confidence")),
        },
    }


def build_explanation_prompt(llm_input):
    llm_input_json = json.dumps(llm_input, indent=2, ensure_ascii=False)
    return EXPLANATION_PROMPT_TEMPLATE.replace("{{llm_input_json}}", llm_input_json)


def latest_audit_block():
    blocks = sorted(AUDIT_DIR.glob("*.json"))
    if not blocks:
        return None
    with open(blocks[-1], encoding="utf-8") as file:
        block = json.load(file)
    block["path"] = str(blocks[-1])
    return block


def build_run_record(query, location_text, country_code, results, suggestions):
    result_ids = results["Operation_Unique_Identifier"].fillna("").tolist()
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "input_text_hash": stable_hash(clean_text(query)),
        "location_metadata_hash": stable_hash(
            {"raw_location": clean_text(location_text), "country_code": country_code}
        ),
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "top_k_result_ids": result_ids,
        "llm_input_hash": None,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "llm_model_version": LOCAL_LLM_MODEL_VERSION,
        "explanation_hash": stable_hash(
            [
                explain_match(query, location_text, result_row)
                for _, result_row in results.iterrows()
            ]
        ),
        "suggestions_hash": stable_hash(suggestions),
    }


def render_loading_indicator():
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div class="loading-screen" role="status" aria-label="Loading">
            <div class="loading-indicator">{icon_svg("compass")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return placeholder


def run_search(query, location_text, top_k, show_loading=False):
    loading_indicator = render_loading_indicator() if show_loading else None
    results, country_code = retrieve_projects(query, location_text, top_k)
    if loading_indicator is not None:
        loading_indicator.empty()

    suggestions = summarize_suggestions(results)
    run_record = build_run_record(query, location_text, country_code, results, suggestions)
    return {
        "query": query,
        "location": location_text,
        "top_k": top_k,
        "country_code": country_code,
        "results": results,
        "suggestions": suggestions,
        "run_record": run_record,
    }


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --ink: #080d31;
            --muted: #5f6687;
            --teal: #087d82;
            --teal-soft: #e8f5f5;
            --line: #dfe6ef;
            --panel: #ffffff;
            --surface: #f7fafc;
        }

        .stApp {
            background: var(--surface);
            color: var(--ink);
        }

        h1, h2, h3, p, label {
            letter-spacing: 0;
        }

        header[data-testid="stHeader"],
        #MainMenu,
        footer {
            display: none;
        }

        .block-container {
            max-width: 1520px;
            padding: 1.1rem 2rem 2rem;
        }

        .app-header {
            align-items: center;
            background: transparent;
            border-bottom: 0;
            display: flex;
            gap: 1.1rem;
            margin: -1.1rem auto 1.1rem;
            max-width: 62rem;
            padding: 1.05rem 0;
            width: 100%;
        }

        .app-header.is-centered {
            animation: workspace-center 180ms ease both;
        }

        .app-header.is-split {
            animation: workspace-slide-left 420ms cubic-bezier(0.16, 1, 0.3, 1) both;
            max-width: 92rem;
        }

        .logo-mark {
            align-items: center;
            color: var(--teal);
            display: flex;
            height: 3.1rem;
            justify-content: center;
            width: 3.1rem;
        }

        .logo-mark svg {
            display: block;
            height: 3rem;
            width: 3rem;
        }

        .brand {
            color: var(--ink);
            font-size: clamp(2rem, 3vw, 3rem);
            font-weight: 800;
            letter-spacing: 0;
            line-height: 1;
        }

        div[class*="st-key-workspace_centered"] {
            animation: workspace-center 180ms ease both;
            gap: 0.65rem;
            margin: 0 auto;
            max-width: 62rem;
        }

        div[class*="st-key-workspace_split"] {
            margin: 0 auto;
            max-width: 92rem;
        }

        div[class*="st-key-workspace_split"] div[data-testid="column"]:first-of-type {
            animation: workspace-slide-left 420ms cubic-bezier(0.16, 1, 0.3, 1) both;
        }

        div[class*="st-key-workspace_split"] div[data-testid="column"]:first-of-type
        > div[data-testid="stVerticalBlock"] {
            gap: 0.65rem;
        }

        div[class*="st-key-workspace_split"] div[data-testid="column"]:last-of-type {
            animation: none;
        }

        @keyframes workspace-center {
            from {
                opacity: 0.96;
                transform: translateX(-0.75rem);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes workspace-slide-left {
            from {
                transform: translateX(10rem);
            }
            to {
                transform: translateX(0);
            }
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 12px 28px rgba(8, 13, 49, 0.04);
            margin-bottom: 1.1rem;
        }

        div[data-testid="stForm"] {
            border: 0;
            padding: 0;
        }

        div[data-testid="stTextArea"] label p,
        div[data-testid="stTextInput"] label p {
            color: var(--ink);
            font-weight: 700;
        }

        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input {
            background-color: #fbffff !important;
            border: 1px solid #bfdee1 !important;
            border-radius: 8px !important;
            color: var(--ink) !important;
            font-size: 1rem !important;
        }

        div[data-testid="stTextInput"] input {
            background-image: url("__LOCATION_PIN_DATA_URL__") !important;
            background-position: 0.9rem center !important;
            background-repeat: no-repeat !important;
            background-size: 1.15rem 1.15rem !important;
            padding-left: 2.55rem !important;
        }

        div[data-testid="stTextArea"] textarea {
            min-height: 5.6rem !important;
        }

        .fake-upload-label {
            color: var(--ink);
            font-size: 0.875rem;
            font-weight: 700;
            line-height: 1.35;
            margin: 0 0 0.375rem;
        }

        .fake-upload {
            align-items: center;
            background: #fbffff;
            border: 1.5px dashed #7dbcc0;
            border-radius: 8px;
            color: var(--ink);
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
            justify-content: center;
            margin-bottom: 1.05rem;
            min-height: 5.6rem;
            padding: 1rem;
            text-align: center;
            transition:
                background 140ms ease,
                border-color 140ms ease;
        }

        .fake-upload:hover {
            background: #f2fbfb;
            border-color: var(--teal);
        }

        .fake-upload-icon {
            align-items: center;
            color: var(--teal);
            display: flex;
            height: 1.35rem;
            justify-content: center;
            width: 1.35rem;
        }

        .fake-upload-icon svg {
            display: block;
            height: 1.25rem;
            width: 1.25rem;
        }

        .fake-upload-text {
            color: var(--ink);
            font-size: 0.95rem;
            font-weight: 800;
            line-height: 1.25;
        }

        div[data-testid="stTextArea"] textarea:focus,
        div[data-testid="stTextInput"] input:focus {
            border-color: var(--teal) !important;
            box-shadow: 0 0 0 1px var(--teal) !important;
        }

        div[data-testid="stFormSubmitButton"] button {
            background: var(--teal) !important;
            border: 1px solid var(--teal) !important;
            border-radius: 8px !important;
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background: #06676b !important;
            border-color: #06676b !important;
        }

        .loading-screen {
            align-items: center;
            background: var(--surface);
            display: flex;
            inset: 0;
            justify-content: center;
            position: fixed;
            z-index: 9999;
        }

        .loading-indicator {
            align-items: center;
            color: var(--teal);
            display: flex;
            height: 7rem;
            justify-content: center;
            position: relative;
            width: 7rem;
        }

        .loading-indicator::before {
            animation: loading-spin 820ms linear infinite;
            border: 0.45rem solid #d8eeee;
            border-radius: 50%;
            border-top-color: var(--teal);
            content: "";
            inset: 0;
            position: absolute;
        }

        .loading-indicator svg {
            display: block;
            height: 3.1rem;
            position: relative;
            width: 3.1rem;
        }

        @keyframes loading-spin {
            to {
                transform: rotate(360deg);
            }
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 12px 28px rgba(8, 13, 49, 0.04);
            min-height: 33rem;
            padding: 1.45rem 1.55rem;
        }

        div[class*="st-key-details_card"] {
            animation: detail-reveal 260ms ease both;
            position: relative;
        }

        @keyframes detail-reveal {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }

        div[class*="st-key-close_details"] {
            position: absolute;
            right: 0.7rem;
            top: 0.65rem;
            z-index: 4;
        }

        div[class*="st-key-close_details"] button {
            align-items: center;
            background: transparent !important;
            border: 0 !important;
            border-radius: 999px !important;
            box-shadow: none !important;
            color: var(--muted) !important;
            display: flex;
            font-size: 1.05rem;
            font-weight: 800;
            height: 2rem;
            justify-content: center;
            min-height: 2rem;
            padding: 0 !important;
            width: 2rem;
        }

        div[class*="st-key-close_details"] button:hover {
            background: #eef7f7 !important;
            color: var(--teal) !important;
        }

        div[class*="st-key-close_details"] button p {
            margin: 0;
        }

        div[class*="st-key-results_panel"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 12px 28px rgba(8, 13, 49, 0.04);
            padding: 1.45rem 1.55rem;
        }

        .panel-title {
            align-items: center;
            display: flex;
            gap: 0.65rem;
            margin-bottom: 1.25rem;
        }

        .icon {
            align-items: center;
            color: var(--teal);
            display: flex;
            flex: 0 0 auto;
            font-size: 1rem;
            font-weight: 900;
            height: 1.45rem;
            justify-content: center;
            width: 1.45rem;
        }

        .icon svg {
            display: block;
            height: 1.35rem;
            width: 1.35rem;
        }

        .panel-title h2 {
            color: var(--ink);
            font-size: 1.45rem;
            line-height: 1.15;
            margin: 0;
        }

        .why-box {
            background: #fbffff;
            border: 1px solid #bfdee1;
            border-radius: 8px;
            color: var(--ink);
            font-size: 1.05rem;
            line-height: 1.55;
            padding: 1.2rem 1.35rem;
        }

        .llm-speaker {
            margin: 0.85rem 0 0.75rem;
        }

        .prompt-disclosure {
            margin-top: 0.5rem;
        }

        .prompt-disclosure > summary {
            list-style: none;
            transition:
                background 140ms ease,
                border-color 140ms ease,
                box-shadow 140ms ease;
            user-select: none;
            -webkit-user-select: none;
        }

        .prompt-disclosure > summary::-webkit-details-marker {
            display: none;
        }

        .prompt-disclosure > summary * {
            user-select: none;
            -webkit-user-select: none;
        }

        .prompt-disclosure > summary:focus {
            outline: 0;
        }

        .prompt-disclosure > summary:focus-visible {
            box-shadow: 0 0 0 3px rgba(8, 125, 130, 0.18);
        }

        .prompt-disclosure > summary:hover {
            background: #f6fbfb;
            border-color: #bfdee1;
            box-shadow: 0 0 0 1px rgba(8, 125, 130, 0.08);
        }

        .prompt-disclosure > summary:hover .trace-key,
        .prompt-disclosure > summary:hover .trace-hash {
            color: var(--teal);
        }

        .prompt-row {
            cursor: pointer;
            grid-template-columns: 1.5rem minmax(0, 1fr) auto;
            margin-top: 0;
        }

        .prompt-arrow {
            color: var(--teal);
            display: inline-block;
            font-size: 1rem;
            font-weight: 900;
            line-height: 1;
            margin-left: 0.35rem;
        }

        .prompt-arrow::before {
            content: "→";
        }

        .prompt-disclosure[open] .prompt-arrow::before {
            content: "↓";
        }

        .prompt-details {
            background: #fbffff;
            border: 1px solid var(--line);
            border-radius: 7px;
            color: var(--ink);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.76rem;
            line-height: 1.45;
            margin-top: 0.5rem;
            max-height: 15rem;
            overflow: auto;
            padding: 0.85rem;
            white-space: pre-wrap;
        }

        .score-breakdown {
            border: 1px solid var(--line);
            border-radius: 7px;
            display: grid;
            gap: 0.75rem;
            margin: 0.75rem 0;
            padding: 0.85rem;
        }

        .score-breakdown-title,
        .matched-terms-title {
            color: var(--ink);
            font-size: 0.84rem;
            font-weight: 850;
            line-height: 1.2;
        }

        .score-components {
            display: grid;
            gap: 0.7rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .score-component {
            min-width: 0;
        }

        .score-component-head {
            display: block;
        }

        .score-component-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
        }

        .score-component-points {
            color: var(--teal);
            display: block;
            font-size: 0.82rem;
            font-weight: 900;
            margin-top: 0.15rem;
            white-space: nowrap;
        }

        .score-track {
            background: #edf3f6;
            border-radius: 999px;
            height: 0.42rem;
            margin-top: 0.45rem;
            overflow: hidden;
        }

        .score-track-fill {
            background: var(--teal);
            border-radius: inherit;
            height: 100%;
        }

        .matched-terms {
            margin: 0.75rem 0;
        }

        .matched-term-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.45rem;
        }

        .matched-term {
            background: var(--teal-soft);
            border: 1px solid #bfdee1;
            border-radius: 999px;
            color: var(--teal);
            font-size: 0.78rem;
            font-weight: 850;
            line-height: 1;
            padding: 0.38rem 0.6rem;
        }

        .summary-strip {
            align-items: center;
            border-bottom: 1px solid var(--line);
            display: grid;
            gap: 0;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-bottom: 0.4rem;
            padding: 0 0 1rem;
        }

        .summary-item {
            min-width: 0;
            padding: 0 1.1rem;
        }

        .summary-item:first-child {
            padding-left: 0;
        }

        .summary-item + .summary-item {
            border-left: 1px solid var(--line);
        }

        .summary-value {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .summary-label {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.2;
            margin-top: 0.25rem;
        }

        .match {
            align-items: center;
            border-top: 1px solid var(--line);
            color: inherit;
            cursor: pointer;
            display: grid;
            gap: 0.9rem;
            grid-template-columns: 3.3rem minmax(0, 1fr) 6.3rem;
            margin: 0 -0.75rem;
            padding: 1.1rem 0.75rem;
            text-decoration: none;
            transition:
                background 140ms ease,
                border-color 140ms ease,
                box-shadow 140ms ease,
                transform 140ms ease;
        }

        .match:hover {
            background: #fbffff;
            transform: translateY(-1px);
        }

        .match:focus-visible {
            box-shadow: 0 0 0 3px rgba(8, 125, 130, 0.18);
            outline: 0;
        }

        .match.selected {
            border: 1px solid var(--teal);
            border-radius: 8px;
            box-shadow: 0 0 0 1px rgba(8, 125, 130, 0.12);
            margin-bottom: 0.25rem;
            padding: 1rem;
        }

        .match-action {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 750;
            margin-top: 0.45rem;
            text-transform: uppercase;
        }

        .match.selected .match-action {
            color: var(--teal);
        }

        div[class*="st-key-match_row_"] {
            position: relative;
        }

        .match-row {
            align-items: center;
            border-top: 1px solid var(--line);
            color: var(--ink);
            cursor: pointer;
            display: grid;
            gap: 1rem;
            grid-template-columns: minmax(0, 1fr) auto;
            min-height: 4.25rem;
            padding: 0.78rem 0.25rem;
            position: relative;
            transition:
                background 140ms ease,
                border-color 140ms ease,
                box-shadow 140ms ease;
        }

        .match-row::before {
            background: var(--teal);
            border-radius: 999px;
            bottom: 0.75rem;
            content: "";
            left: -0.25rem;
            opacity: 0;
            pointer-events: none;
            position: absolute;
            top: 0.75rem;
            transform: scaleY(0.65);
            transition:
                opacity 140ms ease,
                transform 140ms ease;
            width: 3px;
        }

        div[class*="st-key-match_row_1"] .match-row {
            border-top: 0;
        }

        div[class*="st-key-match_row_"]:hover .match-row,
        .match-row:hover {
            background: #f6fbfb;
        }

        div[class*="st-key-match_row_"]:hover .match-row::before,
        .match-row:hover::before {
            opacity: 0.45;
            transform: scaleY(1);
        }

        .match-row.selected {
            background: #fbffff;
            box-shadow: none;
        }

        .match-row.selected::before {
            opacity: 1;
            transform: scaleY(1);
        }

        div[class*="st-key-match_row_"]:hover .match-row-title,
        .match-row:hover .match-row-title {
            color: var(--teal);
        }

        .match-copy {
            min-width: 0;
        }

        .match-row-title {
            color: var(--ink);
            font-size: 0.92rem;
            font-weight: 700;
            line-height: 1.32;
        }

        .match-row-meta {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.35;
            margin-top: 0.28rem;
        }

        .match-row-meta span {
            color: var(--teal);
            font-weight: 800;
            margin: 0 0.35rem;
        }

        .match-row-score {
            color: var(--teal);
            font-size: 0.98rem;
            font-weight: 900;
            line-height: 1;
            text-align: right;
            white-space: nowrap;
        }

        div[class*="st-key-match_row_"] div[class*="st-key-select_project_"] {
            cursor: pointer;
            height: calc(100% + 1rem) !important;
            inset: 0;
            left: 0 !important;
            margin: 0;
            position: absolute;
            top: 0 !important;
            width: 100% !important;
            z-index: 2;
        }

        div[class*="st-key-match_row_"] div[data-testid="stButton"] {
            height: 100% !important;
            margin: 0;
            width: 100% !important;
        }

        div[class*="st-key-match_row_"] div[data-testid="stButton"] button {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            height: 100% !important;
            min-height: 100% !important;
            opacity: 0;
            padding: 0 !important;
            width: 100% !important;
        }

        div[class*="st-key-match_row_"] div[data-testid="stButton"] button p {
            font-size: 0;
        }

        div[class*="st-key-match_row_"]:focus-within .match-row {
            box-shadow: 0 0 0 3px rgba(8, 125, 130, 0.18);
        }

        div[class*="st-key-show_more_control"] button {
            background: transparent !important;
            border: 0 !important;
            border-radius: 0 !important;
            color: var(--teal);
            font-size: 0.9rem;
            font-weight: 800;
            justify-content: flex-start;
            min-height: auto;
            padding: 0.4rem 0 0 !important;
            text-align: left;
            text-decoration: underline;
            width: auto;
        }

        div[class*="st-key-show_more_control"] button p {
            color: var(--teal);
            font-size: 0.9rem;
            font-weight: 800;
            text-decoration: underline;
        }

        div[class*="st-key-show_more_control"] button:hover p {
            color: #06676b;
        }

        div[class*="st-key-show_more_control"] button p {
            margin: 0;
        }

        .rank {
            align-items: center;
            background: var(--teal-soft);
            border-radius: 50%;
            color: var(--ink);
            display: flex;
            font-size: 1.15rem;
            font-weight: 800;
            height: 2.75rem;
            justify-content: center;
            width: 2.75rem;
        }

        .match-title {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 800;
            line-height: 1.25;
        }

        .match-meta {
            color: var(--ink);
            font-size: 0.9rem;
            margin-top: 0.35rem;
        }

        .match-meta span {
            color: var(--teal);
            font-weight: 800;
            margin: 0 0.35rem;
        }

        .score {
            color: var(--teal);
            font-size: 1.05rem;
            font-weight: 800;
            text-align: right;
        }

        .bar {
            background: #e3e7ec;
            border-radius: 999px;
            height: 0.48rem;
            margin-left: auto;
            margin-top: 0.8rem;
            overflow: hidden;
            width: 5.8rem;
        }

        .bar-fill {
            background: var(--teal);
            border-radius: inherit;
            height: 100%;
        }

        .trace-row {
            align-items: center;
            border: 1px solid var(--line);
            border-radius: 7px;
            display: grid;
            gap: 0.75rem;
            grid-template-columns: 1.5rem minmax(0, 1fr) auto;
            margin-top: 0.5rem;
            padding: 0.75rem;
        }

        .trace-row.prompt-row {
            grid-template-columns: 1.5rem minmax(0, 1fr) auto;
            margin-top: 0;
        }

        .data-note {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.45;
            margin-top: 1rem;
        }

        .trace-key {
            color: var(--ink);
            font-size: 0.95rem;
            font-weight: 750;
        }

        .trace-hash {
            color: var(--ink);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.85rem;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .app-header {
                margin-left: auto;
                margin-right: auto;
            }

            .summary-strip {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }

            .summary-item {
                padding: 0 0.65rem;
            }

            .summary-value {
                font-size: 0.96rem;
            }

            .summary-label {
                font-size: 0.78rem;
            }

            .score {
                text-align: left;
            }

            .bar {
                margin-left: 0;
            }
        }
        </style>
        """.replace("__LOCATION_PIN_DATA_URL__", LOCATION_PIN_DATA_URL),
        unsafe_allow_html=True,
    )


def render_header(is_split=False):
    layout_class = "is-split" if is_split else "is-centered"
    st.markdown(
        f"""
        <div class="app-header {layout_class}">
            <div class="logo-mark" aria-label="Compass logo">
                {icon_svg("compass")}
            </div>
            <div class="brand">Funding Compass</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def selected_rank(state):
    try:
        rank = int(st.session_state.get("selected_project_rank", 0) or 0)
    except (TypeError, ValueError):
        return None

    visible_ranks = set(state["results"]["rank"].astype(int).tolist())
    if rank in visible_ranks:
        return rank

    st.session_state.selected_project_rank = None
    return None


def selected_result(state):
    rank = selected_rank(state)
    if rank is None:
        return None

    matches = state["results"][state["results"]["rank"].astype(int) == rank]
    if matches.empty:
        return None
    return matches.iloc[0]


def select_project(rank):
    st.session_state.selected_project_rank = rank


def expand_matches():
    st.session_state.show_all_matches = True


def close_details():
    st.session_state.selected_project_rank = None


def render_result_panel(state, active_rank):
    results = state["results"]
    suggestions = state["suggestions"]
    if results.empty:
        with st.container(key="results_panel"):
            st.markdown(
                """
                <div class="panel-title">
                    <div class="icon">R</div>
                    <h2>No matches</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    top_fund = short_fund(suggestions.get("fund_name", [{}])[0].get("value", "n/a"))
    median_budget = format_money(suggestions.get("budget_median"))
    match_count = len(results)
    expanded = st.session_state.get("show_all_matches", False)
    visible_results = results if expanded else results.head(3)

    with st.container(key="results_panel"):
        st.markdown(
            (
                '<div class="summary-strip">'
                '<div class="summary-item">'
                f'<div class="summary-value">{safe(top_fund)}</div>'
                '<div class="summary-label">Top fund</div>'
                "</div>"
                '<div class="summary-item">'
                f'<div class="summary-value">{safe(median_budget)}</div>'
                '<div class="summary-label">Median budget</div>'
                "</div>"
                '<div class="summary-item">'
                f'<div class="summary-value">{match_count}</div>'
                '<div class="summary-label">Matches</div>'
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        for _, row in visible_results.iterrows():
            rank = int(row["rank"])
            title, _ = split_title_summary(row.get("search_text", ""))
            score = float(row.get("confidence", 0))
            selected = rank == active_rank
            selected_class = " selected" if selected else ""
            row_html = (
                f'<div class="match-row{selected_class}">'
                '<div class="match-copy">'
                f'<div class="match-row-title">#{rank} {safe(title)}</div>'
                '<div class="match-row-meta">'
                f"{safe(row.get('Country'))}<span>*</span>"
                f"{safe(short_fund(row.get('Fund_Name')))}"
                "</div>"
                "</div>"
                f'<div class="match-row-score">{score * 1000:.0f} pt.</div>'
                "</div>"
            )
            with st.container(key=f"match_row_{rank}"):
                st.markdown(row_html, unsafe_allow_html=True)
                st.button(
                    f"Select project {rank}",
                    key=f"select_project_{rank}",
                    on_click=select_project,
                    args=(rank,),
                )

        if not expanded and len(results) > len(visible_results):
            with st.container(key="show_more_control"):
                st.button(
                    "Show more ->",
                    key="show_more_matches",
                    on_click=expand_matches,
                    type="tertiary",
                )


def render_explain_panel(state, selected_row):
    results = state["results"]
    if results.empty or selected_row is None:
        return

    rank = int(selected_row.get("rank", 0) or 0)
    explanation = explain_match(state["query"], state["location"], selected_row)
    prompt_text = build_explanation_prompt(
        build_llm_input(state["query"], state["location"], selected_row)
    )
    breakdown_items = score_breakdown(selected_row)
    breakdown_html = "".join(
        (
            '<div class="score-component">'
            '<div class="score-component-head">'
            f'<span class="score-component-label">{safe(item["label"])}</span>'
            f'<span class="score-component-points">{item["points"]:.0f} pt.</span>'
            "</div>"
            '<div class="score-track">'
            f'<div class="score-track-fill" style="width: {item["width"]:.0f}%;"></div>'
            "</div>"
            "</div>"
        )
        for item in breakdown_items
    )
    matched_terms = overlap_terms(state["query"], selected_row.get("search_text", ""), limit=6)
    matched_terms_html = "".join(
        f'<span class="matched-term">{safe(term)}</span>' for term in matched_terms
    )
    source_url = clean_text(selected_row.get("Operation_Unique_Identifier"))
    run_record = state["run_record"]

    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">
                <h2>Why #{rank} matched</h2>
            </div>
            <div class="trace-row">
                <div class="icon">{icon_svg("database")}</div>
                <div class="trace-key">Dataset version</div>
                <div class="trace-hash">{safe(short_hash(run_record["suggestions_hash"]))}</div>
            </div>
            <details class="prompt-disclosure">
                <summary class="trace-row prompt-row">
                    <div class="icon">{icon_svg("file")}</div>
                    <div class="trace-key">
                        Prompt <span class="prompt-arrow" aria-hidden="true"></span>
                    </div>
                    <div class="trace-hash">{safe(short_hash(stable_hash(prompt_text)))}</div>
                </summary>
                <div class="prompt-details">{html.escape(prompt_text)}</div>
            </details>
            <div class="trace-row llm-speaker">
                <div class="icon">{icon_svg("brain")}</div>
                <div class="trace-key">LLM {safe(run_record["llm_model_version"])}</div>
                <div class="trace-hash">{safe(short_hash(stable_hash(explanation)))}</div>
            </div>
            <div class="score-breakdown">
                <div class="score-breakdown-title">Score breakdown</div>
                <div class="score-components">{breakdown_html}</div>
            </div>
            <div class="matched-terms">
                <div class="matched-terms-title">Matched terms</div>
                <div class="matched-term-list">{matched_terms_html}</div>
            </div>
            <div class="why-box">{safe(explanation)}</div>
            <div style="text-align: right; margin: 0.9rem 0 1.2rem;">
                <a href="{safe(source_url)}" target="_blank" style="color: var(--teal);
                    font-weight: 800; text-decoration: underline;">Project source →</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_search_controls(state):
    with st.container(border=True):
        with st.form("search_form"):
            query = st.text_area(
                "Project idea",
                value=state["query"],
                height=120,
                placeholder="Describe the project idea",
            )
            st.markdown(
                f"""
                <div class="fake-upload-label">Project file</div>
                <div class="fake-upload" aria-label="Fake file upload area">
                    <div class="fake-upload-icon">{icon_svg("file")}</div>
                    <div class="fake-upload-text">Move the file there to upload</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            location = st.text_input("Location", value=state["location"])
            submitted = st.form_submit_button("Search", width="stretch", type="primary")

    if submitted:
        st.session_state.search_state = run_search(query, location, DEFAULT_TOP_K)
        st.session_state.selected_project_rank = None
        st.session_state.show_all_matches = False
        st.rerun()


def render_search_tab(state):
    selected_row = selected_result(state)

    if selected_row is None:
        with st.container(key="workspace_centered"):
            render_search_controls(state)
            render_result_panel(state, None)
        return

    with st.container(key="workspace_split"):
        left, right = st.columns([1.45, 1], gap="large")

        with left:
            render_search_controls(state)
            active_rank = int(selected_row["rank"])
            render_result_panel(state, active_rank)

        with right:
            with st.container(key="details_card"):
                st.button(
                    "x",
                    key="close_details",
                    help="Close details",
                    on_click=close_details,
                    type="tertiary",
                )
                render_explain_panel(state, selected_row)


def main():
    inject_styles()

    missing = missing_artifacts()
    if missing:
        render_header()
        missing_list = "\n".join(f"- `{path.relative_to(ROOT_DIR)}`" for path in missing)
        st.error(f"Required cached artifacts are missing:\n\n{missing_list}")
        st.stop()

    if "search_state" not in st.session_state:
        st.session_state.search_state = run_search(
            DEFAULT_QUERY,
            DEFAULT_LOCATION,
            DEFAULT_TOP_K,
            show_loading=True,
        )
    if "show_all_matches" not in st.session_state:
        st.session_state.show_all_matches = False

    state = st.session_state.search_state
    render_header(selected_result(state) is not None)
    render_search_tab(state)


if __name__ == "__main__":
    main()
