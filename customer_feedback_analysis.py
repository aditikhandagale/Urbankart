"""
UrbanKart Customer Feedback Analysis
Run from the project root:
    python -m streamlit run customer_feedback_analysis.py

Expected dataset in the project root:
    UrbanKart_HuggingFace_NLP_Customer Reviews.csv

The app also supports uploading the CSV from the interface.
"""

from pathlib import Path
import io
import logging

import pandas as pd
import streamlit as st

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET_NAME = "UrbanKart_HuggingFace_NLP_Customer Reviews.csv"
ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / DATASET_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="UrbanKart | Customer Feedback Analysis",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .main .block-container {padding-top: 1.8rem; padding-bottom: 2.5rem;}
      .hero {
        padding: 1.35rem 1.5rem; border-radius: 16px;
        background: linear-gradient(120deg, #14213d 0%, #254b75 100%);
        color: white; margin-bottom: 1.2rem;
      }
      .hero h1 {color: white; margin: 0 0 .35rem 0; font-size: 2rem;}
      .hero p {color: #e8f0fb; margin: 0;}
      .small-note {color: #64748b; font-size: .9rem;}
      div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 12px; padding: 14px 16px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🛍️ UrbanKart Customer Intelligence Hub</h1>
      <p>Customer Feedback Analysis · AI-powered sentiment classification</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Analyze individual reviews or classify a complete review dataset using a "
    "pretrained Hugging Face sentiment model."
)


@st.cache_resource(show_spinner="Loading the Hugging Face sentiment model…")
def load_sentiment_pipeline():
    """Load the model once per Streamlit process."""
    from transformers import pipeline

    # Explicit CPU mode avoids requiring a CUDA/GPU setup.
    return pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1,
    )


def normalize_label(label):
    """Return a clean display label from the model's output."""
    return str(label).strip().lower().capitalize()


def analyze_texts(texts, analyzer, batch_size=16):
    """Predict sentiment and confidence for a list of review strings."""
    predictions = []
    # Pipeline handles tokenization/truncation; batches improve throughput.
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_predictions = analyzer(
            batch,
            batch_size=batch_size,
            truncation=True,
            max_length=512,
        )
        predictions.extend(batch_predictions)

    return pd.DataFrame(
        {
            "predicted_sentiment": [normalize_label(item["label"]) for item in predictions],
            "confidence_score": [float(item["score"]) for item in predictions],
            "confidence_percent": [round(float(item["score"]) * 100, 2) for item in predictions],
        }
    )


def read_uploaded_csv(uploaded_file):
    """Read CSV with a few common encodings and return helpful errors."""
    raw_bytes = uploaded_file.getvalue()
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError(f"Could not read the uploaded CSV. Please check its format. Details: {last_error}")


def load_root_dataset():
    """Load the expected dataset from the existing project root if present."""
    if not DATASET_PATH.is_file():
        return None
    return pd.read_csv(DATASET_PATH, encoding="utf-8-sig")


def sentiment_chart_data(results):
    counts = (
        results["predicted_sentiment"]
        .value_counts()
        .rename_axis("Sentiment")
        .reset_index(name="Reviews")
    )
    return counts


# Sidebar
with st.sidebar:
    st.header("Analysis settings")
    st.markdown("**Model**")
    st.code(MODEL_NAME, language="text")
    st.caption("Model predictions are estimates, not verified customer labels.")
    st.divider()
    st.markdown("**Dataset**")
    st.write(DATASET_NAME)
    st.caption("Expected in the project root. You can also upload it below.")

tab_single, tab_batch = st.tabs(["✍️ Analyze a review", "📊 Analyze a CSV dataset"])

with tab_single:
    st.subheader("Single review analysis")
    review_text = st.text_area(
        "Enter customer review",
        placeholder="Example: The delivery was quick and the product quality was excellent.",
        height=150,
        key="single_review_text",
    )
    analyze_one = st.button(
        "Analyze sentiment",
        type="primary",
        key="analyze_single",
        disabled=not review_text.strip(),
    )

    if analyze_one:
        try:
            with st.spinner("Analyzing review…"):
                analyzer = load_sentiment_pipeline()
                prediction = analyzer(
                    review_text.strip(),
                    truncation=True,
                    max_length=512,
                )[0]
            sentiment = normalize_label(prediction["label"])
            confidence = float(prediction["score"])
            left, middle, right = st.columns(3)
            left.metric("Predicted sentiment", sentiment)
            middle.metric("Confidence", f"{confidence:.1%}")
            right.metric("Model", "RoBERTa")
            st.progress(min(max(confidence, 0.0), 1.0))
            st.caption(
                "Confidence is the model's score for its predicted class; it is not a guarantee "
                "that the prediction is correct."
            )
        except ImportError as exc:
            st.error(
                "A required ML package is missing in the Python environment running Streamlit. "
                "Install the packages in that same environment, then restart the app."
            )
            st.code("python -m pip install streamlit pandas transformers torch")
            st.exception(exc)
        except Exception as exc:
            logger.exception("Single-review sentiment analysis failed")
            st.error("The review could not be analyzed. Check the model download, internet connection, and package versions.")
            st.exception(exc)


with tab_batch:
    st.subheader("Batch sentiment analysis")
    st.write(
        "Load the UrbanKart customer reviews CSV, identify the `review_text` column, "
        "classify each non-empty review, and download the enriched results."
    )

    uploaded_csv = st.file_uploader(
        "Upload customer reviews CSV (optional if the file is in the project root)",
        type=["csv"],
        key="reviews_csv_upload",
    )

    dataset = None
    source_label = None
    if uploaded_csv is not None:
        try:
            dataset = read_uploaded_csv(uploaded_csv)
            source_label = uploaded_csv.name
        except Exception as exc:
            st.error("The uploaded file could not be read as a CSV.")
            st.exception(exc)
    else:
        try:
            dataset = load_root_dataset()
            if dataset is not None:
                source_label = DATASET_NAME
        except Exception as exc:
            st.error(f"Could not read `{DATASET_NAME}` from the project root.")
            st.exception(exc)

    if dataset is None:
        st.info(
            f"Place **{DATASET_NAME}** beside `customer_feedback_analysis.py`, "
            "or upload it using the control above."
        )
    else:
        st.success(f"Loaded **{source_label}**")
        st.write(f"Dataset shape: **{dataset.shape[0]:,} rows × {dataset.shape[1]:,} columns**")
        with st.expander("Inspect dataset before analysis", expanded=True):
            st.write("**Columns detected**")
            st.code(", ".join(map(str, dataset.columns)), language="text")
            st.write("**Sample rows**")
            st.dataframe(dataset.head(5), use_container_width=True, hide_index=True)
            if "review_text" in dataset.columns:
                st.write("**Review text quality**")
                missing_count = dataset["review_text"].isna().sum()
                blank_count = (
                    dataset["review_text"].fillna("").astype(str).str.strip().eq("").sum()
                    - missing_count
                )
                st.write(
                    f"Missing review_text values: **{missing_count:,}** · "
                    f"Blank review_text values: **{max(int(blank_count), 0):,}**"
                )

        if "review_text" not in dataset.columns:
            st.error(
                "Could not find the required `review_text` column. "
                "Please check the CSV header. The expected column is exactly `review_text`."
            )
        else:
            available_rows = int(
                dataset["review_text"].fillna("").astype(str).str.strip().ne("").sum()
            )
            st.info(f"Non-empty reviews available for analysis: **{available_rows:,}**")

            batch_size = st.select_slider(
                "Prediction batch size",
                options=[4, 8, 16, 32],
                value=16,
                help="Larger batches can be faster but may use more memory.",
            )
            run_batch = st.button(
                "Run batch sentiment analysis",
                type="primary",
                key="run_batch_analysis",
                disabled=available_rows == 0,
            )

            if run_batch:
                try:
                    working = dataset.copy()
                    review_series = working["review_text"].fillna("").astype(str).str.strip()
                    valid_mask = review_series.ne("")
                    valid_texts = review_series[valid_mask].tolist()

                    progress = st.progress(0, text="Loading sentiment model…")
                    analyzer = load_sentiment_pipeline()
                    progress.progress(10, text=f"Analyzing {len(valid_texts):,} reviews…")

                    result_parts = []
                    total = len(valid_texts)
                    for start in range(0, total, batch_size):
                        batch_texts = valid_texts[start : start + batch_size]
                        batch_predictions = analyzer(
                            batch_texts,
                            batch_size=batch_size,
                            truncation=True,
                            max_length=512,
                        )
                        result_parts.extend(batch_predictions)
                        pct = 10 + int(85 * min(start + len(batch_texts), total) / total)
                        progress.progress(
                            min(pct, 95),
                            text=f"Analyzed {min(start + len(batch_texts), total):,} of {total:,} reviews…",
                        )

                    working["predicted_sentiment"] = pd.NA
                    working["confidence_score"] = pd.NA
                    working["confidence_percent"] = pd.NA

                    predictions_df = pd.DataFrame(
                        {
                            "predicted_sentiment": [
                                normalize_label(item["label"]) for item in result_parts
                            ],
                            "confidence_score": [
                                float(item["score"]) for item in result_parts
                            ],
                            "confidence_percent": [
                                round(float(item["score"]) * 100, 2) for item in result_parts
                            ],
                        }
                    )
                    working.loc[
                        valid_mask, ["predicted_sentiment", "confidence_score", "confidence_percent"]
                    ] = predictions_df[
                        ["predicted_sentiment", "confidence_score", "confidence_percent"]
                    ].to_numpy()

                    st.session_state["urban_kart_feedback_results"] = working
                    progress.progress(100, text="Batch analysis complete.")
                except ImportError as exc:
                    st.error(
                        "A required ML package is missing in the Python environment running Streamlit. "
                        "Install packages in that same environment, then restart the app."
                    )
                    st.code("python -m pip install streamlit pandas transformers torch")
                    st.exception(exc)
                except Exception as exc:
                    logger.exception("Batch sentiment analysis failed")
                    st.error(
                        "Batch analysis failed. Check that the CSV is valid, the model can be downloaded, "
                        "and the active Python environment has compatible dependencies."
                    )
                    st.exception(exc)

            results = st.session_state.get("urban_kart_feedback_results")
            if results is not None:
                st.divider()
                st.subheader("Sentiment overview")
                analyzed = results["predicted_sentiment"].notna()
                analyzed_results = results.loc[analyzed].copy()

                if analyzed_results.empty:
                    st.warning("No non-empty reviews were available to analyze.")
                else:
                    counts = sentiment_chart_data(analyzed_results)
                    total_analyzed = len(analyzed_results)
                    dominant = counts.iloc[0]["Sentiment"]
                    avg_confidence = pd.to_numeric(
                        analyzed_results["confidence_score"], errors="coerce"
                    ).mean()

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Reviews analyzed", f"{total_analyzed:,}")
                    m2.metric("Positive", f"{(analyzed_results['predicted_sentiment'].eq('Positive').sum()):,}")
                    m3.metric("Negative", f"{(analyzed_results['predicted_sentiment'].eq('Negative').sum()):,}")
                    m4.metric("Average confidence", f"{avg_confidence:.1%}" if pd.notna(avg_confidence) else "N/A")

                    chart_col, table_col = st.columns([1, 1])
                    with chart_col:
                        st.markdown("**Sentiment distribution**")
                        st.bar_chart(counts.set_index("Sentiment"), y="Reviews", use_container_width=True)
                    with table_col:
                        st.markdown("**Distribution summary**")
                        counts["Share (%)"] = (counts["Reviews"] / total_analyzed * 100).round(2)
                        st.dataframe(counts, use_container_width=True, hide_index=True)
                        st.caption(f"Most common predicted sentiment: **{dominant}**")

                    st.subheader("Classified customer reviews")
                    sentiment_filter = st.multiselect(
                        "Filter by predicted sentiment",
                        options=sorted(analyzed_results["predicted_sentiment"].dropna().unique().tolist()),
                        default=sorted(analyzed_results["predicted_sentiment"].dropna().unique().tolist()),
                    )
                    filtered = analyzed_results[
                        analyzed_results["predicted_sentiment"].isin(sentiment_filter)
                    ]
                    st.dataframe(filtered, use_container_width=True, hide_index=True)

                    csv_bytes = results.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="⬇️ Download analyzed results (CSV)",
                        data=csv_bytes,
                        file_name="UrbanKart_customer_feedback_sentiment_analysis.csv",
                        mime="text/csv",
                        type="primary",
                    )
                    st.caption(
                        "The original dataset columns are preserved. Added columns are "
                        "`predicted_sentiment`, `confidence_score` (0–1), and "
                        "`confidence_percent` (0–100). Existing `sentiment_label` values are not overwritten."
                    )

st.divider()
st.markdown(
    '<p class="small-note">UrbanKart Customer Intelligence Hub · Sentiment predictions are automated '
    'and should be reviewed before making high-impact customer decisions.</p>',
    unsafe_allow_html=True,
)
