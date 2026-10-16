"""UrbanKart Customer Feedback Analysis page.

This file is registered by app.py using st.Page. Do not call st.set_page_config
here; page configuration is centralized in app.py.
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

st.markdown(
    """
    <div style="padding:1.3rem 1.5rem;border-radius:16px;
    background:linear-gradient(120deg,#14213d 0%,#254b75 100%);
    color:white;margin-bottom:1rem;">
      <h1 style="color:white;margin:0 0 .3rem 0;">💬 Customer Feedback Analysis</h1>
      <p style="color:#e8f0fb;margin:0;">AI-powered sentiment classification for UrbanKart reviews</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Use the pretrained Hugging Face model to classify individual reviews or a complete CSV dataset."
)


@st.cache_resource(show_spinner="Loading the Hugging Face sentiment model…")
def load_sentiment_pipeline():
    from transformers import pipeline

    return pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1,
    )


def normalize_label(label):
    """Normalize model output labels for display."""
    label = str(label).strip().lower()
    mapping = {
        "label_0": "Negative",
        "label_1": "Neutral",
        "label_2": "Positive",
    }
    return mapping.get(label, label.capitalize())


def read_uploaded_csv(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError(f"Could not parse the CSV. Details: {last_error}")


def load_root_dataset():
    if not DATASET_PATH.is_file():
        return None
    return pd.read_csv(DATASET_PATH, encoding="utf-8-sig")


tab_single, tab_batch = st.tabs(["✍️ Analyze a review", "📊 Analyze a CSV dataset"])

with tab_single:
    st.subheader("Single review analysis")
    review_text = st.text_area(
        "Customer review",
        placeholder="Example: Delivery was quick and the product quality was excellent.",
        height=140,
        key="feedback_single_review",
    )
    if st.button(
        "Analyze sentiment",
        type="primary",
        key="feedback_analyze_one",
        disabled=not review_text.strip(),
    ):
        try:
            with st.spinner("Analyzing review…"):
                analyzer = load_sentiment_pipeline()
                result = analyzer(review_text.strip(), truncation=True, max_length=512)[0]
            sentiment = normalize_label(result["label"])
            confidence = float(result["score"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted sentiment", sentiment)
            c2.metric("Confidence", f"{confidence:.1%}")
            c3.metric("Model", "RoBERTa")
            st.progress(min(max(confidence, 0.0), 1.0))
            st.caption("Confidence is the model's score for its predicted class, not a guarantee of correctness.")
        except ImportError as exc:
            st.error("A required package is missing from the active Streamlit environment.")
            st.code("python -m pip install transformers torch")
            st.exception(exc)
        except Exception as exc:
            logger.exception("Single-review analysis failed")
            st.error("Could not analyze this review. Check the internet connection and model dependencies.")
            st.exception(exc)

with tab_batch:
    st.subheader("Batch sentiment analysis")
    st.write(
        "The actual review text field in this dataset is `review_text`. "
        "Existing `sentiment_label` values are retained for later comparison."
    )
    uploaded_csv = st.file_uploader(
        "Upload customer reviews CSV (optional if present in the project root)",
        type=["csv"],
        key="feedback_csv_upload",
    )

    dataset = None
    if uploaded_csv is not None:
        try:
            dataset = read_uploaded_csv(uploaded_csv)
            source_name = uploaded_csv.name
        except Exception as exc:
            st.error("The uploaded file could not be read.")
            st.exception(exc)
    else:
        try:
            dataset = load_root_dataset()
            source_name = DATASET_NAME
        except Exception as exc:
            st.error(f"Could not read {DATASET_NAME} from the project root.")
            st.exception(exc)

    if dataset is None:
        st.info(
            f"Place `{DATASET_NAME}` beside `app.py`, or upload the CSV above."
        )
    else:
        st.success(f"Loaded {source_name}")
        st.write(f"Dataset: **{len(dataset):,} rows × {len(dataset.columns):,} columns**")
        with st.expander("Inspect source data", expanded=True):
            st.code(", ".join(map(str, dataset.columns)), language="text")
            st.dataframe(dataset.head(5), use_container_width=True, hide_index=True)

        if "review_text" not in dataset.columns:
            st.error("Required column `review_text` was not found in the CSV.")
        else:
            reviews = dataset["review_text"].fillna("").astype(str).str.strip()
            valid_mask = reviews.ne("")
            valid_count = int(valid_mask.sum())
            st.info(f"Non-empty reviews available: **{valid_count:,}**")
            batch_size = st.select_slider(
                "Batch size", options=[4, 8, 16, 32], value=16,
                help="Larger batches may use more memory.",
            )
            if st.button(
                "Run batch sentiment analysis",
                type="primary",
                key="feedback_run_batch",
                disabled=valid_count == 0,
            ):
                try:
                    analyzer = load_sentiment_pipeline()
                    texts = reviews[valid_mask].tolist()
                    predictions = []
                    progress = st.progress(0, text="Starting sentiment analysis…")
                    for start in range(0, len(texts), batch_size):
                        batch = texts[start:start + batch_size]
                        predictions.extend(
                            analyzer(batch, batch_size=batch_size, truncation=True, max_length=512)
                        )
                        done = min(start + len(batch), len(texts))
                        progress.progress(
                            int(done / len(texts) * 100),
                            text=f"Analyzed {done:,} of {len(texts):,} reviews…",
                        )

                    output = dataset.copy()
                    output["predicted_sentiment"] = pd.NA
                    output["confidence_score"] = pd.NA
                    output["confidence_percent"] = pd.NA
                    prediction_frame = pd.DataFrame({
                        "predicted_sentiment": [normalize_label(p["label"]) for p in predictions],
                        "confidence_score": [float(p["score"]) for p in predictions],
                        "confidence_percent": [round(float(p["score"]) * 100, 2) for p in predictions],
                    })
                    output.loc[
                        valid_mask,
                        ["predicted_sentiment", "confidence_score", "confidence_percent"],
                    ] = prediction_frame.to_numpy()

                    st.session_state["urbankart_feedback_results"] = output
                    st.success("Batch sentiment analysis completed.")
                except ImportError as exc:
                    st.error("A required package is missing from the active Streamlit environment.")
                    st.code("python -m pip install transformers torch")
                    st.exception(exc)
                except Exception as exc:
                    logger.exception("Batch sentiment analysis failed")
                    st.error("Batch analysis failed. Check the CSV, model download, and dependencies.")
                    st.exception(exc)

            output = st.session_state.get("urbankart_feedback_results")
            if output is not None:
                analyzed = output["predicted_sentiment"].notna()
                analyzed_frame = output.loc[analyzed].copy()
                if analyzed_frame.empty:
                    st.warning("No non-empty reviews were classified.")
                else:
                    counts = (
                        analyzed_frame["predicted_sentiment"]
                        .value_counts()
                        .rename_axis("Sentiment")
                        .reset_index(name="Reviews")
                    )
                    total = len(analyzed_frame)
                    avg_conf = pd.to_numeric(analyzed_frame["confidence_score"], errors="coerce").mean()
                    positive = int(analyzed_frame["predicted_sentiment"].eq("Positive").sum())
                    negative = int(analyzed_frame["predicted_sentiment"].eq("Negative").sum())
                    neutral = int(analyzed_frame["predicted_sentiment"].eq("Neutral").sum())
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Reviews analyzed", f"{total:,}")
                    m2.metric("Positive", f"{positive:,}")
                    m3.metric("Negative", f"{negative:,}")
                    m4.metric("Average confidence", f"{avg_conf:.1%}" if pd.notna(avg_conf) else "N/A")

                    chart_col, summary_col = st.columns(2)
                    with chart_col:
                        st.markdown("**Sentiment distribution**")
                        st.bar_chart(counts.set_index("Sentiment"), y="Reviews", use_container_width=True)
                    with summary_col:
                        st.markdown("**Distribution summary**")
                        counts["Share (%)"] = (counts["Reviews"] / total * 100).round(2)
                        st.dataframe(counts, use_container_width=True, hide_index=True)
                        st.caption(f"Neutral predictions: {neutral:,}")

                    st.subheader("Classified reviews")
                    options = sorted(analyzed_frame["predicted_sentiment"].dropna().unique().tolist())
                    chosen = st.multiselect("Filter sentiment", options, default=options)
                    st.dataframe(
                        analyzed_frame[analyzed_frame["predicted_sentiment"].isin(chosen)],
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "⬇️ Download analyzed CSV",
                        data=output.to_csv(index=False).encode("utf-8-sig"),
                        file_name="UrbanKart_customer_feedback_sentiment_analysis.csv",
                        mime="text/csv",
                        type="primary",
                    )
                    st.caption(
                        "Original columns are preserved. The app adds predicted_sentiment, "
                        "confidence_score (0–1), and confidence_percent (0–100)."
                    )

st.divider()
st.caption(
    "Automated sentiment predictions may be wrong. Review results before using them for consequential customer decisions."
)
