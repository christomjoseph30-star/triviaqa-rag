"""
Streamlit app to explore RAG pipeline results:
  - Browse questions, gold answers, predicted answers
  - View retrieved chunks per question
  - Filter by retrieval hit / exact match
"""
import os
import glob
import pandas as pd
import streamlit as st

RESULTS_DIR = "./results"

st.set_page_config(page_title="RAG Results Explorer", layout="wide")
st.title("RAG Results Explorer")

# ── Load results file ──────────────────────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv")), reverse=True)
if not csv_files:
    st.error("No results CSV found in ./results/. Run the pipeline first.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Results file",
    csv_files,
    format_func=lambda x: os.path.basename(x),
)
df = pd.read_csv(selected_file)

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("Filters")

filter_retrieval = st.sidebar.selectbox(
    "Retrieval hit", ["All", "Hit only", "Miss only"]
)
filter_em = st.sidebar.selectbox(
    "Exact match", ["All", "Correct only", "Wrong only"]
)
search_query = st.sidebar.text_input("Search question", "")

filtered = df.copy()
if filter_retrieval == "Hit only":
    filtered = filtered[filtered["retrieval_hit"] == True]
elif filter_retrieval == "Miss only":
    filtered = filtered[filtered["retrieval_hit"] == False]

if filter_em == "Correct only":
    filtered = filtered[filtered["exact_match"] == True]
elif filter_em == "Wrong only":
    filtered = filtered[filtered["exact_match"] == False]

if search_query:
    filtered = filtered[filtered["question"].str.contains(search_query, case=False, na=False)]

# ── Metrics summary ────────────────────────────────────────────────────────────
st.subheader("Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Recall@5", f"{df['retrieval_hit'].mean():.2%}")
col2.metric("Exact Match", f"{df['exact_match'].mean():.2%}")
col3.metric("Mean Token F1", f"{df['token_f1'].mean():.2%}")
col4.metric("Questions", len(df))

st.markdown("---")

# ── Question list ──────────────────────────────────────────────────────────────
st.subheader(f"Questions ({len(filtered)} shown)")

if filtered.empty:
    st.info("No questions match the current filters.")
    st.stop()

for _, row in filtered.iterrows():
    # Header label with status indicators
    ret_icon = "✅" if row["retrieval_hit"] else "❌"
    em_icon  = "✅" if row["exact_match"]   else "❌"
    label = f"{ret_icon} Ret  {em_icon} EM  F1:{row['token_f1']:.2f}  |  {row['question']}"

    with st.expander(label):
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("**Gold Answer**")
            st.success(row["gold_answer"])

            st.markdown("**Predicted Answer**")
            if row["exact_match"]:
                st.success(row["predicted_answer"])
            elif row["retrieval_hit"]:
                st.warning(row["predicted_answer"])
            else:
                st.error(row["predicted_answer"])

        with col_right:
            st.markdown("**Retrieved Chunks**")
            if "retrieved_chunks" in row and pd.notna(row["retrieved_chunks"]):
                chunks = row["retrieved_chunks"].split("\n---\n")
                for chunk in chunks:
                    lines = chunk.strip().split("\n", 1)
                    title = lines[0] if lines else ""
                    text  = lines[1] if len(lines) > 1 else ""
                    with st.expander(title, expanded=False):
                        st.write(text)
            else:
                for title in row["top_passages"].split(" | "):
                    st.write(f"- {title}")
