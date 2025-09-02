import streamlit as st
import requests
import plotly.express as px

st.set_page_config(page_title="Text Summarizer & Sentiment Analyzer", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stTextInput>div>div>input {
        color: white;
    }
    .stTextArea>div>textarea {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

API_URL = "https://huggingface-api.onrender.com"

st.title("Text Summarizer & Sentiment Analyzer")

task = st.selectbox("Select Task", ["Summarization", "Sentiment"])

examples = {
    "Summarization": "NASA's Artemis program aims to land the first woman and next man on the Moon by 2024, paving the way for sustainable lunar exploration and missions to Mars.",
    "Sentiment": "I absolutely love the new features of this product. It's intuitive and powerful!"
}

if "text_input" not in st.session_state:
    st.session_state.text_input = ""

if st.button("🧪 Load Example Text"):
    st.session_state.text_input = examples[task]

text_input = st.text_area("Enter your text", value=st.session_state.text_input, height=200)

if st.button("🔍 Analyze"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        try:
            if task == "Summarization":
                response = requests.post(f"{API_URL}/summarize", json={"text": text_input})
                if response.status_code == 200:
                    summary = response.json()["summary"]
                    st.success("✅ Summary:")
                    st.write(summary)
                else:
                    st.error(f"❌ Error fetching summary. Status code: {response.status_code}")

            elif task == "Sentiment":
                response = requests.post(f"{API_URL}/sentiment", json={"text": text_input})
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result["sentiment"]
                    scores = result["confidence_scores"]

                    st.success(f"🧠 Sentiment: **{sentiment.upper()}**")
                    st.write("Confidence Scores:")
                    st.json(scores)

                    fig = px.bar(
                        x=list(scores.keys()),
                        y=list(scores.values()),
                        labels={'x': 'Sentiment Class', 'y': 'Confidence'},
                        title="Sentiment Confidence Scores",
                        color=list(scores.keys()),
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                    fig.update_layout(showlegend=False, yaxis_range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"❌ Error fetching sentiment. Status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("🚫 Could not connect to FastAPI backend. Is the server running?")
