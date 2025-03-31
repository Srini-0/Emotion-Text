import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the model
pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

# Emotion dictionary
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Prediction functions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Set page configuration
st.set_page_config(page_title="Text Emotion Detection", layout="centered")

# Apply custom CSS for black background
st.markdown(
    """
    <style>
        /* Base Dark Theme */
        body, .stApp {
            background-color: #1a1a1a;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }

        /* Gradient Headings */
        h1 {
            background: linear-gradient(45deg, #00ff87, #60efff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        /* Neon Accent Borders */
        .stTextArea textarea, .stTextInput input {
            background-color: #2d2d2d !important;
            border: 2px solid #3a3a3a !important;
            border-radius: 10px !important;
            color: #ffffff !important;
            padding: 12px !important;
            transition: all 0.3s ease;
        }

        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: #00ff87 !important;
            box-shadow: 0 0 15px rgba(0,255,135,0.2);
        }

        /* Cyberpunk Button */
        .stButton>button {
            background: linear-gradient(45deg, #ff003c, #c70039) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 32px !important;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(255,0,60,0.3);
        }

        /* Glowing Alert */
        .stAlert {
            background-color: #2d2d2d !important;
            border-left: 4px solid #00ff87 !important;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,255,135,0.1);
        }

        /* Modern Label Style */
        .stTextArea label, .stForm label {
            color: #ffffff !important;
            font-weight: 600;
            letter-spacing: 0.03em;
            opacity: 0.9;
        }

        /* Card-like Sections */
        .stSuccess {
            background-color: #2d2d2d !important;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}: {}".format(prediction, emoji_icon))
            st.write("Confidence: {:.2f}".format(np.max(probability)))

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions', 
                y='probability', 
                color='emotions'
            )
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
