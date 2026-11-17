
import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Water Leakage Detection",
    page_icon="💧",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "water_leakage_model.pkl"
)

METADATA_FILE = os.path.join(
    BASE_DIR,
    "model_metadata.json"
)

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_FILE)

    with open(METADATA_FILE, "r") as file:
        metadata = json.load(file)

    return model, metadata


try:
    model, metadata = load_model()

except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


st.title("💧 Water Leakage Detection System")

st.write(
    "Enter the pipeline operating parameters "
    "to predict a possible leakage condition."
)

st.info(
    "This application uses a trained machine learning "
    "model to classify pipeline conditions."
)

st.divider()

numerical_features = metadata["numerical_features"]
categorical_features = metadata["categorical_features"]

numerical_defaults = metadata["numerical_defaults"]
categorical_options = metadata["categorical_options"]

user_input = {}

col1, col2 = st.columns(2)

for index, feature in enumerate(numerical_features):

    container = col1 if index % 2 == 0 else col2

    with container:

        default_value = numerical_defaults[feature]

        user_input[feature] = st.number_input(
            feature,
            value=float(default_value),
            format="%.4f"
        )


for index, feature in enumerate(categorical_features):

    container = col1 if index % 2 == 0 else col2

    with container:

        options = categorical_options[feature]

        if not options:
            st.error(f"No available options for {feature}")
            st.stop()

        user_input[feature] = st.selectbox(
            feature,
            options=options
        )


st.divider()

if st.button(
    "Predict Leakage Status",
    type="primary",
    use_container_width=True
):

    input_df = pd.DataFrame(
        [user_input],
        columns=metadata["input_features"]
    )

    probability = model.predict_proba(
        input_df
    )[0][1]

    threshold = metadata["threshold"]

    st.subheader("Prediction Result")

    st.metric(
        "Predicted Leakage Probability",
        f"{probability * 100:.2f}%"
    )

    if probability >= threshold:

        st.error("⚠️ Possible Leakage Detected!")

        st.write(
            "The model classifies these pipeline conditions "
            "as possible leakage."
        )

    else:

        st.success("✅ No Leakage Detected")

        st.write(
            "The model classifies these pipeline conditions "
            "as no leakage."
        )

    st.caption(
        "This is a machine learning prediction, not a confirmed "
        "physical inspection. Actual leakage must be verified "
        "using appropriate monitoring and inspection methods."
    )


with st.sidebar:

    st.title("About the Project")

    st.write(
        "Water Leakage Detection using Machine Learning."
    )

    st.write("Algorithms evaluated:")

    st.write(
        "- Decision Tree\n"
        "- Random Forest\n"
        "- AdaBoost"
    )

    st.write(
        f"Selected Model: {metadata['model_name']}"
    )

    st.write(
        "Target classes:\n"
        "- 0: No Leakage\n"
        "- 1: Leakage"
    )

    st.divider()

    st.write(
        "Academic Machine Learning Project"
    )
