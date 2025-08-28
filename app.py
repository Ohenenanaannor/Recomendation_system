import streamlit as st
import pandas as pd
import joblib

# ===== LOAD TRAINED MODEL AND ENCODERS =====
@st.cache_resource
def load_models():
    rf_model = joblib.load("rf_model_small.joblib")
    label_encoders = joblib.load("label_encoders.pkl")
    return rf_model, label_encoders

rf_model, label_encoders = load_models()

feature_cols = ["last_view_cat", "most_freq_cat", "unique_cats_viewed", "total_views_before_cart"]

# ===== PREDICTION FUNCTION =====
def recommend_category(last_view_cat, most_freq_cat, unique_cats_viewed, total_views_before_cart):
    last_view_encoded = label_encoders["last_view_cat"].transform([str(last_view_cat)])[0]
    most_freq_encoded = label_encoders["most_freq_cat"].transform([str(most_freq_cat)])[0]

    input_df = pd.DataFrame([[last_view_encoded, most_freq_encoded,
                              unique_cats_viewed, total_views_before_cart]],
                            columns=feature_cols)

    pred_encoded = rf_model.predict(input_df)[0]
    pred_category = label_encoders["target_category"].inverse_transform([pred_encoded])[0]

    proba_df = pd.DataFrame(rf_model.predict_proba(input_df), columns=rf_model.classes_)
    top3 = proba_df.T.sort_values(by=0, ascending=False).head(3)
    top3_results = [
        (label_encoders["target_category"].inverse_transform([idx])[0], f"{prob*100:.2f}%")
        for idx, prob in zip(top3.index, top3[0])
    ]
    return pred_category, top3_results

# ===== STREAMLIT UI =====
st.set_page_config(page_title="E-commerce Recommender", page_icon="🛒", layout="wide")
st.title("🛒 Recommendation System")

# Only Task 1 Tab
st.subheader("Predict Category of Item Added to Cart")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        last_view_cat = st.text_input("Last Viewed Category ID", "")
    with col2:
        most_freq_cat = st.text_input("Most Frequently Viewed Category ID", "")

    col3, col4 = st.columns(2)
    with col3:
        unique_cats_viewed = st.number_input("Number of Unique Categories Viewed", min_value=1, value=3)
    with col4:
        total_views_before_cart = st.number_input("Total Views Before Add-to-Cart", min_value=1, value=5)

    submit = st.form_submit_button("Predict")

if submit:
    try:
        pred_category, top3_results = recommend_category(last_view_cat, most_freq_cat, unique_cats_viewed, total_views_before_cart)
        st.success(f"**Predicted Category:** {pred_category}")
        st.markdown("### 📊 Top 3 Predictions")
        st.table(pd.DataFrame(top3_results, columns=["Category", "Probability"]))
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
