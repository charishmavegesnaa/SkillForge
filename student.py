import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("student_logreg_model.pkl")
    except Exception as e:
        st.error("❌ Failed to load model file")
        st.code(str(e))
        st.stop()

model_data = load_model()

# ---------------- Title ----------------
st.markdown(
    '<h1 class="main-header">🎓 Student Performance Predictor</h1>',
    unsafe_allow_html=True
)

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("📊 Student Information")

age = st.sidebar.slider("Age", 15, 22, 17)
studytime = st.sidebar.slider("Study Time (1–4)", 1, 4, 2)
failures = st.sidebar.slider("Past Failures (0–3)", 0, 3, 0)
absences = st.sidebar.slider("Absences (0–32)", 0, 32, 4)
freetime = st.sidebar.slider("Free Time (1–5)", 1, 5, 3)
goout = st.sidebar.slider("Going Out (1–5)", 1, 5, 2)
dalc = st.sidebar.slider("Workday Alcohol (1–5)", 1, 5, 1)
walc = st.sidebar.slider("Weekend Alcohol (1–5)", 1, 5, 2)
g1 = st.sidebar.slider("G1 Grade (0–20)", 0, 20, 12)
g2 = st.sidebar.slider("G2 Grade (0–20)", 0, 20, 13)

# ---------------- Prediction ----------------
if st.sidebar.button("🔮 Predict Learning Level", type="primary"):

    student_data = {
        "age": age,
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "freetime": freetime,
        "goout": goout,
        "Dalc": dalc,
        "Walc": walc,
        "G1": g1,
        "G2": g2
    }

    df = pd.DataFrame([student_data])

    X = df[model_data["features"]]
    X_poly = model_data["poly_features"].transform(X)
    X_scaled = model_data["scaler"].transform(X_poly)

    pred = model_data["model"].predict(X_scaled)[0]
    prob = model_data["model"].predict_proba(X_scaled)[0]

    level = model_data["target_mapping"][pred]

    st.success(f"🎯 Predicted Learning Level: **{level}**")
    st.info(f"📊 Model Accuracy: **{model_data['accuracy']:.2%}**")

    prob_df = pd.DataFrame({
        "Level": model_data["target_mapping"].values(),
        "Probability": prob
    })

    st.bar_chart(prob_df.set_index("Level"))

    st.subheader("👤 Student Profile")
    st.dataframe(df, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Developed by INFERRIX")

