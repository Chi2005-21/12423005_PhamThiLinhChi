import streamlit as st
import pandas as pd
import joblib
import gzip

# ================================
# Page config
# ================================
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)

# ================================
# Load trained pipeline
# ================================
@st.cache_resource
def load_model():
    with gzip.open("lgbm_pipeline.pkl.gz", "rb") as f:
        model = joblib.load(f)
    return model

model = load_model()

# ================================
# Title & description
# ================================
st.title("💳 Credit Risk Prediction")
st.markdown(
    """
    Ứng dụng dự đoán **rủi ro tín dụng (Credit Default Risk)**  
    dựa trên mô hình **LightGBM** đã được huấn luyện.

    👉 Kết quả trả về **xác suất khách hàng thuộc nhóm rủi ro cao**.
    """
)

# ================================
# Sidebar – Threshold
# ================================
st.sidebar.header("⚙️ Cài đặt dự đoán")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.4,
    step=0.05
)

st.sidebar.markdown(
    f"""
    **Ngưỡng hiện tại:** `{threshold}`  
    - Threshold thấp → ưu tiên **Recall**  
    - Threshold cao → ưu tiên **Precision**
    """
)

# ================================
# Input form
# ================================
st.subheader("📥 Nhập thông tin khách hàng")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Age", 18, 100, 30)
        person_income = st.number_input(
            "Annual Income (USD)", min_value=0, value=50000
        )
        person_emp_length = st.number_input(
            "Employment Length (years)", min_value=0.0, value=3.0
        )
        cb_person_cred_hist_length = st.number_input(
            "Credit History Length (years)", min_value=0.0, value=5.0
        )
        person_home_ownership = st.selectbox(
            "Home Ownership",
            ["RENT", "OWN", "MORTGAGE", "OTHER"]
        )

    with col2:
        loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
        loan_int_rate = st.number_input(
            "Interest Rate (%)", min_value=0.0, value=10.0
        )
        loan_percent_income = st.number_input(
            "Loan / Income Ratio",
            min_value=0.0, max_value=1.0, value=0.2
        )
        loan_intent = st.selectbox(
            "Loan Intent",
            [
                "PERSONAL",
                "MEDICAL",
                "EDUCATION",
                "VENTURE",
                "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION"
            ]
        )

    cb_person_default_on_file = st.selectbox(
        "Previous Default on File",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    loan_grade = st.selectbox(
        "Loan Grade",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: chr(65 + x)  # 0->A, 1->B, ...
    )

    submit = st.form_submit_button("🚀 Predict")

# ================================
# Prediction
# ================================
if submit:
    input_df = pd.DataFrame({
        "person_age": [person_age],
        "person_income": [person_income],
        "person_emp_length": [person_emp_length],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length],
        "loan_amnt": [loan_amnt],
        "loan_int_rate": [loan_int_rate],
        "loan_percent_income": [loan_percent_income],
        "person_home_ownership": [person_home_ownership],
        "loan_intent": [loan_intent],
        "cb_person_default_on_file": [cb_person_default_on_file],
        "loan_grade": [loan_grade],
    })

    # Predict probability
    prob = model.predict_proba(input_df)[0, 1]
    pred = int(prob >= threshold)

    st.markdown("---")
    st.subheader("📊 Kết quả dự đoán")

    st.metric(
        label="Xác suất rủi ro (High Risk Probability)",
        value=f"{prob:.3f}"
    )

    if pred == 1:
        st.error("⚠️ Khách hàng được phân loại: **RỦI RO CAO**")
    else:
        st.success("✅ Khách hàng được phân loại: **RỦI RO THẤP**")

    st.caption(
        "⚠️ Kết quả chỉ mang tính hỗ trợ ra quyết định, không thay thế đánh giá nghiệp vụ."
    )
