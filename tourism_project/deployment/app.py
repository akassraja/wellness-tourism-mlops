import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Visit with Us — Wellness Tourism Predictor",
    page_icon="✈️",
    layout="wide",
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_tourism_model_v1.joblib",
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Ensure the MLOps pipeline has trained and committed the model."
        )
    return joblib.load(MODEL_PATH)


st.title("Visit with Us — Wellness Tourism Package Predictor")
st.markdown(
    "Predict whether a customer will purchase the **Wellness Tourism Package** "
    "before initiating a sales outreach campaign."
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load trained model: {e}")
    st.stop()

with st.form("prediction_form"):
    st.subheader("1. Customer Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=90, value=36, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox(
            "Marital Status", ["Married", "Single", "Divorced", "Unmarried"]
        )
        city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)

    with col2:
        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Small Business", "Large Business", "Free Lancer"],
        )
        designation = st.selectbox(
            "Designation",
            ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
        )
        monthly_income = st.number_input(
            "Monthly Income ($)",
            min_value=1000.0,
            max_value=200000.0,
            value=22500.0,
            step=500.0,
        )
        own_car_label = st.selectbox("Owns a Car?", ["Yes", "No"])

    with col3:
        passport_label = st.selectbox("Holds Valid Passport?", ["Yes", "No"], index=1)
        num_person_visiting = st.number_input(
            "Total People Visiting", min_value=1, max_value=10, value=3, step=1
        )
        num_children_visiting = st.number_input(
            "Children Under Age 5 Visiting", min_value=0, max_value=5, value=1, step=1
        )
        num_trips = st.number_input(
            "Average Annual Trips", min_value=0, max_value=30, value=3, step=1
        )
        preferred_property_star = st.selectbox(
            "Preferred Hotel Star Rating", [3.0, 4.0, 5.0], index=0
        )

    st.subheader("2. Customer Interaction Data")
    icol1, icol2, icol3 = st.columns(3)

    with icol1:
        type_of_contact = st.selectbox(
            "Type of Contact", ["Self Enquiry", "Company Invited"]
        )
        product_pitched = st.selectbox(
            "Product Pitched",
            ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
        )

    with icol2:
        duration_of_pitch = st.number_input(
            "Duration of Pitch (minutes)",
            min_value=1.0,
            max_value=150.0,
            value=14.0,
            step=1.0,
        )
        num_followups = st.number_input(
            "Number of Follow-ups", min_value=1, max_value=10, value=4, step=1
        )

    with icol3:
        pitch_satisfaction_score = st.selectbox(
            "Pitch Satisfaction Score (1–5)", [1, 2, 3, 4, 5], index=2
        )

    submitted = st.form_submit_button("Predict Purchase Likelihood", use_container_width=True)

if submitted:
    # Assemble user inputs into a DataFrame matching the training feature schema
    input_df = pd.DataFrame(
        [
            {
                "Age": float(age),
                "TypeofContact": type_of_contact,
                "CityTier": int(city_tier),
                "DurationOfPitch": float(duration_of_pitch),
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": int(num_person_visiting),
                "NumberOfFollowups": float(num_followups),
                "ProductPitched": product_pitched,
                "PreferredPropertyStar": float(preferred_property_star),
                "MaritalStatus": marital_status,
                "NumberOfTrips": float(num_trips),
                "Passport": 1 if passport_label == "Yes" else 0,
                "PitchSatisfactionScore": int(pitch_satisfaction_score),
                "OwnCar": 1 if own_car_label == "Yes" else 0,
                "NumberOfChildrenVisiting": float(num_children_visiting),
                "Designation": designation,
                "MonthlyIncome": float(monthly_income),
            }
        ]
    )

    prediction = int(model.predict(input_df)[0])
    purchase_prob = float(model.predict_proba(input_df)[0][1])

    st.divider()
    st.subheader("Prediction Result")

    res_col1, res_col2 = st.columns([2, 1])
    with res_col1:
        if prediction == 1:
            st.success(
                "**High Potential Buyer (ProdTaken = 1):** This customer is likely to "
                "purchase the Wellness Tourism Package. Prioritize outreach!"
            )
        else:
            st.warning(
                "**Low Purchase Likelihood (ProdTaken = 0):** This customer is unlikely to "
                "purchase the Wellness Tourism Package under current parameters."
            )
    with res_col2:
        st.metric("Purchase Probability", f"{purchase_prob * 100:.1f}%")

    with st.expander("View Input DataFrame Sent to Model"):
        st.dataframe(input_df, use_container_width=True)
