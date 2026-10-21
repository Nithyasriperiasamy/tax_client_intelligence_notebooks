import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    accuracy_score,
    roc_auc_score
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Opportunity Intelligence",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 16px;
        color: #666666;
        margin-bottom: 25px;
    }

    .model-header {
        font-size: 28px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .model-description {
        color: #666666;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .result-box {
        padding: 18px;
        border: 1px solid #dddddd;
        border-radius: 10px;
        background-color: #fafafa;
    }

    .insight-box {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        background-color: #f8f8f8;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA
# ============================================================

DATA_PATH = "client_data.csv"


@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


try:

    df = load_data()

except Exception as e:

    st.error("Unable to load client_data.csv")
    st.error(str(e))
    st.stop()


# ============================================================
# FEATURE LISTS
# ============================================================

LINEAR_FEATURES = [
    "inquiry_source_score",
    "mentioned_business_income",
    "response_time_hours",
    "requested_pricing_info",
    "lead_hot",
    "years_as_client",
    "filing_complexity_score",
    "addons_purchased",
    "referrals_given",
    "num_income_sources",
    "lifetime_revenue",
    "past_filing_errors",
    "time_spent_selfiling_minutes",
    "deductions_claimed_count",
    "dependents_count",
    "support_contacts_count",
    "premium_upgrade",
    "acquisition_channel_cost",
    "onboarding_hours",
    "expected_annual_fee",
    "expected_retention_years",
    "payback_period_months"
]


LOGISTIC_FEATURES = [
    "inquiry_source_score",
    "mentioned_business_income",
    "response_time_hours",
    "requested_pricing_info",
    "lead_hot",
    "years_as_client",
    "filing_complexity_score",
    "addons_purchased",
    "referrals_given",
    "num_income_sources",
    "past_filing_errors",
    "time_spent_selfiling_minutes",
    "deductions_claimed_count",
    "dependents_count",
    "support_contacts_count",
    "acquisition_channel_cost",
    "onboarding_hours",
    "expected_annual_fee"
]


RIDGE_FEATURES = [
    "acquisition_channel_cost",
    "onboarding_hours",
    "expected_annual_fee",
    "expected_retention_years"
]


# ============================================================
# TARGETS
# ============================================================

LINEAR_TARGET = "next_year_income"
LOGISTIC_TARGET = "premium_upgrade"
RIDGE_TARGET = "payback_period_months"


# ============================================================
# CHECK DATASET
# ============================================================

required_columns = list(
    set(
        LINEAR_FEATURES
        + LOGISTIC_FEATURES
        + RIDGE_FEATURES
        + [
            LINEAR_TARGET,
            LOGISTIC_TARGET,
            RIDGE_TARGET
        ]
    )
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error("Missing columns in client_data.csv")

    for column in missing_columns:
        st.write(f"- {column}")

    st.stop()


# ============================================================
# LINEAR REGRESSION
# REVENUE GROWTH INTELLIGENCE
# ============================================================

@st.cache_resource
def train_linear_model(data):

    X = data[
        LINEAR_FEATURES
    ].astype(float).values

    y = data[
        LINEAR_TARGET
    ].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    mean = X_train.mean(axis=0)

    std = X_train.std(axis=0)

    std[std == 0] = 1.0

    X_train_scaled = (
        X_train - mean
    ) / std

    X_test_scaled = (
        X_test - mean
    ) / std

    X_train_bias = np.c_[
        np.ones(X_train_scaled.shape[0]),
        X_train_scaled
    ]

    X_test_bias = np.c_[
        np.ones(X_test_scaled.shape[0]),
        X_test_scaled
    ]

    weights = np.linalg.solve(
        X_train_bias.T @ X_train_bias,
        X_train_bias.T @ y_train
    )

    train_prediction = (
        X_train_bias @ weights
    )

    test_prediction = (
        X_test_bias @ weights
    )

    train_r2 = r2_score(
        y_train,
        train_prediction
    )

    test_r2 = r2_score(
        y_test,
        test_prediction
    )

    test_mae = mean_absolute_error(
        y_test,
        test_prediction
    )

    return {
        "weights": weights,
        "mean": mean,
        "std": std,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_mae": test_mae
    }


linear_model = train_linear_model(df)


# ============================================================
# LOGISTIC REGRESSION
# PREMIUM CONVERSION INTELLIGENCE
# ============================================================

@st.cache_resource
def train_logistic_model(data):

    X = data[
        LOGISTIC_FEATURES
    ].astype(float).values

    y = data[
        LOGISTIC_TARGET
    ].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy,
        "auc": auc
    }


logistic_model = train_logistic_model(df)


# ============================================================
# PERCEPTRON
# PREMIUM CONVERSION INTELLIGENCE
# ============================================================

@st.cache_resource
def train_perceptron_model(data):

    X = data[
        LOGISTIC_FEATURES
    ].astype(float).values

    y = data[
        LOGISTIC_TARGET
    ].astype(int).values

    y = np.where(
        y == 1,
        1,
        -1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    weights = np.zeros(
        X_train_scaled.shape[1]
    )

    bias = 0.0

    learning_rate = 0.01
    epochs = 50

    for epoch in range(epochs):

        for i in range(
            len(X_train_scaled)
        ):

            linear_output = (
                np.dot(
                    X_train_scaled[i],
                    weights
                )
                + bias
            )

            prediction = (
                1
                if linear_output >= 0
                else -1
            )

            if prediction != y_train[i]:

                weights += (
                    learning_rate
                    * y_train[i]
                    * X_train_scaled[i]
                )

                bias += (
                    learning_rate
                    * y_train[i]
                )

    test_predictions = []

    for row in X_test_scaled:

        output = (
            np.dot(
                row,
                weights
            )
            + bias
        )

        prediction = (
            1
            if output >= 0
            else -1
        )

        test_predictions.append(
            prediction
        )

    accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    return {
        "weights": weights,
        "bias": bias,
        "scaler": scaler,
        "accuracy": accuracy
    }


perceptron_model = train_perceptron_model(df)


# ============================================================
# RIDGE REGRESSION
# ACQUISITION PAYBACK INTELLIGENCE
# ============================================================

@st.cache_resource
def train_ridge_model(data):

    X = data[
        RIDGE_FEATURES
    ].astype(float).values

    y = data[
        RIDGE_TARGET
    ].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = Ridge(
        alpha=1.0
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    predictions = np.maximum(
        predictions,
        0
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    return {
        "model": model,
        "scaler": scaler,
        "r2": r2,
        "mae": mae
    }


ridge_model = train_ridge_model(df)


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_income(values):

    X = np.array(
        [
            values[feature]
            for feature in LINEAR_FEATURES
        ],
        dtype=float
    ).reshape(1, -1)

    X_scaled = (
        X - linear_model["mean"]
    ) / linear_model["std"]

    X_bias = np.c_[
        np.ones(1),
        X_scaled
    ]

    prediction = (
        X_bias @ linear_model["weights"]
    )[0]

    return max(
        float(prediction),
        0
    )


def predict_logistic(values):

    X = np.array(
        [
            values[feature]
            for feature in LOGISTIC_FEATURES
        ],
        dtype=float
    ).reshape(1, -1)

    X_scaled = (
        logistic_model["scaler"]
        .transform(X)
    )

    probability = (
        logistic_model["model"]
        .predict_proba(X_scaled)[0][1]
    )

    prediction = (
        1
        if probability >= 0.5
        else 0
    )

    return (
        float(probability),
        prediction
    )


def predict_perceptron(values):

    X = np.array(
        [
            values[feature]
            for feature in LOGISTIC_FEATURES
        ],
        dtype=float
    ).reshape(1, -1)

    X_scaled = (
        perceptron_model["scaler"]
        .transform(X)
    )

    output = (
        np.dot(
            X_scaled[0],
            perceptron_model["weights"]
        )
        + perceptron_model["bias"]
    )

    return (
        1
        if output >= 0
        else 0
    )


def predict_payback(values):

    X = np.array(
        [
            values[feature]
            for feature in RIDGE_FEATURES
        ],
        dtype=float
    ).reshape(1, -1)

    X_scaled = (
        ridge_model["scaler"]
        .transform(X)
    )

    prediction = (
        ridge_model["model"]
        .predict(X_scaled)[0]
    )

    return max(
        float(prediction),
        0
    )


# ============================================================
# BUSINESS INTERPRETATION FUNCTIONS
# ============================================================

def income_interpretation(change):

    if change >= 20:
        return "Strong revenue growth opportunity"

    elif change >= 5:
        return "Positive revenue growth opportunity"

    elif change >= -5:
        return "Stable revenue outlook"

    else:
        return "Revenue decline signal"


def premium_interpretation(
    probability,
    perceptron_signal
):

    if (
        probability >= 0.70
        and perceptron_signal == 1
    ):

        return "Strong premium conversion signal"

    elif (
        probability >= 0.50
        or perceptron_signal == 1
    ):

        return "Moderate premium conversion opportunity"

    else:

        return "Lower premium conversion signal"


def payback_interpretation(
    months
):

    if months <= 6:

        return "Fast investment recovery"

    elif months <= 12:

        return "Reasonable investment recovery"

    elif months <= 24:

        return "Longer investment recovery"

    else:

        return "Extended investment recovery"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Customer Opportunity Intelligence"
)

st.sidebar.caption(
    "ML-powered customer intelligence platform"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "Executive Dashboard",
        "Revenue Growth Intelligence",
        "Premium Conversion Intelligence",
        "Acquisition Payback Intelligence",
        "Customer 360"
    ]
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.markdown(
        '<div class="main-title">'
        'Customer Opportunity Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Customer revenue forecasting, premium conversion analysis, '
        'and acquisition payback intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Customers",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Average Next-Year Income",
            f"₹{df['next_year_income'].mean():,.0f}"
        )

    with col3:

        st.metric(
            "Premium Conversion",
            f"{df['premium_upgrade'].mean() * 100:.1f}%"
        )

    with col4:

        st.metric(
            "Average Payback",
            f"{df['payback_period_months'].mean():.1f} months"
        )

    st.divider()

    st.subheader(
        "Machine Learning Modules"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            "### Revenue Growth Intelligence"
        )

        st.write(
            "Linear Regression"
        )

        st.write(
            "Predicts next-year customer income."
        )

        st.metric(
            "Test R²",
            f"{linear_model['test_r2']:.3f}"
        )

    with col2:

        st.markdown(
            "### Premium Conversion Intelligence"
        )

        st.write(
            "Logistic Regression + Perceptron"
        )

        st.write(
            "Identifies premium conversion signals."
        )

        st.metric(
            "Logistic AUC",
            f"{logistic_model['auc']:.3f}"
        )

    with col3:

        st.markdown(
            "### Acquisition Payback Intelligence"
        )

        st.write(
            "Ridge Regression"
        )

        st.write(
            "Predicts acquisition investment recovery time."
        )

        st.metric(
            "Test R²",
            f"{ridge_model['r2']:.3f}"
        )

    st.divider()

    st.subheader(
        "Model Performance"
    )

    performance = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Logistic Regression",
                "Perceptron",
                "Ridge Regression"
            ],
            "Metric": [
                "R²",
                "ROC AUC",
                "Accuracy",
                "R²"
            ],
            "Value": [
                round(linear_model["test_r2"], 3),
                round(logistic_model["auc"], 3),
                round(perceptron_model["accuracy"], 3),
                round(ridge_model["r2"], 3)
            ]
        }
    )

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# REVENUE GROWTH INTELLIGENCE
# ============================================================

elif page == "Revenue Growth Intelligence":

    st.markdown(
        '<div class="model-header">'
        'Revenue Growth Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="model-description">'
        'Linear Regression | Predicts Next-Year Income'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader(
        "Customer Revenue Information"
    )

    with st.form("linear_prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            inquiry_source_score = st.number_input(
                "Inquiry Source Score",
                min_value=0.0,
                value=5.0,
                step=0.1
            )

            mentioned_business_income = st.number_input(
                "Mentioned Business Income",
                min_value=0.0,
                value=5000.0,
                step=100.0
            )

            response_time_hours = st.number_input(
                "Response Time (Hours)",
                min_value=0.0,
                value=4.0,
                step=0.5
            )

            requested_pricing_info = st.selectbox(
                "Requested Pricing Information",
                [0, 1],
                format_func=lambda x:
                    "Yes" if x == 1 else "No"
            )

            lead_hot = st.selectbox(
                "Lead Status",
                [0, 1],
                format_func=lambda x:
                    "Hot Lead" if x == 1 else "Normal Lead"
            )

            years_as_client = st.number_input(
                "Years as Client",
                min_value=0.0,
                value=3.0,
                step=0.5
            )

            filing_complexity_score = st.number_input(
                "Filing Complexity Score",
                min_value=0.0,
                value=5.0,
                step=0.1
            )

        with col2:

            addons_purchased = st.number_input(
                "Add-ons Purchased",
                min_value=0,
                value=1,
                step=1
            )

            referrals_given = st.number_input(
                "Referrals Given",
                min_value=0,
                value=1,
                step=1
            )

            num_income_sources = st.number_input(
                "Number of Income Sources",
                min_value=1,
                value=2,
                step=1
            )

            lifetime_revenue = st.number_input(
                "Lifetime Revenue",
                min_value=0.0,
                value=10000.0,
                step=500.0
            )

            past_filing_errors = st.number_input(
                "Past Filing Errors",
                min_value=0,
                value=0,
                step=1
            )

            time_spent_selfiling_minutes = st.number_input(
                "Self-Filing Time (Minutes)",
                min_value=0.0,
                value=60.0,
                step=5.0
            )

            deductions_claimed_count = st.number_input(
                "Deductions Claimed",
                min_value=0,
                value=5,
                step=1
            )

        with col3:

            dependents_count = st.number_input(
                "Dependents",
                min_value=0,
                value=1,
                step=1
            )

            support_contacts_count = st.number_input(
                "Support Contacts",
                min_value=0,
                value=2,
                step=1
            )

            premium_upgrade = st.selectbox(
                "Current Premium Status",
                [0, 1],
                format_func=lambda x:
                    "Premium" if x == 1 else "Standard"
            )

            acquisition_channel_cost = st.number_input(
                "Acquisition Channel Cost",
                min_value=0.0,
                value=100.0,
                step=10.0
            )

            onboarding_hours = st.number_input(
                "Onboarding Hours",
                min_value=0.0,
                value=2.0,
                step=0.5
            )

            expected_annual_fee = st.number_input(
                "Expected Annual Fee",
                min_value=0.0,
                value=5000.0,
                step=100.0
            )

            expected_retention_years = st.number_input(
                "Expected Retention Years",
                min_value=0.1,
                value=3.0,
                step=0.5
            )

            payback_period_months = st.number_input(
                "Observed Payback Period (Months)",
                min_value=0.0,
                value=10.0,
                step=0.1
            )

        submitted = st.form_submit_button(
            "Predict Next-Year Income",
            use_container_width=True
        )

    if submitted:

        values = {
            "inquiry_source_score": inquiry_source_score,
            "mentioned_business_income": mentioned_business_income,
            "response_time_hours": response_time_hours,
            "requested_pricing_info": requested_pricing_info,
            "lead_hot": lead_hot,
            "years_as_client": years_as_client,
            "filing_complexity_score": filing_complexity_score,
            "addons_purchased": addons_purchased,
            "referrals_given": referrals_given,
            "num_income_sources": num_income_sources,
            "lifetime_revenue": lifetime_revenue,
            "past_filing_errors": past_filing_errors,
            "time_spent_selfiling_minutes": time_spent_selfiling_minutes,
            "deductions_claimed_count": deductions_claimed_count,
            "dependents_count": dependents_count,
            "support_contacts_count": support_contacts_count,
            "premium_upgrade": premium_upgrade,
            "acquisition_channel_cost": acquisition_channel_cost,
            "onboarding_hours": onboarding_hours,
            "expected_annual_fee": expected_annual_fee,
            "expected_retention_years": expected_retention_years,
            "payback_period_months": payback_period_months
        }

        prediction = predict_income(
            values
        )

        difference = (
            prediction
            - lifetime_revenue
        )

        if lifetime_revenue != 0:

            percentage = (
                difference
                / lifetime_revenue
            ) * 100

        else:

            percentage = 0

        st.divider()

        st.subheader(
            "Linear Regression Prediction"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Next-Year Income",
                f"₹{prediction:,.0f}"
            )

        with col2:

            st.metric(
                "Revenue Difference",
                f"₹{difference:,.0f}"
            )

        with col3:

            st.metric(
                "Change vs Lifetime Revenue",
                f"{percentage:.1f}%"
            )

        st.divider()

        if percentage >= 20:

            outlook = "Strong revenue growth opportunity"

        elif percentage >= 5:

            outlook = "Positive revenue growth opportunity"

        elif percentage >= -5:

            outlook = "Stable revenue outlook"

        else:

            outlook = "Revenue decline signal"

        st.info(
            f"Revenue Outlook: {outlook}"
        )


# ============================================================
# PREMIUM CONVERSION INTELLIGENCE
# ============================================================

elif page == "Premium Conversion Intelligence":

    st.markdown(
        '<div class="model-header">'
        'Premium Conversion Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="model-description">'
        'Logistic Regression + Perceptron | Predicts Premium Conversion'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader(
        "Customer Conversion Information"
    )

    with st.form("premium_prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            inquiry_source_score = st.number_input(
                "Inquiry Source Score",
                min_value=0.0,
                value=5.0,
                step=0.1
            )

            mentioned_business_income = st.number_input(
                "Mentioned Business Income",
                min_value=0.0,
                value=5000.0,
                step=100.0
            )

            response_time_hours = st.number_input(
                "Response Time (Hours)",
                min_value=0.0,
                value=4.0,
                step=0.5
            )

            requested_pricing_info = st.selectbox(
                "Requested Pricing Information",
                [0, 1],
                format_func=lambda x:
                    "Yes" if x == 1 else "No"
            )

            lead_hot = st.selectbox(
                "Lead Status",
                [0, 1],
                format_func=lambda x:
                    "Hot Lead" if x == 1 else "Normal Lead"
            )

            years_as_client = st.number_input(
                "Years as Client",
                min_value=0.0,
                value=3.0,
                step=0.5
            )

        with col2:

            filing_complexity_score = st.number_input(
                "Filing Complexity Score",
                min_value=0.0,
                value=5.0,
                step=0.1
            )

            addons_purchased = st.number_input(
                "Add-ons Purchased",
                min_value=0,
                value=1,
                step=1
            )

            referrals_given = st.number_input(
                "Referrals Given",
                min_value=0,
                value=1,
                step=1
            )

            num_income_sources = st.number_input(
                "Number of Income Sources",
                min_value=1,
                value=2,
                step=1
            )

            past_filing_errors = st.number_input(
                "Past Filing Errors",
                min_value=0,
                value=0,
                step=1
            )

            time_spent_selfiling_minutes = st.number_input(
                "Self-Filing Time (Minutes)",
                min_value=0.0,
                value=60.0,
                step=5.0
            )

        with col3:

            deductions_claimed_count = st.number_input(
                "Deductions Claimed",
                min_value=0,
                value=5,
                step=1
            )

            dependents_count = st.number_input(
                "Dependents",
                min_value=0,
                value=1,
                step=1
            )

            support_contacts_count = st.number_input(
                "Support Contacts",
                min_value=0,
                value=2,
                step=1
            )

            acquisition_channel_cost = st.number_input(
                "Acquisition Channel Cost",
                min_value=0.0,
                value=100.0,
                step=10.0
            )

            onboarding_hours = st.number_input(
                "Onboarding Hours",
                min_value=0.0,
                value=2.0,
                step=0.5
            )

            expected_annual_fee = st.number_input(
                "Expected Annual Fee",
                min_value=0.0,
                value=5000.0,
                step=100.0
            )

        submitted = st.form_submit_button(
            "Predict Premium Conversion",
            use_container_width=True
        )

    if submitted:

        values = {
            "inquiry_source_score": inquiry_source_score,
            "mentioned_business_income": mentioned_business_income,
            "response_time_hours": response_time_hours,
            "requested_pricing_info": requested_pricing_info,
            "lead_hot": lead_hot,
            "years_as_client": years_as_client,
            "filing_complexity_score": filing_complexity_score,
            "addons_purchased": addons_purchased,
            "referrals_given": referrals_given,
            "num_income_sources": num_income_sources,
            "past_filing_errors": past_filing_errors,
            "time_spent_selfiling_minutes": time_spent_selfiling_minutes,
            "deductions_claimed_count": deductions_claimed_count,
            "dependents_count": dependents_count,
            "support_contacts_count": support_contacts_count,
            "acquisition_channel_cost": acquisition_channel_cost,
            "onboarding_hours": onboarding_hours,
            "expected_annual_fee": expected_annual_fee
        }

        # ====================================================
        # LOGISTIC REGRESSION
        # ====================================================

        probability, logistic_prediction = (
            predict_logistic(values)
        )

        # ====================================================
        # PERCEPTRON
        # ====================================================

        perceptron_prediction = (
            predict_perceptron(values)
        )

        st.divider()

        # ====================================================
        # LOGISTIC SECTION
        # ====================================================

        st.subheader(
            "Logistic Regression"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Premium Conversion Probability",
                f"{probability * 100:.1f}%"
            )

        with col2:

            st.metric(
                "Prediction",
                "Premium Conversion"
                if logistic_prediction == 1
                else "Lower Conversion Signal"
            )

        st.progress(
            float(probability)
        )

        st.divider()

        # ====================================================
        # PERCEPTRON SECTION
        # ====================================================

        st.subheader(
            "Perceptron"
        )

        if perceptron_prediction == 1:

            perceptron_text = (
                "Positive Premium Conversion Signal"
            )

        else:

            perceptron_text = (
                "Negative Premium Conversion Signal"
            )

        st.metric(
            "Perceptron Prediction",
            perceptron_text
        )

        st.divider()

        # ====================================================
        # COMPARISON
        # ====================================================

        st.subheader(
            "Premium Conversion Model Comparison"
        )

        comparison = pd.DataFrame(
            {
                "Model": [
                    "Logistic Regression",
                    "Perceptron"
                ],
                "Output": [
                    f"{probability * 100:.1f}% probability",
                    (
                        "Positive Signal"
                        if perceptron_prediction == 1
                        else "Negative Signal"
                    )
                ]
            }
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        if (
            probability >= 0.70
            and perceptron_prediction == 1
        ):

            insight = (
                "Strong premium conversion signal"
            )

        elif (
            probability >= 0.50
            or perceptron_prediction == 1
        ):

            insight = (
                "Moderate premium conversion opportunity"
            )

        else:

            insight = (
                "Lower premium conversion signal"
            )

        st.info(
            f"Premium Conversion Insight: {insight}"
        )


# ============================================================
# ACQUISITION PAYBACK INTELLIGENCE
# ============================================================

elif page == "Acquisition Payback Intelligence":

    st.markdown(
        '<div class="model-header">'
        'Acquisition Payback Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="model-description">'
        'Ridge Regression | Predicts Customer Acquisition Payback Period'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader(
        "Acquisition & Financial Information"
    )

    with st.form("ridge_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            acquisition_channel_cost = st.number_input(
                "Acquisition Channel Cost",
                min_value=0.0,
                value=100.0,
                step=10.0
            )

            onboarding_hours = st.number_input(
                "Onboarding Hours",
                min_value=0.0,
                value=2.0,
                step=0.5
            )

        with col2:

            expected_annual_fee = st.number_input(
                "Expected Annual Fee",
                min_value=0.0,
                value=5000.0,
                step=100.0
            )

            expected_retention_years = st.number_input(
                "Expected Retention Years",
                min_value=0.1,
                value=3.0,
                step=0.5
            )

        submitted = st.form_submit_button(
            "Predict Payback Period",
            use_container_width=True
        )

    if submitted:

        values = {
            "acquisition_channel_cost":
                acquisition_channel_cost,

            "onboarding_hours":
                onboarding_hours,

            "expected_annual_fee":
                expected_annual_fee,

            "expected_retention_years":
                expected_retention_years
        }

        prediction = predict_payback(
            values
        )

        st.divider()

        st.subheader(
            "Ridge Regression Prediction"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Predicted Payback Period",
                f"{prediction:.2f} months"
            )

        with col2:

            st.metric(
                "Payback in Years",
                f"{prediction / 12:.2f} years"
            )

        with col3:

            st.metric(
                "Expected Annual Fee",
                f"₹{expected_annual_fee:,.0f}"
            )

        st.divider()

        if prediction <= 6:

            outlook = (
                "Fast investment recovery"
            )

        elif prediction <= 12:

            outlook = (
                "Reasonable investment recovery"
            )

        elif prediction <= 24:

            outlook = (
                "Longer investment recovery"
            )

        else:

            outlook = (
                "Extended investment recovery"
            )

        st.info(
            f"Payback Outlook: {outlook}"
        )

        st.divider()

        st.subheader(
            "Input Summary"
        )

        input_summary = pd.DataFrame(
            {
                "Feature": RIDGE_FEATURES,
                "Value": [
                    acquisition_channel_cost,
                    onboarding_hours,
                    expected_annual_fee,
                    expected_retention_years
                ]
            }
        )

        st.dataframe(
            input_summary,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# CUSTOMER 360
# ============================================================

elif page == "Customer 360":

    st.markdown(
        '<div class="model-header">'
        'Customer 360'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="model-description">'
        'Complete customer profile using all three machine learning models'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    selected_client = st.selectbox(
        "Select Client",
        df["client_id"].astype(str).tolist()
    )

    customer = df[
        df["client_id"].astype(str)
        == selected_client
    ].iloc[0]

    values = customer.to_dict()

    # ========================================================
    # ALL PREDICTIONS
    # ========================================================

    predicted_payback = predict_payback(
        values
    )

    income_values = values.copy()

    income_values[
        "payback_period_months"
    ] = predicted_payback

    predicted_income = predict_income(
        income_values
    )

    premium_probability, premium_prediction = (
        predict_logistic(values)
    )

    perceptron_prediction = (
        predict_perceptron(values)
    )

    st.subheader(
        "Customer Profile"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Client ID",
            str(customer["client_id"])
        )

    with col2:

        st.metric(
            "Years as Client",
            f"{customer['years_as_client']:.1f}"
        )

    with col3:

        st.metric(
            "Lifetime Revenue",
            f"₹{customer['lifetime_revenue']:,.0f}"
        )

    with col4:

        st.metric(
            "Premium Status",
            "Premium"
            if customer["premium_upgrade"] == 1
            else "Standard"
        )

    # ========================================================
    # REVENUE
    # ========================================================

    st.divider()

    st.subheader(
        "Revenue Growth Intelligence"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Actual Next-Year Income",
            f"₹{customer['next_year_income']:,.0f}"
        )

    with col2:

        st.metric(
            "Predicted Next-Year Income",
            f"₹{predicted_income:,.0f}"
        )

    with col3:

        difference = (
            predicted_income
            - customer["next_year_income"]
        )

        st.metric(
            "Prediction Difference",
            f"₹{difference:,.0f}"
        )

    # ========================================================
    # PREMIUM
    # ========================================================

    st.divider()

    st.subheader(
        "Premium Conversion Intelligence"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Logistic Probability",
            f"{premium_probability * 100:.1f}%"
        )

    with col2:

        st.metric(
            "Logistic Prediction",
            "Premium Conversion"
            if premium_prediction == 1
            else "Lower Conversion Signal"
        )

    with col3:

        st.metric(
            "Perceptron Signal",
            "Positive"
            if perceptron_prediction == 1
            else "Negative"
        )

    st.progress(
        float(premium_probability)
    )

    # ========================================================
    # PAYBACK
    # ========================================================

    st.divider()

    st.subheader(
        "Acquisition Payback Intelligence"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Actual Payback",
            f"{customer['payback_period_months']:.2f} months"
        )

    with col2:

        st.metric(
            "Predicted Payback",
            f"{predicted_payback:.2f} months"
        )

    with col3:

        st.metric(
            "Expected Annual Fee",
            f"₹{customer['expected_annual_fee']:,.0f}"
        )

    # ========================================================
    # COMPLETE DATA
    # ========================================================

    st.divider()

    st.subheader(
        "Complete Customer Information"
    )

    st.dataframe(
        customer.to_frame(),
        use_container_width=True
    )