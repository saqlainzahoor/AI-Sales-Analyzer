import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Sales Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .main-title {
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 20px;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        background: white;
        min-height: 125px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.04);
    }

    .metric-label {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 750;
    }

    .prediction-card {
        padding: 30px;
        text-align: center;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        background: white;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
    }

    .prediction-label {
        color: #6b7280;
        font-size: 16px;
    }

    .prediction-value {
        font-size: 40px;
        font-weight: 800;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("AI-Sales-Analyzer/sales_data.csv")

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    return df


data = load_data()


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

features = [
    "Product",
    "Quantity",
    "Price",
    "Discount",
    "Region",
    "Advertising_Spend",
    "Customer_Type"
]

X = data[features]
y = data["Total_Sales"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


categorical_features = [
    "Product",
    "Region",
    "Customer_Type"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


model.fit(X_train, y_train)


# =========================================================
# MODEL EVALUATION
# =========================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📊 AI Sales Analyzer")

    st.caption(
        "Sales Intelligence Dashboard"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "AI Prediction",
            "Sales Data",
            "Model Performance"
        ]
    )

    st.divider()

    st.markdown("### Dashboard Filters")

    min_date = data["Date"].min().date()
    max_date = data["Date"].max().date()

    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    product_filter = st.multiselect(
        "Product",
        sorted(data["Product"].unique()),
        default=[]
    )

    region_filter = st.multiselect(
        "Region",
        sorted(data["Region"].unique()),
        default=[]
    )

    customer_filter = st.multiselect(
        "Customer Type",
        sorted(data["Customer_Type"].unique()),
        default=[]
    )

    st.divider()

    st.markdown("### Project")

    st.write("Python")
    st.write("Pandas")
    st.write("Scikit-learn")
    st.write("Streamlit")

    st.divider()

    st.caption("Portfolio Project")


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_data = data.copy()


if len(date_range) == 2:

    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

    filtered_data = filtered_data[
        (filtered_data["Date"] >= start_date)
        & (filtered_data["Date"] <= end_date)
    ]


if product_filter:

    filtered_data = filtered_data[
        filtered_data["Product"].isin(product_filter)
    ]


if region_filter:

    filtered_data = filtered_data[
        filtered_data["Region"].isin(region_filter)
    ]


if customer_filter:

    filtered_data = filtered_data[
        filtered_data["Customer_Type"].isin(customer_filter)
    ]


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📊 AI Sales Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Sales analytics, business insights and AI-powered prediction'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.subheader("Business Overview")

    if filtered_data.empty:

        st.warning(
            "No sales data found for the selected filters."
        )

    else:

        total_revenue = filtered_data[
            "Total_Sales"
        ].sum()

        total_orders = len(filtered_data)

        average_sales = filtered_data[
            "Total_Sales"
        ].mean()

        best_product = (
            filtered_data
            .groupby("Product")["Total_Sales"]
            .sum()
            .idxmax()
        )

        # =================================================
        # KPI CARDS
        # =================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        💰 Total Revenue
                    </div>
                    <div class="metric-value">
                        {total_revenue:,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        🧾 Total Orders
                    </div>
                    <div class="metric-value">
                        {total_orders}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        📦 Best Product
                    </div>
                    <div class="metric-value">
                        {best_product}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col4:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        📊 Average Sale
                    </div>
                    <div class="metric-value">
                        {average_sales:,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        st.write("")


        # =================================================
        # CHART 1 - PRODUCT SALES
        # =================================================

        chart1, chart2 = st.columns(2)

        with chart1:

            st.markdown("### 📈 Sales by Product")

            product_sales = (
                filtered_data
                .groupby("Product")["Total_Sales"]
                .sum()
                .sort_values(ascending=False)
            )

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.bar(
                product_sales.index,
                product_sales.values
            )

            ax.set_xlabel("Product")
            ax.set_ylabel("Total Sales")
            ax.tick_params(
                axis="x",
                rotation=45
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


        # =================================================
        # CHART 2 - REGION SALES
        # =================================================

        with chart2:

            st.markdown("### 🌍 Sales by Region")

            region_sales = (
                filtered_data
                .groupby("Region")["Total_Sales"]
                .sum()
                .sort_values(ascending=False)
            )

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.bar(
                region_sales.index,
                region_sales.values
            )

            ax.set_xlabel("Region")
            ax.set_ylabel("Total Sales")
            ax.tick_params(
                axis="x",
                rotation=45
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


        # =================================================
        # SALES TREND
        # =================================================

        st.markdown("### 📅 Sales Trend")

        daily_sales = (
            filtered_data
            .groupby("Date")["Total_Sales"]
            .sum()
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.plot(
            daily_sales.index,
            daily_sales.values,
            marker="o"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.set_title("Sales Trend Over Time")

        plt.xticks(rotation=45)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


# =========================================================
# AI PREDICTION
# =========================================================

elif page == "AI Prediction":

    st.subheader("🤖 AI Sales Prediction")

    st.write(
        "Enter sales information to estimate expected sales."
    )

    st.divider()

    col1, col2 = st.columns(2)


    with col1:

        product = st.selectbox(
            "Product",
            sorted(data["Product"].unique())
        )

        quantity = st.number_input(
            "Quantity",
            min_value=1.0,
            value=10.0,
            step=1.0
        )

        price = st.number_input(
            "Price",
            min_value=0.0,
            value=80000.0,
            step=1000.0
        )

        discount = st.number_input(
            "Discount (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0
        )


    with col2:

        region = st.selectbox(
            "Region",
            sorted(data["Region"].unique())
        )

        advertising_spend = st.number_input(
            "Advertising Spend",
            min_value=0.0,
            value=25000.0,
            step=1000.0
        )

        customer_type = st.selectbox(
            "Customer Type",
            sorted(data["Customer_Type"].unique())
        )


    st.write("")


    if st.button(
        "🔮 Predict Sales",
        use_container_width=True
    ):

        new_data = pd.DataFrame({

            "Product": [product],

            "Quantity": [quantity],

            "Price": [price],

            "Discount": [discount],

            "Region": [region],

            "Advertising_Spend": [
                advertising_spend
            ],

            "Customer_Type": [
                customer_type
            ]
        })


        prediction = model.predict(
            new_data
        )[0]


        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-label">
                    Estimated Sales
                </div>

                <div class="prediction-value">
                    {prediction:,.2f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.success(
            "Prediction generated successfully."
        )


# =========================================================
# SALES DATA
# =========================================================

elif page == "Sales Data":

    st.subheader("📋 Sales Dataset")

    st.write(
        "Explore the sales records used by the application."
    )

    st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True
    )


    st.divider()


    # DOWNLOAD BUTTON

    csv_data = filtered_data.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Filtered Data",
        data=csv_data,
        file_name="filtered_sales_data.csv",
        mime="text/csv",
        use_container_width=True
    )

# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.subheader("🧠 Machine Learning Performance")

    st.write(
        "Performance of the Linear Regression model on the test dataset."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Mean Absolute Error",
            value=f"{mae:,.2f}"
        )

    with col2:
        st.metric(
            label="R² Score",
            value=f"{r2:.4f}"
        )

    st.write("")

    # =========================================================
    # ACTUAL VS PREDICTED
    # =========================================================

    st.markdown("### 🔬 Actual vs Predicted Sales")

    comparison = pd.DataFrame({
        "Actual Sales": y_test.values,
        "Predicted Sales": y_pred
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "This project uses a small synthetic practice dataset "
        "created for learning and portfolio development. "
        "The model metrics should not be interpreted as "
        "real-world business performance."
    )
        


    st.write("")


    # ACTUAL VS PREDICTED

    st.markdown(
        "### 🔬 Actual vs Predicted Sales"
    )

    comparison = pd.DataFrame({

        "Actual Sales": y_test.values,

        "Predicted Sales": y_pred

    })


    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        "This project uses a small synthetic practice dataset "
        "created for learning and portfolio development. "
        "The model metrics should not be interpreted as "
        "real-world business performance."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Sales Analyzer • Python • Pandas • Scikit-learn • Streamlit"
)