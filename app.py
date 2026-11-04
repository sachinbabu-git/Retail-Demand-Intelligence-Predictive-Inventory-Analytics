
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math
import plotly.express as px
from pathlib import Path

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Retail Sales Analytics",
    page_icon="📊",
    layout="wide"
)

DATA_DIR = Path("data/processed")

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_sales():
    df = pd.read_csv(DATA_DIR / "cleaned_sales.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def load_comparison():
    return pd.read_csv(DATA_DIR / "model_comparison.csv")


@st.cache_data
def load_predictions():
    df = pd.read_csv(DATA_DIR / "model_predictions.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def load_coefficients():
    return pd.read_csv(DATA_DIR / "feature_coefficients.csv")


@st.cache_resource
def load_model():
    return joblib.load(DATA_DIR / "best_demand_model.pkl")


sales = load_sales()
comparison = load_comparison()
predictions = load_predictions()
coefficients = load_coefficients()
model = load_model()

# ==================================================
# COMMON CHART SETTINGS
# ==================================================

CHART_TEMPLATE = "plotly_dark"

# ==================================================
# DEMAND FORECAST FUNCTION
# ==================================================

def generate_forecast(product, variant, forecast_days=7):

    history = sales[
        (sales["Product"] == product)
        & (sales["Variant / Pack Size"] == variant)
    ]

    if history.empty:
        return pd.DataFrame()

    daily = (
        history.groupby("Date")["Quantity Sold"]
        .sum()
        .sort_index()
    )

    start_date = daily.index.min()
    end_date = daily.index.max()

    date_index = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    daily = daily.reindex(date_index, fill_value=0)

    demand_history = daily.astype(float).tolist()

    future_dates = pd.date_range(
        start=end_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    results = []

    for forecast_date in future_dates:

        input_data = pd.DataFrame([{
            "Product": product,
            "Variant / Pack Size": variant,
            "Day": forecast_date.day,
            "Month": forecast_date.month,
            "DayOfWeek": forecast_date.dayofweek,
            "IsWeekend": int(forecast_date.dayofweek >= 5),
            "Demand_Lag_1": demand_history[-1],
            "Demand_Lag_7": demand_history[-7],
            "Rolling_Mean_7": np.mean(demand_history[-7:])
        }])

        prediction = max(
            0,
            float(model.predict(input_data)[0])
        )

        results.append({
            "Date": forecast_date,
            "Predicted Demand": round(prediction, 2)
        })

        demand_history.append(prediction)

    return pd.DataFrame(results)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("Retail Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Sales Overview",
        "Sales Analytics",
        "Demand Forecasting",
        "Restock Planning",
        "Model Performance",
        "Model Diagnostics"
    ]
)

st.sidebar.divider()

st.sidebar.subheader("Dataset Information")

st.sidebar.write(
    f"**Period:** {sales['Date'].min().date()} "
    f"to {sales['Date'].max().date()}"
)

st.sidebar.write(
    f"**Transactions:** {len(sales):,}"
)

st.sidebar.write(
    f"**Products:** {sales['Product'].nunique()}"
)

st.sidebar.divider()

st.sidebar.caption(
    "Retail Sales Analytics & Demand Forecasting"
)

st.sidebar.caption(
    "Python | Pandas | SQL | Scikit-learn | Streamlit"
)


# ==================================================
# 1. SALES OVERVIEW
# ==================================================

if page == "Sales Overview":

    st.title("Retail Sales Dashboard")
    st.caption("Sales performance and business insights")

    total_revenue = sales["Revenue (₹)"].sum()
    total_units = sales["Quantity Sold"].sum()
    transactions = len(sales)
    products = sales["Product"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Revenue",
        f"₹{total_revenue:,.0f}"
    )

    col2.metric(
        "Units Sold",
        f"{total_units:,}"
    )

    col3.metric(
        "Transactions",
        f"{transactions:,}"
    )

    col4.metric(
        "Products",
        products
    )

    st.divider()

    # Monthly revenue

    st.subheader("Monthly Revenue")

    monthly = (
        sales.groupby(sales["Date"].dt.to_period("M"))["Revenue (₹)"]
        .sum()
        .reset_index()
    )

    monthly["Month"] = monthly["Date"].astype(str)

    fig = px.line(
        monthly,
        x="Month",
        y="Revenue (₹)",
        markers=True,
        title="Monthly Revenue Trend",
        template=CHART_TEMPLATE
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue (₹)"
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Top 10 Products by Revenue")

        top_products = (
            sales.groupby("Product")["Revenue (₹)"]
            .sum()
            .nlargest(10)
            .reset_index()
            .sort_values("Revenue (₹)")
        )

        fig = px.bar(
            top_products,
            x="Revenue (₹)",
            y="Product",
            orientation="h",
            template=CHART_TEMPLATE,
            title="Revenue by Product"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Top 10 Products by Quantity")

        top_quantity = (
            sales.groupby("Product")["Quantity Sold"]
            .sum()
            .nlargest(10)
            .reset_index()
            .sort_values("Quantity Sold")
        )

        fig = px.bar(
            top_quantity,
            x="Quantity Sold",
            y="Product",
            orientation="h",
            template=CHART_TEMPLATE,
            title="Units Sold by Product"
        )

        st.plotly_chart(fig, use_container_width=True)


# ==================================================
# 2. SALES ANALYTICS
# ==================================================

elif page == "Sales Analytics":

    st.title("Sales Analytics")
    st.caption("Explore sales using interactive filters")

    min_date = sales["Date"].min().date()
    max_date = sales["Date"].max().date()

    date_range = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    selected_products = st.multiselect(
        "Filter products",
        sorted(sales["Product"].unique())
    )

    filtered = sales.copy()

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

        filtered = filtered[
            filtered["Date"].between(
                pd.Timestamp(date_range[0]),
                pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
                - pd.Timedelta(seconds=1)
            )
        ]

    if selected_products:
        filtered = filtered[
            filtered["Product"].isin(selected_products)
        ]

    if filtered.empty:

        st.warning("No sales records found for these filters.")

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Revenue",
            f"₹{filtered['Revenue (₹)'].sum():,.0f}"
        )

        col2.metric(
            "Units Sold",
            f"{filtered['Quantity Sold'].sum():,}"
        )

        col3.metric(
            "Transactions",
            f"{len(filtered):,}"
        )

        st.divider()

        st.subheader("Daily Revenue Trend")

        daily = (
            filtered.groupby("Date")["Revenue (₹)"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            daily,
            x="Date",
            y="Revenue (₹)",
            markers=True,
            template=CHART_TEMPLATE
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Revenue (₹)",
            hovermode="x unified"
        )

        fig.update_xaxes(tickformat="%d %b")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Product Performance")

        product_table = (
            filtered.groupby("Product")
            .agg(
                Total_Quantity=("Quantity Sold", "sum"),
                Total_Revenue=("Revenue (₹)", "sum")
            )
            .sort_values("Total_Revenue", ascending=False)
        )

        st.dataframe(
            product_table,
            use_container_width=True
        )

        st.download_button(
            "Download Sales Analysis CSV",
            product_table.to_csv().encode("utf-8"),
            file_name="sales_analysis.csv",
            mime="text/csv"
        )


# ==================================================
# 3. DEMAND FORECASTING
# ==================================================

elif page == "Demand Forecasting":

    st.title("Demand Forecasting")

    st.caption(
        "Seven-day demand predictions using Linear Regression"
    )

    col1, col2 = st.columns(2)

    with col1:

        product = st.selectbox(
            "Select product",
            sorted(sales["Product"].unique())
        )

    variants = sorted(
        sales.loc[
            sales["Product"] == product,
            "Variant / Pack Size"
        ].unique()
    )

    with col2:

        variant = st.selectbox(
            "Select product variant",
            variants
        )

    forecast = generate_forecast(product, variant)

    if not forecast.empty:

        st.divider()

        st.subheader("Next 7 Days Forecast")

        fig = px.line(
            forecast,
            x="Date",
            y="Predicted Demand",
            markers=True,
            template=CHART_TEMPLATE,
            title=f"{product} - {variant}"
        )

        fig.update_layout(
            xaxis_title="Forecast Date",
            yaxis_title="Predicted Units",
            hovermode="x unified"
        )

        fig.update_xaxes(
            tickformat="%d %b",
            dtick=86400000
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        total_forecast = forecast["Predicted Demand"].sum()

        col1, col2 = st.columns(2)

        col1.metric(
            "Expected 7-Day Demand",
            f"{total_forecast:.1f} units"
        )

        col2.metric(
            "Average Daily Demand",
            f"{total_forecast / 7:.2f} units"
        )

        st.subheader("Forecast Details")

        st.dataframe(
            forecast,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Forecast CSV",
            forecast.to_csv(index=False).encode("utf-8"),
            file_name="demand_forecast.csv",
            mime="text/csv"
        )


# ==================================================
# 4. INVENTORY RESTOCK PLANNING
# ==================================================

elif page == "Restock Planning":

    st.title("Inventory Restock Planning")

    st.caption(
        "Estimate replenishment requirements using predicted demand "
        "and manually entered stock levels."
    )

    col1, col2 = st.columns(2)

    with col1:

        product = st.selectbox(
            "Select product",
            sorted(sales["Product"].unique()),
            key="restock_product"
        )

    variants = sorted(
        sales.loc[
            sales["Product"] == product,
            "Variant / Pack Size"
        ].unique()
    )

    with col2:

        variant = st.selectbox(
            "Select variant",
            variants,
            key="restock_variant"
        )

    col1, col2 = st.columns(2)

    with col1:

        current_stock = st.number_input(
            "Current available stock (units)",
            min_value=0,
            value=10,
            step=1
        )

    with col2:

        safety_buffer = st.slider(
            "Safety stock buffer (%)",
            min_value=0,
            max_value=50,
            value=20
        )

    forecast = generate_forecast(product, variant)

    if not forecast.empty:

        expected_demand = forecast["Predicted Demand"].sum()

        # Calculate safety stock
        safety_stock = math.ceil(
            expected_demand * safety_buffer / 100
        )

        # Calculate target inventory
        recommended_stock = (
            math.ceil(expected_demand) + safety_stock
        )

        # Calculate replenishment requirement
        reorder_quantity = max(
            0,
            recommended_stock - current_stock
        )

        st.divider()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Forecast Demand",
            f"{expected_demand:.1f} units"
        )

        col2.metric(
            "Safety Stock",
            f"{safety_stock} units"
        )

        col3.metric(
            "Available Stock",
            f"{current_stock} units"
        )

        col4.metric(
            "Recommended Order",
            f"{reorder_quantity} units"
        )

        st.subheader("Inventory Recommendation")

        if reorder_quantity > 0:

            st.warning(
                f"Restock recommended: order {reorder_quantity} units."
            )

        else:

            st.success(
                "Current stock covers forecast demand and safety stock."
            )

        st.info(
            f"Target inventory: {recommended_stock} units "
            f"(forecast demand + safety stock)."
        )

        st.subheader("Seven-Day Demand Forecast")

        fig = px.bar(
            forecast,
            x="Date",
            y="Predicted Demand",
            template=CHART_TEMPLATE,
            title="Expected Daily Demand"
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Predicted Units"
        )

        fig.update_xaxes(tickformat="%d %b")

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Planning estimate based on manually entered inventory. "
            "This is not connected to live stock records."
        )


# ==================================================
# 5. MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    st.title("Machine Learning Model Evaluation")

    st.caption(
        "Comparison of regression models using MAE and RMSE"
    )

    st.subheader("Model Comparison")

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("RMSE Comparison")

        fig = px.bar(
            comparison,
            x="Model",
            y="RMSE",
            color="Model",
            template=CHART_TEMPLATE,
            title="Root Mean Squared Error"
        )

        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="RMSE (Lower is Better)",
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("MAE Comparison")

        fig = px.bar(
            comparison,
            x="Model",
            y="MAE",
            color="Model",
            template=CHART_TEMPLATE,
            title="Mean Absolute Error"
        )

        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="MAE (Lower is Better)",
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    regression_results = comparison[
        comparison["Model"] != "7-Day Moving Average Baseline"
    ]

    best_model_result = regression_results.sort_values(
        "RMSE"
    ).iloc[0]

    st.divider()

    st.success(
        f"Best regression model: {best_model_result['Model']}"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Best Model MAE",
        f"{best_model_result['MAE']:.4f}"
    )

    col2.metric(
        "Best Model RMSE",
        f"{best_model_result['RMSE']:.4f}"
    )


# ==================================================
# 6. MODEL DIAGNOSTICS
# ==================================================

elif page == "Model Diagnostics":

    st.title("Model Diagnostics")
    st.caption(
        "Evaluate predicted demand against actual sales observations"
    )

    diagnostic_data = predictions.copy()
    diagnostic_data["Residual"] = (
        diagnostic_data["Actual Demand"]
        - diagnostic_data["Predicted Demand"]
    )

    mae = diagnostic_data["Residual"].abs().mean()
    rmse = np.sqrt((diagnostic_data["Residual"] ** 2).mean())
    mean_error = diagnostic_data["Residual"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.4f}")
    col2.metric("RMSE", f"{rmse:.4f}")
    col3.metric("Mean Error", f"{mean_error:.4f}")

    st.divider()
    st.subheader("Actual vs Predicted Demand")

    daily_comparison = (
        diagnostic_data.groupby("Date")[
            ["Actual Demand", "Predicted Demand"]
        ].sum().reset_index()
    )

    fig = px.line(
        daily_comparison,
        x="Date",
        y=["Actual Demand", "Predicted Demand"],
        markers=True,
        template=CHART_TEMPLATE,
        title="Daily Actual vs Predicted Demand"
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode="x unified"
    )
    fig.update_xaxes(tickformat="%d %b")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Accuracy")

    fig = px.scatter(
        diagnostic_data,
        x="Actual Demand",
        y="Predicted Demand",
        opacity=0.5,
        template=CHART_TEMPLATE,
        title="Actual Demand vs Predicted Demand"
    )

    max_value = max(
        diagnostic_data["Actual Demand"].max(),
        diagnostic_data["Predicted Demand"].max()
    )
    fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=max_value, y1=max_value,
        line=dict(dash="dash")
    )
    fig.update_layout(
        xaxis_title="Actual Units",
        yaxis_title="Predicted Units"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Prediction Error Distribution")

    fig = px.histogram(
        diagnostic_data,
        x="Residual",
        nbins=30,
        template=CHART_TEMPLATE,
        title="Residual Distribution"
    )
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(
        xaxis_title="Actual Demand - Predicted Demand",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Feature Coefficient Analysis")

    top_features = (
        coefficients.sort_values(
            "Absolute Coefficient",
            ascending=False
        )
        .head(15)
        .sort_values("Coefficient")
    )

    fig = px.bar(
        top_features,
        x="Coefficient",
        y="Feature",
        orientation="h",
        template=CHART_TEMPLATE,
        title="Top 15 Linear Regression Coefficients"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Coefficient magnitude shows the model's learned linear "
        "relationship with each encoded feature. It does not establish "
        "causation."
    )

    st.subheader("Test Prediction Records")
    st.dataframe(
        diagnostic_data,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Model Predictions",
        diagnostic_data.to_csv(index=False).encode("utf-8"),
        file_name="model_predictions_with_residuals.csv",
        mime="text/csv"
    )


# ==================================================
# FOOTER
# ==================================================

st.sidebar.divider()

st.sidebar.caption(
    "Retail Sales Analytics & Demand Forecasting"
)

st.sidebar.caption(
    "Built using Python, Pandas, SQL, Scikit-learn and Streamlit"
)
