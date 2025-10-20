#!/usr/bin/env python3

"""Interactive Streamlit app for the house equity projection tool."""

from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from house_equity_projection import (
    compute_monthly_payment,
    create_scenario,
    format_currency,
    project_scenario,
    summarize_by_year,
)


def project_investment_balances(initial_capital, monthly_contributions, annual_return_percent):
    """Deterministic investment projection using constant average returns."""
    monthly_rate = (1.0 + annual_return_percent / 100.0) ** (1.0 / 12.0) - 1.0
    balance = initial_capital
    balances = []
    for contribution in monthly_contributions:
        balance *= 1.0 + monthly_rate
        balance += contribution
        balances.append(balance)
    return balances


def main() -> None:
    st.set_page_config(page_title="House Equity Projection", layout="wide")
    st.title("House Equity and Value Projection")
    st.markdown(
        "Adjust the assumptions in the sidebar to model mortgage amortization, "
        "equity growth, and carrying costs over time. "
        "Compare those cash flows with investing the same upfront capital and monthly cost difference."
    )

    scenario, rent_amount = collect_inputs()
    if scenario is None:
        st.stop()

    st.markdown("### Investment Return")
    investment_return = st.radio(
        "Select the expected annual investment return:",
        options=[3.0, 5.0, 7.0, 10.0],
        index=2,
        horizontal=True,
        key="investment_return_choice",
        format_func=lambda x: f"{x:.0f}%",
    )

    snapshots = project_scenario(scenario)
    monthly_df = build_monthly_dataframe(snapshots)
    monthly_df = add_investment_projection(
        monthly_df=monthly_df,
        scenario=scenario,
        investment_return=investment_return,
        rent_amount=rent_amount,
    )
    yearly_df = pd.DataFrame(summarize_by_year(snapshots))

    investment_yearly = (
        monthly_df.groupby("Year", as_index=False)["Investment Balance"]
        .last()
        .rename(columns={"Year": "year", "Investment Balance": "investment_balance"})
    )
    yearly_df = yearly_df.merge(investment_yearly, on="year", how="left")

    render_key_metrics(monthly_df, scenario, investment_return)
    render_plot(monthly_df)
    render_yearly_table(yearly_df)
    render_download(yearly_df)


def collect_inputs():
    st.sidebar.header("Inputs")
    purchase_price = st.sidebar.number_input(
        "Purchase price ($)",
        min_value=50000.0,
        max_value=5_000_000.0,
        value=800_000.0,
        step=5_000.0,
        format="%.2f",
    )

    closing_cost_percent = st.sidebar.number_input(
        "Closing costs (% of purchase price)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.25,
        format="%.2f",
    )
    closing_costs = purchase_price * (closing_cost_percent / 100.0)
    st.sidebar.caption(f"Closing costs: {format_currency(closing_costs)}")

    down_payment_mode = st.sidebar.selectbox(
        "Down payment input",
        options=["Percent of purchase price", "Dollar amount"],
    )
    if down_payment_mode == "Percent of purchase price":
        down_payment_percent = st.sidebar.number_input(
            "Down payment (%)",
            min_value=0.0,
            max_value=100.0,
            value=3.0,
            step=0.5,
            format="%.2f",
        )
        down_payment_amount = None
    else:
        down_payment_amount = st.sidebar.number_input(
            "Down payment ($)",
            min_value=0.0,
            max_value=float(purchase_price),
            value=float(purchase_price * 0.03),
            step=5_000.0,
            format="%.2f",
        )
        down_payment_percent = None

    rent_amount = st.sidebar.number_input(
        "Monthly rent assumption ($)",
        min_value=0.0,
        max_value=20_000.0,
        value=4_000.0,
        step=100.0,
        format="%.2f",
    )

    interest_rate = st.sidebar.number_input(
        "Mortgage interest rate (annual %)",
        min_value=0.0,
        max_value=25.0,
        value=6.25,
        step=0.05,
        format="%.3f",
    )

    loan_term_years = st.sidebar.slider("Loan term (years)", min_value=10, max_value=40, value=30, step=5)
    appreciation_rate = st.sidebar.number_input(
        "Appreciation rate (annual %)",
        min_value=-10.0,
        max_value=20.0,
        value=3.0,
        step=0.25,
        format="%.2f",
    )
    property_tax_rate = st.sidebar.number_input(
        "Property tax rate (annual % of value)",
        min_value=0.0,
        max_value=10.0,
        value=1.2,
        step=0.1,
        format="%.2f",
    )
    pmi_rate = st.sidebar.number_input(
        "PMI rate (annual % of balance)",
        min_value=0.0,
        max_value=3.0,
        value=0.6,
        step=0.05,
        format="%.2f",
    )
    pmi_ltv_threshold = st.sidebar.number_input(
        "PMI removal threshold (loan-to-value %)",
        min_value=50.0,
        max_value=100.0,
        value=80.0,
        step=1.0,
        format="%.1f",
    )
    insurance_percent = st.sidebar.number_input(
        "Homeowner's insurance (annual % of value)",
        min_value=0.0,
        max_value=10.0,
        value=0.3,
        step=0.05,
        format="%.2f",
    )
    insurance_annual = purchase_price * (insurance_percent / 100.0)
    st.sidebar.caption(f"Annual insurance: {format_currency(insurance_annual)}")
    hoa_annual = st.sidebar.number_input(
        "HOA dues (annual $)",
        min_value=0.0,
        max_value=20_000.0,
        value=0.0,
        step=250.0,
        format="%.2f",
    )
    maintenance_rate = st.sidebar.number_input(
        "Maintenance reserve (annual % of value)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.25,
        format="%.2f",
    )
    analysis_years = st.sidebar.slider("Years to project", min_value=1, max_value=40, value=30)
    extra_monthly_principal = st.sidebar.number_input(
        "Extra principal paid monthly ($)",
        min_value=0.0,
        max_value=10_000.0,
        value=0.0,
        step=100.0,
        format="%.2f",
    )

    try:
        scenario = create_scenario(
            purchase_price=purchase_price,
            down_payment_amount=down_payment_amount,
            down_payment_percent=down_payment_percent,
            closing_costs=closing_costs,
            interest_rate=interest_rate,
            loan_term_years=loan_term_years,
            appreciation_rate=appreciation_rate,
            property_tax_rate=property_tax_rate,
            pmi_rate=pmi_rate,
            pmi_ltv_threshold=pmi_ltv_threshold,
            insurance_annual=insurance_annual,
            hoa_annual=hoa_annual,
            maintenance_rate=maintenance_rate,
            analysis_years=analysis_years,
            extra_monthly_principal=extra_monthly_principal,
        )
    except ValueError as exc:
        st.error(f"Input error: {exc}")
        return None, None

    return scenario, rent_amount


def build_monthly_dataframe(snapshots):
    data = {
        "Month": [snap.month_index for snap in snapshots],
        "Year": [snap.year_index for snap in snapshots],
        "Home Value": [snap.home_value for snap in snapshots],
        "Loan Balance": [snap.remaining_balance for snap in snapshots],
        "Equity": [snap.equity for snap in snapshots],
        "Principal Paid": [snap.principal_payment for snap in snapshots],
        "Interest Paid": [snap.interest_payment for snap in snapshots],
        "PMI Paid": [snap.pmi_payment for snap in snapshots],
        "Property Taxes": [snap.property_tax_payment for snap in snapshots],
        "Insurance": [snap.insurance_payment for snap in snapshots],
        "HOA": [snap.hoa_payment for snap in snapshots],
        "Maintenance": [snap.maintenance_payment for snap in snapshots],
        "Total Outlay": [snap.total_monthly_outlay for snap in snapshots],
    }
    df = pd.DataFrame(data)
    df["Years"] = df["Month"] / 12.0
    return df


def add_investment_projection(monthly_df, scenario, investment_return, rent_amount):
    contributions = monthly_df["Total Outlay"] - rent_amount
    balances = project_investment_balances(
        initial_capital=scenario.initial_cash_outlay,
        monthly_contributions=contributions.tolist(),
        annual_return_percent=investment_return,
    )
    monthly_df["Investment Contribution"] = contributions
    monthly_df["Investment Balance"] = balances
    return monthly_df


def render_key_metrics(monthly_df: pd.DataFrame, scenario, investment_return: float) -> None:
    final_row = monthly_df.iloc[-1]
    monthly_pi = compute_monthly_payment(
        scenario.loan_amount,
        scenario.annual_interest_rate,
        scenario.loan_term_months,
    )

    investment_final = final_row["Investment Balance"]
    home_value_final = final_row["Home Value"]

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Loan amount", format_currency(scenario.loan_amount))
    col_b.metric("Initial cash outlay", format_currency(scenario.initial_cash_outlay))
    col_c.metric("Monthly P&I payment", format_currency(monthly_pi))
    col_d.metric(
        f"Equity after {scenario.analysis_years} years",
        format_currency(final_row["Equity"]),
    )
    col_e.metric(
        f"Investment balance ({investment_return:.0f}%)",
        format_currency(investment_final),
    )
    st.caption(f"Final home value: {format_currency(home_value_final)}")


def render_plot(monthly_df: pd.DataFrame) -> None:
    st.subheader("Projected Value and Equity")
    fig = px.line(
        monthly_df,
        x="Years",
        y=["Home Value", "Loan Balance", "Equity", "Investment Balance"],
        labels={"value": "Dollars", "variable": "Metric", "Years": "Years"},
        color_discrete_map={
            "Home Value": "#1f77b4",
            "Loan Balance": "#ff7f0e",
            "Equity": "#2ca02c",
            "Investment Balance": "#9467bd",
        },
    )
    fig.update_layout(legend_title="", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def render_yearly_table(yearly_df: pd.DataFrame) -> None:
    st.subheader("Yearly Summary")
    formatted = yearly_df.copy()
    currency_cols = [
        "ending_home_value",
        "ending_balance",
        "ending_equity",
        "principal_paid",
        "interest_paid",
        "pmi_paid",
        "property_taxes_paid",
        "insurance_paid",
        "hoa_paid",
        "maintenance_paid",
        "total_outlay",
        "investment_balance",
    ]
    for col in currency_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(format_currency)
    st.dataframe(formatted.rename(columns=lambda name: name.replace("_", " ").title()), use_container_width=True)


def render_download(yearly_df: pd.DataFrame) -> None:
    csv_buffer = io.StringIO()
    yearly_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download yearly summary (CSV)",
        data=csv_buffer.getvalue(),
        file_name="house_equity_summary.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
