#!/usr/bin/env python3

"""
House equity and value projection tool.

This script models the evolution of a home's value, mortgage balance, and owner
equity over time. It accounts for mortgage amortization, private mortgage
insurance (PMI), appreciation, property taxes, insurance, HOA dues, and
maintenance. The output summarizes the results on a yearly basis.

Example:
    python house_equity_projection.py \
        --purchase-price 450000 \
        --down-payment-percent 10 \
        --interest-rate 6.25 \
        --loan-term-years 30 \
        --appreciation-rate 3 \
        --property-tax-rate 1.2 \
        --pmi-rate 0.6 \
        --analysis-years 30
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class MortgageScenario:
    """Container for the scenario inputs used during the projection run."""

    purchase_price: float
    down_payment: float
    closing_costs: float
    points: float
    points_cost: float
    annual_interest_rate: float
    effective_interest_rate: float
    loan_term_years: int
    annual_appreciation_rate: float
    annual_property_tax_rate: float
    annual_pmi_rate: float
    pmi_ltv_threshold: float
    annual_insurance: float
    annual_hoa: float
    annual_maintenance_rate: float
    analysis_years: int
    extra_monthly_principal: float

    @property
    def loan_amount(self) -> float:
        return max(0.0, self.purchase_price - self.down_payment)

    @property
    def initial_cash_outlay(self) -> float:
        return self.down_payment + self.closing_costs + self.points_cost

    @property
    def monthly_interest_rate(self) -> float:
        return self.effective_interest_rate / 12.0

    @property
    def monthly_appreciation_rate(self) -> float:
        if self.annual_appreciation_rate <= -1.0:
            raise ValueError("Annual appreciation rate must be greater than -100%.")
        return (1.0 + self.annual_appreciation_rate) ** (1.0 / 12.0) - 1.0

    @property
    def loan_term_months(self) -> int:
        return self.loan_term_years * 12

    @property
    def monthly_insurance(self) -> float:
        return self.annual_insurance / 12.0

    @property
    def monthly_hoa(self) -> float:
        return self.annual_hoa / 12.0


@dataclass
class MonthlySnapshot:
    """Stores the state of the projection for one calendar month."""

    month_index: int
    year_index: int
    home_value: float
    remaining_balance: float
    principal_payment: float
    interest_payment: float
    pmi_payment: float
    property_tax_payment: float
    insurance_payment: float
    hoa_payment: float
    maintenance_payment: float

    @property
    def total_monthly_outlay(self) -> float:
        return (
            self.principal_payment
            + self.interest_payment
            + self.pmi_payment
            + self.property_tax_payment
            + self.insurance_payment
            + self.hoa_payment
            + self.maintenance_payment
        )

    @property
    def equity(self) -> float:
        return self.home_value - self.remaining_balance


def compute_monthly_payment(principal: float, annual_rate: float, term_months: int) -> float:
    """Return the fixed monthly principal and interest payment for the loan."""
    if principal <= 0 or term_months <= 0:
        return 0.0

    monthly_rate = annual_rate / 12.0
    if math.isclose(monthly_rate, 0.0, abs_tol=1e-12):
        return principal / term_months

    factor = (1.0 + monthly_rate) ** term_months
    return principal * monthly_rate * factor / (factor - 1.0)


def project_scenario(inputs: MortgageScenario) -> List[MonthlySnapshot]:
    """Generate the month-by-month projection for the provided scenario."""

    loan_balance = inputs.loan_amount
    home_value = inputs.purchase_price
    monthly_pi_payment = compute_monthly_payment(
        principal=loan_balance,
        annual_rate=inputs.effective_interest_rate,
        term_months=inputs.loan_term_months,
    )
    monthly_appreciation = inputs.monthly_appreciation_rate
    monthly_property_tax_rate = inputs.annual_property_tax_rate / 12.0
    snapshots: List[MonthlySnapshot] = []

    months_to_simulate = inputs.analysis_years * 12

    for month in range(1, months_to_simulate + 1):
        year = (month - 1) // 12 + 1

        if loan_balance > 0:
            interest_payment = loan_balance * inputs.monthly_interest_rate
            scheduled_principal = monthly_pi_payment - interest_payment
            if scheduled_principal < 0:
                scheduled_principal = 0.0
            total_principal_payment = min(
                loan_balance,
                scheduled_principal + inputs.extra_monthly_principal,
            )
            new_balance = max(0.0, loan_balance - total_principal_payment)
            loan_to_value = loan_balance / home_value if home_value > 0 else 0.0
            if (
                inputs.annual_pmi_rate > 0
                and loan_to_value > inputs.pmi_ltv_threshold
            ):
                pmi_payment = loan_balance * inputs.annual_pmi_rate / 12.0
            else:
                pmi_payment = 0.0
        else:
            interest_payment = 0.0
            total_principal_payment = 0.0
            pmi_payment = 0.0
            new_balance = 0.0

        property_tax_payment = home_value * monthly_property_tax_rate
        maintenance_payment = home_value * inputs.annual_maintenance_rate / 12.0

        snapshots.append(
            MonthlySnapshot(
                month_index=month,
                year_index=year,
                home_value=home_value,
                remaining_balance=new_balance,
                principal_payment=total_principal_payment,
                interest_payment=interest_payment,
                pmi_payment=pmi_payment,
                property_tax_payment=property_tax_payment,
                insurance_payment=inputs.monthly_insurance,
                hoa_payment=inputs.monthly_hoa,
                maintenance_payment=maintenance_payment,
            )
        )

        loan_balance = new_balance
        home_value *= 1.0 + monthly_appreciation

    return snapshots


def summarize_by_year(snapshots: Iterable[MonthlySnapshot]) -> List[dict]:
    """Aggregate month-level snapshots into yearly summary rows."""
    summary = {}
    for snap in snapshots:
        year = snap.year_index
        if year not in summary:
            summary[year] = {
                "year": year,
                "ending_home_value": snap.home_value,
                "ending_balance": snap.remaining_balance,
                "ending_equity": snap.equity,
                "principal_paid": 0.0,
                "interest_paid": 0.0,
                "pmi_paid": 0.0,
                "property_taxes_paid": 0.0,
                "insurance_paid": 0.0,
                "hoa_paid": 0.0,
                "maintenance_paid": 0.0,
                "total_outlay": 0.0,
            }

        summary_row = summary[year]
        summary_row["ending_home_value"] = snap.home_value
        summary_row["ending_balance"] = snap.remaining_balance
        summary_row["ending_equity"] = snap.equity
        summary_row["principal_paid"] += snap.principal_payment
        summary_row["interest_paid"] += snap.interest_payment
        summary_row["pmi_paid"] += snap.pmi_payment
        summary_row["property_taxes_paid"] += snap.property_tax_payment
        summary_row["insurance_paid"] += snap.insurance_payment
        summary_row["hoa_paid"] += snap.hoa_payment
        summary_row["maintenance_paid"] += snap.maintenance_payment
        summary_row["total_outlay"] += snap.total_monthly_outlay

    return [summary[key] for key in sorted(summary)]


def format_currency(value: float) -> str:
    """Format a numeric value as USD-style currency."""
    return f"${value:,.0f}"


def print_summary_table(summary_rows: List[dict]) -> None:
    """Render the yearly summary as a simple text table."""

    headers = [
        "Year",
        "Home Value",
        "Loan Balance",
        "Equity",
        "Principal Paid",
        "Interest Paid",
        "PMI Paid",
        "Taxes",
        "Insurance",
        "HOA",
        "Maintenance",
        "Total Outlay",
    ]

    column_widths = [max(len(header), 12) for header in headers]

    def format_row(values: List[str]) -> str:
        parts = []
        for value, width in zip(values, column_widths):
            parts.append(value.rjust(width))
        return " ".join(parts)

    print(format_row(headers))
    print(" ".join("-" * width for width in column_widths))

    for row in summary_rows:
        values = [
            str(row["year"]),
            format_currency(row["ending_home_value"]),
            format_currency(row["ending_balance"]),
            format_currency(row["ending_equity"]),
            format_currency(row["principal_paid"]),
            format_currency(row["interest_paid"]),
            format_currency(row["pmi_paid"]),
            format_currency(row["property_taxes_paid"]),
            format_currency(row["insurance_paid"]),
            format_currency(row["hoa_paid"]),
            format_currency(row["maintenance_paid"]),
            format_currency(row["total_outlay"]),
        ]
        print(format_row(values))


def write_csv(path: str, summary_rows: Iterable[dict]) -> None:
    """Persist the summary table to a CSV file."""
    fieldnames = [
        "year",
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
    ]
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project home value, equity, and carrying costs over time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--purchase-price", type=float, required=True, help="Home purchase price.")
    down_payment_group = parser.add_mutually_exclusive_group(required=True)
    down_payment_group.add_argument(
        "--down-payment-amount",
        type=float,
        help="Down payment amount in dollars.",
    )
    down_payment_group.add_argument(
        "--down-payment-percent",
        type=float,
        help="Down payment as a percentage of the purchase price (e.g., 20 for 20%).",
    )

    parser.add_argument("--closing-costs", type=float, default=0.0, help="Total closing costs paid upfront in dollars.")
    parser.add_argument("--interest-rate", type=float, required=True, help="Annual mortgage interest rate as a percentage (e.g., 6.5).")
    parser.add_argument("--points", type=float, default=0.0, help="Mortgage discount points purchased (each point equals 1% of loan amount).")
    parser.add_argument("--loan-term-years", type=int, default=30, help="Length of the mortgage in years.")
    parser.add_argument("--appreciation-rate", type=float, default=3.0, help="Expected annual appreciation rate (%).")
    parser.add_argument("--property-tax-rate", type=float, default=1.0, help="Annual property tax rate as a percentage of home value.")
    parser.add_argument("--pmi-rate", type=float, default=0.5, help="Annual PMI rate as a percentage of outstanding balance.")
    parser.add_argument("--pmi-ltv-threshold", type=float, default=80.0, help="Loan-to-value percentage at which PMI drops off.")
    parser.add_argument("--insurance-annual", type=float, default=1200.0, help="Annual homeowner's insurance cost in dollars.")
    parser.add_argument("--hoa-annual", type=float, default=0.0, help="Annual HOA dues in dollars.")
    parser.add_argument("--maintenance-rate", type=float, default=1.0, help="Annual maintenance reserve as a percentage of home value.")
    parser.add_argument("--analysis-years", type=int, default=30, help="Number of years to project.")
    parser.add_argument("--extra-monthly-principal", type=float, default=0.0, help="Extra principal paid each month.")
    parser.add_argument("--output-csv", type=str, help="Optional path to write a CSV summary.")

    return parser.parse_args(argv)


def create_scenario(
    *,
    purchase_price: float,
    down_payment_amount: Optional[float] = None,
    down_payment_percent: Optional[float] = None,
    closing_costs: float,
    points: float,
    interest_rate: float,
    loan_term_years: int,
    appreciation_rate: float,
    property_tax_rate: float,
    pmi_rate: float,
    pmi_ltv_threshold: float,
    insurance_annual: float,
    hoa_annual: float,
    maintenance_rate: float,
    analysis_years: int,
    extra_monthly_principal: float,
) -> MortgageScenario:
    if down_payment_amount is not None and down_payment_percent is not None:
        raise ValueError("Specify only one of down payment amount or percentage.")

    if down_payment_percent is not None:
        down_payment_amount = purchase_price * (down_payment_percent / 100.0)

    if down_payment_amount is None:
        raise ValueError("A down payment amount or percentage must be provided.")

    if down_payment_amount < 0:
        raise ValueError("Down payment cannot be negative.")
    if down_payment_amount > purchase_price:
        raise ValueError("Down payment cannot exceed the purchase price.")

    if closing_costs < 0:
        raise ValueError("Closing costs cannot be negative.")

    if points < 0:
        raise ValueError("Points cannot be negative.")

    annual_interest_rate = interest_rate / 100.0
    rate_discount = points * 0.0025
    effective_interest_rate = max(0.0, annual_interest_rate - rate_discount)
    annual_appreciation_rate = appreciation_rate / 100.0
    annual_property_tax_rate = property_tax_rate / 100.0
    annual_pmi_rate = pmi_rate / 100.0
    pmi_ltv_threshold_fraction = pmi_ltv_threshold / 100.0
    annual_maintenance_rate = maintenance_rate / 100.0

    if annual_interest_rate < 0:
        raise ValueError("Interest rate cannot be negative.")
    if analysis_years <= 0:
        raise ValueError("Analysis years must be positive.")
    if loan_term_years <= 0:
        raise ValueError("Loan term years must be positive.")

    loan_amount = max(0.0, purchase_price - down_payment_amount)
    points_cost = loan_amount * (points / 100.0)

    return MortgageScenario(
        purchase_price=purchase_price,
        down_payment=down_payment_amount,
        closing_costs=closing_costs,
        points=points,
        points_cost=points_cost,
        annual_interest_rate=annual_interest_rate,
        effective_interest_rate=effective_interest_rate,
        loan_term_years=loan_term_years,
        annual_appreciation_rate=annual_appreciation_rate,
        annual_property_tax_rate=annual_property_tax_rate,
        annual_pmi_rate=annual_pmi_rate,
        pmi_ltv_threshold=pmi_ltv_threshold_fraction,
        annual_insurance=insurance_annual,
        annual_hoa=hoa_annual,
        annual_maintenance_rate=annual_maintenance_rate,
        analysis_years=analysis_years,
        extra_monthly_principal=extra_monthly_principal,
    )


def build_scenario(args: argparse.Namespace) -> MortgageScenario:
    return create_scenario(
        purchase_price=args.purchase_price,
        down_payment_amount=args.down_payment_amount,
        down_payment_percent=args.down_payment_percent,
        closing_costs=args.closing_costs,
        points=args.points,
        interest_rate=args.interest_rate,
        loan_term_years=args.loan_term_years,
        appreciation_rate=args.appreciation_rate,
        property_tax_rate=args.property_tax_rate,
        pmi_rate=args.pmi_rate,
        pmi_ltv_threshold=args.pmi_ltv_threshold,
        insurance_annual=args.insurance_annual,
        hoa_annual=args.hoa_annual,
        maintenance_rate=args.maintenance_rate,
        analysis_years=args.analysis_years,
        extra_monthly_principal=args.extra_monthly_principal,
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    scenario = build_scenario(args)

    snapshots = project_scenario(scenario)
    yearly_summary = summarize_by_year(snapshots)

    print("Scenario Overview")
    print("=================")
    print(f"Purchase price:          {format_currency(scenario.purchase_price)}")
    print(f"Down payment:            {format_currency(scenario.down_payment)}")
    print(f"Closing costs:           {format_currency(scenario.closing_costs)}")
    if scenario.points > 0:
        print(f"Points purchased:        {scenario.points:.2f} (cost {format_currency(scenario.points_cost)})")
    print(f"Initial cash needed:     {format_currency(scenario.initial_cash_outlay)}")
    print(f"Loan amount:             {format_currency(scenario.loan_amount)}")
    print(f"Interest rate (base):    {scenario.annual_interest_rate * 100:.3f}%")
    print(f"Interest rate (effective): {scenario.effective_interest_rate * 100:.3f}%")
    print(f"Loan term (years):       {scenario.loan_term_years}")
    print(f"Appreciation rate:       {scenario.annual_appreciation_rate * 100:.3f}%")
    print(f"Property tax rate:       {scenario.annual_property_tax_rate * 100:.3f}%")
    if scenario.annual_pmi_rate > 0:
        print(f"PMI rate:                {scenario.annual_pmi_rate * 100:.3f}% (drops at {scenario.pmi_ltv_threshold * 100:.1f}% LTV)")
    else:
        print("PMI rate:                None")
    print(f"Insurance (annual):      {format_currency(scenario.annual_insurance)}")
    print(f"HOA dues (annual):       {format_currency(scenario.annual_hoa)}")
    print(f"Maintenance reserve:     {scenario.annual_maintenance_rate * 100:.3f}% of value")
    print(f"Extra monthly principal: {format_currency(scenario.extra_monthly_principal)}")
    print()
    print_summary_table(yearly_summary)

    if args.output_csv:
        write_csv(args.output_csv, yearly_summary)
        print(f"\nSummary written to {args.output_csv}")


if __name__ == "__main__":
    main()
