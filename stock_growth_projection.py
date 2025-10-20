#!/usr/bin/env python3

"""
Stock market growth projection tool.

Given an initial capital amount, periodic contributions, and assumptions about
average investment returns and volatility, this script projects the future
value of the portfolio over a specified horizon. It can generate both a single
expected trajectory and Monte Carlo style percentile bands to convey risk.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass
class InvestmentParameters:
    starting_capital: float
    monthly_contribution: float
    expected_annual_return: float
    annual_volatility: float
    analysis_years: int
    inflation_rate: float
    contribution_growth_rate: float
    monte_carlo_trials: int

    @property
    def monthly_return_mean(self) -> float:
        return (1.0 + self.expected_annual_return) ** (1.0 / 12.0) - 1.0

    @property
    def monthly_return_std(self) -> float:
        return self.annual_volatility / math.sqrt(12.0)

    @property
    def monthly_inflation_rate(self) -> float:
        return (1.0 + self.inflation_rate) ** (1.0 / 12.0) - 1.0

    @property
    def monthly_contribution_growth(self) -> float:
        return (1.0 + self.contribution_growth_rate) ** (1.0 / 12.0) - 1.0

    @property
    def months(self) -> int:
        return self.analysis_years * 12


def deterministic_projection(params: InvestmentParameters) -> List[Tuple[int, float]]:
    """Return the expected value trajectory using constant average returns."""
    balance = params.starting_capital
    contribution = params.monthly_contribution
    month_results = []
    for month in range(1, params.months + 1):
        balance *= 1.0 + params.monthly_return_mean
        balance += contribution
        month_results.append((month, balance))
        contribution *= 1.0 + params.monthly_contribution_growth
    return month_results


def monte_carlo_projection(params: InvestmentParameters) -> List[List[float]]:
    """Generate repeated simulations incorporating volatility."""
    if params.monte_carlo_trials <= 0 or math.isclose(params.annual_volatility, 0.0):
        return [value for _, value in deterministic_projection(params)]

    streams: List[List[float]] = []
    import random

    for _ in range(params.monte_carlo_trials):
        balance = params.starting_capital
        contribution = params.monthly_contribution
        trial_values: List[float] = []
        for _ in range(params.months):
            monthly_return = random.gauss(params.monthly_return_mean, params.monthly_return_std)
            balance *= 1.0 + monthly_return
            balance += contribution
            trial_values.append(balance)
            contribution *= 1.0 + params.monthly_contribution_growth
        streams.append(trial_values)
    return streams


def percentile_curve(streams: Sequence[Sequence[float]], percentile: float) -> List[Tuple[int, float]]:
    """Compute the per-month percentile across multiple simulation streams."""
    if not streams:
        return []
    months = len(streams[0])
    curves = []
    for month in range(months):
        month_values = [stream[month] for stream in streams]
        month_values.sort()
        index = int((percentile / 100.0) * (len(month_values) - 1))
        curves.append((month + 1, month_values[index]))
    return curves


def inflate_values(curve: Iterable[Tuple[int, float]], params: InvestmentParameters) -> List[Tuple[int, float]]:
    """Convert future dollars into today's dollars using inflation assumptions."""
    inflated = []
    for month, value in curve:
        real_value = value / ((1.0 + params.monthly_inflation_rate) ** month)
        inflated.append((month, real_value))
    return inflated


def summarize_yearly(curve: Iterable[Tuple[int, float]]) -> List[Tuple[int, float]]:
    results = []
    for month, value in curve:
        if month % 12 == 0:
            year = month // 12
            results.append((year, value))
    return results


def write_csv(path: str, rows: Iterable[Tuple[int, float]], label: str) -> None:
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["year", label])
        for year, value in rows:
            writer.writerow([year, value])


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project investment growth with optional Monte Carlo uncertainty bands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--starting-capital", type=float, default=0.0, help="Initial investment amount.")
    parser.add_argument("--monthly-contribution", type=float, default=1000.0, help="Monthly contribution added at the end of each period.")
    parser.add_argument("--expected-annual-return", type=float, default=7.0, help="Expected average annual return (percent).")
    parser.add_argument("--annual-volatility", type=float, default=15.0, help="Annualized standard deviation of returns (percent).")
    parser.add_argument("--analysis-years", type=int, default=30, help="Number of years to project.")
    parser.add_argument("--inflation-rate", type=float, default=2.0, help="Annual inflation assumption (percent) for real-dollar values.")
    parser.add_argument("--contribution-growth-rate", type=float, default=2.0, help="Annual increase in contributions (percent).")
    parser.add_argument("--monte-carlo-trials", type=int, default=0, help="Number of random simulations to run; 0 disables Monte Carlo.")
    parser.add_argument("--percentiles", type=float, nargs="*", default=[5.0, 50.0, 95.0], help="Percentiles to report when running Monte Carlo.")
    parser.add_argument("--output-csv", type=str, help="Optional path to write yearly summary CSV for the median curve.")
    parser.add_argument("--use-real-dollars", action="store_true", help="Discount results for inflation to show in today's dollars.")
    return parser.parse_args(argv)


def build_params(args: argparse.Namespace) -> InvestmentParameters:
    if args.starting_capital < 0 or args.monthly_contribution < 0:
        raise ValueError("Starting capital and contributions cannot be negative.")
    if args.analysis_years <= 0:
        raise ValueError("Analysis years must be positive.")
    if args.monte_carlo_trials < 0:
        raise ValueError("Monte Carlo trials cannot be negative.")

    expected_return = args.expected_annual_return / 100.0
    volatility = args.annual_volatility / 100.0
    inflation = args.inflation_rate / 100.0
    contribution_growth = args.contribution_growth_rate / 100.0

    return InvestmentParameters(
        starting_capital=args.starting_capital,
        monthly_contribution=args.monthly_contribution,
        expected_annual_return=expected_return,
        annual_volatility=volatility,
        analysis_years=args.analysis_years,
        inflation_rate=inflation,
        contribution_growth_rate=contribution_growth,
        monte_carlo_trials=args.monte_carlo_trials,
    )


def report_results(
    params: InvestmentParameters,
    deterministic_curve: List[Tuple[int, float]],
    streams: List[List[float]],
    percentiles: Sequence[float],
    use_real_dollars: bool,
) -> None:
    def adjust(curve: Iterable[Tuple[int, float]]) -> List[Tuple[int, float]]:
        return inflate_values(curve, params) if use_real_dollars else list(curve)

    base_curve = adjust(deterministic_curve)
    yearly_base = summarize_yearly(base_curve)
    print("Deterministic Projection")
    print("========================")
    print(f"Final balance after {params.analysis_years} years: {format_currency(base_curve[-1][1])}")
    print(f"Total contributions: {format_currency(total_contributions(params))}")
    print("Year-by-year (deterministic):")
    for year, value in yearly_base:
        print(f"  Year {year:>2}: {format_currency(value)}")

    if streams:
        print("\nMonte Carlo Summary")
        print("===================")
        for percentile in percentiles:
            percentile_curve_data = percentile_curve(streams, percentile)
            percentile_curve_adjusted = adjust(
                [(month, percentile_curve_data[month - 1][1]) for month, _ in base_curve]
            )
            print(f"{percentile:>5.1f} percentile final balance: {format_currency(percentile_curve_adjusted[-1][1])}")


def total_contributions(params: InvestmentParameters) -> float:
    contribution = params.monthly_contribution
    total = 0.0
    for _ in range(params.months):
        total += contribution
        contribution *= 1.0 + params.monthly_contribution_growth
    return total


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    params = build_params(args)
    deterministic_curve = deterministic_projection(params)
    streams = []
    if params.monte_carlo_trials > 0 and params.annual_volatility > 0:
        streams = monte_carlo_projection(params)

    report_results(
        params=params,
        deterministic_curve=deterministic_curve,
        streams=streams,
        percentiles=args.percentiles,
        use_real_dollars=args.use_real_dollars,
    )

    if args.output_csv:
        base_curve = deterministic_curve
        if args.use_real_dollars:
            base_curve = inflate_values(base_curve, params)
        yearly = summarize_yearly(base_curve)
        write_csv(args.output_csv, yearly, label="balance_real" if args.use_real_dollars else "balance_nominal")
        print(f"\nSummary written to {args.output_csv}")


if __name__ == "__main__":
    main()
