from __future__ import annotations

from shiny import App, render, ui
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from performance_drivers import (
    CORRELATION_WINDOWS,
    CONFIG,
    create_bar_chart_figure,
    create_corr_table_figure,
    create_rebased_figure,
    run_analysis,
)

DEFAULT_TARGET = CONFIG.get("default_target") or "IBE.MC"
DEFAULT_MONTHS = CONFIG.get("default_months", 12)

_ANALYSIS_RESULT = None
_ANALYSIS_ERROR = None
_BAR_FIG = None
_TABLE_FIG = None
_REBASED_FIG = None

app_ui = ui.page_fluid(
    ui.h2("Performance Drivers Dashboard"),
    ui.help_text(
        f"Using CONFIG defaults (target: {DEFAULT_TARGET}, months: {DEFAULT_MONTHS}). Edit configuration in performance_drivers.py to change inputs."
    ),
    ui.output_ui("warnings"),
    ui.hr(),
    ui.row(
        ui.column(
            6,
            ui.h4("Factor Contribution Bars"),
            ui.output_plot("bar_plot", width="100%", height="340px"),
        ),
        ui.column(
            6,
            ui.h4("Correlation Table"),
            ui.output_plot("table_plot", width="100%", height="400px"),
        ),
    ),
    ui.row(
        ui.column(
            6,
            ui.h4("Rebased Comparison"),
            ui.output_plot("rebased_plot", width="100%", height="320px"),
        ),
        ui.column(
            6,
            ui.h4("OLS Regression Summary"),
            ui.output_text_verbatim("model_summary"),
        ),
    ),
)


def placeholder_figure(message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    return fig


def ensure_analysis() -> None:
    global _ANALYSIS_RESULT, _ANALYSIS_ERROR, _BAR_FIG, _TABLE_FIG, _REBASED_FIG
    if _ANALYSIS_RESULT is not None or _ANALYSIS_ERROR is not None:
        return
    try:
        result = run_analysis(DEFAULT_TARGET, DEFAULT_MONTHS)
        _ANALYSIS_RESULT = result
        _ANALYSIS_ERROR = None
        _BAR_FIG = create_bar_chart_figure(result.group_shares, result.total_r2, result.group_order)
        _TABLE_FIG = create_corr_table_figure(result.corr_table, CORRELATION_WINDOWS)
        _REBASED_FIG = create_rebased_figure(result.rebased_prices, result.rebased_series)
    except Exception as exc:  # noqa: BLE001
        _ANALYSIS_RESULT = None
        _ANALYSIS_ERROR = str(exc)
        _BAR_FIG = _TABLE_FIG = _REBASED_FIG = None


def server(input, output, session):  # noqa: ARG001
    @output
    @render.ui
    def warnings():
        ensure_analysis()
        messages = []
        if _ANALYSIS_ERROR:
            messages.append(f"Analysis error: {_ANALYSIS_ERROR}")
        if _ANALYSIS_RESULT:
            messages.extend(_ANALYSIS_RESULT.warnings)
        if not messages:
            return ui.div()
        return ui.div({"class": "alert alert-warning"}, ui.tags.ul(*[ui.tags.li(msg) for msg in messages]))

    @output
    @render.text
    def model_summary():
        ensure_analysis()
        if _ANALYSIS_ERROR:
            return f"Cannot compute model: {_ANALYSIS_ERROR}"
        if not _ANALYSIS_RESULT:
            return "Model is running…"
        return _ANALYSIS_RESULT.model_summary

    @output
    @render.plot
    def bar_plot():
        ensure_analysis()
        if _BAR_FIG is None:
            return placeholder_figure(_ANALYSIS_ERROR or "No data available.")
        return _BAR_FIG

    @output
    @render.plot
    def table_plot():
        ensure_analysis()
        if _TABLE_FIG is None:
            return placeholder_figure(_ANALYSIS_ERROR or "No data available.")
        return _TABLE_FIG

    @output
    @render.plot
    def rebased_plot():
        ensure_analysis()
        if _REBASED_FIG is None:
            return placeholder_figure(_ANALYSIS_ERROR or "No data available.")
        return _REBASED_FIG



app = App(app_ui, server)
