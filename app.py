from __future__ import annotations

import asyncio
from typing import Dict, List, Tuple
import contextlib

import matplotlib

matplotlib.use("Agg")

from shiny import App, reactive, render, ui

from performance_drivers import (
    CONFIG,
    CORRELATION_WINDOWS,
    create_bar_chart_figure,
    create_corr_table_figure,
    create_rebased_figure,
    run_analysis,
)

DEFAULT_TARGET = CONFIG.get("default_target") or "IBE.MC"
DEFAULT_MONTHS = CONFIG.get("default_months", 12)
GROUP_SPECS: List[Tuple[str, str]] = [
    (group_name, group_name.lower().replace(" ", "_"))
    for group_name in CONFIG["groups"].keys()
]


def _default_group_entries(group_name: str) -> List[Dict[str, str]]:
    entries = [dict(entry) for entry in CONFIG["groups"].get(group_name, [])]
    return entries or [{"name": "", "ticker": ""}]


def feature_group_card(group_name: str, slug: str) -> ui.Tag:
    return ui.card(
        ui.h4(group_name),
        ui.output_ui(f"{slug}_fields"),
        ui.div(
            ui.input_action_button(
                f"{slug}_add",
                "Add feature",
                class_="btn btn-secondary btn-sm mt-2 me-2",
            ),
            ui.input_action_button(
                f"{slug}_remove",
                "Remove feature",
                class_="btn btn-outline-danger btn-sm mt-2",
            ),
        ),
    )


group_cards = [feature_group_card(name, slug) for name, slug in GROUP_SPECS]


app_ui = ui.page_fluid(
    ui.h2("Performance Drivers Dashboard"),
    ui.row(
        ui.column(
            8,
            ui.input_text("target_equity", "Target Equity", DEFAULT_TARGET),
        ),
        ui.column(
            4,
            ui.input_action_button(
                "run_analysis",
                "Run analysis",
                class_="btn btn-primary mt-4 float-end",
            ),
        ),
    ),
    ui.p("Enter factor names and tickers for each group. Expand the panel to edit the explaining features."),
    ui.accordion(
        ui.accordion_panel(
            "Explaining Features",
            ui.layout_column_wrap(
                width="420px",
                *group_cards,
            ),
        ),
        open=False,
    ),
    ui.hr(),
    ui.output_ui("warnings"),
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
            ui.output_plot("rebased_plot", width="100%", height="640px"),
        ),
        ui.column(
            6,
            ui.h4("OLS Regression Summary"),
            ui.output_text_verbatim("model_summary"),
        ),
    ),
)


def placeholder_figure(message: str) -> matplotlib.figure.Figure:
    fig = matplotlib.figure.Figure(figsize=(6, 2.5))
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    return fig


def build_group_fields(slug: str) -> ui.Tag:
    entries = group_states[slug]()
    rows = []
    for idx, entry in enumerate(entries):
        rows.append(
            ui.row(
                ui.column(
                    6,
                    ui.input_text(
                        f"{slug}_name_{idx}",
                        "Name",
                        entry.get("name", ""),
                    ),
                ),
                ui.column(
                    6,
                    ui.input_text(
                        f"{slug}_ticker_{idx}",
                        "Ticker",
                        entry.get("ticker", ""),
                    ),
                ),
            )
        )
    return ui.TagList(*rows)


group_states = {
    slug: reactive.Value(_default_group_entries(group_name))
    for group_name, slug in GROUP_SPECS
}
analysis_result = reactive.Value(None)
analysis_error = reactive.Value(None)
progress_percent = reactive.Value(0)


def collect_group_payload(input, group_states) -> Dict[str, List[Dict[str, str]]]:
    payload: Dict[str, List[Dict[str, str]]] = {}
    for group_name, slug in GROUP_SPECS:
        entries_meta = group_states[slug]()
        rows: List[Dict[str, str]] = []
        for idx in range(len(entries_meta)):
            name_val = input[f"{slug}_name_{idx}"]().strip()
            ticker_val = input[f"{slug}_ticker_{idx}"]().strip()
            if ticker_val:
                rows.append(
                    {
                        "name": name_val or f"{slug}_{idx + 1}",
                        "ticker": ticker_val,
                    }
                )
        payload[group_name] = rows
    return payload


def server(input, output, session):  # noqa: ARG001
    def register_group_handlers(slug: str):
        state = group_states[slug]

        @reactive.effect
        @reactive.event(input[f"{slug}_add"])
        def _add_feature() -> None:
            current = list(state())
            current.append({"name": "", "ticker": ""})
            state.set(current)

        @reactive.effect
        @reactive.event(input[f"{slug}_remove"])
        def _remove_feature() -> None:
            current = list(state())
            if current:
                current.pop()
            if not current:
                current.append({"name": "", "ticker": ""})
            state.set(current)

        @output(id=f"{slug}_fields")
        @render.ui
        def _render_fields():
            return build_group_fields(slug)

    for _, slug in GROUP_SPECS:
        register_group_handlers(slug)

    @reactive.effect
    @reactive.event(input.run_analysis)
    async def perform_analysis():
        target = input.target_equity().strip() or DEFAULT_TARGET
        groups_payload = collect_group_payload(input, group_states)
        analysis_error.set(None)
        try:
            result = await asyncio.to_thread(run_analysis, target, DEFAULT_MONTHS, groups_payload)
            analysis_result.set(result)
        except Exception as exc:  # noqa: BLE001
            analysis_result.set(None)
            analysis_error.set(str(exc))

    @output
    @render.ui
    def warnings():
        msgs = []
        if analysis_error():
            msgs.append(f"Analysis error: {analysis_error()}")
        result = analysis_result()
        if result:
            msgs.extend(result.warnings)
        if not msgs:
            return ui.div()
        return ui.div({"class": "alert alert-warning"}, ui.tags.ul(*[ui.tags.li(msg) for msg in msgs]))

    @output
    @render.text
    def model_summary():
        if analysis_error():
            return f"Cannot compute model: {analysis_error()}"
        result = analysis_result()
        if not result:
            return "Press 'Run analysis' after entering tickers."
        return result.model_summary

    @output
    @render.plot
    def bar_plot():
        result = analysis_result()
        if not result:
            return placeholder_figure("Run the analysis to view contributions.")
        return create_bar_chart_figure(result.group_shares, result.total_r2, result.group_order)

    @output
    @render.plot
    def table_plot():
        result = analysis_result()
        if not result:
            return placeholder_figure("Run the analysis to view correlations.")
        return create_corr_table_figure(result.corr_table, CORRELATION_WINDOWS)

    @output
    @render.plot
    def rebased_plot():
        result = analysis_result()
        if not result:
            return placeholder_figure("Run the analysis to view rebased series.")
        return create_rebased_figure(result.rebased_prices, result.rebased_series)



app = App(app_ui, server)
