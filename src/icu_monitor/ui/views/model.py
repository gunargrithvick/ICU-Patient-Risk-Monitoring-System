"""Model Insights: the honest numbers, and the reasons not to trust them too far.

This view leads with what the model gets wrong. Macro-F1 0.52 on a three-class problem is
modestly better than the 0.33 a coin flip gives and nowhere near good enough to act on
alone, which is exactly why the fusion layer weights it at 0.45 and lets a published,
validated score (NEWS2) hold a veto through the override rules. A dashboard that showed only
accuracy 0.57 without the class breakdown would be technically true and practically
misleading, because the HIGH class - the one that matters - is 14% of the data.
"""

from __future__ import annotations

import streamlit as st

from icu_monitor.ui import charts, theme
from icu_monitor.ui import components as ui
from icu_monitor.ui import state as app_state


def _headline(metrics: dict) -> None:
    ui.kpi_row(
        [
            (
                "Macro F1",
                f"{metrics.get('macro_f1', 0):.3f}",
                "Unweighted mean F1 across the three classes. A coin flip scores ~0.33.",
            ),
            (
                "Balanced accuracy",
                f"{metrics.get('balanced_accuracy', 0):.3f}",
                "Mean per-class recall; immune to class imbalance.",
            ),
            (
                "ROC-AUC (macro)",
                f"{(metrics.get('roc_auc') or {}).get('macro', 0):.3f}",
                "Ranking quality averaged over classes.",
            ),
            (
                "Cohen's κ",
                f"{metrics.get('cohen_kappa', 0):.3f}",
                "Agreement above chance. 0.30 is 'fair'.",
            ),
            (
                "Held-out patients",
                f"{metrics.get('n_patients', 0):,}",
                "Patient-disjoint from every training window.",
            ),
        ]
    )


def _per_class(metrics: dict) -> None:
    per_class = metrics.get("per_class") or {}
    roc = metrics.get("roc_auc") or {}
    ap = metrics.get("average_precision") or {}
    if not per_class:
        return
    with st.container(border=True):
        st.markdown("**Per-class performance**")
        ui.caption(
            "HIGH is the class the system exists to catch and the hardest one: it is a "
            "minority of the data, and its precision is the number to read before trusting "
            "any single prediction."
        )
        rows = []
        for label, scores in per_class.items():
            support = int(scores.get("support", 0))
            total = max(1, int(metrics.get("n_samples", 0)))
            rows.append(
                {
                    "Class": f"{theme.level_glyph(label)} {label}",
                    "Precision": round(float(scores.get("precision", 0)), 3),
                    "Recall": round(float(scores.get("recall", 0)), 3),
                    "F1": round(float(scores.get("f1", 0)), 3),
                    "ROC-AUC": round(float(roc.get(label, 0)), 3),
                    "Avg precision": round(float(ap.get(label, 0)), 3),
                    "Windows": support,
                    "Prevalence": f"{support / total:.1%}",
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")


def render() -> None:
    state = app_state.state()
    meta = state.metadata()
    metrics = meta.get("metrics") or {}
    card = meta.get("card") or {}

    ui.page_header(
        "Model insights",
        "Held-out performance, calibration, and the model card - including its limits.",
        right=f"{meta.get('version') or 'no artefact'}",
    )

    if not meta.get("available"):
        ui.empty_state(
            "No trained artefact is loaded. Scoring still runs on NEWS2 and the vision "
            "signal; train one with `python -m icu_monitor train`.",
            icon="◇",
        )
        return

    with st.container(border=True):
        left, right = st.columns([0.62, 0.38])
        with left:
            ui.definition_list(
                {
                    "Version": meta.get("version") or "—",
                    "Algorithm": meta.get("algorithm") or "—",
                    "Trained at": str(meta.get("trained_at") or "—")[:19].replace("T", " "),
                    "Features": meta.get("feature_count") or "—",
                    "Classes": ", ".join(meta.get("classes") or []),
                }
            )
        with right:
            ui.caption(
                "Reloading re-reads the artefact from disk, so a model retrained in another "
                "process is picked up without restarting the dashboard."
            )
            if st.button("Reload artefact", width="stretch"):
                # ``reload_model`` returns the model, not its version - naming the version
                # explicitly, because interpolating the object prints a Python repr at the
                # reader.
                reloaded = app_state.engine().reload_model()
                st.success(
                    f"Loaded {reloaded.version}." if reloaded else "No artefact found on disk."
                )

    _headline(metrics)
    _per_class(metrics)

    left, right = st.columns([0.44, 0.56])
    with left:
        labels = metrics.get("confusion_labels") or []
        matrix = metrics.get("confusion") or []
        ui.chart_panel(
            "Confusion matrix (row-normalised)",
            charts.confusion_heatmap(matrix, labels) if matrix and labels else None,
            note="Normalised by actual class. Raw counts would flatter a model that leans "
            "towards the majority class.",
        )
    with right:
        calibration = metrics.get("calibration") or {}
        brier = calibration.get("brier")
        ece = calibration.get("expected_calibration_error")
        ui.chart_panel(
            f"Calibration · P({calibration.get('target_class', 'HIGH')})",
            charts.calibration_curve(
                calibration.get("bins") or [], target=str(calibration.get("target_class", "HIGH"))
            ),
            note=(
                f"Brier {brier:.4f} · expected calibration error {ece:.1%}. Points above the "
                "diagonal mean the model is under-confident in that band."
                if brier is not None and ece is not None
                else "Predicted probability against observed frequency."
            ),
        )

    ui.chart_panel(
        "Permutation importance · top features",
        charts.importance_bars(metrics.get("importances") or []),
        note="Measured on held-out data by shuffling one feature at a time. The whiskers are "
        "one standard deviation across repeats; overlapping bars are not reliably ordered.",
    )

    st.markdown("#### Model card")
    ui.caption(
        "Written at training time and read from disk here, so it cannot drift from the "
        "artefact it describes."
    )
    sections = [
        ("Intended use and what is out of scope", "intended_use"),
        ("Training data and split", "training_data"),
        ("Label definitions", "labels"),
        ("Candidates considered", "candidates_considered"),
        ("Ethical considerations", "ethical_considerations"),
        ("Caveats and recommendations", "caveats_and_recommendations"),
    ]
    for title, key in sections:
        payload = card.get(key)
        if not payload:
            continue
        with st.expander(title, expanded=key == "caveats_and_recommendations"):
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        ui.definition_list(item)
                        st.markdown("")
                    else:
                        st.markdown(f"- {item}")
            elif isinstance(payload, dict):
                for name, value in payload.items():
                    if isinstance(value, list):
                        st.markdown(f"**{name.replace('_', ' ').title()}**")
                        for item in value:
                            st.markdown(f"- {item}")
                    elif isinstance(value, dict):
                        st.markdown(f"**{name.replace('_', ' ').title()}**")
                        ui.definition_list(value)
                    else:
                        st.markdown(f"**{name.replace('_', ' ').title()}** — {value}")
            else:
                st.markdown(str(payload))


__all__ = ["render"]
