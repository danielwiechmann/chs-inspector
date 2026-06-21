#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import mannwhitneyu, kruskal, spearmanr, pearsonr


DEFAULT_CSV = (
    "/Users/daniel24/Documents/0_Exaia/0_Research/NOESAIS/CHS_composite_clean/"
    "0_handover/CHS_composite_handover/validation/all_subject_chs_scores_for_distributions.csv"
)

SCORE_COLS = [
    "chs_global_v1_1_z",
    "chs_complexity_readiness_weighted_z",
    "chs_information_density_readiness_weighted_z",
    "whisper_fluency_primary_health_z",
]

PERCENTILE_COLS = [
    "chs_global_v1_1_z_hc_percentile",
    "chs_complexity_readiness_weighted_z_hc_percentile",
    "chs_information_density_readiness_weighted_z_hc_percentile",
    "whisper_fluency_primary_health_z_hc_percentile",
]

OBJECTIVE_SCORE_MAP = {
    "chs_global_v1_1_z": "chs_global_v1_1_score_0_100",
    "chs_complexity_readiness_weighted_z": "chs_complexity_score_0_100",
    "chs_information_density_readiness_weighted_z": "chs_information_density_score_0_100",
    "whisper_fluency_primary_health_z": "whisper_fluency_score_0_100",
}

OBJECTIVE_SCORE_COLS = list(OBJECTIVE_SCORE_MAP.values())

LABELS = {
    "chs_global_v1_1_z": "Global CHS v1.1",
    "chs_complexity_readiness_weighted_z": "Complexity",
    "chs_information_density_readiness_weighted_z": "Information density",
    "whisper_fluency_primary_health_z": "Whisper fluency",
    "chs_global_v1_1_z_hc_percentile": "Global CHS HC percentile",
    "chs_complexity_readiness_weighted_z_hc_percentile": "Complexity HC percentile",
    "chs_information_density_readiness_weighted_z_hc_percentile": "Information density HC percentile",
    "whisper_fluency_primary_health_z_hc_percentile": "Whisper fluency HC percentile",
    "chs_global_v1_1_score_0_100": "Global CHS score 0–100",
    "chs_complexity_score_0_100": "Complexity score 0–100",
    "chs_information_density_score_0_100": "Information density score 0–100",
    "whisper_fluency_score_0_100": "Whisper fluency score 0–100",
    "moca_total": "MoCA total",
}

GROUP_ORDER = ["HC", "AD", "PD", "VCI", "CAADMI"]
TASK_ORDER = ["cookie_theft", "free_speech"]
DATASET_ORDER = ["AixCog", "CAADMI"]


def pretty(col: str) -> str:
    return LABELS.get(col, col)


def score_help_text(score: str, center: float = 70.0, sd_points: float = 15.0) -> str:
    if score in OBJECTIVE_SCORE_COLS:
        return objective_score_formula_text(center, sd_points)
    if score.endswith("_hc_percentile"):
        return (
            "Empirical HC percentile = 100 × P_HC(score ≤ observed score). "
            "This is based on the small AixCog HC ECDF and can show floor effects."
        )
    if score == "chs_global_v1_1_z":
        return (
            "Scientific global CHS z-score = mean(complexity z, information-density z, fluency z). "
            "Positive values are healthier than the AixCog HC reference mean; negative values are lower."
        )
    if score == "chs_complexity_readiness_weighted_z":
        return (
            "Complexity component z-score from MeanLengthWord, MeanLengthClause, and MeanLengthSentence. "
            "Higher values indicate greater linguistic complexity relative to AixCog HC."
        )
    if score == "chs_information_density_readiness_weighted_z":
        return (
            "Information-density component z-score from Base.Kolmogorov_len_resid_CNref only. "
            "Higher values indicate more information density than expected after length residualization."
        )
    if score == "whisper_fluency_primary_health_z":
        return (
            "Whisper fluency health z-score from words_per_min_audio, pause_ratio_1000ms, "
            "mean_run_words_1000ms, and filled_pause_per_100_words. Pause burden and filled pauses are reversed."
        )
    return "Selected score."


def ordered(values, preferred):
    vals = pd.Series(values).dropna().astype(str).unique().tolist()
    out = [x for x in preferred if x in vals]
    out.extend(sorted([x for x in vals if x not in preferred]))
    return out


def safe_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else str(x))
    return out


def z_to_objective_score(z, center: float = 50.0, sd_points: float = 15.0):
    return np.clip(center + sd_points * pd.to_numeric(z, errors="coerce"), 0, 100)


def objective_score_formula_text(center: float, sd_points: float) -> str:
    return (
        f"Objective score = clip({center:.0f} + {sd_points:.0f} × z, 0, 100). "
        f"{center:.0f} equals the AixCog HC reference mean; each 1 SD changes the score by {sd_points:.0f} points."
    )


def prepare_loaded_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize loaded CHS distribution data after reading from path or upload."""
    df = df.copy()

    for c in SCORE_COLS + PERCENTILE_COLS + OBJECTIVE_SCORE_COLS + ["moca_total", "age", "education"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Objective 0–100 scores are computed dynamically in main()
    # because the dashboard allows the center/bias and SD scaling to be adjusted.

    for c in ["dataset", "task", "group", "TID", "ID", "base_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return prepare_loaded_data(df)


@st.cache_data
def load_uploaded_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    return prepare_loaded_data(df)


def summary_table(df: pd.DataFrame, score_cols: list[str], by_cols: list[str]) -> pd.DataFrame:
    rows = []

    for keys, g in df.groupby(by_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_dict = dict(zip(by_cols, keys))

        for score in score_cols:
            if score not in g.columns:
                continue

            vals = pd.to_numeric(g[score], errors="coerce").dropna()

            row = {
                **key_dict,
                "score": pretty(score),
                "score_col": score,
                "n": int(vals.size),
                "mean": vals.mean() if vals.size else np.nan,
                "sd": vals.std(ddof=1) if vals.size > 1 else np.nan,
                "median": vals.median() if vals.size else np.nan,
                "q25": vals.quantile(0.25) if vals.size else np.nan,
                "q75": vals.quantile(0.75) if vals.size else np.nan,
                "iqr": (vals.quantile(0.75) - vals.quantile(0.25)) if vals.size else np.nan,
                "min": vals.min() if vals.size else np.nan,
                "max": vals.max() if vals.size else np.nan,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def pairwise_table(df: pd.DataFrame, score: str, group_col: str, reference_group: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    groups = ordered(df[group_col], GROUP_ORDER)
    if reference_group not in groups:
        return pd.DataFrame()

    ref = pd.to_numeric(df.loc[df[group_col].astype(str) == reference_group, score], errors="coerce").dropna()

    rows = []
    for g in groups:
        if g == reference_group:
            continue

        vals = pd.to_numeric(df.loc[df[group_col].astype(str) == g, score], errors="coerce").dropna()

        if len(ref) < 2 or len(vals) < 2:
            rows.append({
                "reference": reference_group,
                "comparison": g,
                "n_ref": len(ref),
                "n_comparison": len(vals),
                "median_ref": ref.median() if len(ref) else np.nan,
                "median_comparison": vals.median() if len(vals) else np.nan,
                "delta_median": np.nan,
                "mann_whitney_p": np.nan,
                "auc_ref_gt_comparison": np.nan,
                "cliff_delta_ref_gt_comparison": np.nan,
            })
            continue

        u, p = mannwhitneyu(ref, vals, alternative="two-sided")
        auc = u / (len(ref) * len(vals))
        cliff = 2 * auc - 1

        rows.append({
            "reference": reference_group,
            "comparison": g,
            "n_ref": len(ref),
            "n_comparison": len(vals),
            "median_ref": ref.median(),
            "median_comparison": vals.median(),
            "delta_median": ref.median() - vals.median(),
            "mann_whitney_p": p,
            "auc_ref_gt_comparison": auc,
            "cliff_delta_ref_gt_comparison": cliff,
        })

    return pd.DataFrame(rows)


def kruskal_table(df: pd.DataFrame, score: str, group_col: str) -> pd.DataFrame:
    groups = []
    names = []

    for g in ordered(df[group_col], GROUP_ORDER):
        vals = pd.to_numeric(df.loc[df[group_col].astype(str) == g, score], errors="coerce").dropna()
        if len(vals) >= 2:
            groups.append(vals)
            names.append(g)

    if len(groups) < 2:
        return pd.DataFrame([{"score": pretty(score), "groups": ", ".join(names), "H": np.nan, "p": np.nan}])

    h, p = kruskal(*groups)

    return pd.DataFrame([{
        "score": pretty(score),
        "groups": ", ".join(names),
        "H": h,
        "p": p,
    }])


def correlation_table(df: pd.DataFrame, score_cols: list[str], y_col: str, method: str) -> pd.DataFrame:
    rows = []

    if y_col not in df.columns:
        return pd.DataFrame()

    for s in score_cols:
        if s not in df.columns:
            continue

        w = df[[s, y_col]].dropna()

        if len(w) < 3 or w[s].nunique() < 2 or w[y_col].nunique() < 2:
            rows.append({"score": pretty(s), "score_col": s, "y": y_col, "n": len(w), "r": np.nan, "p": np.nan})
            continue

        if method == "Spearman":
            r, p = spearmanr(w[s], w[y_col])
        else:
            r, p = pearsonr(w[s], w[y_col])

        rows.append({"score": pretty(s), "score_col": s, "y": y_col, "n": len(w), "r": r, "p": p})

    return pd.DataFrame(rows)


def add_hover(df: pd.DataFrame, score: str) -> list[str]:
    out = []
    for _, r in df.iterrows():
        out.append(
            f"dataset={r.get('dataset', '')}<br>"
            f"group={r.get('group', '')}<br>"
            f"task={r.get('task', '')}<br>"
            f"TID={r.get('TID', '')}<br>"
            f"ID={r.get('ID', '')}<br>"
            f"{pretty(score)}={r.get(score, np.nan):.3f}<br>"
            f"MoCA={r.get('moca_total', '')}"
        )
    return out


def plot_distribution(
    df: pd.DataFrame,
    score: str,
    group_col: str,
    plot_type: str,
    facet_col: str | None,
    show_hc_reference: bool = True,
    show_hc_lines: bool = True,
    objective_center: float = 70.0,
    objective_sd_points: float = 15.0,
) -> go.Figure:
    fig = go.Figure()

    if facet_col and facet_col in df.columns and facet_col != "None":
        facet_values = ordered(
            df[facet_col],
            TASK_ORDER if facet_col == "task" else DATASET_ORDER if facet_col == "dataset" else GROUP_ORDER,
        )
    else:
        facet_values = ["All"]

    # HC reference is always AixCog HC within the current task-filtered context.
    hc_ref = df[df["group"].astype(str) == "HC"].copy() if "group" in df.columns else pd.DataFrame()
    hc_vals = pd.to_numeric(hc_ref[score], errors="coerce").dropna() if not hc_ref.empty else pd.Series(dtype=float)

    if show_hc_reference and len(hc_vals) > 0:
        if plot_type in ["Violin", "Box"]:
            label = "HC reference"
            if plot_type == "Violin":
                fig.add_trace(go.Violin(
                    y=hc_vals,
                    x=[label] * len(hc_vals),
                    name=label,
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    jitter=0.25,
                    text=add_hover(hc_ref.loc[hc_vals.index], score),
                    hovertemplate="%{text}<extra></extra>",
                    line=dict(width=3),
                ))
            else:
                fig.add_trace(go.Box(
                    y=hc_vals,
                    x=[label] * len(hc_vals),
                    name=label,
                    boxmean=True,
                    boxpoints="all",
                    jitter=0.25,
                    text=add_hover(hc_ref.loc[hc_vals.index], score),
                    hovertemplate="%{text}<extra></extra>",
                    line=dict(width=3),
                ))

        elif plot_type == "Histogram":
            fig.add_trace(go.Histogram(
                x=hc_vals,
                name=f"HC reference (n={len(hc_vals)})",
                opacity=0.35,
                marker=dict(line=dict(width=2)),
                hovertemplate=f"HC reference<br>{pretty(score)}=%{{x:.3f}}<br>count=%{{y}}<extra></extra>",
            ))

        elif plot_type == "EPDF":
            fig.add_trace(go.Histogram(
                x=hc_vals,
                name=f"HC reference (n={len(hc_vals)})",
                histnorm="probability density",
                opacity=0.35,
                marker=dict(line=dict(width=2)),
                hovertemplate=f"HC reference<br>{pretty(score)}=%{{x:.3f}}<br>density=%{{y:.3f}}<extra></extra>",
            ))

        elif plot_type == "ECDF":
            x = np.sort(hc_vals.to_numpy())
            y = np.arange(1, len(x) + 1) / len(x) * 100
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=f"HC reference (n={len(x)})",
                line=dict(width=4),
                hovertemplate=f"HC reference<br>{pretty(score)}=%{{x:.3f}}<br>ECDF=%{{y:.1f}}%<extra></extra>",
            ))

    for facet in facet_values:
        sub = df.copy() if facet == "All" else df[df[facet_col].astype(str) == str(facet)].copy()

        groups = ordered(
            sub[group_col],
            GROUP_ORDER if group_col == "group" else DATASET_ORDER if group_col == "dataset" else TASK_ORDER,
        )

        for g in groups:
            # Do not duplicate HC if the dedicated HC reference trace is active.
            if show_hc_reference and group_col == "group" and str(g) == "HC":
                continue

            gd = sub[sub[group_col].astype(str) == str(g)].copy()
            vals = pd.to_numeric(gd[score], errors="coerce")
            gd = gd.loc[vals.notna()].copy()
            vals = vals.dropna()

            if gd.empty:
                continue

            name = str(g) if facet == "All" else f"{facet} | {g}"

            if plot_type == "Violin":
                fig.add_trace(go.Violin(
                    y=vals,
                    x=[name] * len(vals),
                    name=name,
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    jitter=0.35,
                    text=add_hover(gd, score),
                    hovertemplate="%{text}<extra></extra>",
                ))

            elif plot_type == "Box":
                fig.add_trace(go.Box(
                    y=vals,
                    x=[name] * len(vals),
                    name=name,
                    boxmean=True,
                    boxpoints="all",
                    jitter=0.35,
                    pointpos=0,
                    text=add_hover(gd, score),
                    hovertemplate="%{text}<extra></extra>",
                ))

            elif plot_type == "Histogram":
                fig.add_trace(go.Histogram(
                    x=vals,
                    name=f"{name} (n={len(vals)})",
                    opacity=0.65,
                    hovertemplate=f"{name}<br>{pretty(score)}=%{{x:.3f}}<br>count=%{{y}}<extra></extra>",
                ))

            elif plot_type == "EPDF":
                fig.add_trace(go.Histogram(
                    x=vals,
                    name=f"{name} (n={len(vals)})",
                    histnorm="probability density",
                    opacity=0.55,
                    hovertemplate=f"{name}<br>{pretty(score)}=%{{x:.3f}}<br>density=%{{y:.3f}}<extra></extra>",
                ))

            elif plot_type == "ECDF":
                x = np.sort(vals.to_numpy())
                y = np.arange(1, len(x) + 1) / len(x) * 100
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=f"{name} (n={len(x)})",
                    hovertemplate=f"{name}<br>{pretty(score)}=%{{x:.3f}}<br>ECDF=%{{y:.1f}}%<extra></extra>",
                ))

    if show_hc_lines and len(hc_vals) > 0:
        hc_mean = float(hc_vals.mean())
        hc_median = float(hc_vals.median())

        if plot_type in ["Violin", "Box"]:
            fig.add_hline(y=hc_mean, line_dash="dash", annotation_text=f"HC mean={hc_mean:.2f}")
            fig.add_hline(y=hc_median, line_dash="dot", annotation_text=f"HC median={hc_median:.2f}")
        else:
            fig.add_vline(x=hc_mean, line_dash="dash", annotation_text=f"HC mean={hc_mean:.2f}")
            fig.add_vline(x=hc_median, line_dash="dot", annotation_text=f"HC median={hc_median:.2f}")

    if plot_type in ["Violin", "Box"]:
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(yaxis_title=pretty(score), xaxis_title=group_col)
    else:
        if score.endswith("_hc_percentile"):
            fig.update_xaxes(range=[0, 100])
            fig.add_vline(x=50, line_dash="dot", annotation_text="50th HC pct")
            fig.add_vline(x=25, line_dash="dash")
            fig.add_vline(x=75, line_dash="dash")
        elif score.endswith("_0_100"):
            fig.update_xaxes(range=[0, 100])
            fig.add_vline(x=objective_center, line_dash="dot", annotation_text=f"HC mean = {objective_center:.0f}")
            fig.add_vline(x=max(0, objective_center - objective_sd_points), line_dash="dash", annotation_text="-1 SD")
            fig.add_vline(x=min(100, objective_center + objective_sd_points), line_dash="dash", annotation_text="+1 SD")
        else:
            fig.add_vline(x=0, line_dash="dash")
        if plot_type == "Histogram":
            yaxis_title = "Count"
        elif plot_type == "EPDF":
            yaxis_title = "Empirical probability density"
        else:
            yaxis_title = "Cumulative percent"

        fig.update_layout(xaxis_title=pretty(score), yaxis_title=yaxis_title)

    if score.endswith("_hc_percentile"):
        title_suffix = "HC percentile; empirical and coarse"
    elif score.endswith("_0_100"):
        title_suffix = f"objective user-facing score; {objective_center:.0f} = HC mean, {objective_sd_points:.0f} points = 1 SD"
    else:
        title_suffix = "scientific z-score"

    fig.update_layout(
        title=f"{pretty(score)} distribution ({title_suffix})",
        height=700,
        margin=dict(l=40, r=20, t=80, b=130),
        legend_title="Group / reference",
        barmode="overlay" if plot_type in ["Histogram", "EPDF"] else None,
    )

    return fig


def plot_mean_bars(summary: pd.DataFrame, score: str, x_col: str) -> go.Figure:
    s = summary[summary["score_col"] == score].copy()

    if s.empty:
        fig = go.Figure()
        fig.add_annotation(text="No summary data", x=0.5, y=0.5, showarrow=False)
        return fig

    label = s[x_col].astype(str)

    fig = go.Figure(go.Bar(
        x=label,
        y=s["mean"],
        error_y=dict(type="data", array=s["sd"].fillna(0)),
        text=[f"n={int(n)}<br>mean={m:.2f}" for n, m in zip(s["n"], s["mean"])],
        hovertemplate="%{text}<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        title=f"Mean {pretty(score)} with SD",
        xaxis_title=x_col,
        yaxis_title=pretty(score),
        height=500,
        margin=dict(l=40, r=20, t=70, b=80),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="CHS Distribution Dashboard", layout="wide")

    st.title("CHS v1.1 Distribution Dashboard")
    st.caption("Generate distribution plots and summary tables across datasets, tasks, and groups.")

    with st.sidebar:
        st.header("Data source")
        uploaded_csv = st.file_uploader(
            "Upload CHS distribution CSV",
            type=["csv"],
            help=(
                "Use this when the app is deployed online. The selected CSV is uploaded "
                "from the user's local computer to the Streamlit session."
            ),
        )
        csv_path = st.text_input(
            "Fallback/local CSV path",
            DEFAULT_CSV,
            help=(
                "Used only when no CSV is uploaded. In online deployments, this path refers "
                "to the server/repository filesystem, not the user's local computer."
            ),
        )

    if uploaded_csv is not None:
        df = load_uploaded_data(uploaded_csv.getvalue())
        st.sidebar.success(f"Using uploaded CSV: {uploaded_csv.name}")
    else:
        df = load_data(csv_path)

    with st.sidebar:
        st.header("Objective score calibration")
        objective_center = st.number_input(
            "Objective score center / bias",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=5.0,
            help="Score assigned to z=0, i.e. the AixCog HC reference mean. A higher value reduces floor effects for impaired participants.",
        )
        objective_sd_points = st.number_input(
            "Points per 1 z-SD",
            min_value=5.0,
            max_value=30.0,
            value=15.0,
            step=1.0,
            help="How many 0–100 score points correspond to one z-score unit. Larger values increase separation but also increase clipping at 0/100.",
        )
        st.caption(objective_score_formula_text(objective_center, objective_sd_points))

    for z_col, score_col in OBJECTIVE_SCORE_MAP.items():
        if z_col in df.columns:
            df[score_col] = z_to_objective_score(df[z_col], center=objective_center, sd_points=objective_sd_points)

    available_scores = [c for c in SCORE_COLS + OBJECTIVE_SCORE_COLS + PERCENTILE_COLS if c in df.columns]

    with st.sidebar:
        st.header("Filters")

        datasets = ordered(df["dataset"], DATASET_ORDER) if "dataset" in df.columns else []
        selected_dataset = st.selectbox("Dataset", ["All"] + datasets)

        if selected_dataset == "All":
            dataset_filtered = df.copy()
        else:
            dataset_filtered = df[df["dataset"].astype(str) == selected_dataset].copy()

        tasks = ordered(dataset_filtered["task"], TASK_ORDER) if "task" in dataset_filtered.columns else []
        selected_task = st.selectbox("Task", ["All"] + tasks)

        if selected_task == "All":
            filtered = dataset_filtered.copy()
        else:
            filtered = dataset_filtered[dataset_filtered["task"].astype(str) == selected_task].copy()

        groups = ordered(filtered["group"], GROUP_ORDER) if "group" in filtered.columns else []
        selected_groups = st.multiselect("Groups", groups, default=groups)

        if selected_groups:
            filtered = filtered[filtered["group"].astype(str).isin(selected_groups)].copy()

        st.write(f"Rows after filters: {len(filtered)}")

        st.header("HC reference display")
        show_hc_reference = st.checkbox("Show HC as separate reference group", value=True)
        show_hc_lines = st.checkbox("Show HC mean/median lines", value=True)

    tab_plot, tab_summary, tab_stats, tab_export = st.tabs([
        "Plots",
        "Summary tables",
        "Statistics",
        "Export",
    ])

    with tab_plot:
        c1, c2, c3, c4 = st.columns(4)

        score_scale = c1.selectbox(
            "Score scale",
            [
                "Objective CHS score 0–100",
                "Scientific z-score",
                "HC percentile 0–100",
            ],
        )

        if score_scale.startswith("Objective"):
            score_pool = [c for c in OBJECTIVE_SCORE_COLS if c in available_scores]
        elif score_scale.startswith("Scientific"):
            score_pool = [c for c in SCORE_COLS if c in available_scores]
        else:
            score_pool = [c for c in PERCENTILE_COLS if c in available_scores]

        score = c1.selectbox("Score", score_pool, format_func=pretty)
        plot_type = c2.selectbox(
            "Plot type",
            ["Violin", "Box", "Histogram", "ECDF", "EPDF"],
            help=(
                "ECDF shows cumulative probability. EPDF shows the empirical density/mass "
                "of observed scores and is useful for detecting clustering and floor effects."
            ),
        )
        group_col = c3.selectbox("Group by", [c for c in ["group", "dataset", "task"] if c in filtered.columns])
        facet_col = c4.selectbox("Separate traces by", ["None"] + [c for c in ["dataset", "task", "group"] if c in filtered.columns and c != group_col])

        st.plotly_chart(
            plot_distribution(
                filtered,
                score,
                group_col,
                plot_type,
                facet_col,
                show_hc_reference=show_hc_reference,
                show_hc_lines=show_hc_lines,
                objective_center=objective_center,
                objective_sd_points=objective_sd_points,
            ),
            use_container_width=True,
        )

        if plot_type == "ECDF":
            st.caption(
                "ECDF shows the cumulative percentage of observations with scores less than or equal to each value."
            )
        elif plot_type == "EPDF":
            st.caption(
                "EPDF shows the empirical density of observed scores along the score axis. "
                "For HC-percentile scores, this view makes coarse steps, clustering, and floor effects visible."
            )

        by_cols = [group_col]
        if facet_col != "None":
            by_cols = [facet_col, group_col]

        summary = summary_table(filtered, [score], by_cols)

        st.plotly_chart(
            plot_mean_bars(summary, score, x_col=group_col),
            use_container_width=True,
        )

    with tab_summary:
        st.subheader("Distribution summary table")

        by_cols = st.multiselect(
            "Summarize by",
            [c for c in ["dataset", "task", "group"] if c in filtered.columns],
            default=[c for c in ["dataset", "task", "group"] if c in filtered.columns],
        )

        selected_scores = st.multiselect(
            "Scores",
            available_scores,
            default=[c for c in OBJECTIVE_SCORE_COLS if c in available_scores],
            format_func=pretty,
        )

        summary = summary_table(filtered, selected_scores, by_cols)
        rounded = summary.round({
            "mean": 3,
            "sd": 3,
            "median": 3,
            "q25": 3,
            "q75": 3,
            "iqr": 3,
            "min": 3,
            "max": 3,
        })

        st.dataframe(safe_df(rounded), use_container_width=True, hide_index=True)

        st.download_button(
            "Download summary CSV",
            rounded.to_csv(index=False).encode("utf-8"),
            file_name="chs_distribution_summary.csv",
            mime="text/csv",
        )

    with tab_stats:
        st.subheader("Group comparison statistics")

        c1, c2, c3 = st.columns(3)

        stat_score = c1.selectbox("Score for tests", available_scores, format_func=pretty, key="stat_score")
        group_col = c2.selectbox("Compare groups using", [c for c in ["group", "dataset", "task"] if c in filtered.columns], key="stat_group_col")
        reference_group = c3.selectbox("Reference group", ordered(filtered[group_col], GROUP_ORDER if group_col == "group" else DATASET_ORDER if group_col == "dataset" else TASK_ORDER))

        ktab = kruskal_table(filtered, stat_score, group_col).round({"H": 3, "p": 5})
        ptab = pairwise_table(filtered, stat_score, group_col, reference_group).round({
            "median_ref": 3,
            "median_comparison": 3,
            "delta_median": 3,
            "mann_whitney_p": 5,
            "auc_ref_gt_comparison": 3,
            "cliff_delta_ref_gt_comparison": 3,
        })

        st.markdown("#### Overall Kruskal-Wallis test")
        st.dataframe(safe_df(ktab), use_container_width=True, hide_index=True)

        st.markdown("#### Pairwise reference comparisons")
        st.dataframe(safe_df(ptab), use_container_width=True, hide_index=True)

        if "moca_total" in filtered.columns:
            st.markdown("#### Correlations with MoCA")
            method = st.selectbox("Correlation method", ["Spearman", "Pearson"])
            ctab = correlation_table(filtered, [c for c in SCORE_COLS if c in filtered.columns], "moca_total", method).round({"r": 3, "p": 5})
            st.dataframe(safe_df(ctab), use_container_width=True, hide_index=True)

    with tab_export:
        st.subheader("Filtered data")
        cols_default = [c for c in ["dataset", "task", "group", "TID", "ID", "base_id", "moca_total"] + OBJECTIVE_SCORE_COLS + SCORE_COLS + PERCENTILE_COLS if c in filtered.columns]

        cols = st.multiselect(
            "Columns to show/export",
            filtered.columns.tolist(),
            default=cols_default,
        )

        out = filtered[cols].copy() if cols else filtered.copy()
        st.dataframe(safe_df(out), use_container_width=True, hide_index=True)

        st.download_button(
            "Download filtered rows CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name="chs_filtered_distribution_rows.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
