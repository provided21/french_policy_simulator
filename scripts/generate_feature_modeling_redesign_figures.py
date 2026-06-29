from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "feature_modeling_redesign_figures"
OUT.mkdir(parents=True, exist_ok=True)

STRUCTURED = ROOT / "output" / "structured_baselines" / "structured_baseline_results.csv"
TEXT = ROOT / "output" / "text_feature_ablation" / "text_feature_ablation_results.csv"

PURPLE = "#6F006F"
LIGHT = "#E8D8E8"
PALE = "#F7F2F7"
MID = "#B987B9"
DARK = "#222222"
GREY = "#666666"
BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"


plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def annotate_heatmap(ax, data: pd.DataFrame, fmt: str = ".3f") -> None:
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            if pd.isna(val):
                continue
            text_color = "white" if val >= np.nanmax(data.values) * 0.82 else DARK
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=9, color=text_color)


def heatmap(ax, data: pd.DataFrame, title: str, cmap: str = "Purples") -> None:
    im = ax.imshow(data.values, cmap=cmap, aspect="auto")
    ax.set_title(title, loc="left", fontsize=15, color=PURPLE, weight="bold", pad=12)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, fontsize=9)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index, fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    annotate_heatmap(ax, data)
    return im


def main() -> None:
    s = pd.read_csv(STRUCTURED)
    t = pd.read_csv(TEXT)

    # 1. Structured feature x model heatmap for regression R2.
    reg = s[s["task"] == "regression"].copy()
    model_order = ["LinearRegression", "Ridge", "RandomForest", "XGBoost", "LightGBM"]
    app_order = ["A0_label", "A1_onehot", "A2_age_engineered", "A3_social_grouped"]
    short_app = {
        "A0_label": "A0 label",
        "A1_onehot": "A1 one-hot",
        "A2_age_engineered": "A2 age+",
        "A3_social_grouped": "A3 social",
    }
    short_model = {
        "LinearRegression": "Linear",
        "Ridge": "Ridge",
        "RandomForest": "RF",
        "XGBoost": "XGB",
        "LightGBM": "LGBM",
    }
    pivot = (
        reg.pivot_table(index="approach", columns="model", values="r2", aggfunc="max")
        .reindex(app_order)[model_order]
        .rename(index=short_app, columns=short_model)
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    im = heatmap(ax, pivot, "结构化特征 × 模型：Test R² 热力图")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Test R²", fontsize=9)
    fig.text(
        0.12,
        0.035,
        "读法：颜色越深，R²越高；A3社会分组 + XGBoost 是结构化阶段最好组合。",
        fontsize=10,
        color=GREY,
    )
    save(fig, "structured_r2_heatmap.png")

    # 2. Structured feature ladder: best regression and best classification by feature scheme.
    cls = s[s["task"] == "classification"].copy()
    best_reg = reg.sort_values("r2").groupby("approach", as_index=False).tail(1).set_index("approach").reindex(app_order)
    best_cls = cls.sort_values("macro_f1").groupby("approach", as_index=False).tail(1).set_index("approach").reindex(app_order)
    x = np.arange(len(app_order))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharex=True)
    axes[0].plot(x, best_reg["r2"], marker="o", lw=2.8, color=PURPLE)
    axes[0].fill_between(x, best_reg["r2"], best_reg["r2"].min() - 0.01, color=LIGHT, alpha=0.75)
    axes[0].set_title("回归：结构化特征逐步增强", loc="left", fontsize=14, color=PURPLE, weight="bold")
    axes[0].set_ylabel("Best Test R²")
    axes[0].set_ylim(0.16, 0.21)
    axes[1].plot(x, best_cls["macro_f1"], marker="o", lw=2.8, color=BLUE)
    axes[1].fill_between(x, best_cls["macro_f1"], best_cls["macro_f1"].min() - 0.015, color="#D9E8F5", alpha=0.9)
    axes[1].set_title("分类：Macro F1 变化", loc="left", fontsize=14, color=PURPLE, weight="bold")
    axes[1].set_ylabel("Best Test Macro F1")
    axes[1].set_ylim(0.36, 0.44)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(["A0\nlabel", "A1\none-hot", "A2\nage+", "A3\nsocial"], fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for i, row in enumerate(best_reg.itertuples()):
        axes[0].text(i, row.r2 + 0.003, f"{row.r2:.3f}", ha="center", fontsize=9, color=DARK)
    for i, row in enumerate(best_cls.itertuples()):
        axes[1].text(i, row.macro_f1 + 0.004, f"{row.macro_f1:.3f}", ha="center", fontsize=9, color=DARK)
    fig.suptitle("结构化基线：one-hot / 年龄工程 / 社会分组只带来小幅提升", fontsize=16, color=PURPLE, weight="bold", y=1.04)
    save(fig, "structured_feature_ladder.png")

    # 3. Text feature ablation line chart, showing small increment over structured baseline.
    treg = t[t["task"] == "regression"].copy()
    tcls = t[t["task"] == "classification"].copy()
    scheme_order = [
        "T0_structured_A3",
        "T1_full_persona_text",
        "T2_career_text",
        "T3_lifestyle_text",
        "T4_field_pca",
    ]
    short_scheme = {
        "T0_structured_A3": "T0\nstructured",
        "T1_full_persona_text": "T1\nfull text",
        "T2_career_text": "T2\ncareer",
        "T3_lifestyle_text": "T3\nlifestyle",
        "T4_field_pca": "T4\nfield PCA",
    }
    best_treg = treg.sort_values("r2").groupby("feature_scheme", as_index=False).tail(1).set_index("feature_scheme").reindex(scheme_order)
    best_tcls = tcls.sort_values("macro_f1").groupby("feature_scheme", as_index=False).tail(1).set_index("feature_scheme").reindex(scheme_order)
    x = np.arange(len(scheme_order))
    fig, ax1 = plt.subplots(figsize=(9.2, 4.6))
    ax1.plot(x, best_treg["r2"], marker="o", lw=3, color=PURPLE, label="Best R²")
    ax1.set_ylabel("Best Test R²", color=PURPLE)
    ax1.tick_params(axis="y", labelcolor=PURPLE)
    ax1.set_ylim(0.18, 0.212)
    ax1.grid(axis="y", alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(x, best_tcls["macro_f1"], marker="s", lw=2.6, color=ORANGE, label="Best Macro F1")
    ax2.set_ylabel("Best Test Macro F1", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.set_ylim(0.38, 0.44)
    ax1.set_xticks(x)
    ax1.set_xticklabels([short_scheme[s] for s in scheme_order], fontsize=10)
    ax1.set_title("文本特征消融：文本有增量，但幅度很小", loc="left", fontsize=16, color=PURPLE, weight="bold", pad=12)
    baseline = best_treg.loc["T0_structured_A3", "r2"]
    best = best_treg["r2"].max()
    ax1.annotate(
        f"R²提升 +{best - baseline:.4f}",
        xy=(4, best),
        xytext=(2.6, best + 0.006),
        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5),
        fontsize=11,
        color=PURPLE,
        weight="bold",
    )
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], frameon=False, loc="lower right")
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
    save(fig, "text_feature_increment_lines.png")

    # 4. All feature schemes as tradeoff scatter: regression vs classification, size = feature count.
    struct_reg_best = reg.sort_values("r2").groupby("approach", as_index=False).tail(1)
    struct_cls_best = cls.sort_values("macro_f1").groupby("approach", as_index=False).tail(1)
    struct_merge = pd.merge(
        struct_reg_best[["approach", "r2", "n_features_after_encoding"]],
        struct_cls_best[["approach", "macro_f1"]],
        on="approach",
    )
    struct_merge["scheme"] = struct_merge["approach"]
    struct_merge["family"] = "structured"
    struct_merge["features"] = struct_merge["n_features_after_encoding"]

    text_reg_best = treg.sort_values("r2").groupby("feature_scheme", as_index=False).tail(1)
    text_cls_best = tcls.sort_values("macro_f1").groupby("feature_scheme", as_index=False).tail(1)
    text_merge = pd.merge(
        text_reg_best[["feature_scheme", "r2", "n_features"]],
        text_cls_best[["feature_scheme", "macro_f1"]],
        on="feature_scheme",
    )
    text_merge["scheme"] = text_merge["feature_scheme"]
    text_merge["family"] = "text"
    text_merge["features"] = text_merge["n_features"]

    all_schemes = pd.concat(
        [
            struct_merge[["scheme", "family", "r2", "macro_f1", "features"]],
            text_merge[["scheme", "family", "r2", "macro_f1", "features"]],
        ],
        ignore_index=True,
    )
    label_map = {
        "A0_label": "A0",
        "A1_onehot": "A1",
        "A2_age_engineered": "A2",
        "A3_social_grouped": "A3",
        "T0_structured_A3": "T0",
        "T1_full_persona_text": "T1",
        "T2_career_text": "T2",
        "T3_lifestyle_text": "T3",
        "T4_field_pca": "T4",
    }
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for fam, color in [("structured", PURPLE), ("text", ORANGE)]:
        sub = all_schemes[all_schemes["family"] == fam]
        sizes = 70 + np.sqrt(sub["features"].astype(float)) * 10
        ax.scatter(sub["r2"], sub["macro_f1"], s=sizes, color=color, alpha=0.78, edgecolor="white", linewidth=1.2, label=fam)
        for _, row in sub.iterrows():
            dx, dy = 0.0008, 0.001
            label = label_map.get(row["scheme"], row["scheme"])
            if label == "T0":
                dy = -0.0016
            if label == "A3":
                dx = -0.0022
                dy = -0.0017
            if label == "T4":
                dx = 0.0008
                dy = 0.0012
            ax.text(row["r2"] + dx, row["macro_f1"] + dy, label, fontsize=10, weight="bold", color=DARK)
    ax.set_title("特征方案取舍：回归解释力 vs 分类表现", loc="left", fontsize=16, color=PURPLE, weight="bold", pad=12)
    ax.set_xlabel("Best Test R²")
    ax.set_ylabel("Best Test Macro F1")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="lower right", title="feature family")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.189,
        0.4214,
        "点越大：编码后特征维度越高\n文本特征整体右移很小，说明增量有限",
        fontsize=10,
        color=GREY,
        bbox=dict(boxstyle="round,pad=0.35", fc=PALE, ec=PURPLE, lw=1),
    )
    save(fig, "feature_scheme_tradeoff_scatter.png")

    # Save a compact table for slide labels.
    summary = all_schemes.copy()
    summary["label"] = summary["scheme"].map(label_map)
    summary.sort_values(["family", "scheme"]).to_csv(OUT / "feature_scheme_plot_summary.csv", index=False, encoding="utf-8-sig")

    # 5. Model complexity ladder under structured features: simple -> ensemble -> boosting.
    model_family = {
        "LinearRegression": "1 Linear",
        "Ridge": "1 Linear",
        "RandomForest": "2 Tree ensemble",
        "XGBoost": "3 Boosting",
        "LightGBM": "3 Boosting",
    }
    reg2 = reg[reg["model"].isin(model_family)].copy()
    reg2["family"] = reg2["model"].map(model_family)
    fam_best = (
        reg2.groupby(["approach", "family"], as_index=False)
        .agg(r2=("r2", "max"))
        .assign(approach=lambda d: pd.Categorical(d["approach"], categories=app_order, ordered=True))
        .sort_values(["approach", "family"])
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = [PURPLE, BLUE, GREEN, ORANGE]
    for approach, color in zip(app_order, colors):
        sub = fam_best[fam_best["approach"] == approach]
        xs = np.arange(len(sub))
        ax.plot(xs, sub["r2"], marker="o", lw=2.5, color=color, label=short_app[approach])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["线性模型\nLinear/Ridge", "树模型\nRandomForest", "Boosting\nXGBoost/LightGBM"], fontsize=10)
    ax.set_ylabel("Best Test R²")
    ax.set_ylim(0.09, 0.215)
    ax.set_title("模型阶梯：复杂模型能提升，但很快进入平台期", loc="left", fontsize=16, color=PURPLE, weight="bold", pad=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate(
        "Boosting 是主要提升来源",
        xy=(2, fam_best["r2"].max()),
        xytext=(1.05, 0.207),
        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.4),
        color=PURPLE,
        fontsize=11,
        weight="bold",
    )
    save(fig, "structured_model_complexity_ladder.png")

    # 6. Feature dimension vs regression performance: more features are not automatically better.
    dim_df = all_schemes.copy()
    dim_df["label"] = dim_df["scheme"].map(label_map)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for fam, color in [("structured", PURPLE), ("text", ORANGE)]:
        sub = dim_df[dim_df["family"] == fam]
        ax.scatter(sub["features"], sub["r2"], s=120, color=color, alpha=0.82, edgecolor="white", linewidth=1.2, label=fam)
        for _, row in sub.iterrows():
            ax.text(row["features"] * 1.04, row["r2"] + 0.0008, row["label"], fontsize=10, weight="bold", color=DARK)
    ax.set_xscale("log")
    ax.set_xlabel("编码后特征维度（log scale）")
    ax.set_ylabel("Best Test R²")
    ax.set_title("特征维度 ≠ 效果提升：高维文本只带来有限增量", loc="left", fontsize=16, color=PURPLE, weight="bold", pad=12)
    ax.grid(alpha=0.24, which="both")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, "feature_dimension_vs_r2.png")

    # 7. Class-level F1 comparison for selected baselines: support dominates, oppose weak.
    selected = [
        ("A0_label", "RandomForest", "结构化最佳\nA0+RF"),
        ("A1_onehot", "XGBoost", "one-hot\nA1+XGB"),
    ]
    rows_cls = []
    for approach, model, label in selected:
        row = cls[(cls["approach"] == approach) & (cls["model"] == model)].iloc[0]
        rows_cls.extend(
            [
                {"model": label, "class": "support", "f1": row["support_f1"]},
                {"model": label, "class": "neutral", "f1": row["neutral_f1"]},
                {"model": label, "class": "oppose", "f1": row["oppose_f1"]},
            ]
        )
    for scheme, model, label in [
        ("T4_field_pca", "XGBoost", "文本最佳\nT4+XGB"),
        ("T1_full_persona_text", "LogisticRegression", "文本线性\nT1+LR"),
    ]:
        row = tcls[(tcls["feature_scheme"] == scheme) & (tcls["model"] == model)].iloc[0]
        rows_cls.extend(
            [
                {"model": label, "class": "support", "f1": row["support_f1"]},
                {"model": label, "class": "neutral", "f1": row["neutral_f1"]},
                {"model": label, "class": "oppose", "f1": row["oppose_f1"]},
            ]
        )
    cf = pd.DataFrame(rows_cls)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    class_colors = {"support": "#7E57C2", "neutral": "#4C78A8", "oppose": "#F58518"}
    y_positions = np.arange(cf["model"].nunique())
    labels = list(dict.fromkeys(cf["model"]))
    for i, label in enumerate(labels):
        sub = cf[cf["model"] == label]
        for klass in ["support", "neutral", "oppose"]:
            val = sub[sub["class"] == klass]["f1"].iloc[0]
            ax.scatter(val, i, s=130, color=class_colors[klass], label=klass if i == 0 else None, zorder=3)
            ax.plot([0, val], [i, i], color=class_colors[klass], alpha=0.23, lw=5, solid_capstyle="round")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(-0.02, 0.86)
    ax.set_xlabel("Test F1")
    ax.set_title("分类不是平均地差：support 易学，oppose 几乎学不到", loc="left", fontsize=16, color=PURPLE, weight="bold", pad=12)
    ax.grid(axis="x", alpha=0.22)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.55, -0.10),
        borderaxespad=0,
    )
    fig.subplots_adjust(bottom=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    save(fig, "class_f1_lollipop_selected_models.png")

    # 8. Text scheme rank table as a compact visual table.
    rank = best_treg[["r2", "mae"]].join(best_tcls[["macro_f1", "oppose_f1"]], how="left")
    rank = rank.reindex(scheme_order)
    rank.index = [short_scheme[i].replace("\n", " ") for i in rank.index]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.axis("off")
    table_data = []
    for idx, row in rank.iterrows():
        table_data.append([idx, f"{row['r2']:.4f}", f"{row['mae']:.4f}", f"{row['macro_f1']:.4f}", f"{row['oppose_f1']:.4f}"])
    table = ax.table(
        cellText=table_data,
        colLabels=["方案", "Best R²", "MAE", "Macro F1", "Oppose F1"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(PURPLE)
        if r == 0:
            cell.set_facecolor(LIGHT)
            cell.set_text_props(weight="bold", color=DARK)
        elif r == len(table_data):
            cell.set_facecolor("#F1E1F1")
        else:
            cell.set_facecolor("white")
    ax.set_title("文本方案结果表：T4 最优，但 Oppose F1 仍为 0", loc="left", fontsize=16, color=PURPLE, weight="bold", pad=16)
    save(fig, "text_scheme_compact_result_table.png")


if __name__ == "__main__":
    main()
