from pathlib import Path
import shutil
import textwrap

import nbformat as nbf
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn


SRC = Path(r"D:/Sophomore Year/Spring/Data Science and Data Analysis/french_policy_simulator")
DST = Path("D:/Sophomore Year/Spring/" + "\u6570\u667a\u4f20\u64ad\u5bfc\u8bba")
SUB = DST / "final_submission_french_policy"
FIG_SRC = SRC / "report" / "figures"
RES_SRC = SRC / "submission" / "results"
DATA_SRC = SRC / "submission" / "data" / "main_dataset.parquet"


def ensure_dirs() -> None:
    for path in [SUB, SUB / "figures", SUB / "results", SUB / "data"]:
        path.mkdir(exist_ok=True)


def copy_assets() -> None:
    figure_names = [
        "huggingface_source.png",
        "system_architecture.png",
        "frontend_screenshot.png",
        "prompt_sampling_mae_heatmap.png",
        "stance_vs_ifop.png",
        "age_group_support.png",
        "score_distribution.png",
        "reason_wordclouds_panel.png",
        "reason_theme_by_stance.png",
        "data_feature_map.png",
        "structured_r2_heatmap.png",
        "feature_dimension_vs_r2.png",
        "text_feature_increment_lines.png",
        "class_f1_lollipop_selected_models.png",
        "direct_multiclass_confusion_matrix.png",
        "absolute_residual_by_stance.png",
        "label_stability_by_stance.png",
        "oppose_recall_accuracy_tradeoff.png",
    ]
    for name in figure_names:
        src = FIG_SRC / name
        if src.exists():
            shutil.copy2(src, SUB / "figures" / name)

    if DATA_SRC.exists():
        shutil.copy2(DATA_SRC, SUB / "data" / "main_dataset.parquet")

    result_names = [
        "prompt_sampling_summary.csv",
        "prompt_sampling_detailed_summary.csv",
        "summary_1w_main.csv",
        "structured_baseline_results.csv",
        "text_feature_ablation_results.csv",
        "agreement_by_original_stance.csv",
        "reasonable_oppose_recovery.csv",
    ]
    for name in result_names:
        src = RES_SRC / name
        if src.exists():
            shutil.copy2(src, SUB / "results" / name)

    (SUB / "requirements.txt").write_text(
        "\n".join(
            [
                "pandas>=2.0",
                "numpy>=1.24",
                "matplotlib>=3.7",
                "seaborn>=0.12",
                "pyarrow>=14.0",
                "jupyter>=1.0",
                "nbconvert>=7.0",
                "python-docx>=1.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def make_nb(title: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb["cells"] = [
        md(
            f"""# {title}

            本 notebook 使用课程项目冻结版主数据和已经在汇报/报告中采用的标准图表，
            确保图表口径与最终 Word 报告一致。
            """
        )
    ] + cells
    return nb


COMMON_SETUP = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 120)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
PURPLE = '#6f0070'
PPT_PURPLE = '#6F006F'
PPT_LIGHT = '#E8D8E8'
PPT_PALE = '#F7F2F7'
PPT_DARK = '#222222'
PPT_GREY = '#666666'
BLUE = '#4c78a8'
ORANGE = '#f58518'
GREEN = '#54A24B'
GRAY = '#9a9a9a'
FIG_OUT = Path('generated_figures')
FIG_OUT.mkdir(exist_ok=True)
FINAL_FIG = Path('figures')
FINAL_FIG.mkdir(exist_ok=True)
DATA = 'data/main_dataset.parquet'
df = pd.read_parquet(DATA)
print(f'主数据规模: {len(df):,} 行, {df.shape[1]} 列')
display(df.head(3))
"""


def build_notebooks() -> None:
    nb1 = make_nb(
        "01 宏观真实性评估：LLM 模拟民调与真实民调对照",
        [
            code(COMMON_SETUP),
            md(
                """## 1. Prompt × Sampling 交叉实验

                交叉实验用于决定主数据生成方案。下面的热力图由 `results/prompt_sampling_detailed_summary.csv`
                现场计算得到，指标是三分类分布相对 IFOP 基准的平均绝对误差，越低越接近真实民调。
                """
            ),
            code(
                """
                ps = pd.read_csv('results/prompt_sampling_detailed_summary.csv')
                cols = ['sampling','prompt_version','rows','parse_success_rate','support_pct',
                        'neutral_pct','oppose_pct','distribution_mae','js_distance']
                display(ps[cols].round(3))

                order_sampling = ['random', 'faiss']
                order_prompt = ['P0_demo', 'P1_career', 'P2_full_text', 'P3_compact_all']
                pivot = ps.pivot(index='sampling', columns='prompt_version', values='distribution_mae').loc[order_sampling, order_prompt]

                fig, ax = plt.subplots(figsize=(8, 3.2), dpi=160)
                im = ax.imshow(pivot.values, cmap='Purples_r', vmin=0, vmax=max(15, pivot.values.max()))
                ax.set_xticks(range(len(order_prompt)))
                ax.set_xticklabels(['P0\\ndemo', 'P1\\ncareer', 'P2\\nfull text', 'P3\\ncompact'])
                ax.set_yticks(range(len(order_sampling)))
                ax.set_yticklabels(['Random', 'FAISS'])
                ax.set_title('Prompt x Sampling: distribution MAE vs IFOP (lower is better)', pad=10)
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        val = pivot.values[i, j]
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                fontweight='bold' if val == pivot.values.min() else 'normal')
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(length=0)
                cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
                cbar.set_label('Distribution MAE')
                fig.tight_layout()
                fig.savefig(FIG_OUT / 'prompt_sampling_mae_heatmap.png', bbox_inches='tight', facecolor='white')
                fig.savefig(FINAL_FIG / 'prompt_sampling_mae_heatmap.png', bbox_inches='tight', facecolor='white')
                plt.show()
                """
            ),
            md(
                """## 2. 宏观分布拟合

                选定 FAISS + P2 full text 后，主数据总体分布与 IFOP 的方向一致，
                但 support 仍略高，oppose 仍偏少。
                """
            ),
            code(
                """
                order = ['support', 'neutral', 'oppose']
                sim = df['stance'].value_counts(normalize=True).reindex(order).mul(100)
                ifop = pd.Series({'support': 61, 'neutral': 34, 'oppose': 5})
                compare = pd.DataFrame({'LLM simulated': sim, 'IFOP benchmark': ifop})
                display(compare.round(2))

                x = np.arange(len(order))
                fig, ax = plt.subplots(figsize=(7.2, 4), dpi=160)
                ax.bar(x - 0.18, compare['LLM simulated'], width=0.36, color=PURPLE, label='LLM simulated')
                ax.bar(x + 0.18, compare['IFOP benchmark'], width=0.36, color=GRAY, label='IFOP benchmark')
                ax.set_xticks(x)
                ax.set_xticklabels(order)
                ax.set_ylabel('Share (%)')
                ax.set_title('Stance distribution: LLM simulation vs IFOP')
                ax.legend(frameon=False)
                ax.spines[['top','right']].set_visible(False)
                for i, v in enumerate(compare['LLM simulated']):
                    ax.text(i - 0.18, v + 1, f'{v:.1f}%', ha='center', fontsize=9)
                for i, v in enumerate(compare['IFOP benchmark']):
                    ax.text(i + 0.18, v + 1, f'{v:.0f}%', ha='center', fontsize=9)
                fig.tight_layout()
                fig.savefig(FIG_OUT / 'stance_vs_ifop.png', bbox_inches='tight', facecolor='white')
                fig.savefig(FINAL_FIG / 'stance_vs_ifop.png', bbox_inches='tight', facecolor='white')
                plt.show()
                """
            ),
            md(
                """## 4. 小结

                宏观层面，LLM 生成数据不是随机噪声，能够呈现接近真实民调的多数/中间/少数结构；
                但它仍有系统偏差，因此需要继续做 EDA、建模和稳定性诊断。
                """
            ),
        ],
    )

    nb2 = make_nb(
        "02 主数据探索性分析：人群结构、评分与理由文本",
        [
            code(COMMON_SETUP),
            md("## 1. 支持度评分分布\n\n先观察 0-10 支持度评分是否集中，以及是否存在明显极端化或中间化。"),
            code(
                """
                score_col = 'support_score_0_10' if 'support_score_0_10' in df.columns else 'support_score'
                display(df[score_col].describe())

                fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
                bins = np.arange(df[score_col].min(), df[score_col].max() + 1.5, 1)
                ax.hist(df[score_col], bins=bins, color=PURPLE, alpha=0.82, edgecolor='white')
                ax.set_title('Support score distribution')
                ax.set_xlabel('support_score (0-10)')
                ax.set_ylabel('Count')
                ax.spines[['top','right']].set_visible(False)
                fig.tight_layout()
                fig.savefig(FIG_OUT / 'score_distribution.png', bbox_inches='tight', facecolor='white')
                fig.savefig(FINAL_FIG / 'score_distribution.png', bbox_inches='tight', facecolor='white')
                plt.show()
                """
            ),
            md("## 2. 年龄与态度关系\n\n退休改革议题天然与年龄阶段相关，因此年龄是检验模拟数据是否存在社会结构信号的关键维度。"),
            code(
                """
                age_bins = [17, 24, 34, 44, 54, 61, 64, 74, 120]
                age_labels = ['18-24', '25-34', '35-44', '45-54', '55-61', '62-64', '65-74', '75+']
                tmp = df.copy()
                tmp['age_group'] = pd.cut(tmp['age'], bins=age_bins, labels=age_labels, right=False)
                age_support = tmp.groupby('age_group', observed=True).apply(lambda x: (x['stance'] == 'support').mean() * 100)
                display(age_support.round(2).rename('support_pct').reset_index())

                fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
                ax.plot(age_support.index.astype(str), age_support.values, color=PURPLE, marker='o', linewidth=2)
                ax.axhline(61, color=GRAY, linestyle='--', linewidth=1, label='IFOP support 61%')
                ax.set_title('Support rate by age group')
                ax.set_xlabel('Age group')
                ax.set_ylabel('Support (%)')
                ax.legend(frameon=False)
                ax.spines[['top','right']].set_visible(False)
                fig.tight_layout()
                fig.savefig(FIG_OUT / 'age_group_support.png', bbox_inches='tight', facecolor='white')
                fig.savefig(FINAL_FIG / 'age_group_support.png', bbox_inches='tight', facecolor='white')
                plt.show()
                """
            ),
            md(
                """## 3. reason 文本解释

                LLM 不只输出标签和分数，也输出理由文本。词云和主题占比用于检查不同立场是否存在可解释的语义差异。
                """
            ),
            code(
                """
                themes = {
                    'age_retirement': ['年龄','退休','年纪','岁','提前退休'],
                    'job_work': ['工作','职业','体力','员工','护理','劳动'],
                    'economy': ['经济','收入','财政','成本','养老金','负担'],
                    'family_life': ['家庭','生活','孩子','健康','质量'],
                    'uncertain': ['未知','不确定','无法判断']
                }
                reason = df['reason'].fillna('').astype(str)
                rows = []
                for stance, g in df.assign(reason_text=reason).groupby('stance'):
                    n = len(g)
                    for theme, kws in themes.items():
                        pct = g['reason_text'].apply(lambda s: any(k in s for k in kws)).mean() * 100
                        rows.append({'stance': stance, 'theme': theme, 'share_pct': pct})
                theme_df = pd.DataFrame(rows)
                display(theme_df.pivot(index='theme', columns='stance', values='share_pct').round(1))

                order_themes = list(themes.keys())
                stances = ['support','neutral','oppose']
                colors = {'support': PURPLE, 'neutral': BLUE, 'oppose': ORANGE}
                x = np.arange(len(order_themes))
                width = 0.25
                fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
                for i, stance in enumerate(stances):
                    vals = theme_df[theme_df['stance'] == stance].set_index('theme').reindex(order_themes)['share_pct']
                    ax.bar(x + (i-1)*width, vals, width=width, label=stance, color=colors[stance], alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels(order_themes, rotation=25, ha='right')
                ax.set_ylabel('Share of reasons (%)')
                ax.set_title('Reason theme share by stance')
                ax.legend(frameon=False)
                ax.spines[['top','right']].set_visible(False)
                fig.tight_layout()
                fig.savefig(FIG_OUT / 'reason_theme_by_stance.png', bbox_inches='tight', facecolor='white')
                fig.savefig(FINAL_FIG / 'reason_theme_by_stance.png', bbox_inches='tight', facecolor='white')
                plt.show()
                """
            ),
            md(
                """## 4. 小结

                EDA 显示主数据包含可解释的社会结构信号：年龄与支持度存在方向性关系，
                reason 中也出现年龄、工作、经济压力等主题。但 oppose 的文本来源更复杂，
                为后续分类困难埋下伏笔。
                """
            ),
        ],
    )

    nb3 = make_nb(
        "03 特征工程与建模：从结构化变量到文本画像",
        [
            code(COMMON_SETUP),
            md(
                """## 1. 特征来源

                主数据固定后，建模阶段不再改变 LLM 标签，而是比较不同人物画像表示方式能否解释这些标签和评分。
                """
            ),
            code(
                """
                feature_blocks = pd.DataFrame({
                    'block': ['demographic', 'social position', 'career text', 'lifestyle text', 'full persona text'],
                    'examples': ['age, sex, household', 'occupation, education, income, region',
                                 'professional_persona, skills, career_goals',
                                 'sports, arts, travel, culinary, hobbies',
                                 'merged persona_text embedding'],
                    'role': ['structured baseline', 'social grouping', 'text embedding', 'text embedding', 'full text embedding']
                })
                display(feature_blocks)
                """
            ),
            md(
                """## 2. 第一轮：结构化特征 × 多类模型

                第一轮实验同时改变结构化特征方案和模型复杂度，用来判断瓶颈来自“特征不够”还是“模型不够”。
                """
            ),
            code(
                """
                res = pd.read_csv('results/structured_baseline_results.csv')
                reg = res[res['task'] == 'regression'].copy()
                cls = res[res['task'] == 'classification'].copy()
                display(reg[['approach','model','n_features_after_encoding','r2','mae','rmse']].head())

                model_order = ['LinearRegression', 'Ridge', 'RandomForest', 'XGBoost', 'LightGBM']
                app_order = ['A0_label', 'A1_onehot', 'A2_age_engineered', 'A3_social_grouped']
                short_app = {
                    'A0_label': 'A0 label',
                    'A1_onehot': 'A1 one-hot',
                    'A2_age_engineered': 'A2 age+',
                    'A3_social_grouped': 'A3 social',
                }
                short_model = {
                    'LinearRegression': 'Linear',
                    'Ridge': 'Ridge',
                    'RandomForest': 'RF',
                    'XGBoost': 'XGB',
                    'LightGBM': 'LGBM',
                }
                heat = (
                    reg.pivot_table(index='approach', columns='model', values='r2', aggfunc='max')
                    .reindex(app_order)[model_order]
                    .rename(index=short_app, columns=short_model)
                )

                fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=160)
                im = ax.imshow(heat.values, cmap='Purples', aspect='auto')
                ax.set_title('结构化特征 × 模型：Test R² 热力图', loc='left', fontsize=15, color=PPT_PURPLE, weight='bold', pad=12)
                ax.set_xticks(range(heat.shape[1]))
                ax.set_xticklabels(heat.columns, fontsize=9)
                ax.set_yticks(range(heat.shape[0]))
                ax.set_yticklabels(heat.index, fontsize=10)
                ax.tick_params(length=0)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xticks(np.arange(-0.5, heat.shape[1], 1), minor=True)
                ax.set_yticks(np.arange(-0.5, heat.shape[0], 1), minor=True)
                ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
                for i in range(heat.shape[0]):
                    for j in range(heat.shape[1]):
                        val = heat.iloc[i, j]
                        txt_color = 'white' if val >= np.nanmax(heat.values) * 0.82 else PPT_DARK
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, color=txt_color)
                cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
                cbar.ax.tick_params(labelsize=8)
                cbar.set_label('Test R²', fontsize=9)
                fig.text(0.12, 0.035, '读法：颜色越深，R²越高；A3社会分组 + XGBoost 是结构化阶段最好组合。', fontsize=10, color=PPT_GREY)
                fig.savefig(FIG_OUT / 'structured_r2_heatmap.png', bbox_inches='tight', facecolor='white')
                fig.savefig(FINAL_FIG / 'structured_r2_heatmap.png', bbox_inches='tight', facecolor='white')
                plt.show()

                # 原始 PPT 图逻辑：横轴是编码后的特征维度（log scale），纵轴是每个方案的最佳 Test R²。
                txt_all = pd.read_csv('results/text_feature_ablation_results.csv')
                treg = txt_all[txt_all['task'] == 'regression'].copy()
                tcls = txt_all[txt_all['task'] == 'classification'].copy()
                struct_reg_best = reg.sort_values('r2').groupby('approach', as_index=False).tail(1)
                struct_cls_best = cls.sort_values('macro_f1').groupby('approach', as_index=False).tail(1)
                struct_merge = pd.merge(
                    struct_reg_best[['approach', 'r2', 'n_features_after_encoding']],
                    struct_cls_best[['approach', 'macro_f1']],
                    on='approach',
                )
                struct_merge['scheme'] = struct_merge['approach']
                struct_merge['family'] = 'structured'
                struct_merge['features'] = struct_merge['n_features_after_encoding']
                text_reg_best = treg.sort_values('r2').groupby('feature_scheme', as_index=False).tail(1)
                text_cls_best = tcls.sort_values('macro_f1').groupby('feature_scheme', as_index=False).tail(1)
                text_merge = pd.merge(
                    text_reg_best[['feature_scheme', 'r2', 'n_features']],
                    text_cls_best[['feature_scheme', 'macro_f1']],
                    on='feature_scheme',
                )
                text_merge['scheme'] = text_merge['feature_scheme']
                text_merge['family'] = 'text'
                text_merge['features'] = text_merge['n_features']
                all_schemes = pd.concat(
                    [
                        struct_merge[['scheme', 'family', 'r2', 'macro_f1', 'features']],
                        text_merge[['scheme', 'family', 'r2', 'macro_f1', 'features']],
                    ],
                    ignore_index=True,
                )
                label_map = {
                    'A0_label': 'A0',
                    'A1_onehot': 'A1',
                    'A2_age_engineered': 'A2',
                    'A3_social_grouped': 'A3',
                    'T0_structured_A3': 'T0',
                    'T1_full_persona_text': 'T1',
                    'T2_career_text': 'T2',
                    'T3_lifestyle_text': 'T3',
                    'T4_field_pca': 'T4',
                }
                dim_df = all_schemes.copy()
                dim_df['label'] = dim_df['scheme'].map(label_map)
                fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=160)
                for fam, color in [('structured', PPT_PURPLE), ('text', ORANGE)]:
                    sub = dim_df[dim_df['family'] == fam]
                    ax.scatter(sub['features'], sub['r2'], s=120, color=color, alpha=0.82, edgecolor='white', linewidth=1.2, label=fam)
                    for _, row in sub.iterrows():
                        ax.text(row['features'] * 1.04, row['r2'] + 0.0008, row['label'], fontsize=10, weight='bold', color=PPT_DARK)
                ax.set_xscale('log')
                ax.set_xlabel('编码后特征维度（log scale）')
                ax.set_ylabel('Best Test R²')
                ax.set_title('特征维度 ≠ 效果提升：高维文本只带来有限增量', loc='left', fontsize=16, color=PPT_PURPLE, weight='bold', pad=12)
                ax.grid(alpha=0.24, which='both')
                ax.legend(frameon=False, loc='lower right')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.savefig(FIG_OUT / 'feature_dimension_vs_r2.png', bbox_inches='tight', facecolor='white')
                fig.savefig(FINAL_FIG / 'feature_dimension_vs_r2.png', bbox_inches='tight', facecolor='white')
                plt.show()
                """
            ),
            md(
                """## 3. 第二轮：文本画像增量

                当结构化特征进入平台期后，进一步测试完整 persona 文本、职业文本、生活方式文本和分字段 embedding + PCA。
                """
            ),
            code(
                """
                txt = pd.read_csv('results/text_feature_ablation_results.csv')
                display(txt.head())
                treg = txt[txt['task'] == 'regression'].copy()
                tcls = txt[txt['task'] == 'classification'].copy()
                order = ['T0_structured_A3','T1_full_persona_text','T2_career_text','T3_lifestyle_text','T4_field_pca']
                best_treg = treg.sort_values('r2').groupby('feature_scheme', as_index=False).tail(1).set_index('feature_scheme').reindex(order)
                best_tcls = tcls.sort_values('macro_f1').groupby('feature_scheme', as_index=False).tail(1).set_index('feature_scheme').reindex(order)
                merged = pd.DataFrame({'r2': best_treg['r2'], 'macro_f1': best_tcls['macro_f1']})
                display(merged.round(4))

                x = np.arange(len(order))
                fig, ax1 = plt.subplots(figsize=(9.2, 4.6), dpi=160)
                ax2 = ax1.twinx()
                ax1.plot(x, best_treg['r2'], marker='o', lw=3, color=PPT_PURPLE, label='Best R²')
                ax2.plot(x, best_tcls['macro_f1'], marker='s', lw=2.6, color=ORANGE, label='Best Macro F1')
                ax1.set_xticks(x)
                ax1.set_xticklabels(['T0\\nstructured', 'T1\\nfull text', 'T2\\ncareer', 'T3\\nlifestyle', 'T4\\nfield PCA'])
                ax1.set_ylabel('Best Test R²', color=PPT_PURPLE)
                ax2.set_ylabel('Best Test Macro F1', color=ORANGE)
                ax1.tick_params(axis='y', labelcolor=PPT_PURPLE)
                ax2.tick_params(axis='y', labelcolor=ORANGE)
                ax1.set_ylim(0.18, 0.212)
                ax2.set_ylim(0.38, 0.44)
                ax1.grid(axis='y', alpha=0.22)
                ax1.set_title('文本特征消融：文本有增量，但幅度很小', loc='left', fontsize=16, color=PPT_PURPLE, weight='bold', pad=12)
                baseline = best_treg.loc['T0_structured_A3', 'r2']
                best = best_treg['r2'].max()
                ax1.annotate(
                    f'R²提升 +{best - baseline:.4f}',
                    xy=(4, best),
                    xytext=(2.6, best + 0.006),
                    arrowprops=dict(arrowstyle='->', color=PPT_PURPLE, lw=1.5),
                    fontsize=11,
                    color=PPT_PURPLE,
                    weight='bold',
                )
                lines = ax1.get_lines() + ax2.get_lines()
                ax1.legend(lines, [l.get_label() for l in lines], frameon=False, loc='lower right')
                ax1.spines['top'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                fig.tight_layout()
                fig.savefig(FIG_OUT / 'text_feature_increment_lines.png', bbox_inches='tight', facecolor='white')
                fig.savefig(FINAL_FIG / 'text_feature_increment_lines.png', bbox_inches='tight', facecolor='white')
                plt.show()
                """
            ),
            md(
                """## 4. 小结

                结构化变量和复杂模型都能带来一定提升，但最佳 R² 仍约 0.20；
                文本画像有增量但幅度有限，说明更多维度不等于更有效的信息。
                """
            ),
        ],
    )

    nb4 = make_nb(
        "04 诊断与优化：少数类、标签稳定性和方法边界",
        [
            code(COMMON_SETUP),
            md(
                """## 1. 分类表现：oppose 最难

                Accuracy 容易被多数类 support 抬高，因此重点观察 Macro F1 和类别级 F1。
                """
            ),
            code(
                """
                structured = pd.read_csv('results/structured_baseline_results.csv')
                text = pd.read_csv('results/text_feature_ablation_results.csv')
                rows = []
                def add_pick(df0, label, key_col, key, model):
                    r = df0[(df0[key_col] == key) & (df0['task'] == 'classification') & (df0['model'] == model)].iloc[0]
                    for klass in ['support', 'neutral', 'oppose']:
                        rows.append({'model': label, 'class': klass, 'f1': r[f'{klass}_f1']})
                add_pick(structured, '结构化最佳\\nA0+RF', 'approach', 'A0_label', 'RandomForest')
                add_pick(structured, 'one-hot\\nA1+XGB', 'approach', 'A1_onehot', 'XGBoost')
                add_pick(text, '文本最佳\\nT4+XGB', 'feature_scheme', 'T4_field_pca', 'XGBoost')
                add_pick(text, '文本线性\\nT1+LR', 'feature_scheme', 'T1_full_persona_text', 'LogisticRegression')
                cf = pd.DataFrame(rows)
                display(cf.pivot(index='model', columns='class', values='f1').round(3))

                fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=160)
                class_colors = {'support': '#7E57C2', 'neutral': '#4C78A8', 'oppose': '#F58518'}
                labels = list(dict.fromkeys(cf['model']))
                for i, label in enumerate(labels):
                    sub = cf[cf['model'] == label]
                    for klass in ['support', 'neutral', 'oppose']:
                        val = sub[sub['class'] == klass]['f1'].iloc[0]
                        ax.scatter(val, i, s=130, color=class_colors[klass], label=klass if i == 0 else None, zorder=3)
                        ax.plot([0, val], [i, i], color=class_colors[klass], alpha=0.23, lw=5, solid_capstyle='round')
                ax.set_yticks(np.arange(len(labels)))
                ax.set_yticklabels(labels, fontsize=10)
                ax.set_xlim(-0.02, 0.86)
                ax.set_xlabel('Class-level F1')
                ax.set_title('分类不是平均地差：support 易学，oppose 几乎学不到', loc='left', fontsize=16, color=PPT_PURPLE, weight='bold', pad=12)
                ax.grid(axis='x', alpha=0.22)
                ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.55, -0.10), borderaxespad=0)
                fig.subplots_adjust(bottom=0.22)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.invert_yaxis()
                fig.savefig(FIG_OUT / 'class_f1_lollipop_selected_models.png', bbox_inches='tight', facecolor='white')
                fig.savefig(FINAL_FIG / 'class_f1_lollipop_selected_models.png', bbox_inches='tight', facecolor='white')
                plt.show()
                """
            ),
            md("## 2. 误差结构\n\n混淆矩阵与残差图用于判断模型到底是整体都差，还是某个类别存在边界问题。"),
            code(
                """
                score_col = 'support_score_0_10' if 'support_score_0_10' in df.columns else 'support_score'
                by_stance = df.groupby('stance')[score_col].describe()[['count','mean','std','min','25%','50%','75%','max']]
                display(by_stance.round(2))

                fig, ax = plt.subplots(figsize=(6.5, 4), dpi=160)
                order = ['support','neutral','oppose']
                data = [df.loc[df['stance'] == s, score_col].dropna() for s in order]
                ax.boxplot(data, patch_artist=True,
                           boxprops=dict(facecolor='#eee6f5', color=PURPLE),
                           medianprops=dict(color=ORANGE, linewidth=2))
                ax.set_xticks(range(1, len(order) + 1))
                ax.set_xticklabels(order)
                ax.set_title('Score distribution by stance')
                ax.set_ylabel('support_score (0-10)')
                ax.spines[['top','right']].set_visible(False)
                fig.tight_layout()
                fig.savefig(FIG_OUT / 'score_distribution_by_stance.png', bbox_inches='tight', facecolor='white')
                plt.show()
                """
            ),
            md("## 3. 标签稳定性\n\n对同一批 persona 重复调用 LLM，检验标签是否能够稳定复现。"),
            code(
                """
                stab = pd.read_csv('results/agreement_by_original_stance.csv')
                display(stab)
                order = ['support','neutral','oppose']
                stab['original_stance'] = pd.Categorical(stab['original_stance'], categories=order, ordered=True)
                stab = stab.sort_values('original_stance')

                fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=160)
                colors = [PURPLE if s == 'support' else GRAY if s == 'neutral' else ORANGE for s in stab['original_stance']]
                axes[0].bar(stab['original_stance'].astype(str), stab['stance_agreement'] * 100, color=colors)
                axes[0].set_title('Label agreement drops for neutral / oppose')
                axes[0].set_ylabel('Retest stance agreement (%)')
                axes[1].bar(stab['original_stance'].astype(str), stab['score_mae_0_10'], color=colors)
                axes[1].set_title('Score error is largest for oppose')
                axes[1].set_ylabel('Score MAE vs original label (0-10)')
                for ax in axes:
                    ax.spines[['top','right']].set_visible(False)
                fig.tight_layout()
                fig.savefig(FIG_OUT / 'label_stability_by_stance.png', bbox_inches='tight', facecolor='white')
                fig.savefig(FINAL_FIG / 'label_stability_by_stance.png', bbox_inches='tight', facecolor='white')
                plt.show()
                """
            ),
            md("## 4. 优化尝试\n\n通过阈值调整、类别加权和两阶段分类提高 oppose 召回，但这种优化存在准确率代价。"),
            code(
                """
                opt = pd.read_csv('results/reasonable_oppose_recovery.csv')
                display(opt[['method','accuracy','macro_f1','oppose_recall','oppose_f1']].round(3).head(12))
                methods = opt['method'].unique()
                palette = {m: c for m, c in zip(methods, [PURPLE, BLUE, ORANGE, '#59a14f', '#e15759'])}
                fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
                for method, g in opt.groupby('method'):
                    ax.scatter(g['accuracy'], g['oppose_recall'], label=method, alpha=0.8, color=palette[method])
                ax.set_xlabel('Accuracy')
                ax.set_ylabel('Oppose recall')
                ax.set_title('Oppose recovery tradeoff')
                ax.legend(frameon=False, fontsize=8)
                ax.spines[['top','right']].set_visible(False)
                fig.tight_layout()
                fig.savefig(FIG_OUT / 'oppose_recall_accuracy_tradeoff.png', bbox_inches='tight', facecolor='white')
                fig.savefig(FINAL_FIG / 'oppose_recall_accuracy_tradeoff.png', bbox_inches='tight', facecolor='white')
                plt.show()
                """
            ),
            md(
                """## 5. 小结

                建模失败并非简单算法问题，而暴露了生成式社会调查数据的边界：
                少数类样本少、语义边界复杂、标签自身重复生成稳定性不足。
                """
            ),
        ],
    )

    notebooks = {
        "01_macro_comparison.ipynb": nb1,
        "02_data_and_eda.ipynb": nb2,
        "03_modeling.ipynb": nb3,
        "04_diagnosis.ipynb": nb4,
    }
    for name, nb in notebooks.items():
        nbf.write(nb, SUB / name)


def add_title(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.name = "黑体"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    r.font.size = Pt(18)
    r.font.color.rgb = RGBColor(102, 0, 102)


def add_para(doc: Document, text: str) -> None:
    p = doc.add_paragraph(text)
    p.paragraph_format.first_line_indent = Pt(21)
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_after = Pt(5)


def add_fig(doc: Document, name: str, caption: str, width: float = 5.6) -> None:
    fp = SUB / "figures" / name
    if not fp.exists():
        return
    doc.add_picture(str(fp), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cap.runs:
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(90, 90, 90)


def build_docx() -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.8)
    sec.bottom_margin = Inches(0.8)
    sec.left_margin = Inches(0.9)
    sec.right_margin = Inches(0.9)

    styles = doc.styles
    styles["Normal"].font.name = "宋体"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    styles["Normal"].font.size = Pt(10.5)
    for style_name in ["Heading 1", "Heading 2", "Heading 3"]:
        style = styles[style_name]
        style.font.name = "黑体"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        style.font.color.rgb = RGBColor(102, 0, 102)

    add_title(doc, "大语言模型生成社会调查数据的统计真实性评估：基于法国退休改革议题的模拟民调研究")
    p = doc.add_paragraph("课程：数智传播导论｜小组成员：胡浩然、何俊豪、仇炎｜提交材料：Word 报告、Jupyter Notebook、HTML 输出与数据文件")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sections = {
        "摘要": [
            "本文基于一个已经完成的数据科学项目，重新从数智传播导论的视角讨论大语言模型生成社会调查数据的可能性与边界。项目使用 Nemotron-Personas-France 合成法国人物画像数据集，将其中的个体画像视为虚拟受访者，并以法国退休改革为案例，通过抽样策略、Prompt 构建、LLM 调用、JSON 响应解析和质量审计，生成 9,960 条结构化模拟民调记录。每条记录包含人口统计属性、文本画像、支持度评分、三分类立场标签以及理由文本。",
            "研究以 IFOP 真实民调作为外部基准，从宏观分布拟合、年龄结构、reason 文本语义、机器学习可预测性、标签重复稳定性和少数类识别等角度评估生成数据的统计真实性。结果显示，LLM 生成数据能够在宏观层面复现一定社会结构，例如 support 是多数类、年龄越高越倾向支持回到 62 岁；但在个体层面，最佳回归模型的测试集 R² 约为 0.206，直接三分类模型几乎无法稳定识别 oppose。进一步诊断表明，问题并不只是算法性能不足，也与 persona 缺少近端政治态度变量、少数类语义边界复杂、LLM 标签重复生成不稳定有关。本文认为，LLM 模拟社会调查可以作为探索性传播研究工具，但不能直接替代真实民调。"
        ],
        "引言": [
            "生成式人工智能进入社会科学和传播研究之后，一个重要变化是：研究者不仅可以用模型处理既有文本，也开始尝试用模型生成新的社会数据。传统社会调查依赖真实样本、问卷设计、抽样执行和数据清洗，具有较高成本和较长周期；而 LLM 模拟调查则可以基于人物画像快速生成大规模态度数据，帮助研究者在正式调查之前形成假设，观察不同群体对公共议题的可能反应，并为舆论结构、政策传播和受众画像研究提供新的实验材料。",
            "但是，LLM 生成的回答并不等同于真实人的意见。它可能受到训练语料、Prompt 表述、抽样策略、人物画像信息量、输出格式要求和模型随机性的共同影响。如果只看单条回答是否“像人”，很容易忽略更重要的问题：当大量回答被汇总后，生成数据是否接近真实社会调查？不同群体之间的差异是否被保留？少数意见是否被压缩？同一批 persona 重复生成时，标签是否能稳定复现？因此，本文关注的核心问题不是“LLM 能否替代真实民调”，而是“LLM 生成的社会调查数据在什么意义上可信，又在哪些地方失真”。",
            "本项目的起点是一条数据驱动路径。我们最初在 Hugging Face 上发现 Nemotron-Personas-France 数据集，它包含约百万级合成法国居民画像，既有人口统计字段，也有职业、生活方式、兴趣、文化背景等文本描述。看到这批数据之后，我们首先思考的是：如此丰富的个体画像可以支持什么研究？相比只做传统人口统计分析，这批数据更适合作为一组具有不同社会背景的虚拟受访者。因此，我们提出设想：能否让大语言模型基于每个人的画像，对具体公共政策作出态度判断，从而模拟一次社会调查？",
            "法国退休改革适合作为本项目的测试场景。一方面，它具有明确的政策选项，即回到 62 岁、维持 64 岁、进一步提高退休年龄；另一方面，它又与年龄、职业、体力劳动、收入和社会福利预期高度相关。这使得我们可以同时观察 LLM 是否复现宏观民意分布，也可以进一步检验 persona 中的人口统计和文本画像是否能够解释模型生成的态度。"
        ],
        "研究设计": [
            "本研究的整体流程可以概括为：找到人物画像数据，确定真实社会调查基准，构建 LLM 模拟民调系统，进行 Prompt × Sampling 交叉实验，生成主数据，再围绕主数据开展探索性分析、特征工程、机器学习建模和误差诊断。这一路径不是先给出一个固定理论模型再套用数据，而是从数据资源出发，逐步提出可检验的问题，并通过实验反复修正研究框架。",
            "第一，构建人物画像数据库。基础数据来自 Nemotron-Personas-France。记录包含年龄、性别、婚姻、家庭、职业、学历、收入、地区等结构化变量，也包含职业画像、生活方式、文化背景、兴趣爱好等文本字段。我们并不把这些 persona 当作真实法国人口，而是把它们视为生成模拟调查数据的输入材料。换句话说，它们不是调查对象本身，而是让 LLM 扮演受访者时所依据的社会背景材料。",
            "第二，确定真实民调基准。为了避免生成数据只在内部自洽而缺少外部参照，我们选取 IFOP 关于法国退休改革的真实民调作为宏观基准。该民调围绕法定退休年龄提出三个选项：回到 62 岁、维持 64 岁、进一步提高。我们将其映射为 support、neutral、oppose 三类标签，其中 support 约为 61%，neutral 约为 34%，oppose 约为 5%。这个基准提供了一个宏观标尺，使后续实验不只是比较模型输出好不好看，而是比较它是否在总体分布上接近真实调查。",
            "第三，设计 LLM 社会调查模拟系统。系统从人物画像数据库出发，经过抽样、Prompt 构建、LLM API 调用、JSON 解析、质量审计和数据集构建，最终得到包含 support_score、stance 和 reason 的结构化记录。为了保证输出可分析，我们要求模型返回严格 JSON，包括 0-10 的支持度评分、三分类立场和中文理由。系统还记录 parse success、API error、stance consistency 等质量字段，用于排除无法解析或明显异常的结果。我们还实现了前端界面，用于展示样本检索、调用设置、个体回答和历史结果，使数据生成过程更可观察。",
            "第四，进行 Prompt × Sampling 交叉实验。Sampling 回答“问谁”的问题，Prompt Context 回答“给模型哪些信息”的问题。Sampling 包括随机抽样和 FAISS 相似度检索；Prompt Context 包括只给人口统计信息、加入职业信息、给完整 persona 文本、以及压缩后的全字段画像。交叉实验记录 API 成功率、JSON 解析成功率、三分类分布、相对 IFOP 的分布误差和理由文本质量。结果显示，FAISS + P2 完整画像方案相对 IFOP 的三类分布 MAE 最低，因此被选为主数据生成方案。",
            "第五，围绕主数据开展探索性分析和机器学习诊断。EDA 用于观察生成数据是否存在可解释社会结构；特征工程和建模用于测试 persona 信息能解释多少 LLM 态度输出；分类诊断、标签稳定性实验和优化尝试则用于识别生成数据的边界。这个阶段不是为了追求一个最高分模型，而是把模型结果当作诊断工具，反过来理解生成数据哪里可靠、哪里不稳定。"
        ],
    }
    for heading, paras in sections.items():
        doc.add_heading(heading, level=1)
        for para in paras:
            add_para(doc, para)

    add_fig(doc, "system_architecture.png", "图 1  LLM 社会调查模拟系统架构", 5.9)
    add_fig(doc, "prompt_sampling_mae_heatmap.png", "图 2  Prompt × Sampling 交叉实验的分布误差，数值越低表示越接近 IFOP 基准", 5.3)

    doc.add_heading("发现", level=1)
    findings = [
        "第一，LLM 生成数据在宏观分布上具有一定真实性。主数据中 support 是多数类，neutral 次之，oppose 最少，这与 IFOP 真实民调的大方向一致。虽然 support 占比仍高于真实基准，oppose 占比也容易偏低，但生成数据不是完全随机噪声，而是能够形成接近真实议题结构的总体分布。这说明 LLM 在处理公共政策问题时，会调用某种社会常识和议题知识，而不是只机械复制输入字段。",
        "第二，年龄变量呈现清晰的社会结构信号。退休改革议题与生命周期高度相关，年龄越高越可能支持回到 62 岁。主数据 EDA 中，年龄组与支持率之间呈现方向性关系，说明模型并非只是在复制总体标签比例，而是把部分常识性社会机制编码进了回答。这个结果也说明，人物画像中的基本人口统计变量对 LLM 态度生成具有实际影响。",
        "第三，reason 文本具有解释价值，但也暴露了语义边界问题。不同立场的词云和主题占比显示，年龄、工作、改革、退休、经济压力等词在三类回答中反复出现。support 往往围绕体力劳动、退休公平和生活质量展开；neutral 常常强调谨慎、现实可行和维持平衡；oppose 则更复杂，它既可能表示维持现状，也可能表示财政压力、代际公平或改革必要性。也就是说，oppose 不是 support 的简单反向，而是一个语义来源更复杂的少数类。这种语义混合会导致普通分类模型很难学习稳定边界。",
        "第四，结构化特征能够解释一部分态度，但解释力有限。第一轮建模比较了 A0 到 A3 四种结构化特征表示和 Linear、Ridge、Random Forest、XGBoost、LightGBM 等模型。A0 是最低复杂度基线，A1 使用 one-hot 展开类别变量，A2 加入年龄段和临近退休等年龄机制，A3 进一步进行职业、学历、社会分组归并。结果显示，复杂模型特别是 Boosting 方法能够带来提升，但最佳测试集 R² 大约停留在 0.20。这意味着 persona 中的年龄、职业、教育等信息确实有用，但不足以充分预测个体层面的支持度。",
        "第五，加入文本画像只带来有限增量。第二轮特征工程进一步尝试完整 persona 文本、职业文本、生活方式文本以及分字段 embedding + PCA。最佳方案 T4 只把 R² 从约 0.200 提升到约 0.206，Macro F1 也只小幅改善。这个结果说明，更多文本维度并不自动等于更有效的信息。persona 中大量文本是兴趣、生活方式或一般背景，未必直接对应政治态度形成机制。对退休改革这样的议题，真正关键的可能是工会倾向、政策信任、养老金依赖、职业风险和收入压力等更近端变量，而这些变量在原始 persona 中并不充分。",
        "第六，分类问题不是平均地差，而是 oppose 最难。代表模型的类别 F1 显示，support 较容易识别，neutral 处于中等水平，oppose 在多数直接三分类模型中几乎识别不出来。混淆矩阵进一步表明，真实 oppose 样本往往被吸收到 neutral 或 support 中。这一现象不能简单解释为样本量少，因为模型对少数类的错误并非随机分散，而是系统性地向多数类和边界更模糊的中间类靠拢。",
        "第七，标签稳定性限制了下游模型上限。重复调用实验显示，同一批 persona 再次交给 LLM 生成标签时，support 的一致率达到 90.4%，但 neutral 和 oppose 的一致率只有约 43%。这说明监督模型面对的标签并不是完全稳定的真实标签，而是 LLM 再生成时也会波动的模拟标签。如果上游标签本身在边界处摇摆，下游模型很难通过更复杂算法获得稳定高分。",
        "第八，优化少数类识别存在明显代价。我们尝试类别加权、概率阈值调整和两阶段分类等方案。它们可以提高 oppose 召回或 F1，但通常会牺牲总体准确率。两阶段分类在综合表现上较好，Macro F1 和 Oppose F1 相比原始三分类有所提升，但仍不能根本解决少数类边界不稳定的问题。这说明模型优化可以缓解问题，却不能替代更好的数据构造和标签设计。"
    ]
    for item in findings:
        add_para(doc, item)

    for name, cap in [
        ("stance_vs_ifop.png", "图 3  模拟立场分布与 IFOP 基准对比"),
        ("age_group_support.png", "图 4  不同年龄组的 support 比例"),
        ("reason_theme_by_stance.png", "图 5  不同 stance 的 reason 主题占比"),
        ("structured_r2_heatmap.png", "图 6  结构化特征 × 模型的 Test R² 热力图"),
        ("feature_dimension_vs_r2.png", "图 7  特征维度与最佳 Test R² 的关系"),
        ("text_feature_increment_lines.png", "图 8  文本特征增量实验"),
        ("class_f1_lollipop_selected_models.png", "图 9  代表模型的类别级 F1"),
        ("label_stability_by_stance.png", "图 10  同一批 persona 重复生成标签的一致率"),
        ("oppose_recall_accuracy_tradeoff.png", "图 11  提高 oppose 识别的准确率-召回率权衡"),
    ]:
        add_fig(doc, name, cap, 5.5)

    doc.add_heading("结论", level=1)
    conclusions = [
        "本文从数智传播导论的视角，将 LLM 模拟调查视为一种新的传播数据生成方式，并通过法国退休改革案例评估其统计真实性。研究表明，LLM 能够基于人物画像生成具有一定宏观结构的模拟民调数据，尤其能够复现多数立场分布和年龄相关趋势。这说明 LLM 生成数据可以作为探索性工具，用于政策传播、舆论结构和受众画像研究中的前期假设生成。",
        "但本研究也表明，LLM 模拟调查不能被直接当作真实民调。其局限不只来自模型算法，也来自数据基础和标签机制：persona 缺少政治倾向、政府信任、政策利益关系等近端变量；少数类态度语义来源复杂；LLM 生成标签本身存在重复不稳定。正因为如此，使用 LLM 生成社会调查数据时，必须同时报告真实基准、抽样策略、Prompt 设置、解析成功率、标签稳定性和误差诊断，而不能只呈现一个看似漂亮的总体比例。",
        "对传播研究而言，本项目的价值在于提供了一条可复用的方法链：先用真实调查建立外部基准，再通过 Prompt × Sampling 选择主数据生成方案，随后结合 EDA、机器学习建模和稳定性实验诊断生成数据的可用边界。未来可以进一步扩展到不同国家、不同公共议题和不同 LLM 模型，比较中文、法语、英语 Prompt 是否会系统影响政策态度输出；也可以引入跨模型验证、跨语言验证、更多真实民调基准和人工复核标签。更重要的是，应将 LLM 模拟调查定位为真实社会调查的补充和诊断工具，而不是替代品。"
    ]
    for para in conclusions:
        add_para(doc, para)

    doc.save(SUB / "course_report_french_policy.docx")

def main() -> None:
    ensure_dirs()
    copy_assets()
    build_notebooks()
    build_docx()
    print(SUB)


if __name__ == "__main__":
    main()
