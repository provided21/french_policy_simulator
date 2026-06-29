"""
Generate word clouds for support/neutral/oppose LLM reasons.

Outputs PPT-ready PNGs plus token frequency tables.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from wordcloud import WordCloud


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "modeling_samples" / "main_1w_faiss_p2_clean.parquet"
OUT_DIR = ROOT / "output" / "reason_wordclouds"
FIG_DIR = OUT_DIR / "figures"

FONT_PATHS = [
    Path(r"C:\Windows\Fonts\Noto Sans SC Medium (TrueType).otf"),
    Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
]

STANCE_ORDER = ["support", "neutral", "oppose"]
STANCE_META = {
    "support": {
        "title": "支持：恢复 62 岁",
        "color": "Greens",
        "bg": "#F7FBF7",
    },
    "neutral": {
        "title": "中立：维持 64 岁",
        "color": "Blues",
        "bg": "#F7FAFF",
    },
    "oppose": {
        "title": "反对：进一步提高",
        "color": "Oranges",
        "bg": "#FFF9F3",
    },
}

CUSTOM_WORDS = [
    "退休年龄",
    "提前退休",
    "延迟退休",
    "维持现状",
    "现行制度",
    "财政可持续",
    "养老金",
    "政策稳定",
    "工作压力",
    "体力劳动",
    "职业背景",
    "生活负担",
    "家庭责任",
    "信息不足",
    "无法判断",
    "改革",
    "稳定",
    "谨慎",
    "财政",
    "工作",
]

STOPWORDS = {
    "的",
    "了",
    "和",
    "与",
    "及",
    "或",
    "在",
    "对",
    "于",
    "为",
    "是",
    "有",
    "因",
    "因其",
    "由于",
    "可能",
    "倾向",
    "倾向于",
    "考虑",
    "基于",
    "较",
    "更",
    "其",
    "这",
    "该",
    "一个",
    "一种",
    "方面",
    "相关",
    "背景",
    "个人",
    "作为",
    "明确",
    "立场",
    "画像",
    "态度",
    "表态",
    "直接",
    "提及",
    "显示",
    "体现",
    "影响",
    "关联",
    "普通",
    "缺乏",
    "没有",
    "比较",
    "来自",
    "认为",
    "表达",
    "因素",
    "问题",
    "身份",
    "未知",
    "支持",
    "反对",
    "中立",
    "政策",
    "改革",
}

KEEP_SINGLE_CHARS = {"稳", "老"}


def setup() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for word in CUSTOM_WORDS:
        jieba.add_word(word, freq=10000)
    for font_path in FONT_PATHS:
        if font_path.exists():
            return font_path
    raise FileNotFoundError("No Chinese font found in C:\\Windows\\Fonts.")


def clean_token(token: str) -> str | None:
    token = token.strip()
    token = token.strip("，。；：、,.!?！？（）()《》“”\"'[]【】")
    if not token:
        return None
    if token in STOPWORDS:
        return None
    if len(token) == 1 and token not in KEEP_SINGLE_CHARS:
        return None
    if token.isnumeric():
        return None
    if all(not ("\u4e00" <= ch <= "\u9fff") for ch in token):
        return None
    return token


def tokenize_reasons(reasons: pd.Series) -> Counter[str]:
    counter: Counter[str] = Counter()
    for reason in reasons.fillna("").astype(str):
        for raw_token in jieba.cut(reason, cut_all=False):
            token = clean_token(raw_token)
            if token:
                counter[token] += 1
    return counter


def save_frequency(counter: Counter[str], stance: str) -> None:
    freq = pd.DataFrame(counter.most_common(80), columns=["token", "count"])
    freq.to_csv(OUT_DIR / f"{stance}_reason_word_frequency.csv", index=False, encoding="utf-8-sig")


def make_cloud(counter: Counter[str], stance: str, font_path: Path) -> WordCloud:
    meta = STANCE_META[stance]
    wc = WordCloud(
        font_path=str(font_path),
        width=1600,
        height=1000,
        background_color=meta["bg"],
        colormap=meta["color"],
        max_words=80,
        prefer_horizontal=0.88,
        relative_scaling=0.45,
        collocations=False,
        random_state=42,
        margin=8,
        contour_width=0,
    )
    return wc.generate_from_frequencies(dict(counter))


def save_single_cloud(wc: WordCloud, stance: str, font_prop: FontProperties) -> None:
    meta = STANCE_META[stance]
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=180)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(meta["title"], fontsize=20, pad=14, fontweight="bold", fontproperties=font_prop)
    fig.savefig(FIG_DIR / f"{stance}_reason_wordcloud.png", bbox_inches="tight", facecolor=meta["bg"])
    plt.close(fig)


def save_panel(clouds: dict[str, WordCloud], font_prop: FontProperties) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), dpi=180)
    for ax, stance in zip(axes, STANCE_ORDER):
        ax.imshow(clouds[stance], interpolation="bilinear")
        ax.axis("off")
        ax.set_title(STANCE_META[stance]["title"], fontsize=17, pad=10, fontweight="bold", fontproperties=font_prop)
    fig.suptitle("三类立场的 reason 词云对比", fontsize=22, fontweight="bold", y=0.98, fontproperties=font_prop)
    fig.savefig(FIG_DIR / "reason_wordclouds_panel.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(df: pd.DataFrame, counters: dict[str, Counter[str]], font_path: Path) -> None:
    lines = [
        "# Reason Word Clouds",
        "",
        f"Data: `{DATA_PATH.name}`",
        f"Font: `{font_path}`",
        "",
        "## Top Tokens",
        "",
    ]
    for stance in STANCE_ORDER:
        n = int((df["stance"] == stance).sum())
        lines.extend(
            [
                f"### {stance}",
                "",
                f"Rows: {n}",
                "",
                "| token | count |",
                "|---|---:|",
            ]
        )
        for token, count in counters[stance].most_common(15):
            lines.append(f"| {token} | {count} |")
        lines.append("")
    lines.extend(
        [
            "## PPT Note",
            "",
            "Use `figures/reason_wordclouds_panel.png` for a compact comparison slide. Use the single stance images if the slide needs larger labels.",
            "",
        ]
    )
    (OUT_DIR / "reason_wordclouds_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    font_path = setup()
    font_prop = FontProperties(fname=str(font_path))
    df = pd.read_parquet(DATA_PATH)
    counters: dict[str, Counter[str]] = {}
    clouds: dict[str, WordCloud] = {}

    for stance in STANCE_ORDER:
        counter = tokenize_reasons(df.loc[df["stance"] == stance, "reason"])
        counters[stance] = counter
        save_frequency(counter, stance)
        cloud = make_cloud(counter, stance, font_path)
        clouds[stance] = cloud
        save_single_cloud(cloud, stance, font_prop)

    save_panel(clouds, font_prop)
    write_summary(df, counters, font_path)
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
