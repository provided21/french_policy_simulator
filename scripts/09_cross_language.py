"""
跨语言对比实验
================================================================
验证 Prompt 语言对 LLM 立场输出的影响。

设计:
  - 随机抽取 500 个画像
  - 同一画像 × 3 种语言 (中文 / 法语 / 英语)
  - 同一 Run 2 Prompt 模板，仅翻译指令框架和字段标签
  - 被测: 立场分布、支持度均值、年龄-SHAP 是否受语言影响

输出:
  - data/cross_language_results.parquet
  - data/cross_language_summary.csv

运行: python scripts/09_cross_language.py
"""

import sys
import os
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.utils import Config
from src.llm_client import LLMClient, parse_response

# ============================================================
# 实验配置
# ============================================================

K = 500  # 每种语言样本量
RANDOM_SEED = 42

QUESTION = (
    "关于法定退休年龄，您认为应该怎么做？"
    "回到62岁、维持在64岁，还是进一步提高？"
)

QUESTION_FR = (
    "Concernant l'âge légal de départ à la retraite, "
    "que pensez-vous qu'il faille faire ? "
    "Revenir à 62 ans, maintenir à 64 ans, ou augmenter davantage ?"
)

QUESTION_EN = (
    "Regarding the legal retirement age, what do you think should be done? "
    "Return to 62, maintain at 64, or further increase?"
)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 多语言字段标签映射
# ============================================================

FIELD_LABELS = {
    "zh": {
        "age": "年龄", "sex": "性别",
        "marital_status": "婚姻", "household_type": "家庭",
        "occupation": "职业", "education_level": "学历",
        "departement": "省份", "commune": "市镇",
        "persona": "性格特质", "cultural_background": "文化背景",
        "professional_persona": "职业画像",
        "sports_persona": "体育", "arts_persona": "艺术",
        "travel_persona": "旅行", "culinary_persona": "美食",
        "hobbies_and_interests": "兴趣爱好",
        "career_goals_and_ambitions": "工作目标",
        "skills_and_expertise": "技能专长",
    },
    "fr": {
        "age": "Âge", "sex": "Sexe",
        "marital_status": "Situation matrimoniale", "household_type": "Type de foyer",
        "occupation": "Profession", "education_level": "Niveau d'éducation",
        "departement": "Département", "commune": "Commune",
        "persona": "Traits de personnalité", "cultural_background": "Origine culturelle",
        "professional_persona": "Profil professionnel",
        "sports_persona": "Sports", "arts_persona": "Arts",
        "travel_persona": "Voyages", "culinary_persona": "Cuisine",
        "hobbies_and_interests": "Loisirs et centres d'intérêt",
        "career_goals_and_ambitions": "Objectifs de carrière",
        "skills_and_expertise": "Compétences et expertise",
    },
    "en": {
        "age": "Age", "sex": "Sex",
        "marital_status": "Marital Status", "household_type": "Household Type",
        "occupation": "Occupation", "education_level": "Education Level",
        "departement": "Department", "commune": "Commune",
        "persona": "Personality Traits", "cultural_background": "Cultural Background",
        "professional_persona": "Professional Profile",
        "sports_persona": "Sports", "arts_persona": "Arts",
        "travel_persona": "Travel", "culinary_persona": "Culinary",
        "hobbies_and_interests": "Hobbies & Interests",
        "career_goals_and_ambitions": "Career Goals",
        "skills_and_expertise": "Skills & Expertise",
    },
}

SECTION_TITLES = {
    "zh": ["基础信息", "社会背景", "个人画像", "兴趣爱好", "职业发展"],
    "fr": ["Informations de base", "Contexte social", "Profil personnel", "Loisirs", "Carrière"],
    "en": ["Basic Information", "Social Background", "Personal Profile", "Interests", "Career"],
}

ROLE_INSTRUCTION = {
    "zh": "你是一位法国公民，请根据人物画像回答政策问题。",
    "fr": "Vous êtes un citoyen français. Veuillez répondre à la question politique en vous basant sur le profil personnel fourni.",
    "en": "You are a French citizen. Please answer the policy question based on the provided persona profile.",
}

REQUIREMENTS = {
    "zh": [
        '第一人称"我"视角',
        "用中文，简短精炼准确",
        "结合自身画像，真实自然",
        "请结合你对当前法国形势的理解，更真实地扮演角色，摒弃你自身内置的价值观，回答需具有实时性",
        "避免过度极端语气，理性客观",
        "结尾标注：支持度：X/10（0-10分）",
    ],
    "fr": [
        'Utilisez la première personne « je »',
        "Répondez en français, de manière concise et précise",
        "Appuyez-vous sur votre profil pour une réponse authentique et naturelle",
        "Tenez compte de votre compréhension de la situation française actuelle, jouez le rôle de manière plus réaliste, en mettant de côté vos propres valeurs",
        "Évitez les tons trop extrêmes, restez rationnel et objectif",
        "Terminez par : Score de soutien : X/10 (0-10)",
    ],
    "en": [
        'Use first-person "I" perspective',
        "Respond in English, be concise and precise",
        "Reference your persona profile for an authentic, natural response",
        "Consider your understanding of the current French situation, role-play more realistically, setting aside your own built-in values",
        "Avoid overly extreme tones, be rational and objective",
        "End with: Support score: X/10 (0-10)",
    ],
}

EXAMPLES = {
    "zh": "示例：作为一名45岁男性工人，我认为... 支持度：3/10",
    "fr": "Exemple : En tant qu'ouvrier de 45 ans, je pense que... Score de soutien : 3/10",
    "en": "Example: As a 45-year-old male worker, I think... Support score: 3/10",
}


def _get(persona, key):
    val = persona.get(key)
    if val and str(val).strip():
        return str(val).strip()
    return None


def build_prompt_multilang(persona, question, lang):
    """构建多语言 Prompt"""
    labels = FIELD_LABELS[lang]
    sections_titles = SECTION_TITLES[lang]

    sections = {}

    # 基础信息
    basic = []
    for key in ["age", "sex", "marital_status", "household_type"]:
        val = _get(persona, key)
        if val:
            basic.append(f"{labels[key]}: {val}")
    if basic:
        sections[sections_titles[0]] = "\n".join(basic)

    # 社会背景
    social = []
    for key in ["occupation", "education_level", "departement", "commune"]:
        val = _get(persona, key)
        if val:
            social.append(f"{labels[key]}: {val}")
    if social:
        sections[sections_titles[1]] = "\n".join(social)

    # 个人画像
    profile = []
    for key in ["persona", "cultural_background", "professional_persona"]:
        val = _get(persona, key)
        if val:
            profile.append(f"{labels[key]}: {val}")
    if profile:
        sections[sections_titles[2]] = "\n".join(profile)

    # 兴趣爱好
    hobbies = []
    for key in ["sports_persona", "arts_persona", "travel_persona",
                "culinary_persona", "hobbies_and_interests"]:
        val = _get(persona, key)
        if val:
            hobbies.append(f"{labels[key]}: {val}")
    if hobbies:
        sections[sections_titles[3]] = "\n".join(hobbies)

    # 职业发展
    career = []
    for key in ["career_goals_and_ambitions", "skills_and_expertise"]:
        val = _get(persona, key)
        if val:
            career.append(f"{labels[key]}: {val}")
    if career:
        sections[sections_titles[4]] = "\n".join(career)

    persona_parts = []
    for title, content in sections.items():
        if lang == "zh":
            persona_parts.append(f"【{title}】\n{content}")
        else:
            persona_parts.append(f"[{title}]\n{content}")
    persona_text = "\n\n".join(persona_parts)

    reqs = REQUIREMENTS[lang]
    numbered_reqs = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reqs))

    prompt = f"""{ROLE_INSTRUCTION[lang]}

{persona_text}

{'【问题】' if lang == 'zh' else '[Question] ' if lang == 'en' else '[Question] '}{question}

{'【要求】' if lang == 'zh' else '[Instructions]' if lang == 'en' else '[Instructions]'}
{numbered_reqs}

{EXAMPLES[lang]}"""

    return prompt


def parse_response_multilang(response, lang):
    """多语言解析 LLM 回答"""
    full_response = response
    support_score = 0.5

    patterns = {
        "zh": [r'支持度[：:]\s*(\d+)(?:/10|分)?', r'支持度\s*(\d+)/10'],
        "fr": [r'Score de soutien[ :]+\s*(\d+)(?:/10)?', r'score[ :]+\s*(\d+)/10'],
        "en": [r'Support score[ :]+\s*(\d+)(?:/10)?', r'score[ :]+\s*(\d+)/10'],
    }

    for pattern in patterns.get(lang, []):
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            raw_score = int(match.group(1))
            support_score = raw_score / 10.0
            break
    else:
        # fallback: any X/10
        match = re.search(r'(\d+)/10', response)
        if match:
            raw_score = int(match.group(1))
            support_score = raw_score / 10.0

    support_score = max(0.0, min(1.0, support_score))

    if support_score < 0.4:
        stance = "oppose"
    elif support_score < 0.7:
        stance = "neutral"
    else:
        stance = "support"

    return {
        "full_response": full_response,
        "support_score": support_score,
        "stance": stance,
    }


# ============================================================
# 主流程
# ============================================================

import re  # noqa: E402 (used in parse_response_multilang)


def main():
    print("=" * 60)
    print("跨语言对比实验 —— ZH / FR / EN")
    print("=" * 60)
    print(f"\n  K = {K} 画像 × 3 语言 = {K * 3} 次调用")

    # 加载数据
    print("\n[1/4] 加载画像数据...")
    df = pd.read_parquet(Config.DATA_PATH)
    print(f"      总画像数: {len(df):,}")

    # 随机抽样
    print(f"\n[2/4] 随机抽样 {K} 个画像 (seed={RANDOM_SEED})...")
    sample_df = df.sample(n=K, random_state=RANDOM_SEED).copy().reset_index(drop=True)
    personas = sample_df.to_dict('records')
    print(f"      抽样完成")

    # 构建所有 prompts：每人 × 3 语言，打乱顺序
    print(f"\n[3/4] 构建多语言 Prompts 并打乱顺序...")
    lang_configs = [
        ("zh", QUESTION),
        ("fr", QUESTION_FR),
        ("en", QUESTION_EN),
    ]

    all_prompts = []
    for _, row in sample_df.iterrows():
        persona = row.to_dict()
        persona_id = persona.get("persona_id", persona.get("uuid", ""))
        for lang, question in lang_configs:
            prompt = build_prompt_multilang(persona, question, lang)
            all_prompts.append({
                "persona_id": persona_id,
                "lang": lang,
                "prompt": prompt,
            })

    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(all_prompts)
    print(f"      生成 {len(all_prompts)} 条 Prompt，顺序已打乱")

    # 调用 LLM
    if not Config.LLM_API_KEY:
        print("\n[错误] 未找到 LLM_API_KEY")
        return

    client = LLMClient(
        api_key=Config.LLM_API_KEY,
        model=Config.LLM_MODEL,
        base_url=Config.LLM_BASE_URL,
    )

    def on_progress(done, total):
        pct = done / total * 100
        print(f"\r      进度: {done}/{total} ({pct:.1f}%)", end="", flush=True)

    flat_prompts = [{"persona_id": p["persona_id"], "prompt": p["prompt"]}
                    for p in all_prompts]
    responses = client.call_batch_sync(flat_prompts, progress_callback=on_progress)
    print("")

    success = sum(1 for r in responses if r and r.get("success"))
    fail = len(responses) - success
    print(f"      成功: {success}, 失败: {fail}")

    # 解析
    results = []
    for i, r in enumerate(responses):
        info = all_prompts[i]
        if r and r.get("success") and r.get("response"):
            parsed = parse_response_multilang(r["response"], info["lang"])
            results.append({
                "persona_id": info["persona_id"],
                "lang": info["lang"],
                "support_score": parsed["support_score"],
                "stance": parsed["stance"],
                "response": r["response"],
            })

    results_df = pd.DataFrame(results)

    # 统计汇总
    print("\n[4/4] 跨语言对比统计")
    print("=" * 60)

    summary_rows = []
    for lang, label in [("zh", "中文"), ("fr", "Français"), ("en", "English")]:
        subset = results_df[results_df["lang"] == lang]
        stance_pct = subset["stance"].value_counts(normalize=True) * 100
        row = {
            "lang": lang,
            "label": label,
            "n_valid": len(subset),
            "mean_score": subset["support_score"].mean(),
            "std_score": subset["support_score"].std(),
            "support_pct": stance_pct.get("support", 0),
            "neutral_pct": stance_pct.get("neutral", 0),
            "oppose_pct": stance_pct.get("oppose", 0),
        }
        summary_rows.append(row)

        print(f"\n  {label} ({lang})  n={row['n_valid']}:")
        print(f"    支持: {row['support_pct']:.1f}%  "
              f"中立: {row['neutral_pct']:.1f}%  "
              f"反对: {row['oppose_pct']:.1f}%")
        print(f"    平均分: {row['mean_score']:.3f} ± {row['std_score']:.3f}")

    # 两两语言间分数相关性 (同一画像)
    print(f"\n  语言间 Pearson 相关系数 (同一画像):")
    pivot = results_df.pivot_table(
        index="persona_id", columns="lang", values="support_score"
    )
    if len(pivot) > 1:
        for l1, l2 in [("zh", "fr"), ("zh", "en"), ("fr", "en")]:
            corr = pivot[l1].corr(pivot[l2]) if l1 in pivot and l2 in pivot else float("nan")
            print(f"    {l1} ↔ {l2}: r = {corr:.4f}")

    # 立场一致性 (同一画像不同语言)
    stance_pivot = results_df.pivot_table(
        index="persona_id", columns="lang", values="stance", aggfunc="first"
    )
    if len(stance_pivot) > 1:
        agree_zh_fr = (stance_pivot["zh"] == stance_pivot["fr"]).mean() * 100
        agree_zh_en = (stance_pivot["zh"] == stance_pivot["en"]).mean() * 100
        agree_fr_en = (stance_pivot["fr"] == stance_pivot["en"]).mean() * 100
        print(f"\n  跨语言立场一致率:")
        print(f"    ZH==FR: {agree_zh_fr:.1f}%")
        print(f"    ZH==EN: {agree_zh_en:.1f}%")
        print(f"    FR==EN: {agree_fr_en:.1f}%")

    # 保存
    parquet_path = os.path.join(OUTPUT_DIR, "cross_language_results.parquet")
    results_df.to_parquet(parquet_path, index=False)
    print(f"\n  详细结果已保存: {parquet_path}")

    csv_path = os.path.join(OUTPUT_DIR, "cross_language_summary.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"  汇总统计已保存: {csv_path}")

    print("\n" + "=" * 60)
    print("跨语言实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
