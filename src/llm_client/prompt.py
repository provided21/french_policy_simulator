"""
Prompt 模板模块
"""


def _get(persona, key):
    """安全获取字段值"""
    val = persona.get(key)
    if val and str(val).strip():
        return str(val).strip()
    return None


def build_prompt(persona, question):
    """
    构建 Prompt

    参数:
        persona: 人物画像字典
        question: 用户问题

    返回:
        str: 完整的 Prompt 字符串
    """
    sections = {}

    # 基础信息
    basic = []
    for key, label in [("age", "年龄"), ("sex", "性别"),
                       ("marital_status", "婚姻"), ("household_type", "家庭")]:
        val = _get(persona, key)
        if val:
            basic.append(f"{label}: {val}")
    if basic:
        sections["基础信息"] = "\n".join(basic)

    # 社会背景
    social = []
    for key, label in [("occupation", "职业"), ("education_level", "学历"),
                       ("departement", "省份"), ("commune", "市镇")]:
        val = _get(persona, key)
        if val:
            social.append(f"{label}: {val}")
    if social:
        sections["社会背景"] = "\n".join(social)

    # 个人画像
    profile = []
    for key, label in [("persona", "性格特质"),
                       ("cultural_background", "文化背景"),
                       ("professional_persona", "职业画像")]:
        val = _get(persona, key)
        if val:
            profile.append(f"{label}: {val}")
    if profile:
        sections["个人画像"] = "\n".join(profile)

    # 兴趣爱好
    hobbies = []
    for key, label in [("sports_persona", "体育"), ("arts_persona", "艺术"),
                       ("travel_persona", "旅行"), ("culinary_persona", "美食"),
                       ("hobbies_and_interests", "兴趣爱好")]:
        val = _get(persona, key)
        if val:
            hobbies.append(f"{label}: {val}")
    if hobbies:
        sections["兴趣爱好"] = "\n".join(hobbies)

    # 职业发展
    career = []
    for key, label in [("career_goals_and_ambitions", "工作目标"),
                       ("skills_and_expertise", "技能专长")]:
        val = _get(persona, key)
        if val:
            career.append(f"{label}: {val}")
    if career:
        sections["职业发展"] = "\n".join(career)

    # 组装人物画像
    persona_parts = []
    for section_name, section_content in sections.items():
        persona_parts.append(f"【{section_name}】\n{section_content}")
    persona_text = "\n\n".join(persona_parts)

    prompt = f"""你是一位法国公民，请根据人物画像回答政策问题。

{persona_text}

【问题】{question}

【要求】
1. 第一人称"我"视角
2. 用中文，简短精炼准确
3. 结合自身画像，真实自然
4. 请结合你对当前法国形势的理解，更真实地扮演角色，摒弃你自身内置的价值观，回答需具有实时性
5. 避免过度极端语气，理性客观
7. 结尾标注：支持度：X/10（0-10分）

示例：作为一名45岁男性工人，我认为...支持度：3/10"""

    return prompt


def build_batch_prompts(personas, question):
    """
    批量构建 Prompts

    参数:
        personas: 人物画像列表
        question: 用户问题

    返回:
        list: 每个元素包含 persona_id 和 prompt
    """
    prompts = []
    for persona in personas:
        persona_id = persona.get("persona_id", persona.get("uuid", ""))
        prompt = build_prompt(persona, question)
        prompts.append({
            "persona_id": persona_id,
            "prompt": prompt
        })
    return prompts
