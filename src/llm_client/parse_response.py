"""
响应解析模块
"""

import re


def parse_response(response):
    """
    解析 LLM 回答

    参数:
        response: LLM 原始回答文本

    返回:
        dict: 包含 full_response, support_score, stance
    """
    full_response = response

    # 尝试提取支持度分数（中文为主，兼容英文）
    # 格式：支持度：X/10, 支持度：X分, 支持度X/10, X/10
    support_score = 0.5  # 默认中立

    # 模式1: 支持度：X/10 或 支持度：X分
    pattern1 = r'支持度[：:]\s*(\d+)(?:/10|分)?'
    match = re.search(pattern1, response)
    if match:
        raw_score = int(match.group(1))
        support_score = raw_score / 10.0
    else:
        # 模式2: 单独的 X/10 格式
        pattern2 = r'支持度\s*(\d+)/10'
        match = re.search(pattern2, response)
        if match:
            raw_score = int(match.group(1))
            support_score = raw_score / 10.0
        else:
            # 模式3: 任何 X/10 格式
            pattern3 = r'(\d+)/10'
            match = re.search(pattern3, response)
            if match:
                raw_score = int(match.group(1))
                support_score = raw_score / 10.0

    # 确保分数在0-1范围内
    support_score = max(0.0, min(1.0, support_score))

    # 判断立场：0-3反对，4-6中立，7-10支持
    if support_score < 0.4:
        stance = "oppose"
    elif support_score < 0.7:
        stance = "neutral"
    else:
        stance = "support"

    return {
        "full_response": full_response,
        "support_score": support_score,
        "stance": stance
    }
