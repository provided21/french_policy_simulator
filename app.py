"""
法国政策模拟器 - Streamlit 前端

运行: streamlit run app.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows UTF-8 输出（跳过 Streamlit 和已关闭的 stdout）
if sys.platform == "win32":
    try:
        import io
        if hasattr(sys.stdout, 'buffer') and not sys.stdout.closed:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass

# 离线模式加载模型（跳过 HuggingFace 网络检查，大幅加快加载速度）
os.environ["HF_HUB_OFFLINE"] = "1"

from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils import Config
from src.retriever.build_index import load_index
from src.retriever.search import search_similar
from src.retriever import create_database_adapter
from src.llm_client import LLMClient, build_batch_prompts, parse_response, build_prompt
from src.data_pipeline import load_local_data

st.set_page_config(
    page_title="法国政策模拟器",
    page_icon="🇫🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 样式 - 自适应深浅色主题
# ============================================
st.markdown("""
<style>
/* ===== 深色主题 ===== */
@media (prefers-color-scheme: dark) {
    .stApp { background: #0f1117 !important; color: #e0e0e0 !important; }
    .main-header {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sub-header { color: #888 !important; }
    .question-card { background: #1a1f2e !important; border-left-color: #3a7bd5 !important; }
    .question-card pre { color: #d0d0d0 !important; }
    .kpi-card.blue { background: linear-gradient(135deg, #1a3a5c, #1e4976) !important; }
    .kpi-card.green { background: linear-gradient(135deg, #1a3c2a, #1e6942) !important; }
    .kpi-card.orange { background: linear-gradient(135deg, #3c2a1a, #69421e) !important; }
    .kpi-card.red { background: linear-gradient(135deg, #3c1a1a, #691e1e) !important; }
    .kpi-label { color: #bbb !important; }
    .chart-box { background: #1a1f2e !important; }
    .detail-prompt { background: #1a2744 !important; }
    .detail-response { background: #1a3c2a !important; }
    .history-card { background: #1a1f2e !important; border-left-color: #3a7bd5 !important; }
    section[data-testid="stSidebar"] { background-color: #141824 !important; }
    .stTabs [data-baseweb="tab"] { background-color: #1a1f2e !important; color: #aaa !important; border-color: #2a2f3e !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #1a3a5c, #1e4976) !important; color: #fff !important; }
}

/* ===== 浅色主题 ===== */
@media (prefers-color-scheme: light) {
    .stApp { background: linear-gradient(135deg, #f5f7fa, #e8ecf1) !important; color: #333 !important; }
    .main-header {
        background: linear-gradient(90deg, #1a73e8, #34a853);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sub-header { color: #666 !important; }
    .question-card { background: #ffffff !important; border-left-color: #1a73e8 !important; box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important; }
    .question-card pre { color: #333 !important; }
    .kpi-card.blue { background: linear-gradient(135deg, #e3f2fd, #bbdefb) !important; }
    .kpi-card.green { background: linear-gradient(135deg, #e8f5e9, #c8e6c9) !important; }
    .kpi-card.orange { background: linear-gradient(135deg, #fff3e0, #ffe0b2) !important; }
    .kpi-card.red { background: linear-gradient(135deg, #fce4ec, #f8bbd0) !important; }
    .kpi-label { color: #555 !important; }
    .chart-box { background: #ffffff !important; box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important; }
    .detail-prompt { background: #f8f9fa !important; }
    .detail-response { background: #f0faf0 !important; }
    .history-card { background: #ffffff !important; border-left-color: #1a73e8 !important; box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important; }
    section[data-testid="stSidebar"] { background-color: #f0f2f6 !important; }
    .stTabs [data-baseweb="tab"] { background-color: #f0f2f6 !important; color: #666 !important; border-color: #ddd !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #e3f2fd, #bbdefb) !important; color: #1a73e8 !important; }
}

/* 公共样式 */
.main-header { font-size: 2.5rem; font-weight: 800; margin-bottom: 0.3rem; }
.sub-header { font-size: 1rem; margin-bottom: 0.5rem; }
.question-card { border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
.question-card pre { white-space: pre-wrap; font-size: 0.95rem; margin: 0.5rem 0 0 0; }
.kpi-card { padding: 1.2rem; border-radius: 12px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.3); }
.kpi-value { font-size: 2.2rem; font-weight: 800; }
.kpi-label { font-size: 0.85rem; margin-top: 0.2rem; }
.chart-box { border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem; }
.detail-prompt { padding: 0.8rem 1rem; border-radius: 8px; border-left: 4px solid #3a7bd5; }
.detail-response { padding: 0.8rem 1rem; border-radius: 8px; border-left: 4px solid #22c55e; }
.history-card { border-radius: 10px; padding: 1rem; margin-bottom: 0.5rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { padding: 12px 28px; border-radius: 10px 10px 0 0; border: 1px solid; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# 检测浏览器主题
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"  # 默认深色


# ============================================
# 翻译映射
# ============================================

OCCUPATION_ZH = {
    "Ingénieur logiciel": "软件工程师", "Développeur web": "网站开发",
    "Enseignant": "教师", "Professeur": "教授", "Médecin": "医生",
    "Infirmier": "护士", "Comptable": "会计", "Avocat": "律师",
    "Architecte": "建筑师", "Commercial": "销售", "Consultant": "顾问",
    "Chef de projet": "项目经理", "Directeur": "总监", "Manager": "经理",
    "Responsable marketing": "市场主管", "Analyste financier": "金融分析师",
    "Data scientist": "数据科学家", "Graphiste": "平面设计师",
    "Journaliste": "记者", "Photographe": "摄影师",
    "Cuisinier": "厨师", "Boulanger": "面包师", "Serveur": "服务员",
    "Caissier": "收银员", "Vendeur": "售货员", "Électricien": "电工",
    "Plombier": "水管工", "Menuisier": "木匠", "Mécanicien": "机械师",
    "Conducteur": "司机", "Policier": "警察", "Pompier": "消防员",
    "Agent de sécurité": "保安", "Retraité": "退休", "Étudiant": "学生",
    "Au foyer": "家庭主妇/主夫", "Chômeur": "失业",
    "Agriculteur": "农民", "Ouvrier": "工人", "Employé de bureau": "文员",
    "Secrétaire": "秘书", "Assistante": "助理", "Technicien": "技术员",
    "Ingénieur": "工程师", "Pharmacien": "药剂师", "Dentiste": "牙医",
    "Psychologue": "心理学家", "Kinésithérapeute": "理疗师",
    "Avocat d'affaires": "商务律师", "Notaire": "公证人",
    "Artisan": "手工艺人", "Entrepreneur": "企业家",
    "Freelance": "自由职业者", "Fonctionnaire": "公务员",
    "Militaire": "军人", "Pilote": "飞行员",
    "Directeur financier": "财务总监", "Directeur commercial": "销售总监",
    "Directeur technique": "技术总监", "Directeur des ressources humaines": "人力资源总监",
    "Responsable logistique": "物流主管", "Responsable qualité": "质量主管",
    "Chercheur": "研究员", "Biologiste": "生物学家",
    "Chimiste": "化学家", "Mathématicien": "数学家",
    "Statisticien": "统计学家", "Économiste": "经济学家",
    "Sociologue": "社会学家", "Historien": "历史学家",
    "Écrivain": "作家", "Musicien": "音乐家",
    "Acteur": "演员", "Réalisateur": "导演",
    "Producteur": "制作人", "Animateur": "主持人",
    "Agent immobilier": "房产经纪", "Banquier": "银行家",
    "Assureur": "保险经纪", "Courtier": "经纪人",
    "Juriste": "法务", "Traducteur": "翻译",
    "Interprète": "口译", "Professeur de lycée": "高中教师",
    "Professeur des écoles": "小学教师", "Professeur d'université": "大学教授",
    "Éducateur": "教育工作者", "Animateur socio-culturel": "社会文化工作者",
    "Aide-soignant": "护工", "Sage-femme": "助产士",
    "Vétérinaire": "兽医", "Opticien": "眼镜师",
    "Coiffeur": "理发师", "Esthéticienne": "美容师",
    "Agent d'entretien": "保洁", "Garde d'enfants": "保姆",
    "Livreur": "快递员", "Agent de voyage": "旅行社职员",
    "Guide touristique": "导游", "Hôtelier": "酒店经理",
    "Réceptionniste": "前台", "Gérant de restaurant": "餐厅经理",
    "Sommelier": "侍酒师", "Pâtissier": "糕点师",
    "Boucher": "屠夫", "Fromager": "奶酪师",
    "Viticulteur": "葡萄种植者", "Pêcheur": "渔民",
    "Sylviculteur": "林业工人", "Jardinier": "园丁",
    "Paysagiste": "景观设计师", "Géomètre": "测量师",
    "Urbaniste": "城市规划师",
}

DEPARTEMENT_ZH = {
    "Ain": "安省(01)", "Aisne": "埃纳省(02)", "Allier": "阿列省(03)",
    "Alpes-de-Haute-Provence": "上普罗旺斯阿尔卑斯(04)",
    "Hautes-Alpes": "上阿尔卑斯省(05)", "Alpes-Maritimes": "滨海阿尔卑斯省(06)",
    "Ardèche": "阿尔代什省(07)", "Ardennes": "阿登省(08)",
    "Ariège": "阿列日省(09)", "Aube": "奥布省(10)",
    "Aude": "奥德省(11)", "Aveyron": "阿韦龙省(12)",
    "Bouches-du-Rhône": "罗讷河口省(13)", "Calvados": "卡尔瓦多斯省(14)",
    "Cantal": "康塔尔省(15)", "Charente": "夏朗德省(16)",
    "Charente-Maritime": "滨海夏朗德省(17)", "Cher": "谢尔省(18)",
    "Corrèze": "科雷兹省(19)", "Corse-du-Sud": "南科西嘉省(2A)",
    "Haute-Corse": "上科西嘉省(2B)", "Côte-d'Or": "科多尔省(21)",
    "Côtes-d'Armor": "阿摩尔滨海省(22)", "Creuse": "克勒兹省(23)",
    "Dordogne": "多尔多涅省(24)", "Doubs": "杜省(25)",
    "Drôme": "德龙省(26)", "Eure": "厄尔省(27)",
    "Eure-et-Loir": "厄尔-卢瓦尔省(28)", "Finistère": "菲尼斯泰尔省(29)",
    "Gard": "加尔省(30)", "Haute-Garonne": "上加龙省(31)",
    "Gers": "热尔省(32)", "Gironde": "吉伦特省(33)",
    "Hérault": "埃罗省(34)", "Ille-et-Vilaine": "伊勒-维莱讷省(35)",
    "Indre": "安德尔省(36)", "Indre-et-Loire": "安德尔-卢瓦尔省(37)",
    "Isère": "伊泽尔省(38)", "Jura": "汝拉省(39)",
    "Landes": "朗德省(40)", "Loir-et-Cher": "卢瓦-谢尔省(41)",
    "Loire": "卢瓦尔省(42)", "Haute-Loire": "上卢瓦尔省(43)",
    "Loire-Atlantique": "大西洋卢瓦尔省(44)", "Loiret": "卢瓦雷省(45)",
    "Lot": "洛特省(46)", "Lot-et-Garonne": "洛特-加龙省(47)",
    "Lozère": "洛泽尔省(48)", "Maine-et-Loire": "曼恩-卢瓦尔省(49)",
    "Manche": "芒什省(50)", "Marne": "马恩省(51)",
    "Haute-Marne": "上马恩省(52)", "Mayenne": "马耶讷省(53)",
    "Meurthe-et-Moselle": "默尔特-摩泽尔省(54)", "Meuse": "默兹省(55)",
    "Morbihan": "莫尔比昂省(56)", "Moselle": "摩泽尔省(57)",
    "Nièvre": "涅夫勒省(58)", "Nord": "北部省(59)",
    "Oise": "瓦兹省(60)", "Orne": "奥恩省(61)",
    "Pas-de-Calais": "加来海峡省(62)", "Puy-de-Dôme": "多姆山省(63)",
    "Pyrénées-Atlantiques": "大西洋比利牛斯省(64)",
    "Hautes-Pyrénées": "上比利牛斯省(65)",
    "Pyrénées-Orientales": "东比利牛斯省(66)",
    "Bas-Rhin": "下莱茵省(67)", "Haut-Rhin": "上莱茵省(68)",
    "Rhône": "罗讷省(69)", "Haute-Saône": "上索恩省(70)",
    "Saône-et-Loire": "索恩-卢瓦尔省(71)", "Sarthe": "萨尔特省(72)",
    "Savoie": "萨瓦省(73)", "Haute-Savoie": "上萨瓦省(74)",
    "Paris": "巴黎(75)", "Seine-Maritime": "滨海塞纳省(76)",
    "Seine-et-Marne": "塞纳-马恩省(77)", "Yvelines": "伊夫林省(78)",
    "Deux-Sèvres": "德塞夫勒省(79)", "Somme": "索姆省(80)",
    "Tarn": "塔恩省(81)", "Tarn-et-Garonne": "塔恩-加龙省(82)",
    "Var": "瓦尔省(83)", "Vaucluse": "沃克吕兹省(84)",
    "Vendée": "旺代省(85)", "Vienne": "维埃纳省(86)",
    "Haute-Vienne": "上维埃纳省(87)", "Vosges": "孚日省(88)",
    "Yonne": "约讷省(89)", "Territoire de Belfort": "贝尔福地区省(90)",
    "Essonne": "埃松省(91)", "Hauts-de-Seine": "上塞纳省(92)",
    "Seine-Saint-Denis": "塞纳-圣但尼省(93)",
    "Val-de-Marne": "马恩河谷省(94)", "Val-d'Oise": "瓦兹河谷省(95)",
    # 海外省
    "Guadeloupe": "瓜德罗普(971)", "Martinique": "马提尼克(972)",
    "Guyane": "法属圭亚那(973)", "La Réunion": "留尼汪(974)",
}

EDUCATION_ZH = {
    "Bac+5 ou plus": "硕士及以上(Bac+5+)", "Bac+5": "硕士/研二(Bac+5)",
    "Bac+3 ou Bac+4": "本科至研一(Bac+3~4)", "Bac+3": "学士/大三(Bac+3)", "Bac+4": "研一(Bac+4)",
    "Bac+2": "大二(Bac+2)", "Bac+1": "大一(Bac+1)",
    "Baccalauréat": "高中毕业(Bac)", "CAP ou BEP": "职业证书(CAP/BEP)",
    "Sans diplôme ou CEP": "无学历", "Brevet": "初中毕业",
    "Doctorat": "博士", "Master": "硕士", "Licence": "学士",
    "Post-doctorat": "博士后", "Primaire": "小学", "Secondaire": "中学",
}

MARITAL_ZH = {
    "Célibataire": "单身", "Marié(e)": "已婚", "Pacsé(e)": "民事结合",
    "Divorcé(e)": "离婚", "Veuf/Veuve": "丧偶", "En couple": "同居",
    "Séparé(e)": "分居",
}

# PCS 职业社会职业类别 (法国国家统计局 INSEE 分类)
PCS_ZH = {
    "Agriculteurs exploitants": "农业从业者",
    "Artisans, commerçants, chefs d'entreprise": "工商业主/企业主",
    "Cadres et professions intellectuelles supérieures": "高管/高级知识分子",
    "Professions intermédiaires": "中级职业",
    "Employés": "雇员/职员",
    "Ouvriers": "工人",
    "Retraités": "退休人员",
    "Autres sans activité professionnelle": "其他无业人员",
}

HOUSEHOLD_ZH = {
    "Personne seule": "独居", "Couple sans enfant": "无子女夫妻",
    "Couple avec enfant(s)": "有子女夫妻", "Famille monoparentale": "单亲家庭",
    "Colocation": "合租", "Autre": "其他",
}

SEX_ZH = {"M": "男", "F": "女", "Homme": "男", "Femme": "女"}


def _normalize_fr(text):
    """去除法语重音，用于模糊匹配"""
    import unicodedata
    n = unicodedata.normalize('NFD', text)
    n = ''.join(c for c in n if unicodedata.category(c) != 'Mn')
    return n.strip().lower()


# 构建无重音版本的映射，处理变音缺失问题
OCCUPATION_NORM = {_normalize_fr(k): v for k, v in OCCUPATION_ZH.items()}
PCS_NORM = {_normalize_fr(k): v for k, v in PCS_ZH.items()}
EDUCATION_NORM = {_normalize_fr(k): v for k, v in EDUCATION_ZH.items()}
DEPARTEMENT_NORM = {_normalize_fr(k): v for k, v in DEPARTEMENT_ZH.items()}
MARITAL_NORM = {_normalize_fr(k): v for k, v in MARITAL_ZH.items()}
HOUSEHOLD_NORM = {_normalize_fr(k): v for k, v in HOUSEHOLD_ZH.items()}


def zh_translate(value, field_type="occupation"):
    """将法文字段翻译为中文（支持精确和模糊匹配）"""
    if pd.isna(value) or not value:
        return ""
    val = str(value).strip()
    if not val:
        return ""

    if field_type == "occupation":
        return (OCCUPATION_ZH.get(val)
                or PCS_ZH.get(val)
                or OCCUPATION_NORM.get(_normalize_fr(val))
                or val)
    elif field_type == "departement":
        return DEPARTEMENT_ZH.get(val) or DEPARTEMENT_NORM.get(_normalize_fr(val), val)
    elif field_type == "education_level":
        return EDUCATION_ZH.get(val) or EDUCATION_NORM.get(_normalize_fr(val), val)
    elif field_type == "marital_status":
        return MARITAL_ZH.get(val) or MARITAL_NORM.get(_normalize_fr(val), val)
    elif field_type == "household_type":
        return HOUSEHOLD_ZH.get(val) or HOUSEHOLD_NORM.get(_normalize_fr(val), val)
    elif field_type == "sex":
        return SEX_ZH.get(val, val)
    return val


def get_theme_colors():
    """根据当前主题返回配色"""
    if st.session_state.get("theme", "dark") == "dark":
        return {
            "bg": "#0f1117", "card": "#1a1f2e", "plot_bg": "#141824",
            "text": "#d0d0d0", "accent": "#3a7bd5", "green": "#4caf50",
            "orange": "#ff9800", "red": "#ef5350", "blue": "#5ba3e6"
        }
    else:
        return {
            "bg": "#ffffff", "card": "#ffffff", "plot_bg": "#f4f5f7",
            "text": "#333333", "accent": "#1a73e8", "green": "#2e7d32",
            "orange": "#e65100", "red": "#c62828", "blue": "#1565c0"
        }


# ============================================
# 工具函数
# ============================================


# 基础列（缓存在内存，~150 MB）：用于检索 + 筛选 + 图表
_SLIM_COLS = [
    "persona_id", "uuid", "age", "sex", "marital_status", "household_type",
    "occupation", "education_level", "departement", "commune",
]

# 人物画像文本列（按需加载）：仅对检索结果加载
_PERSONA_COLS = [
    "persona", "cultural_background", "professional_persona",
    "sports_persona", "arts_persona", "travel_persona", "culinary_persona",
    "hobbies_and_interests", "career_goals_and_ambitions", "skills_and_expertise",
]

# 转 category 的列，进一步压缩内存
_CATEGORY_COLS = ["sex", "marital_status", "household_type", "occupation",
                  "education_level", "departement"]


@st.cache_resource
def load_data_cache():
    """缓存基础数据（仅 ~150 MB，不加载人物画像文本列）"""
    parquet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "df_full.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path, columns=_SLIM_COLS, engine="pyarrow")
    else:
        csv_cols = [c for c in _SLIM_COLS if c != "persona_id"]
        df = pd.read_csv(Config.DATA_PATH, usecols=csv_cols, engine="pyarrow")
        df["persona_id"] = df["uuid"].astype(str)
    # 低基数列转 category，字符串变整数索引
    for col in _CATEGORY_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _load_persona_text(persona_ids):
    """按需加载人物画像文本列（仅对检索到的 k 条结果）"""
    parquet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "df_full.parquet")
    if not os.path.exists(parquet_path):
        return None
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path, columns=["persona_id"] + _PERSONA_COLS)
    df_text = table.to_pandas()
    del table
    df_text["persona_id_str"] = df_text["persona_id"].astype(str)
    target = set(str(pid) for pid in persona_ids)
    return df_text[df_text["persona_id_str"].isin(target)]


@st.cache_resource
def load_model():
    """缓存 SentenceTransformer 模型（延迟导入 torch，避免 Streamlit 启动卡顿）"""
    from sentence_transformers import SentenceTransformer
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        import os
        MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all-MiniLM-L6-v2")
        model = SentenceTransformer(MODEL_DIR)
    finally:
        sys.stdout = old_stdout
    return model


@st.cache_resource
def load_faiss_index():
    """缓存 FAISS 索引"""
    if not os.path.exists(Config.INDEX_PATH):
        return None
    return load_index(Config.INDEX_PATH)


def run_pipeline(question, k, api_key=None, base_url=None, model=None, rpm=3000, tpm=500000, temperature=0.7):
    """执行完整流程：加载数据 -> 检索 -> LLM -> 保存（带进度展示）"""
    if not api_key:
        st.error("请输入 API Key")
        return None

    active_model_name = model or Config.LLM_MODEL

    run_id = st.session_state.get("run_counter", 0)
    with st.status(f"🔄 正在执行模拟 (第 {run_id} 次)...", expanded=True) as status:
        # 释放可能残留的内存
        import gc
        gc.collect()

        # --- 步骤 1: 加载数据 ---
        st.write("📂 正在加载人物数据...")
        progress_bar = st.progress(5)
        df = load_data_cache()
        st.caption(f"✅ 已加载 {len(df):,} 条人物数据")
        progress_bar.progress(15)

        # --- 步骤 2: 加载模型 ---
        st.write("🧠 正在加载检索模型...")
        embed_model = load_model()
        progress_bar.progress(20)

        # --- 步骤 3: 加载 FAISS 索引 ---
        st.write("📇 正在加载向量索引...")
        index = load_faiss_index()
        if index is None:
            st.error(f"FAISS 索引不存在: {Config.INDEX_PATH}\n请先运行 scripts/02_build_index.py")
            status.update(label="❌ 索引不存在", state="error")
            return None
        st.caption("✅ FAISS 索引加载成功")
        progress_bar.progress(25)

        # --- 步骤 4: 向量检索 ---
        st.write(f"🔍 正在检索 {k:,} 个相似人物...")
        results = search_similar(question, embed_model, index, df, k=k)
        st.caption(f"✅ 找到 {len(results):,} 条相似人物记录")
        progress_bar.progress(35)

        # --- 步骤 4.5: 按需加载人物画像文本 ---
        st.write("📝 正在加载检索结果的人物画像...")
        df_text = _load_persona_text(results["persona_id"].tolist())
        if df_text is not None:
            results = results.merge(
                df_text[["persona_id_str"] + _PERSONA_COLS],
                left_on="persona_id", right_on="persona_id_str",
                how="left", suffixes=("", "_text")
            ).drop(columns=["persona_id_str"])
        progress_bar.progress(40)

        # --- 步骤 5: 构建 Prompt ---
        st.write("✍️ 正在构建 Prompt...")
        prompts = build_batch_prompts(results.to_dict('records'), question)
        st.caption(f"已构建 {len(prompts)} 个提示词")
        progress_bar.progress(50)

        # --- 步骤 6: 调用 LLM ---
        st.write(f"🤖 正在调用 LLM (模型: {active_model_name})...")
        # 使用 st.empty() 占位，进度只更新这一行
        llm_status = st.empty()
        llm_status.caption("准备发送请求...")

        client = LLMClient(
            api_key=api_key,
            model=active_model_name,
            base_url=base_url or Config.LLM_BASE_URL,
            rpm=rpm,
            tpm=tpm,
            temperature=temperature,
        )

        def llm_progress(done, total):
            pct = 50 + int(35 * done / total)
            progress_bar.progress(pct)
            # 检查中断
            if st.session_state.get("interrupt", False):
                raise KeyboardInterrupt("用户中断")
            llm_status.caption(f"已完成 {done:,}/{total:,} 个请求 ({done/total*100:.1f}%)")

        try:
            llm_responses = client.call_batch_sync(prompts, progress_callback=llm_progress)
        except KeyboardInterrupt:
            st.warning("⚠️ 模拟已中断")
            status.update(label="⚠️ 模拟已中断", state="error")
            return None
        finally:
            # 清空占位符，避免 DOM 冲突
            llm_status.empty()
        # 调试：检查 LLM 返回状态
        ok_count = sum(1 for r in llm_responses if r and r.get("success"))
        fail_count = sum(1 for r in llm_responses if not r or not r.get("success"))
        empty_count = sum(1 for r in llm_responses if r and r.get("success") and not r.get("response", "").strip())
        st.caption(f"✅ 成功 {ok_count:,} / 失败 {fail_count:,} / 空回复 {empty_count:,}")
        if fail_count > 0:
            fail_reasons = [r.get("error", "unknown") for r in llm_responses if r and not r.get("success")][:3]
            for fr in fail_reasons:
                st.error(f"LLM 错误: {fr}")
        llm_status.caption(f"✅ 收到 {len(llm_responses)} 条 LLM 回答")
        progress_bar.progress(85)

        # --- 步骤 7: 解析回答 ---
        st.write("📊 正在解析回答并计算支持度...")
        # 只处理成功的请求
        response_dict = {}
        for r in llm_responses:
            if r and r.get("success") and r.get("response"):
                response_dict[str(r['persona_id'])] = r['response']
        # 确保 DataFrame 的 persona_id 也是 str 类型，避免类型不匹配
        results['persona_id_str'] = results['persona_id'].astype(str)
        results['llm_response'] = results['persona_id_str'].map(response_dict)

        # 只解析有成功回复的行，失败请求不参与统计
        mask = results['llm_response'].notna() & (results['llm_response'] != '')
        parsed = results.loc[mask, 'llm_response'].apply(parse_response)
        results["support_score"] = None
        results["stance"] = None
        results.loc[mask, 'support_score'] = parsed.apply(lambda x: x['support_score'])
        results.loc[mask, 'stance'] = parsed.apply(lambda x: x['stance'])
        progress_bar.progress(95)

        # --- 步骤 8: 保存结果 ---
        st.write("💾 正在保存结果到数据库...")
        db = create_database_adapter()
        query_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db.save(query_id, question, results)
        db.close()
        progress_bar.progress(100)

        st.caption("✅ 模拟执行完毕！")
        status.update(label="✅ 模拟执行完毕！", state="complete", expanded=False)

    return results


# ============================================
# 可视化函数
# ============================================

DIMENSION_OPTIONS = {
    "职业": "occupation",
    "省份": "departement",
    "性别": "sex",
    "教育程度": "education_level",
    "婚姻状况": "marital_status",
    "家庭类型": "household_type",
}


def render_charts(df):
    """渲染结果总览图表（仅统计成功请求）"""
    # 过滤掉失败的请求
    df = df[df["stance"].notna()].copy()
    if len(df) == 0:
        st.error("无有效回答，所有请求均失败")
        return
    c = get_theme_colors()

    # 立场分布饼图
    stance_counts = df["stance"].value_counts()
    stance_labels_map = {"oppose": "反对", "neutral": "中立", "support": "支持"}
    labels = [stance_labels_map.get(s, s) for s in stance_counts.index]
    colors = {"反对": "#ef4444", "中立": "#f59e0b", "支持": "#22c55e"}
    pie_colors = [colors.get(l, "#888") for l in labels]

    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=stance_counts.values,
        hole=0.4,
        marker_colors=pie_colors,
        textinfo="label+percent",
        textfont=dict(color=c["text"])
    )])
    fig_pie.update_layout(
        title="立场分布", showlegend=False,
        paper_bgcolor=c["plot_bg"], plot_bgcolor=c["plot_bg"],
        font=dict(color=c["text"])
    )

    # 支持度分布直方图
    fig_hist = px.histogram(
        df, x="support_score", nbins=20,
        title="支持度分布",
        labels={"support_score": "支持度", "count": "人数"},
        color_discrete_sequence=["#636efa"]
    )
    fig_hist.add_vline(x=df["support_score"].mean(), line_dash="dash", line_color="red",
                       annotation_text=f"平均值: {df['support_score'].mean():.2f}")
    fig_hist.update_layout(
        paper_bgcolor=c["plot_bg"], plot_bgcolor=c["plot_bg"],
        font=dict(color=c["text"])
    )

    # 并排显示
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_right:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 关键指标卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="kpi-card blue">
            <div class="kpi-value" style="color:{c['blue']}">{len(df):,}</div>
            <div class="kpi-label">总回答数</div></div>""", unsafe_allow_html=True)
    with col2:
        avg = df["support_score"].mean()
        st.markdown(f"""<div class="kpi-card green">
            <div class="kpi-value" style="color:{c['green']}">{avg:.2f}</div>
            <div class="kpi-label">平均支持度</div></div>""", unsafe_allow_html=True)
    with col3:
        pct_support = (df["stance"] == "support").sum() / len(df) * 100
        st.markdown(f"""<div class="kpi-card orange">
            <div class="kpi-value" style="color:{c['orange']}">{pct_support:.1f}%</div>
            <div class="kpi-label">支持率</div></div>""", unsafe_allow_html=True)
    with col4:
        pct_oppose = (df["stance"] == "oppose").sum() / len(df) * 100
        st.markdown(f"""<div class="kpi-card red">
            <div class="kpi-value" style="color:{c['red']}">{pct_oppose:.1f}%</div>
            <div class="kpi-label">反对率</div></div>""", unsafe_allow_html=True)

    # --- 多维度分析选择器 ---
    st.markdown("---")
    st.markdown("### 📊 多维度支持度分析")
    dim_label = st.selectbox("选择分析维度", list(DIMENSION_OPTIONS.keys()), index=0)
    dim_field = DIMENSION_OPTIONS[dim_label]

    if dim_field in df.columns:
        df_dim = df.copy()
        zh_col_map = {"occupation": "职业", "departement": "省份",
                      "sex": "性别",
                      "education_level": "教育程度", "marital_status": "婚姻状况",
                      "household_type": "家庭类型"}
        col_name = zh_col_map.get(dim_field, dim_field)

        def _zh(v):
            return zh_translate(v, dim_field)

        df_dim["_dim_zh"] = df_dim[dim_field].apply(_zh)

        dim_avg = df_dim.groupby("_dim_zh")["support_score"].agg(["mean", "count"]).reset_index()
        dim_avg.columns = ["维度", "平均支持度", "样本数"]

        # 省份维度不做最小样本过滤，直接显示 Top 15
        if dim_field == "departement":
            dim_avg = dim_avg.sort_values("平均支持度", ascending=False).head(15)
            title_text = f"按{col_name}平均支持度 (Top 15)"
        else:
            min_samples = 10
            dim_avg = dim_avg[dim_avg["样本数"] >= min_samples].sort_values("平均支持度", ascending=False).head(15)
            title_text = f"按{col_name}平均支持度 (Top 15, 最小样本≥{min_samples})"

        if len(dim_avg) > 0:
            fig_dim = px.bar(
                dim_avg, x="平均支持度", y="维度", orientation='h',
                title=title_text,
                labels={"平均支持度": "平均支持度", "维度": col_name},
                color="平均支持度", color_continuous_scale="RdYlGn"
            )
            fig_dim.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                paper_bgcolor=c["plot_bg"], plot_bgcolor=c["plot_bg"],
                font=dict(color=c["text"]),
                height=max(400, len(dim_avg) * 32)
            )
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.plotly_chart(fig_dim, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info(f"{col_name}维度无有效数据（样本数≥10）")
    else:
        st.warning(f"数据中缺少「{dim_field}」字段")

    # 相似度 vs 支持度散点图（采样）
    st.markdown("---")
    st.markdown("### 🔬 相似度与支持度关系")
    sample_size = min(2000, len(df))
    sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    fig_scatter = px.scatter(
        sample, x="similarity", y="support_score",
        title="相似度 vs 支持度",
        labels={"similarity": "相似度", "support_score": "支持度"},
        opacity=0.5, color_discrete_sequence=["#636efa"]
    )
    fig_scatter.update_layout(
        paper_bgcolor=c["plot_bg"], plot_bgcolor=c["plot_bg"],
        font=dict(color=c["text"])
    )
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_individual_responses(df, question):
    """渲染个体回答详情"""
    # 过滤掉失败的请求
    df = df[df["stance"].notna()].copy()
    c = get_theme_colors()
    # 过滤器（横向排列）
    st.markdown("### 🔍 筛选条件")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stance_filter = st.selectbox("立场", ["全部", "支持", "中立", "反对"])
    with col2:
        score_range = st.slider("支持度范围", 0.0, 1.0, (0.0, 1.0), step=0.05)
    with col3:
        search_term = st.text_input("关键词搜索")
    with col4:
        if "occupation" in df.columns:
            occupations = ["全部"] + sorted(df["occupation"].dropna().unique().tolist())
            occ_filter = st.selectbox("职业", occupations)
        else:
            occ_filter = "全部"

    # 应用过滤
    filtered = df.copy()
    if stance_filter != "全部":
        stance_map = {"支持": "support", "中立": "neutral", "反对": "oppose"}
        filtered = filtered[filtered["stance"] == stance_map[stance_filter]]
    filtered = filtered[
        (filtered["support_score"] >= score_range[0]) &
        (filtered["support_score"] <= score_range[1])
    ]
    if search_term:
        mask = filtered["persona_id"].str.contains(search_term, case=False, na=False)
        if "departement" in filtered.columns:
            mask |= filtered["departement"].str.contains(search_term, case=False, na=False)
        if "occupation" in filtered.columns:
            mask |= filtered["occupation"].str.contains(search_term, case=False, na=False)
        filtered = filtered[mask]
    if occ_filter != "全部" and "occupation" in filtered.columns:
        filtered = filtered[filtered["occupation"] == occ_filter]

    st.caption(f"显示 {len(filtered)} / {len(df)} 条记录")

    # 表格
    display_cols = ["persona_id", "age", "occupation", "education_level", "departement",
                    "support_score", "stance", "similarity"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[display_cols].copy()

    # 翻译列中的法文字段
    if "occupation" in display_df.columns:
        display_df["occupation"] = display_df["occupation"].apply(lambda v: zh_translate(v, "occupation"))
    if "education_level" in display_df.columns:
        display_df["education_level"] = display_df["education_level"].apply(lambda v: zh_translate(v, "education_level"))
    if "departement" in display_df.columns:
        display_df["departement"] = display_df["departement"].apply(lambda v: zh_translate(v, "departement"))

    stance_labels = {"oppose": "反对", "neutral": "中立", "support": "支持"}
    if "stance" in display_df.columns:
        display_df["stance"] = display_df["stance"].map(stance_labels).fillna(display_df["stance"])

    st.dataframe(display_df, use_container_width=True, height=400)

    # 展开详情
    st.markdown("---")
    st.markdown("### 查看个体详情")
    selected_id = st.selectbox("选择人物", filtered["persona_id"].tolist() if len(filtered) > 0 else [])
    if selected_id:
        row = filtered[filtered["persona_id"] == selected_id].iloc[0]

        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.markdown("**📋 人物画像**")
            info_cols = ["age", "sex", "marital_status", "household_type", "occupation",
                         "education_level", "departement", "commune"]
            zh_labels = {"age": "年龄", "sex": "性别", "marital_status": "婚姻",
                         "household_type": "家庭", "occupation": "职业",
                         "education_level": "学历", "departement": "省份",
                         "commune": "市镇"}
            for c_col in info_cols:
                val = row.get(c_col, "")
                if pd.notna(val) and val:
                    if c_col in ("occupation", "education_level", "departement",
                                 "marital_status", "household_type"):
                        val = zh_translate(val, c_col)
                    elif c_col == "sex":
                        val = "男" if str(val).strip().upper() in ("M", "MALE", "HOMME", "男性") else "女"
                    label = zh_labels.get(c_col, c_col)
                    st.write(f"- **{label}**: {val}")

        with col_b:
            st.markdown("**📝 发送给 LLM 的 Prompt**")
            persona_dict = row.to_dict()
            prompt = build_prompt(persona_dict, question)
            st.markdown(f'<div class="detail-prompt"><pre style="white-space:pre-wrap;font-size:0.8rem;color:{c["text"]}">{prompt}</pre></div>', unsafe_allow_html=True)

            st.markdown("**💬 LLM 原始回答**")
            response = row.get("llm_response", "")
            if response:
                st.markdown(f'<div class="detail-response"><pre style="white-space:pre-wrap;font-size:0.8rem;color:{c["text"]}">{response}</pre></div>', unsafe_allow_html=True)
            else:
                st.warning("无回答")

            st.markdown("**📊 解析结果**")
            score_col, stance_col = st.columns(2)
            with score_col:
                st.metric("支持度", f"{row.get('support_score', 0):.2f}")
            with stance_col:
                stance_val = row.get("stance", "")
                stance_labels_detail = {"oppose": "反对", "neutral": "中立", "support": "支持"}
                st.metric("立场", stance_labels_detail.get(stance_val, stance_val))

        # 详细画像
        profile_zh = {"persona": "性格特质", "cultural_background": "文化背景",
                      "professional_persona": "职业画像", "sports_persona": "体育偏好",
                      "arts_persona": "艺术偏好", "travel_persona": "旅行偏好",
                      "culinary_persona": "美食偏好", "hobbies_and_interests": "兴趣爱好",
                      "career_goals_and_ambitions": "工作目标", "skills_and_expertise": "技能专长"}
        profile_cols = list(profile_zh.keys())
        has_profile = False
        for c_col in profile_cols:
            val = row.get(c_col, "")
            if pd.notna(val) and str(val).strip():
                if not has_profile:
                    st.markdown("---")
                    st.markdown("**🎨 详细画像**")
                    has_profile = True
                st.write(f"- **{profile_zh.get(c_col, c_col)}**: {val}")


def get_db_size():
    """获取数据库存储占用"""
    try:
        if Config.DB_TYPE == "sqlite":
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), Config.DB_PATH)
            if os.path.exists(db_path):
                size_bytes = os.path.getsize(db_path)
                if size_bytes < 1024:
                    return f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    return f"{size_bytes / 1024:.1f} KB"
                else:
                    return f"{size_bytes / (1024*1024):.1f} MB"
        elif Config.DB_TYPE == "sqlserver":
            import pyodbc
            conn_str = f"DRIVER={{{Config.SQL_DRIVER}}};SERVER={Config.SQL_SERVER};DATABASE={Config.SQL_DATABASE};UID={Config.SQL_USER};PWD={Config.SQL_PASSWORD}"
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT SUM(reserved_page_count) * 8.0 / 1024 AS size_mb
                FROM sys.dm_db_partition_stats
            """)
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                return f"{row[0]:.1f} MB"
    except Exception:
        pass
    return "未知"


def render_history():
    """渲染历史查询"""
    c = get_theme_colors()
    db = create_database_adapter()
    queries = db.list_queries()
    db.close()

    if queries.empty:
        st.info("暂无历史查询")
        return

    # 存储占用
    db_size = get_db_size()
    st.markdown(f"### 📜 历史查询 ({len(queries)} 条)  |  💾 数据库: {db_size}")

    # 每条历史查询用卡片展示
    for _, row in queries.iterrows():
        col_a, col_b, col_c, col_d = st.columns([4, 1, 1, 2])
        with col_a:
            st.markdown(f"**{row['question']}**")
        with col_b:
            st.caption(str(row.get('timestamp', ''))[:19])
        with col_c:
            st.caption(f"{row.get('total_responses', 0):,} 条")
        with col_d:
            if st.button("📂 加载", key=f"load_{row['query_id']}"):
                db2 = create_database_adapter()
                df = db2.load(row['query_id'])
                question = db2.get_question(row['query_id'])
                db2.close()
                if df is not None and not df.empty:
                    st.session_state["history_df"] = df
                    st.session_state["history_question"] = question
                    st.rerun()
                else:
                    st.error("查询结果为空")
            if st.button("🗑️ 删除", key=f"del_{row['query_id']}", type="secondary"):
                db3 = create_database_adapter()
                db3.delete_query(row['query_id'])
                db3.close()
                st.success(f"已删除查询")
                st.rerun()
        st.markdown("---")


# ============================================
# 主界面
# ============================================

st.markdown('<div class="main-header">🇫🇷 法国政策模拟器</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">基于 600 万法国人物画像的 AI 政策态度分析平台</div>', unsafe_allow_html=True)
st.divider()

# running 状态管理
if "running" not in st.session_state:
    st.session_state["running"] = False
if "interrupt" not in st.session_state:
    st.session_state["interrupt"] = False
if "run_counter" not in st.session_state:
    st.session_state["run_counter"] = 0

tab1, tab2, tab3 = st.tabs(["📊 查询与总览", "👤 个体回答详情", "📜 历史查询"])

# 运行中横幅（所有Tab可见）
if st.session_state.get("running", False):
    st.info("⏳ 模拟正在运行中...请切换到「查询与总览」Tab查看进度")

with tab1:
    with st.sidebar:
        st.markdown("### ⚙️ 查询配置")
        st.markdown("---")
        question = st.text_area("政策问题", height=80,
                                 placeholder="例如: 退休年龄推迟到64岁，你怎么看？",
                                 disabled=st.session_state["running"])
        k = st.number_input("检索数量", value=10000, min_value=1, max_value=100000, step=100,
                           disabled=st.session_state["running"])

        st.markdown("---")
        with st.expander("🔑 LLM 配置", expanded=False):
            api_key = st.text_input("API Key", value=Config.LLM_API_KEY, type="password",
                                   disabled=st.session_state["running"])
            base_url = st.text_input("API Base URL", value=Config.LLM_BASE_URL,
                                    disabled=st.session_state["running"])
            model_name = st.text_input("模型", value=Config.LLM_MODEL,
                                      disabled=st.session_state["running"])
            rpm = st.number_input("RPM (每分钟请求数)", value=3000, min_value=1,
                                 disabled=st.session_state["running"])
            tpm = st.number_input("TPM (每分钟 Token 数)", value=500000, min_value=1,
                                 disabled=st.session_state["running"])
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1,
                                   disabled=st.session_state["running"])

        st.markdown("---")
        st.markdown("### 📊 系统状态")
        if st.session_state["running"]:
            st.warning("⏳ 正在运行模拟，请稍候...")
            if st.button("⏹️ 中断模拟", type="secondary", use_container_width=True):
                st.session_state["interrupt"] = True
                st.rerun()
        st.info(f"**数据库**: {Config.DB_TYPE}")

        run_disabled = st.session_state["running"] or not question
        if st.button("🚀 运行模拟", type="primary",
                     use_container_width=True,
                     disabled=run_disabled):
            st.session_state["running"] = True
            st.session_state["interrupt"] = False
            st.session_state["results_df"] = None
            st.session_state["run_counter"] += 1
            st.rerun()

    # 执行模拟（在 sidebar 外，由 running 状态触发）
    if st.session_state["running"]:
        run_id = st.session_state.get("run_counter", 0)
        with st.container(key=f"run_container_{run_id}"):
            results = run_pipeline(question, k, api_key, base_url, model_name, rpm, tpm, temperature)
        if results is not None:
            st.session_state["results_df"] = results
            st.session_state["question"] = question
        st.session_state["running"] = False
        st.session_state["interrupt"] = False
        st.rerun()

    # 空状态
    has_results = "results_df" in st.session_state and st.session_state["results_df"] is not None
    has_history = "history_df" in st.session_state and st.session_state["history_df"] is not None

    if not has_results and not has_history:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <p style="font-size: 4rem;">🇫🇷</p>
            <h2 style="color: #e0e0e0;">欢迎使用法国政策模拟器</h2>
            <p style="color: #888;">在左侧输入政策问题，点击「运行模拟」开始分析</p>
        </div>
        """, unsafe_allow_html=True)
    elif has_results:
        df = st.session_state["results_df"]
        q = st.session_state.get("question", "未知问题")
        st.markdown(f"""
        <div class="question-card">
            <strong>📋 查询问题</strong><br>
            <pre>{q}</pre>
        </div>
        """, unsafe_allow_html=True)
        render_charts(df)
    elif has_history:
        df = st.session_state["history_df"]
        q = st.session_state.get("history_question", "未知问题")
        st.markdown(f"""
        <div class="question-card">
            <strong>📋 历史查询</strong><br>
            <pre>{q}</pre>
        </div>
        """, unsafe_allow_html=True)
        render_charts(df)

with tab2:
    has_results = "results_df" in st.session_state and st.session_state["results_df"] is not None
    has_history = "history_df" in st.session_state and st.session_state["history_df"] is not None

    if has_results:
        df = st.session_state["results_df"]
        q = st.session_state.get("question", "")
        render_individual_responses(df, q)
    elif has_history:
        df = st.session_state["history_df"]
        q = st.session_state.get("history_question", "")
        render_individual_responses(df, q)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <p style="font-size: 3rem;">👤</p>
            <h3 style="color: #e0e0e0;">暂无查询结果</h3>
            <p style="color: #888;">请先在「查询与总览」Tab 中运行一次查询</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    render_history()
