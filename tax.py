from typing import Union, List, Dict
import unicodedata

# ================== 城市配置 (社保/公积金/税率) ==================
# 包含：社保公积金基数上限、各项费率、固定附加费、以及部分险种的绝对值上限（如有）
CITY_CONFIG: Dict[str, Dict] = {
    'Beijing': {
        'name_cn': '北京',
        'shebao_cap': 35283,    # 社保基数上限
        'shebao_min': 6821,     # 社保基数下限
        'gongjijin_cap': 35283, # 公积金基数上限
        'rate_pension': 0.08,   # 养老个人 8%
        'rate_medical': 0.02,   # 医疗个人 2%
        'medical_fixed': 0,     # 医疗附加
        'rate_unemploy': 0.002, # 失业个人 0.2%
        'limit_unemploy': 70.57,# 失业保险个人缴纳绝对上限
        'rate_housing': 0.12,   # 公积金个人 12%
        'corp_rate_pension': 0.16,
        'corp_rate_medical': 0.088,
        'corp_rate_unemploy': 0.005,
        'corp_rate_injury': 0.004,
        'corp_rate_maternity': 0.0,
        'corp_rate_housing': 0.12,
        'rent_deduction': 0     # 房租专项附加
    },
    'Shanghai': {
        'name_cn': '上海',
        'shebao_cap': 36549,
        'shebao_min': 7310,
        'gongjijin_cap': 36549, # 2024基数
        'rate_pension': 0.08,
        'rate_medical': 0.02,
        'medical_fixed': 0,
        'rate_unemploy': 0.005,
        'limit_unemploy': 186.51,
        'rate_housing': 0.12,
        'corp_rate_pension': 0.16,
        'corp_rate_medical': 0.085,
        'corp_rate_unemploy': 0.005,
        'corp_rate_injury': 0.0026,
        'corp_rate_maternity': 0.0,
        'corp_rate_housing': 0.12,
        'rent_deduction': 0
    },
    'Hangzhou': {
        'name_cn': '杭州',
        'shebao_cap': 24060,
        'shebao_min': 4462,
        'gongjijin_cap': 38390,
        'rate_pension': 0.08,
        'rate_medical': 0.02,
        'medical_fixed': 0,
        'rate_unemploy': 0.005,
        'limit_unemploy': 120.3,
        'rate_housing': 0.12,
        'corp_rate_pension': 0.14,
        'corp_rate_medical': 0.099,
        'corp_rate_unemploy': 0.005,
        'corp_rate_injury': 0.002,
        'corp_rate_maternity': 0.0,
        'corp_rate_housing': 0.12,
        'rent_deduction': 0
    }
}

# 城市别名映射 -> 统一转为 CITY_CONFIG 的 key
CITY_ALIAS_MAP = {
    "北京": "Beijing",
    "Beijing": "Beijing",
    "BJ": "Beijing",
    "上海": "Shanghai",
    "Shanghai": "Shanghai",
    "SH": "Shanghai",
    "杭州": "Hangzhou",
    "Hangzhou": "Hangzhou",
    "HZ": "Hangzhou",
    "杭 州": "Hangzhou",
    "深圳": "Shenzhen",
    "Shenzhen": "Shenzhen",
    "SZ": "Shenzhen",
}

# 年度综合所得税率表（保持不变，用于工资薪金计税）
TAX_RATE_TABLE = [
    (36000, 0.03, 0),
    (144000, 0.10, 2520),
    (300000, 0.20, 16920),
    (420000, 0.25, 31920),
    (660000, 0.30, 52920),
    (960000, 0.35, 85920),
    (float('inf'), 0.45, 181920),
]

# 月度税率表（用于年终奖单独计税，按月换算后的综合所得税率表）
MONTHLY_TAX_RATE_TABLE = [
    (3000, 0.03, 0),
    (12000, 0.10, 210),
    (25000, 0.20, 1410),
    (35000, 0.25, 2660),
    (55000, 0.30, 4410),
    (80000, 0.35, 7160),
    (float('inf'), 0.45, 15160),
]

STARTING_POINT_PER_MONTH = 5000

# ================== Offer ==================
COMPANIES: List[Dict] = [
    {
        "name": "BJ",
        "base": 40000,
        "months": 13,
        "allowance": 0,
        "sign_on": 0,
        "city": "Beijing",
        "stock_annual": 0,
        "stock_flat_rate": None,
    },
     {
        "name": "SH",
        "base": 40000,
        "months": 13,
        "allowance": 0,
        "sign_on": 0,
        "city": "Shanghai",
        "stock_annual": 0,
        "stock_flat_rate": None,
    },
     {
        "name": "SZ",
        "base": 40000,
        "months": 13,
        "allowance": 0,
        "sign_on": 0,
        "city": "Shenzhen",
        "stock_annual": 0,
        "stock_flat_rate": None,
    },
     {
        "name": "HZ",
        "base": 40000,
        "months": 13,
        "allowance": 0,
        "sign_on": 0,
        "city": "Hangzhou",
        "stock_annual": 0,
        "stock_flat_rate": None,
    },
]


def resolve_city_config(city_name: str) -> Dict:
    """根据城市名/别名获取配置"""
    key = CITY_ALIAS_MAP.get(city_name, "Beijing")
    return CITY_CONFIG.get(key, CITY_CONFIG["Beijing"])


def get_tax_rate(amount: float):
    """年度综合所得税率表（工资全年/股票按年算用这个）"""
    for upper, rate, qd in TAX_RATE_TABLE:
        if amount <= upper:
            return rate, qd
    return TAX_RATE_TABLE[-1][1], TAX_RATE_TABLE[-1][2]


def get_monthly_tax_rate(amount: float):
    """月度税率表（年终奖换算 & 实习按月简单算税用这个）"""
    for upper, rate, qd in MONTHLY_TAX_RATE_TABLE:
        if amount <= upper:
            return rate, qd
    return MONTHLY_TAX_RATE_TABLE[-1][1], MONTHLY_TAX_RATE_TABLE[-1][2]


def get_bonus_tax_rate(monthly_amount: float):
    return get_monthly_tax_rate(monthly_amount)


def calculate_monthly_details(
    monthly_salaries: Union[float, List[float]],
    social_security_bases: Union[float, List[float]],
    city: str = "北京",
    five_insurance_rate: float = 0.105,
    housing_fund_rate: float = 0.12,
) -> Dict[str, List[Union[float, Dict]]]:
    cfg = resolve_city_config(city)

    if isinstance(monthly_salaries, (int, float)):
        monthly_salaries = [monthly_salaries] * 12
    elif isinstance(monthly_salaries, list) and len(monthly_salaries) != 12:
        raise ValueError("月薪需为单个数值或12个元素的列表")

    if isinstance(social_security_bases, (int, float)):
        social_security_bases = [social_security_bases] * 12
    elif isinstance(social_security_bases, list) and len(social_security_bases) != 12:
        raise ValueError("社保基数需为单个数值或12个元素的列表")

    # 基数按城市上下限收敛
    social_security_bases = [
        max(min(b, cfg["shebao_cap"]), cfg["shebao_min"]) for b in social_security_bases
    ]

    cumulative_income = 0.0
    cumulative_social_housing = 0.0
    cumulative_housing_fund = 0.0
    cumulative_tax = 0.0
    cumulative_personal_social = 0.0
    cumulative_company_social = 0.0
    cumulative_personal_housing = 0.0
    cumulative_company_housing = 0.0
    cumulative_pension = 0.0
    cumulative_medical = 0.0
    cumulative_unemployment = 0.0
    monthly_details = []
    annual_summary: Dict[str, float] = {}

    for month in range(1, 13):
        current_salary = monthly_salaries[month - 1]
        current_social_base = social_security_bases[month - 1]

        pension = current_social_base * cfg["rate_pension"]
        corp_pension = current_social_base * cfg["corp_rate_pension"]

        medical = current_social_base * cfg["rate_medical"] + cfg["medical_fixed"]
        corp_medical = current_social_base * cfg["corp_rate_medical"]

        # 失业保险可能存在单独的金额上限
        unemployment_limit = cfg.get("limit_unemploy", float("inf"))
        unemployment = min(current_social_base * cfg["rate_unemploy"], unemployment_limit)
        corp_unemployment = min(current_social_base * cfg["corp_rate_unemploy"], unemployment_limit)
        
        corp_injury = current_social_base * cfg.get("corp_rate_injury", 0.0)
        corp_maternity = current_social_base * cfg.get("corp_rate_maternity", 0.0)

        social_total = pension + medical + unemployment
        corp_social_total = corp_pension + corp_medical + corp_unemployment + corp_injury + corp_maternity

        # 公积金基数上限可能独立
        housing_limit = cfg.get("gongjijin_cap", cfg["shebao_cap"])
        housing_base = min(current_social_base, housing_limit)
        
        housing_fund = housing_base * housing_fund_rate
        corp_housing_fund = housing_base * cfg.get("corp_rate_housing", housing_fund_rate)

        total_social_housing = social_total + housing_fund

        cumulative_income += current_salary
        cumulative_social_housing += total_social_housing
        cumulative_housing_fund += housing_fund
        cumulative_personal_social += social_total
        cumulative_company_social += corp_social_total
        cumulative_personal_housing += housing_fund
        cumulative_company_housing += corp_housing_fund
        cumulative_pension += (pension + corp_pension)
        cumulative_medical += (medical + corp_medical)
        cumulative_unemployment += (unemployment + corp_unemployment)

        cumulative_taxable_income = cumulative_income - 5000 * month - cumulative_social_housing
        cumulative_monthly_tax = 0.0
        for limit, rate, deduction in TAX_RATE_TABLE:
            if cumulative_taxable_income <= limit:
                cumulative_monthly_tax = cumulative_taxable_income * rate - deduction
                break
        current_month_tax = cumulative_monthly_tax - cumulative_tax
        current_month_tax = max(current_month_tax, 0.0)
        cumulative_tax = cumulative_monthly_tax

        takehome = current_salary - social_total - housing_fund - current_month_tax
        takehome = max(takehome, 0.0)

        monthly_details.append({
            "month": month,
            "pre_tax_income": round(current_salary, 2),
            "pension": round(pension, 2),
            "medical": round(medical, 2),
            "unemployment": round(unemployment, 2),
            "housing_fund": round(housing_fund, 2),
            "taxable_income": round(cumulative_taxable_income, 2),
            "current_tax": round(current_month_tax, 2),
            "takehome": round(takehome, 2),
        })

    total_pre_tax = round(cumulative_income, 2)
    total_housing_fund = round(cumulative_housing_fund, 2)
    total_tax = round(cumulative_tax, 2)
    total_takehome = round(cumulative_income - cumulative_social_housing - cumulative_tax, 2)
    total_takehome_with_housing = total_takehome + total_housing_fund * 2

    annual_summary = {
        "total_pre_tax": total_pre_tax,
        "total_housing_fund": total_housing_fund,
        "total_tax": total_tax,
        "total_takehome": total_takehome,
        "total_takehome_with_housing": total_takehome_with_housing,
    }

    return {"monthly": monthly_details, "annual": annual_summary}


def calculate_year_end_bonus(year_end_bonus: float) -> Dict[str, float]:
    """计算年终奖单独计税的个税、税率及税后金额"""
    if year_end_bonus <= 0:
        raise ValueError("年终奖金额必须大于0")

    monthly_income = year_end_bonus / 12
    tax_rate = MONTHLY_TAX_RATE_TABLE[-1][1]
    quick_deduction = MONTHLY_TAX_RATE_TABLE[-1][2]
    for limit, rate, deduction in MONTHLY_TAX_RATE_TABLE:
        if monthly_income <= limit:
            tax_rate = rate
            quick_deduction = deduction
            break

    bonus_tax = year_end_bonus * tax_rate - quick_deduction
    bonus_after_tax = year_end_bonus - bonus_tax

    return {
        "tax": round(bonus_tax, 2),
        "after_tax": round(bonus_after_tax, 2),
        "tax_rate": round(tax_rate * 100, 2),
    }


def calc_insurance(income: float, city: str) -> float:
    cfg = resolve_city_config(city)
    base_sb = max(min(income, cfg['shebao_cap']), cfg['shebao_min'])
    base_gjj = max(min(income, cfg['gongjijin_cap']), cfg['shebao_min'])
    deduction = (
        base_sb * (cfg['rate_pension'] + cfg['rate_unemploy'] + cfg['rate_medical']) + cfg['medical_fixed'] + base_gjj * cfg['rate_housing']
    )
    return deduction


def display_width(text: str) -> int:
    width = 0
    for ch in str(text):
        width += 2 if unicodedata.east_asian_width(ch) in ('F', 'W') else 1
    return width


def pad(text: str, width: int, align: str = 'left') -> str:
    cur = display_width(text)
    if cur >= width:
        return text
    spaces = ' ' * (width - cur)
    if align == 'right':
        return spaces + text
    return text + spaces


def run_calculation(company_data: Dict) -> Dict:
    base = company_data['base']
    months = company_data['months']
    allowance = company_data['allowance']
    sign_on = company_data['sign_on']
    city = company_data['city']
    cfg = resolve_city_config(city)
    stock_annual = company_data.get('stock_annual', 0)
    stock_flat_rate = company_data.get('stock_flat_rate')

    monthly_fixed = base + allowance
    bonus_months = max(0, months - 12)
    year_end_bonus = base * bonus_months

    stock_tax = 0.0
    stock_net = 0.0
    if stock_annual > 0:
        stock_tax = stock_annual * 0.20
        stock_net = stock_annual - stock_tax

    # ---- 年终奖（单独计税）----
    bonus_tax = 0.0
    bonus_net = 0.0
    if year_end_bonus > 0:
        rate, qd = get_bonus_tax_rate(year_end_bonus / 12)
        bonus_tax = year_end_bonus * rate - qd
        bonus_net = year_end_bonus - bonus_tax

    # ---- 工资薪金 + 签字费，按累计预扣法 ----
    cumulative_income_net = 0.0
    cumulative_taxable = 0.0
    cumulative_tax_paid = 0.0
    cumulative_social_personal = 0.0
    cumulative_social_company = 0.0
    cumulative_housing_personal = 0.0
    cumulative_housing_company = 0.0
    cumulative_pension = 0.0
    cumulative_medical = 0.0
    cumulative_unemployment = 0.0
    monthly_nets: List[float] = []
    first_month_net = 0.0

    rent_deduction = cfg.get('rent_deduction', 0)

    for m in range(1, 13):
        # 每月固定收入
        current_income = monthly_fixed
        # 签字费简化并入首月工资
        if m == 1:
            current_income += sign_on

        # 五险一金按固定月收入算，不把签字费算进基数
        base_sb = max(min(monthly_fixed, cfg['shebao_cap']), cfg['shebao_min'])
        
        # 公积金基数上限
        housing_limit = cfg.get("gongjijin_cap", cfg["shebao_cap"])
        base_gjj = max(min(monthly_fixed, housing_limit), cfg['shebao_min'])

        pension = base_sb * cfg['rate_pension']
        corp_pension = base_sb * cfg['corp_rate_pension']

        medical = base_sb * cfg['rate_medical'] + cfg['medical_fixed']
        corp_medical = base_sb * cfg['corp_rate_medical']

        unemployment_limit = cfg.get("limit_unemploy", float("inf"))
        unemployment = min(base_sb * cfg['rate_unemploy'], unemployment_limit)
        corp_unemployment = min(base_sb * cfg['corp_rate_unemploy'], unemployment_limit)
        
        corp_injury = base_sb * cfg.get('corp_rate_injury', 0.0)
        corp_maternity = base_sb * cfg.get('corp_rate_maternity', 0.0)

        housing_part = base_gjj * cfg['rate_housing']
        corp_housing_part = base_gjj * cfg.get('corp_rate_housing', cfg['rate_housing'])

        social_part = pension + medical + unemployment
        corp_social_part = corp_pension + corp_medical + corp_unemployment + corp_injury + corp_maternity
        insurance = social_part + housing_part

        # 专项附加只有房租：起征点 + 房租专项
        taxable = max(0, current_income - STARTING_POINT_PER_MONTH - rent_deduction - insurance)

        cumulative_taxable += taxable
        rate, qd = get_tax_rate(cumulative_taxable)
        cum_tax = cumulative_taxable * rate - qd
        cur_tax = max(0, cum_tax - cumulative_tax_paid)

        net = current_income - cur_tax - insurance

        if m == 1:
            first_month_net = net

        cumulative_tax_paid += cur_tax
        cumulative_income_net += net
        cumulative_social_personal += social_part
        cumulative_social_company += corp_social_part
        cumulative_housing_personal += housing_part
        cumulative_housing_company += corp_housing_part
        cumulative_pension += pension + corp_pension
        cumulative_medical += medical + corp_medical
        cumulative_unemployment += unemployment + corp_unemployment
        monthly_nets.append(net)

    monthly_max = max(monthly_nets) if monthly_nets else 0.0
    monthly_min = min(monthly_nets) if monthly_nets else 0.0
    total_net = cumulative_income_net + bonus_net + stock_net

    return {
        "name": company_data['name'],
        "total_gross_w": (monthly_fixed * 12 + year_end_bonus + sign_on + stock_annual) / 10000,
        "stock_gross_w": stock_annual / 10000,
        "stock_net_w": stock_net / 10000,
        "salary_net_w": (cumulative_income_net + bonus_net) / 10000,
        "first_month_w": first_month_net / 10000,
        "monthly_min_w": monthly_min / 10000,
        "monthly_max_w": monthly_max / 10000,
        "monthly_range_w": f"{monthly_min/10000:.1f}~{monthly_max/10000:.1f}",
        "annual_tax": cumulative_tax_paid + bonus_tax + stock_tax,
        "annual_social": cumulative_social_personal + cumulative_social_company,
        "annual_social_personal": cumulative_social_personal,
        "annual_housing": cumulative_housing_personal + cumulative_housing_company,
        "annual_housing_personal": cumulative_housing_personal,
        "pension_total": cumulative_pension,
        "medical_total": cumulative_medical,
        "unemployment_total": cumulative_unemployment,
        "total_net_w": total_net / 10000,
    }


def run_internship_calculation(company_data: Dict, months: int = 3) -> Dict | None:
    """
    假设三月入职实习，连实习 3 个月：
      - 不缴社保公积金；
      - 有房租专项附加；
      - 税按“工资薪金”用月度税率简单算（不按全年累计）。
    """
    city = company_data['city']
    # 实习逻辑中也需要获取城市配置来确定房租扣除（如有）
    cfg = resolve_city_config(city)

    intern_monthly = company_data.get('intern_monthly')
    if intern_monthly is None:
        intern_percent = company_data.get('intern_percent')
        if intern_percent is None:
            return None
        intern_monthly = company_data['base'] * intern_percent + company_data.get('allowance', 0)

    rent_deduction = cfg.get('rent_deduction', 0)

    taxable_per_month = max(0, intern_monthly - STARTING_POINT_PER_MONTH - rent_deduction)

    monthly_nets: List[float] = []
    cumulative_taxable = 0.0
    cumulative_tax_paid = 0.0

    for _ in range(months):
        cumulative_taxable += taxable_per_month
        if cumulative_taxable <= 0:
            cur_tax = 0.0
        else:
            rate, qd = get_monthly_tax_rate(cumulative_taxable)
            cum_tax = cumulative_taxable * rate - qd
            cur_tax = max(0.0, cum_tax - cumulative_tax_paid)
        cumulative_tax_paid += cur_tax
        net = intern_monthly - cur_tax
        monthly_nets.append(net)

    if monthly_nets:
        net_month_avg = sum(monthly_nets) / len(monthly_nets)
        net_total = sum(monthly_nets)
    else:
        net_month_avg = 0.0
        net_total = 0.0

    return {
        "name": company_data['name'],
        "intern_city": city,
        "intern_months": months,
        "intern_gross_month_w": intern_monthly / 10000,
        "intern_net_month_w": net_month_avg / 10000,
        "intern_gross_total_w": intern_monthly * months / 10000,
        "intern_net_total_w": net_total / 10000,
    }


FULLTIME_COLUMNS = [
    {"key": "name",              "title": "公司",         "width": 12, "align": "left"},
    {"key": "total_gross_w",     "title": "税前总包(万)",  "width": 12, "align": "right"},
    {"key": "stock_gross_w",     "title": "税前股票(万)",  "width": 12, "align": "right"},
    {"key": "salary_net_w",      "title": "工资到手(万)",  "width": 12, "align": "right"},
    {"key": "total_net_w",       "title": "总到手(万)",    "width": 10, "align": "right"},
    {"key": "monthly_range_w",   "title": "月到手范围(万)", "width": 14, "align": "right"},
    {"key": "stock_net_w",       "title": "股票到手(万)",  "width": 12, "align": "right"},
    {"key": "annual_tax",        "title": "年个税",        "width": 10, "align": "right"},
    {"key": "annual_social",     "title": "年社保(双边)",   "width": 12, "align": "right"},
    {"key": "annual_social_personal", "title": "个人社保",   "width": 10, "align": "right"},
    {"key": "annual_housing",    "title": "年公积金(双边)", "width": 14, "align": "right"},
    {"key": "annual_housing_personal", "title": "个人公积金", "width": 12, "align": "right"},
    {"key": "pension_total",     "title": "养老(双边)",     "width": 12, "align": "right"},
    {"key": "medical_total",     "title": "医疗(双边)",     "width": 12, "align": "right"},
    {"key": "unemployment_total","title": "失业(双边)",     "width": 12, "align": "right"},
]

INTERNSHIP_COLUMNS = [
    {"key": "name",                 "title": "公司",      "width": 16, "align": "left"},
    {"key": "intern_city",          "title": "城市",      "width": 8,  "align": "left"},
    {"key": "intern_gross_month_w", "title": "实习月薪",   "width": 12, "align": "right"},
    {"key": "intern_net_month_w",   "title": "实习月到手", "width": 12, "align": "right"},
    {"key": "intern_months",        "title": "实习月数",   "width": 8,  "align": "right"},
    {"key": "intern_gross_total_w", "title": "实习总税前", "width": 12, "align": "right"},
    {"key": "intern_net_total_w",   "title": "实习总到手", "width": 12, "align": "right"},
]


def main():
    header = " | ".join(
        pad(col["title"], col["width"], col["align"])
        for col in FULLTIME_COLUMNS
    )
    separator_len = sum(col["width"] for col in FULLTIME_COLUMNS) + 3 * (len(FULLTIME_COLUMNS) - 1)
    print(header)
    print("-" * separator_len)

    results: List[Dict] = []
    for comp in COMPANIES:
        res = run_calculation(comp)
        # 如果有实习 offer，把实习月均到手也挂到总表里；否则置 0
        intern_res = run_internship_calculation(comp, months=3)
        if intern_res:
            res["intern_net_month_w"] = intern_res["intern_net_month_w"]
        else:
            res["intern_net_month_w"] = 0.0
        results.append(res)

    for res in results:
        row_items = []
        for col in FULLTIME_COLUMNS:
            key = col["key"]
            if key == "name":
                value = res[key]
            elif key == "monthly_range_w":
                value = res[key]
            elif key in {"total_gross_w", "stock_gross_w", "salary_net_w", "total_net_w", "stock_net_w"}:
                value = f"{res[key]:.1f}"
            else:
                value = f"{res[key]:.0f}"
            row_items.append(pad(value, col["width"], col["align"]))
        print(" | ".join(row_items))

    print("-" * separator_len)
    best = max(results, key=lambda x: x['total_net_w'])


if __name__ == "__main__":
    main()
