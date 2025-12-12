from typing import Union, List, Dict
import unicodedata

# 城市每月社保上限（养老、失业、医疗）
CITY_SOCIAL_UPPER_LIMITS = {
    "北京": {"pension": 2864.88, "unemployment": 179.06, "medical": 716.22},
    "杭州": {"pension": 1994.4, "unemployment": 124.65, "medical": 498.6},
    "上海": {"pension": 2984.16, "unemployment": 186.51, "medical": 746.04},
    "深圳": {"pension": 2200.08, "unemployment": 221.325, "medical": 673.32},  # 养老保险上限未更新仍按27501计算; 职工一档医保
}

# 城市公积金基数上限 = 城市社保上限 * 公积金比例上限
CITY_HOUSING_FUND_LIMITS = {
    "北京": 35811,
    "杭州": 40694,
    "上海": 37302,  # 上海2024年社平工资为12434元。故2025年度缴存基数上限据此计算为37302元，上海公积金比例上限7%
    "深圳": 44265,
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

# ================== 城市社保 + 房租专项附加配置 ==================
CITY_CONFIG: Dict[str, Dict] = {
    'Beijing': {
        'shebao_cap': 35283,    # 社保缴费上限
        'shebao_min': 6821,     # 社保缴费下限
        'gongjijin_cap': 35283, # 公积金缴费上限
        'rate_pension': 0.08,   # 养老个人
        'rate_medical': 0.02,   # 医疗个人
        'medical_fixed': 3,     # 医疗 3 元大病统筹
        'rate_unemploy': 0.005, # 失业个人
        'rate_housing': 0.12,   # 公积金个人
        'rent_deduction': 1500  # 房租专项附加，单位：元/月
    },
    'Hangzhou': {
        'shebao_cap': 24930,
        'shebao_min': 4462,
        'gongjijin_cap': 39527,
        'rate_pension': 0.08,
        'rate_medical': 0.02,
        'medical_fixed': 0,
        'rate_unemploy': 0.005,
        'rate_housing': 0.12,
        'rent_deduction': 1500  # 房租专项附加，单位：元/月
    }
}

STARTING_POINT_PER_MONTH = 5000

# ================== Offer ==================
# stock_annual: 假设“当年归属”的股票票面价值（税前，元）
# stock_flat_rate: 某些公司 粗暴按 20% 直接扣税；其余 None 走累进
# intern_percent: 实习月薪 = base * intern_percent + allowance；或者 intern_monthly 直接写实习月薪
COMPANIES: List[Dict] = [
    {
        "name": "AAA",
        "base": 23000,
        "months": 15,
        "allowance": 0,
        "sign_on": 0,
        "city": "Beijing",
        "stock_annual": 0,
        "stock_flat_rate": None,
    },
    {
        "name": "BBB",
        "base": 20000,
        "months": 13,
        "allowance": 0,
        "sign_on": 0,
        "city": "Beijing",
        "stock_annual": 200000,
        "stock_flat_rate": 0.2,
    },
]


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
    """
    计算本年每月详细薪资数据

    返回：
        - monthly: 每个月的当月值明细（表格用）
        - annual: 全年累计值汇总（年度区域用）
    """
    if isinstance(monthly_salaries, (int, float)):
        monthly_salaries = [monthly_salaries] * 12
    elif isinstance(monthly_salaries, list) and len(monthly_salaries) != 12:
        raise ValueError("月薪需为单个数值或12个元素的列表")

    if isinstance(social_security_bases, (int, float)):
        social_security_bases = [social_security_bases] * 12
    elif isinstance(social_security_bases, list) and len(social_security_bases) != 12:
        raise ValueError("社保基数需为单个数值或12个元素的列表")

    cumulative_income = 0.0
    cumulative_social_housing = 0.0
    cumulative_housing_fund = 0.0
    cumulative_tax = 0.0
    monthly_details = []
    annual_summary: Dict[str, float] = {}

    for month in range(1, 13):
        current_salary = monthly_salaries[month - 1]
        current_social_base = social_security_bases[month - 1]

        pension_upper = CITY_SOCIAL_UPPER_LIMITS[city]["pension"]
        pension = min(current_social_base * 0.08, pension_upper)

        medical_upper = CITY_SOCIAL_UPPER_LIMITS[city]["medical"]
        medical = min(current_social_base * 0.02, medical_upper)

        unemployment_upper = CITY_SOCIAL_UPPER_LIMITS[city]["unemployment"]
        unemployment = min(current_social_base * 0.005, unemployment_upper)

        social_total = pension + medical + unemployment

        housing_limit = CITY_HOUSING_FUND_LIMITS.get(city, float('inf'))
        housing_fund = min(current_social_base, housing_limit) * housing_fund_rate

        total_social_housing = social_total + housing_fund

        cumulative_income += current_salary
        cumulative_social_housing += total_social_housing
        cumulative_housing_fund += housing_fund

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
    cfg = CITY_CONFIG.get(city, CITY_CONFIG['Beijing'])
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
    cumulative_social = 0.0
    cumulative_housing = 0.0
    monthly_nets: List[float] = []
    first_month_net = 0.0

    rent_deduction = CITY_CONFIG.get(city, CITY_CONFIG['Beijing']).get('rent_deduction', 0)

    for m in range(1, 13):
        # 每月固定收入
        current_income = monthly_fixed
        # 签字费简化并入首月工资
        if m == 1:
            current_income += sign_on

        # 五险一金按固定月收入算，不把签字费算进基数
        cfg = CITY_CONFIG.get(city, CITY_CONFIG['Beijing'])
        base_sb = max(min(monthly_fixed, cfg['shebao_cap']), cfg['shebao_min'])
        base_gjj = max(min(monthly_fixed, cfg['gongjijin_cap']), cfg['shebao_min'])
        housing_part = base_gjj * cfg['rate_housing']
        social_part = base_sb * (cfg['rate_pension'] + cfg['rate_unemploy'] + cfg['rate_medical']) + cfg['medical_fixed']
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
        cumulative_social += social_part
        cumulative_housing += housing_part
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
        "annual_tax_w": cumulative_tax_paid,
        "annual_social_w": cumulative_social,
        "annual_housing_w": cumulative_housing,
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

    intern_monthly = company_data.get('intern_monthly')
    if intern_monthly is None:
        intern_percent = company_data.get('intern_percent')
        if intern_percent is None:
            return None
        intern_monthly = company_data['base'] * intern_percent + company_data.get('allowance', 0)

    rent_deduction = CITY_CONFIG.get(city, CITY_CONFIG['Beijing']).get('rent_deduction', 0)

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
    {"key": "name",              "title": "公司",       "width": 16, "align": "left"},
    {"key": "total_gross_w",     "title": "税前总包",    "width": 10, "align": "right"},
    {"key": "stock_gross_w",     "title": "税前股票",    "width": 10, "align": "right"},
    {"key": "salary_net_w",      "title": "工资到手",    "width": 10, "align": "right"},
    {"key": "first_month_w",     "title": "首月到手",    "width": 12, "align": "right"},
    {"key": "monthly_min_w",     "title": "月到手最小",   "width": 12, "align": "right"},
    {"key": "monthly_max_w",     "title": "月到手最大",   "width": 12, "align": "right"},
    {"key": "stock_net_w",       "title": "股票/期权到手", "width": 12, "align": "right"},
    {"key": "annual_tax_w",      "title": "年个税",      "width": 10, "align": "right"},
    {"key": "annual_social_w",   "title": "年社保",      "width": 10, "align": "right"},
    {"key": "annual_housing_w",  "title": "年公积金",     "width": 10, "align": "right"},
    # {"key": "intern_net_month_w","title": "实习月均到手", "width": 12, "align": "right"},
    {"key": "total_net_w",       "title": "总到手",      "width": 10, "align": "right"},
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
            elif key in {"annual_tax_w", "annual_social_w", "annual_housing_w"}:
                value = f"{res[key]:.0f}"
            else:
                value = f"{res[key]:.1f}"
            row_items.append(pad(value, col["width"], col["align"]))
        print(" | ".join(row_items))

    print("-" * separator_len)
    best = max(results, key=lambda x: x['total_net_w'])


if __name__ == "__main__":
    main()
