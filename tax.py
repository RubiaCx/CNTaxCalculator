from typing import List, Dict
import unicodedata

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

# ================== 个税基本参数 ==================
STARTING_POINT_PER_MONTH = 5000  # 起征点（每月 5000）
# 其他专项附加（子女教育、赡养父母等）这里默认 0，你如果有可以在税式里再加一项


# ================== Offer 配置（含股票预估 & 实习） ==================
# stock_annual: 假设“当年归属”的股票票面价值（税前，元）
# stock_flat_rate: 某些公司 粗暴按 20% 直接扣税；其余 None 走累进
# intern_percent: 实习月薪 = base * intern_percent + allowance；或者 intern_monthly 直接写实习月薪
COMPANIES: List[Dict] = [
    {
        "name": "test",
        "base": 20000,
        "months": 16,
        "allowance": 0,
        "sign_on": 100000,
        "city": "Beijing",
        "stock_annual": 0,
        "stock_flat_rate": None,
        "intern_percent": 0.7,
    },
]


# ================== 税率表 ==================
def get_tax_rate(amount: float):
    """年度综合所得税率表（工资全年/股票按年算用这个）"""
    brackets = [
        (0, 36000, 0.03, 0),
        (36000, 144000, 0.10, 2520),
        (144000, 300000, 0.20, 16920),
        (300000, 420000, 0.25, 31920),
        (420000, 660000, 0.30, 52920),
        (660000, 960000, 0.35, 85920),
        (960000, float('inf'), 0.45, 181920),
    ]
    for lower, upper, rate, qd in brackets:
        if amount <= upper:
            return rate, qd
    return 0.45, 181920


def get_monthly_tax_rate(amount: float):
    """月度税率表（年终奖换算 & 实习按月简单算税用这个）"""
    brackets = [
        (0, 3000, 0.03, 0),
        (3000, 12000, 0.10, 210),
        (12000, 25000, 0.20, 1410),
        (25000, 35000, 0.25, 2660),
        (35000, 55000, 0.30, 4410),
        (55000, 80000, 0.35, 7160),
        (80000, float('inf'), 0.45, 15160),
    ]
    for lower, upper, rate, qd in brackets:
        if lower < amount <= upper:
            return rate, qd
    return 0.45, 15160


def get_bonus_tax_rate(monthly_amount: float):
    """年终奖换算后的月度金额用月度税率表"""
    return get_monthly_tax_rate(monthly_amount)


# ================== 五险一金 ==================
def calc_insurance(income: float, city: str) -> float:
    """计算每月个人五险一金缴纳额（按 base+补贴 做基数，不含签字费）"""
    cfg = CITY_CONFIG.get(city, CITY_CONFIG['Beijing'])
    base_sb = max(min(income, cfg['shebao_cap']), cfg['shebao_min'])
    base_gjj = max(min(income, cfg['gongjijin_cap']), cfg['shebao_min'])
    deduction = (
        base_sb * (cfg['rate_pension'] + cfg['rate_unemploy'] + cfg['rate_medical'])
        + cfg['medical_fixed']
        + base_gjj * cfg['rate_housing']
    )
    return deduction


# ================== 打印对齐工具 ==================
def display_width(text: str) -> int:
    """计算包含中英文的显示宽度（全角按 2）"""
    width = 0
    for ch in str(text):
        width += 2 if unicodedata.east_asian_width(ch) in ('F', 'W') else 1
    return width


def pad(text: str, width: int, align: str = 'left') -> str:
    """按显示宽度补空格对齐"""
    cur = display_width(text)
    if cur >= width:
        return text
    spaces = ' ' * (width - cur)
    if align == 'right':
        return spaces + text
    return text + spaces


# ================== 全职 Offer 计算 ==================
def run_calculation(company_data: Dict) -> Dict:
    base = company_data['base']
    months = company_data['months']
    allowance = company_data['allowance']
    sign_on = company_data['sign_on']
    city = company_data['city']
    stock_annual = company_data.get('stock_annual', 0)
    stock_flat_rate = company_data.get('stock_flat_rate')

    monthly_fixed = base + allowance

    # base 月数 > 12 的部分当作年终奖
    bonus_months = max(0, months - 12)
    year_end_bonus = base * bonus_months

    # ---- 股票 / 期权（首年归属部分，统一按 20% 计税） ----
    stock_tax = 0.0
    stock_net = 0.0
    if stock_annual > 0:
        # 期权统一按 20% 税率粗略估算
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
        insurance = calc_insurance(monthly_fixed, city)

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

    # 优先使用 intern_monthly；否则按 intern_percent * base + allowance
    intern_monthly = company_data.get('intern_monthly')
    if intern_monthly is None:
        intern_percent = company_data.get('intern_percent')
        if intern_percent is None:
            # 没写实习信息，就认为没这个实习 offer
            return None
        intern_monthly = company_data['base'] * intern_percent + company_data.get('allowance', 0)

    rent_deduction = CITY_CONFIG.get(city, CITY_CONFIG['Beijing']).get('rent_deduction', 0)

    # 实习不缴社保，只有起征点 + 房租专项
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


# ================== 表头定义 ==================
FULLTIME_COLUMNS = [
    {"key": "name",              "title": "公司",        "width": 16, "align": "left"},
    {"key": "total_gross_w",     "title": "税前总包",    "width": 10, "align": "right"},
    {"key": "stock_gross_w",     "title": "税前股票",    "width": 10, "align": "right"},
    {"key": "salary_net_w",      "title": "工资到手",    "width": 10, "align": "right"},
    {"key": "first_month_w",     "title": "首月到手",    "width": 12, "align": "right"},
    {"key": "monthly_min_w",     "title": "月到手最小",   "width": 12, "align": "right"},
    {"key": "monthly_max_w",     "title": "月到手最大",   "width": 12, "align": "right"},
    {"key": "stock_net_w",       "title": "股票/期权到手",    "width": 12, "align": "right"},
    # {"key": "intern_net_month_w","title": "实习月均到手", "width": 12, "align": "right"},
    {"key": "total_net_w",       "title": "总到手",      "width": 10, "align": "right"},
]

INTERNSHIP_COLUMNS = [
    {"key": "name",                 "title": "公司",       "width": 16, "align": "left"},
    {"key": "intern_city",          "title": "城市",       "width": 8,  "align": "left"},
    {"key": "intern_gross_month_w", "title": "实习月薪",   "width": 12, "align": "right"},
    {"key": "intern_net_month_w",   "title": "实习月到手", "width": 12, "align": "right"},
    {"key": "intern_months",        "title": "实习月数",   "width": 8,  "align": "right"},
    {"key": "intern_gross_total_w", "title": "实习总税前", "width": 12, "align": "right"},
    {"key": "intern_net_total_w",   "title": "实习总到手", "width": 12, "align": "right"},
]


# ================== 主程序 ==================
def main():
    # -------- 全职 Offer 表 --------
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

    # 找首月到手最大值，做 1st 标记
    global_first_month_max = max(r["first_month_w"] for r in results) if results else 0.0

    for res in results:
        row_items = []
        for col in FULLTIME_COLUMNS:
            key = col["key"]
            if key == "name":
                value = res[key]
            else:
                if key == "first_month_w" and abs(res[key] - global_first_month_max) < 1e-6:
                    value = f"{res[key]:.1f} 1st"
                else:
                    value = f"{res[key]:.1f}"
            row_items.append(pad(value, col["width"], col["align"]))
        print(" | ".join(row_items))

    print("-" * separator_len)
    best = max(results, key=lambda x: x['total_net_w'])


if __name__ == "__main__":
    main()
