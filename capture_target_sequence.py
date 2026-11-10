"""生成与 CCS_FNE 线性基准全周期总量相同的捕集目标序列。

修改文件顶部的四个参数后，直接运行：python capture_target_sequence.py
参数依次为期数、基准线性末期年目标、曲线类别、形状参数。
曲线类别为 linear / exponential / logarithmic；linear 不使用形状参数。

基准 b[h] = round(M * (h / N), 3)，与 experiment_setup 一致。
非线性 q[h] = f[h] * sum(b) / sum(f)，f 使用现有曲线定义。
返回以 1 开始的逐期年捕集目标字典，单位与输入 M 相同（项目中为 kt/年）。
各期等长时，序列和相等即全周期总量相等；例如项目共 30 年时，
全周期总量为 (30 / N) * sum(q)。本工具接受任意正整数期数，
接入现有 30 年求解器时仍须满足其期数整除 30 的要求。

M 是基准的末期值，不是归一化后非线性序列的上限。为保持总量，
非线性结果不逐项取整；浮点计算的总量误差仅在数值精度范围内。
本文件仅使用标准库，不读取网络数据，也不运行优化模型。
"""

from __future__ import annotations

import json
from math import exp, expm1, fsum, isfinite, log1p
from numbers import Integral, Real
from typing import Optional


# ==================== 输入参数（修改后直接运行） ====================
PERIOD_NUM = 30                         # 项目期数
MAXIMUM_CAPTURE_AMOUNT = 2e6             # 基准线性末期年捕集目标，单位 kt/年
CURVE = "exponential"                   # linear / exponential / logarithmic
COEFFICIENT = 2.5                       # 形状参数；linear 时可设为 None


def generate_capture_targets(
        period_num: int,
        maximum_capture_amount: float,
        curve: str = "linear",
        coefficient: Optional[float] = None) -> dict[int, float]:
    """保持线性基准总量，返回 {期号: 年捕集目标}。

    指数参数须为有限非零数；对数参数须有限、>-1 且非零。
    线性曲线不使用形状参数。最大目标须非负且有限。
    """

    if (isinstance(period_num, bool) or not isinstance(period_num, Integral)
            or period_num <= 0):
        raise ValueError("period_num must be a positive integer")
    if (isinstance(maximum_capture_amount, bool)
            or not isinstance(maximum_capture_amount, Real)
            or not isfinite(maximum_capture_amount)
            or maximum_capture_amount < 0):
        raise ValueError("maximum_capture_amount must be finite and non-negative")
    if curve not in ("linear", "exponential", "logarithmic"):
        raise ValueError(f"Unknown capture-target curve: {curve}")
    if coefficient is not None:
        if (isinstance(coefficient, bool) or not isinstance(coefficient, Real)
                or not isfinite(coefficient)):
            raise ValueError("coefficient must be a finite number")
        coefficient = float(coefficient)
    if curve != "linear" and (coefficient is None or coefficient == 0):
        raise ValueError(f"The {curve} curve requires a non-zero coefficient")
    if curve == "logarithmic" and coefficient <= -1:
        raise ValueError("The logarithmic coefficient must be greater than -1")

    periods = range(1, period_num + 1)
    baseline = {h: round((h / period_num) * float(maximum_capture_amount), 3)
                for h in periods}
    if curve == "linear":
        return baseline

    fractions = []
    for h in periods:
        x = h / period_num
        if abs(coefficient) < 1e-12:
            # 极小参数的线性极限，避免次正规浮点数的除法损失。
            fraction = x
        elif curve == "exponential" and coefficient > 0:
            # 与 expm1(k*x)/expm1(k) 等价，避免大正参数导致溢出。
            fraction = (exp(coefficient * (x - 1))
                        * (-expm1(-coefficient * x)) / (-expm1(-coefficient)))
        elif curve == "exponential":
            fraction = expm1(coefficient * x) / expm1(coefficient)
        else:
            fraction = log1p(coefficient * x) / log1p(coefficient)
        fractions.append(fraction)

    scale = fsum(baseline.values()) / fsum(fractions)
    return {h: fraction * scale for h, fraction in zip(periods, fractions)}


def main() -> None:
    target = generate_capture_targets(
        PERIOD_NUM, MAXIMUM_CAPTURE_AMOUNT, CURVE, COEFFICIENT)
    baseline = generate_capture_targets(PERIOD_NUM, MAXIMUM_CAPTURE_AMOUNT)
    print(json.dumps({
        "period_num": PERIOD_NUM,
        "maximum_capture_amount": MAXIMUM_CAPTURE_AMOUNT,
        "curve": CURVE,
        "coefficient": COEFFICIENT,
        "linear_target": baseline,
        "capture_target": target,
        "linear_sum": fsum(baseline.values()),
        "target_sum": fsum(target.values()),
        "terminal_target": target[PERIOD_NUM],
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
