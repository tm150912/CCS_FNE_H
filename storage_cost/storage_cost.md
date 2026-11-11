# 封存容量、成本与生命周期输入

入口为本目录的 `calculate_storage_costs.py`，调用 `storage_capacity.py`、`storage_engineering.py`、`storage_site.py`、`storage_operations.py`、`storage_lifecycle.py` 和 `storage_geography.py`。结果是未经折现或资本年化的原始工程费用，供模型读取及 `cost_amortize.py` 处理。

## 输入与环境

输入路径以 `storage_cost/` 为基准，在主脚本的 `INPUT_FILES` 中配置：

| 文件 | 用途 |
| --- | --- |
| `sink_subsite/subsite_result/sink_subsite.csv` | 子场站几何属性、地质参数、资源与年配额 |
| `sink_subsite/subsite_result/sink_subsite_audit.json` | 子场站数量和空间容差核对 |
| `sink.xlsx` | 原盆地年注入量及地质参数 |
| `SalineStorageCostModel.xlsm` | NETL 工程活动、分类与时间安排 |
| `storage_reference_data.csv` | 省界、汇率、CPI 等已保存参考数据 |
| `electricity_tariffs.csv` | 选定月份的地区电价和需量费 |

费用生成器及上述模块只使用 Python 标准库；脚本使用的 `str.removesuffix` 需要 Python 3.9 或更新版本。工作簿通过 ZIP/XML 读取，不启动 Excel、不执行 VBA，也不需要 ArcPy。重新构建子场站几何所需的 ArcGIS 环境见 `sink_subsite/generate_sink_subsite.md`。

## 运行与口径

从项目根目录运行基准方案：

```powershell
python -B storage_cost/calculate_storage_costs.py --design quota
```

`quota` 是默认值，以各子场站分配到的年注入配额作为费用设计量。`--design netl` 改为每个子场站统一 320 万 t/year；`--pore-fee-usd2008` 可覆盖孔隙使用费。这两项属于情景变更，不应用于重现默认基准结果。

主要计算口径如下：

- 单井年能力为 `365 × min(地层日注入能力, 3660)`，单位 t/year；`Max_Well_Num` 为子场站年配额除以单井年能力后向下取整，不重新分配余数。
- 费用设计采用 30 年注入期、1.25 的峰均比和 10% 备用井系数。设计井数用于设备定价；优化模型的井数上限仍为 `Max_Well_Num`。
- NETL 工程金额从 2008 USD 按 `258.811 / 215.303` 转为 USD2020。电能与需量费按各观测年份的汇率和 CPI 单独换算，不再重复乘该系数。
- 场站和井的初始资本、年度固定费用、四个运行年龄增量数组分别保存。原始数组包含 30 项；年度基线不重复进入增量。
- 需量费按设计峰值泵功率每年计 12 个月，归入场站年度固定费用；按实际注入量支付的费用归入 `V_r_USD2020_per_tCO2`。
- 生命周期字段记录预运行费用、共同规划终点关闭和 50 年 PISC。场站终期资本/费用矩阵均为 30×51，井级终期资本/费用向量均为 51 项。

## 输出、核对与同步

默认输出目录为 `storage_cost/storage_cost_results/`，重跑会更新同名文件：

| 输出 | 内容 |
| --- | --- |
| `sink_subsite_cost.csv` | 模型需要的场站容量、资本、运行费及 JSON 时间安排 |
| `basin_capacity.csv`, `subsite_capacity.csv` | 盆地和子场站的容量分配与井数核对 |
| `cost_components.csv`, `storage_cost_schedule.csv` | 费用组成及逐运行年龄数组 |
| `operating_components.json`, `lifecycle_components.json` | 运行和生命周期活动明细 |
| `calculation_details.json`, `assumptions.json` | 每个子场站的计算细节、来源及假设 |
| `validation.json` | 输入哈希、组件对账、资源/羽流限制及容量检查 |

当前默认数据应覆盖 271 个子场站、14 个有子场站的盆地及 21 个电价地区，井数上限范围为 5–49、合计 7573。首先确认脚本成功结束且 `validation.json` 的 `passed` 为 `true`。这些数值是当前发布输入的核对值，改变输入后不能强行套用。

确认结果后，从项目根目录同步：

```powershell
Copy-Item -LiteralPath 'storage_cost/storage_cost_results/sink_subsite_cost.csv' -Destination 'sink_subsite_cost.csv'
python -B -c "from experiment_setup import load_network_data; load_network_data()"
```

保留原始 t/year、USD2020 和 USD2020/t 数值。`read.py` 负责将年容量和固定金额除以 1000，按吨运行费保持数值；不要在 CSV 中再次换算。若子场站标识或坐标改变，应先同步重新生成的节点、弧和管道成本，再运行全部输入检查。
