# 生成管道成本表

入口为 `pipe_cost_calculation.py`，需要 Python 3 和 pandas。脚本对每条有向弧计算全部 12 个管型的投资及单位运输运行成本；不调用 Gurobi，也不读取或运行 Excel 宏。以下命令均从项目根目录执行。

## 输入与管型

| 默认输入 | 必要内容 |
| --- | --- |
| `triangle network/output/arcs.csv` | `i,j,distance`，其中距离单位为 km。 |
| `triangle network/output/nodes.csv` | 第一列为唯一节点 ID；前缀 `P` 标识封存子场站。 |

管型顺序是 `p4,p6,p10,p12,p16,p20,p24a,p24b,p30a,p30b,p36,p48`。相应年均参考流量写在 `FLOW_RATES_KG_PER_S` 中，来源对应随附 `CO2_T_COM_ver4_final_20230726.xlsm` 的 `Cases!F56:F67`。这些参考值已含 90% 容量系数，不应再次乘以 0.9。`p24a/p24b`、`p30a/p30b` 是不同容量类别。

根目录 `pipeline_capacity.csv` 直接提供优化模型使用的精确年容量，单位为 t/year；它不是此脚本的输出。调整管型时须同时保持成本表、容量表及实验配置一致。

## 计费口径

设候选弧长度为 `L` km，参考流量为 `q` kg/s。计费长度为 `1.12*L`；仅当弧的终点是 `P` 节点时，另加 `15*1.609344=24.14016` km。因此反向弧的几何长度相同，成本仍可能不同；进入封存节点的过境流也使用这一计费规则。

参考日流量为 `q_day=q*86400/1000` t/day。脚本计算的总投资为：

```text
投资 = 计费长度 × 0.9 × 1.1 × 9970 × q_day^0.35
       × 计费长度^0.13 × (17.7263/14.7966) × (21.06/19.38)
单位运行成本 = 投资 × 0.025 / (365 × q_day)
```

这里的 0.9 是地区成本系数，与参考流量已经包含的容量系数是两项不同参数。投资输出口径为 USD2020，运行成本为 USD2020/tCO2。函数内部先用 8%、30 年的 `CRF` 计算年化投资，输出表前再除以同一个 `CRF`，因此导出的 `Investment_cost($)` 仍是一次性总投资。优化模型的 9% 融资率、7% 折现率由 `cost_amortize.py` 另行处理，不应把此 CSV 再当作年金输入。

## 运行与输出

先按 [网络生成指南](network_generate.md) 生成配套的节点和弧表，然后运行：

```powershell
python -B "triangle network/pipe_cost_calculation.py"
```

如需直接以根目录已发布网络重算，可显式指定输入，仍先输出到预处理目录：

```powershell
python -B "triangle network/pipe_cost_calculation.py" --arcs "arcs.csv" --nodes "nodes.csv" --output "triangle network/pipeline_cost.csv"
```

输出默认覆盖 `triangle network/pipeline_cost.csv`，字段如下：

| 字段 | 含义 |
| --- | --- |
| `node1_ID`, `node2_ID` | 有向弧起点、终点。 |
| `CO2 flow rate(kg/s)` | 历史列名；实际值是 `p4` 等管型标识，不是数值流量。 |
| `Operation_cost($/tCO2)` | 每吨运行成本。 |
| `Investment_cost($)` | 每条该管型管道的总投资。 |

两列成本均四舍五入至 2 位小数。每条弧应有 12 行；当前发布网络的 14372 条弧对应 172464 条成本记录。输入检查包括非空且唯一的节点 ID、有效端点、正距离、无重复有向弧及存在 `P` 节点。

## 同步到优化模型

确认成本表对应根目录将采用的同一份网络后，同步成本；若重建了网络，同时同步节点与弧表，再调用加载器检查六张输入表的一致性：

```powershell
Copy-Item -LiteralPath "triangle network/pipeline_cost.csv" -Destination "pipeline_cost.csv" -Force
python -B -c "from read import read_data; read_data(); print('Input validation passed')"
```

加载时投资除以 1000 转为 kUSD2020；单位运行成本保持数值，配合 kt/year 的模型流量使用。成本生成脚本不修改根目录输入，也不生成求解结果。
