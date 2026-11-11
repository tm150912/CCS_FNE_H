# 源点聚类与捕集单元表

入口为本目录的 `cluster.py`。以下命令从项目根目录执行；脚本中的相对路径以脚本所在目录为基准。

## 输入与环境

`source_unit_location_cost.csv` 是已提供的单元级输入，包含 3,069 个单元和 1,115 个不同源点。上游排放数据采集、捕集容量估计及成本估计不由本脚本完成。

| 字段 | 含义 |
| --- | --- |
| `source`, `unit` | 源点和单元标识；`unit` 必须唯一 |
| `longitude`, `latitude` | WGS84 经纬度；同一源点须使用相同坐标 |
| `cap` | 单元年捕集能力，t/year |
| `Total_Investment_cost($)` | 单元总投资，保持输入值 |
| `Operation_cost($/tCO2)` | 单位捕集运行费，保持输入值 |

聚类需要 NumPy、pandas 和 scikit-learn。默认 `OUTPUT_PLOTS=True`，绘图还需要 Matplotlib。只生成 CSV 时，将该开关设为 `False`。`China_boundary/中国轮廓线.shp` 仅用于绘图；缺少该底图不会改变聚类或 hub 坐标。

## 参数与运行

`CLUSTER_RADIUS_KM=20`，地球平均半径为 `6371.0088` km。脚本先取每个源点的一个坐标，以 Haversine 距离运行 `DBSCAN(min_samples=1)`。20 km 是相邻源点的连接阈值，不是簇的最大直径；通过相邻点形成的链可以连接更远的源点。

```powershell
python -B source_cluster/cluster.py
```

如需更改输入、输出或绘图位置，编辑脚本顶部的 `INPUT_FILE`、`OUTPUT_FILE`、`PLOT_FILE`、`HUB_PLOT_FILE`。本脚本不提供命令行参数。

源点归入 hub 后，hub 经纬度取成员 **单元行** 经纬度的算术平均，因此同一源点的多个单元分别参与均值。脚本不将均值坐标向大陆内移动，不合并单元容量或成本。

## 输出与后续步骤

| 默认输出 | 内容 |
| --- | --- |
| `source_cluster/hub_unit_location_cost.csv` | 3,069 条 hub–unit 记录，当前输入对应 580 个 hub |
| `source_cluster/output/source_hubs.png` | 源点、hub 及成员关系图 |
| `source_cluster/output/hub_locations.png` | hub 位置图 |

检查终端输出的源点、hub 和单元数量。CSV 应保留所有原始 `unit`，以及各单元的 `cap`、投资和运行成本。更改输入或聚类半径后，hub 数量和编号可能变化，需重新生成候选网络及管道成本。

生成结果不会自动覆盖项目根目录的模型输入。确认结果后，从项目根目录同步：

```powershell
Copy-Item -LiteralPath 'source_cluster/hub_unit_location_cost.csv' -Destination 'hub_unit_location_cost.csv'
```

网络重建见 `../triangle network/network_generate.md`。全部相关表更新并同步后，可运行 `python -B -c "from experiment_setup import load_network_data; load_network_data()"` 检查六张模型输入表的一致性。
