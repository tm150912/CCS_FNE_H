# 生成候选网络

入口为 `generate_delaunay.py`。常规求解直接使用项目根目录已有的 CSV；只有修改源点、子场站或网络布点参数时才需要重新生成网络。下面的命令均从项目根目录执行。

## 环境与输入

需要 Python 3、NumPy、pandas、Matplotlib 和 Shapely 2.0 及以上版本。脚本自行读取边界 SHP 并实现 Albers 投影，不依赖 ArcPy、GDAL、GeoPandas 或 SciPy。仅在使用 `--reference-nodes`、`--reference-arcs` 检查 Excel 参考表的列结构时需要 openpyxl。

| 默认输入 | 用途 |
| --- | --- |
| `source_cluster/hub_unit_location_cost.csv` | 读取 `hub,longitude,latitude`，每个 hub 保留一个固定源点。 |
| `storage_cost/sink_subsite/subsite_result/sink_subsite.csv` | 读取 `Subsite_ID,Longitude,Latitude`，每个子场站保留一个固定汇点。 |
| `triangle network/China_boundary/中国轮廓线.shp` | 使用随项目提供的轮廓与十段线记录结构；大陆轮廓约束布点和连边，十段线用于绘图。 |

先完成源点聚类及子场站划分，保持同一设施 ID 的坐标一致。默认输入与输出路径相对于脚本所在目录；命令行传入的相对路径按当前工作目录解析。

## 参数和运行

可编辑脚本顶部 `USER CONFIGURATION`，也可使用 `--help` 中的命令行参数覆盖默认值。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `MAX_EMPTY_RADIUS_KM` | 55 km | 初始布点的大陆覆盖半径。 |
| `MAX_INTERSECTION_NODES` | 10000 | 辅助节点数量保护上限。 |
| `INSERTION_BATCH_SIZE` | 256 | 每轮最多插入的候选点数。 |
| `BOUNDARY_SAMPLE_SPACING_KM` | 20 km | 覆盖检查的边界探针间距。 |
| `INTERSECTION_BOUNDARY_SETBACK_KM` | 20 km | 新增辅助点优先保持的内陆距离。 |
| `COASTAL_NODE_TOLERANCE_KM` | 10 km | 接受沿海固定设施的边界外距离。 |
| `ARC_BOUNDARY_TOLERANCE_KM` | 2 km | 候选弧的大陆边界容差。 |
| `MAX_ARC_LENGTH_KM` | 0 | 0 表示不设最大弧长。 |
| `PRUNE_REDUNDANT_INTERSECTIONS` | `True` | 删除覆盖冗余的辅助点。 |
| `REMOVE_UNUSED_INTERSECTION_BRANCHES` | `True` | 最后递归删除无向度数不超过 1 的辅助节点。 |

```powershell
python -B "triangle network/generate_delaunay.py" --help
python -B "triangle network/generate_delaunay.py"
```

生成过程依次进行固定点读取、自适应布点、Delaunay 连边、边界过滤、连通修补和末端辅助分支删除。源点和汇点不随最后的分支删除而移除。初始布点经过 55 km 覆盖检查，但最后删除分支不会重新布点、补边或恢复覆盖；应分别读取终端打印的删除前覆盖结果和最终覆盖诊断。

## 输出、检查与同步

默认写入 `triangle network/output/`，覆盖其中同名文件：

| 文件 | 内容 |
| --- | --- |
| `nodes.csv` | `I,Longitude,Latitude`；WGS84 坐标，保留 8 位小数。 |
| `arcs.csv` | `i,j,distance`；成对有向弧，投影直线距离以 km 计，保留 6 位小数。 |
| `network.png` | 网络图；将 `OUTPUT_NETWORK_PNG=False` 可关闭图片输出。 |

脚本检查节点 ID 唯一、弧无重复及自环、距离为正、反向弧存在和全网连通；统计及覆盖诊断打印到终端，不另写审计 JSON。随项目提供的发布数据包含 580 个源 hub、271 个子场站、1666 个辅助节点，共 2517 个节点和 14372 条有向弧。更换输入、布点参数或数值库版本后，应检查新输出，而不是强制凑到这一规模。

完成网络检查后，按 [管道成本指南](pipe_cost_calculation.md) 为这份新网络重新生成成本。然后把互相对应的网络与成本表一起同步到根目录：

```powershell
Copy-Item -LiteralPath "triangle network/output/nodes.csv" -Destination "nodes.csv" -Force
Copy-Item -LiteralPath "triangle network/output/arcs.csv" -Destination "arcs.csv" -Force
Copy-Item -LiteralPath "triangle network/pipeline_cost.csv" -Destination "pipeline_cost.csv" -Force
python -B -c "from read import read_data; read_data(); print('Input validation passed')"
```

若源点或子场站也有变化，应先按对应指南同步根目录的捕集成本和封存成本表。根目录 `pipeline_capacity.csv` 是直接提供的固定管型容量表，本脚本不生成它。
