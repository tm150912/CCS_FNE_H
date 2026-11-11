# 封存盆地子场站划分

入口为本目录的 `sink_subsite.py`。本阶段生成空间分区、代表点和资源配额；井数上限与封存成本由下一阶段计算。使用已提供的处理后模型 CSV 求解时，无需运行空间划分。

## 输入与环境

| 输入（相对项目根目录） | 用途 |
| --- | --- |
| `storage_cost/sink.xlsx`，工作表 `sink` | 盆地资源、年注入配额及地质参数 |
| `storage_cost/sink_subsite/China_boundary/China_boundary.shp` | 从轮廓线构建中国大陆面 |
| `storage_cost/sink_subsite/sinks_boundary/sinks_boundary.shp` | 盆地边界，按 `USGS_BASIN` 显式映射至 `Sink_ID` |

SHP 的 `.shx`、`.dbf`、`.prj` 等配套文件应与主文件放在一起。当前 16 个盆地中有 14 个具有匹配边界；缺少边界的盆地不会凭空生成子场站。

脚本需要 **ArcGIS Pro 的 Python 3 环境、ArcPy、GDAL/OGR (`osgeo.ogr`)、NumPy、openpyxl 和 Matplotlib**。它同时使用 ArcPy 和 OGR，只有其中一个不够。默认绘图还直接读取 Windows 的 `%WINDIR%/Fonts/msyh.ttc` 字体。普通优化用 Python 环境不一定具备这些依赖。

先在配置好上述依赖的解释器中检查：

```powershell
python -B -c "import arcpy; from osgeo import ogr; import numpy, openpyxl, matplotlib"
```

## 关键配置与运行

路径和参数均在脚本顶部配置，不依赖当前工作目录，也不提供命令行选项。

| 参数 | 默认值与作用 |
| --- | --- |
| `MAX_AREA_KM2` | 8000，逐块测地面积硬上限 |
| `CONSTRUCTION_MARGIN_KM2` | 2，划分时向下留数值余量，不放宽硬上限 |
| `BASIN_AREA_RATIO_LIMIT` | 1.03，同盆地最大/最小子场站面积比上限 |
| `PARTITION_RESTARTS`, `PARTITION_SEED` | 3、47，确定性多起点 |
| `LLOYD_MAX_ITERATIONS` | 30，生成点调整次数上限 |
| `SHAPE_POLISH_ROUNDS`, `MEAN_POLISH_ROUNDS` | 8、4，先改善最差紧凑度，再改善平均紧凑度 |
| `BASIN_MIN_SUBSITE_COUNTS` | `{"P16": 10}`，指定盆地的最少分区数 |
| `COVERAGE_TOL_KM2` | 0.02，覆盖和重叠的数值验收阈值 |

从项目根目录，在 ArcGIS Pro 对应的解释器中运行：

```powershell
python -B storage_cost/sink_subsite/sink_subsite.py
```

流程为大陆裁剪、连通组件数量协调、局部 Lambert 等积投影下的加权 Voronoi 面积平衡、中心及形状改进、空间验收。仅使用轮廓线转面所得的最大陆地面。代表点为局部投影下的内部标注点，再转换到 WGS84，并检查严格位于分区内部。源点 hub 不参与本脚本的划分。

默认 `RESOURCE_BASIS="uniform_original_area"`：盆地总地质资源先按保留的大陆面积比例折减，再按子场站面积占全部子场站面积之比分配。年配额按“原盆地年配额 × 子场站面积 / 原盆地面积”计算，不把被裁去区域的配额重新分配到保留区域。

## 输出与验收

所有结果写入 `storage_cost/sink_subsite/subsite_result/`，重跑会更新同名文件：

- `sink_subsite.csv`：子场站标识、经纬度、面积、地质参数、资源与年配额。
- `sink_subsite.geojson`：实际分区多边形及对应属性。
- `sink_subsite_audit.json`：面积、覆盖、重叠、连通性、代表点与保存后几何验收结果。
- `sink_subsite_map.png`、`sink_subsite_detail.png`：总图和逐盆地图。

当前发布数据为 14 个盆地、271 个子场站；输入、参数或空间运算环境变化后应以新审计结果为准。检查保存后几何验收通过、OGC 问题数为 0、各子场站面积不超过 8000 km²、同盆地面积极差比不超过 1.03。

下一步按 `../storage_cost.md` 计算封存成本，并按 `../../triangle network/network_generate.md` 更新网络。不要将本阶段的 `sink_subsite.csv` 当作根目录模型所需的 `sink_subsite_cost.csv`。
