"""CCS 优化模型共用的内存数据容器。

``read.read_data`` 负责一次性填充下列类属性。保留 ``CCS.Qig``、``CCS.ij`` 等原有访问方式，
可以使现有模型脚本无需修改调用接口。
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional


class CCS:
    """由 :func:`read.read_data` 填充的类级数据命名空间，无需创建实例。"""

    # 排放源参数：固定成本、可变成本和最大捕集能力
    Tig: ClassVar[Optional[dict]] = None
    Vig: ClassVar[Optional[dict]] = None
    Qig: ClassVar[Optional[dict]] = None

    # 管道参数：固定成本、可变成本和不同管径的输送能力
    Tijd: ClassVar[Optional[dict]] = None
    Vijd: ClassVar[Optional[dict]] = None
    Qd: ClassVar[Optional[dict]] = None

    # 封存地参数：开启成本、注入井成本、封存可变成本及井数/单井能力上限。
    # 每个键 j 都是直接出现在网络弧上的物理封存地编号。
    Tj: ClassVar[Optional[dict]] = None
    Twj: ClassVar[Optional[dict]] = None
    Vj: ClassVar[Optional[dict]] = None
    Qw: ClassVar[Optional[dict]] = None  # 每个物理子场 j 的单井年能力（kt/年/井）。
    Pj: ClassVar[Optional[dict]] = None  # 每个封存地全规划期累计最大井数。
    # 固定运维以千美元/年计；追加资本和运维为投运后第 1 至 30 年的千美元序列。
    sink_fixed_om: ClassVar[Optional[dict]] = None
    well_fixed_om: ClassVar[Optional[dict]] = None
    sink_extra_capital: ClassVar[Optional[dict]] = None
    sink_extra_opex: ClassVar[Optional[dict]] = None
    well_extra_capital: ClassVar[Optional[dict]] = None
    well_extra_opex: ClassVar[Optional[dict]] = None
    # 开场前费用与共同规划期末关闭/PISC现金流；金额均为千美元。
    storage_lifecycle: ClassVar[Optional[dict]] = None
    storage_cost_schema: ClassVar[Optional[str]] = None

    # 网络节点、前后向邻接表、候选管径以及各类变量索引集合
    I: ClassVar[Optional[dict]] = None
    S: ClassVar[Optional[dict]] = None
    # 直接出现在网络弧上的物理封存地。
    R: ClassVar[Optional[dict]] = None
    # Intersection/transshipment nodes.  ``I`` remains the complete node set;
    # ``X`` is the disjoint subset whose flow balance has zero supply/demand.
    X: ClassVar[Optional[dict]] = None
    Np: ClassVar[Optional[dict]] = None
    Nn: ClassVar[Optional[dict]] = None
    G: ClassVar[Optional[dict]] = None  # 每个源的候选捕集单元；每个单元全期至多建设一次。
    C: ClassVar[Any] = None
    D: ClassVar[Optional[list]] = None
    ij: ClassVar[Optional[list]] = None
    ijd: ClassVar[Optional[list]] = None
    ig: ClassVar[Optional[list]] = None
