# Appendix Code: Corpus Linguistics & Diachronic Analysis Pipeline

This repository contains the supplementary source code for the paper **"[现代汉语外来词音译与意译的动态消长---保真度与统一性的权衡]"**.

The code implements a complete data processing pipeline for analyzing translation strategies and lexical variation across different regions and time periods. The workflow proceeds from raw corpus data to statistical metrics and change-point detection.

## 📂 File Descriptions (Aligned with Paper)

- **`ccl2ann.py`**
  - **Function**: 将非结构化的CCL语料库检索结果转化为包含`时间轴（Period/Year）`、`地域（Region）`、`来源语（Source）`及`译名策略（Strategy）`等维度的结构化数据集，为历时分布研究奠定数据基础（见论文第二节 研究技术路线）。

- **`normalize_alias_plus.py`**
  - **Function**: 解决外来词引入初期“同词异名”的问题（如“德律风”与“德律丰”），通过人工校准表与模糊匹配算法，将不同变体归并至统一概念节点，有效消除频率稀释，确保“统一性（U值）”计算的准确性（见论文第三章第一节）。

- **`unity_meter_plus.py`**
  - **Function**: 本研究的核心量化逻辑实现。该脚本计算“统一性（Uniformity, U）”指标，即主导译名形式在全量形式中的频率占比 (`Dominance Ratio`)，用以表征译名的集中度与规范化收敛速率（见论文第三章第一节）。

- **`timechunker_changepoint.py`**
  - **Function**: 拒绝主观历史分期，引入变点检测（Changepoint Detection）算法，自动识别音译与意译比例发生显著转折的关键年份（如“德律风→电话”的转变节点），为外部社会因素对语言演化的干扰提供客观参照（见论文第三章第三节）。

- **`case_tracker_from_metrics.py`**
  - **Function**: 追踪典型词项（如telephone、coffee）在F-U坐标系中的完整生命轨迹，构建“时段—频率—策略”三位一体的追踪矩阵，支持对“德律风 vs 电话”、“麦克风 vs 扩音器”等个案的深度透视（见论文第四章第二节）。

## ⚙️ Requirements

- Python 3.8+
- Required libraries:
  ```bash
  pip install pandas numpy
## 📚 Corpus Data

The raw corpus data used in this study is available in the [`corpora/`](./corpora) directory.
