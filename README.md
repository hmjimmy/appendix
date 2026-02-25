# Appendix Code: Corpus Linguistics & Diachronic Analysis Pipeline

This repository contains the supplementary source code for the paper **"[现代汉语外来词音译与意译的动态消长
---保真度与统一性的权衡
]"**.

The code implements a complete data processing pipeline for analyzing translation strategies and lexical variation across different regions and time periods. The workflow proceeds from raw corpus data to statistical metrics and change-point detection.

## 📂 File Descriptions

The scripts are designed to be run sequentially. Each script reads from the output of the previous one.

### 1. Data Conversion
- **`ccl2ann.py`**
  - **Function**: Converts raw CCL corpus text files (`.txt`) into structured annotation formats (`.json`, `.csv`).
  - **Input**: Raw text files from the `GOGOGO` directory.
  - **Output**: Annotated CSV/JSON files with metadata (region, year, domain, snippet).
  - **Key Feature**: Automatically infers region (Mainland, Taiwan, HK) and time period from file paths.

### 2. Normalization & Alias Mapping
- **`normalize_alias_plus.py`**
  - **Function**: Cleans text data, normalizes punctuation/encoding, and maps variant forms (aliases) to a canonical form.
  - **Input**: Output CSVs from `ccl2ann.py`.
  - **Output**: 
    - `*.normalized.csv`: Cleaned data with a `canonical` column.
    - `*.alias_stats.csv`: Statistics on variant frequencies.
  - **Key Feature**: Supports an optional `alias_map.csv` for manual mapping; robust encoding detection.

### 3. Temporal Chunking & Change-Point Detection
- **`timechunker_changepoint.py`**
  - **Function**: Aggregates data into time buckets (e.g., 1901-1920) and calculates strategy shares (Phonetic, Semantic, Mixed). It also detects significant change-points in usage trends.
  - **Input**: `*.normalized.csv` files.
  - **Output**: 
    - `strategy_share.csv`: Time-series data of strategy proportions.
    - `timechunker_changepoint.json`: Detailed JSON including detected change-points for visualization.
  - **Dependencies**: `pandas`, `numpy`.

### 4. Unity & Consistency Metrics
- **`unity_meter_plus.py`**
  - **Function**: Calculates cross-regional and cross-temporal consistency metrics (e.g., Jaccard similarity, top-form dominance).
  - **Input**: `*.normalized.csv` files.
  - **Output**: 
    - `unity_metrics.csv`: Dominance ratios and Wilson confidence intervals.
    - `unity_pairs.csv`: Pairwise region similarity (Jaccard index).
    - `unity_discord_top.csv`: Identification of divergent dominant forms across regions.

### 5. Case Tracking & Timeline Generation
- **`case_tracker_from_metrics.py`**
  - **Function**: Generates dominant form timelines and tracks switches/divergences over time based on the metrics.
  - **Input**: `unity_metrics.csv` (from `unity_meter_plus.py`).
  - **Output**: 
    - `case_dominant_timeline.csv`: Timeline of dominant forms per region.
    - `switches_by_region.csv`: Count and details of strategy switches.
    - `divergences_by_period.csv`: Periods with high regional divergence.

## ⚙️ Requirements

- Python 3.8+
- Required libraries:
  ```bash
  pip install pandas numpy
