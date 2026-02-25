#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# timechunker_changepoint.py — 最小依赖版 时间切片 + 变点检测（含编码回退 & 年份区间排序）
# --------------------------------------------------------------------------------------
# 输入目录 (默认，可覆盖 --indir)：
#   C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\normalize_alias_plus-output
#
# 输出目录 (默认，可覆盖 --outdir)：
#   C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\timechunker_changepoint-out
#
# 功能：
# 1) 批量读取 <indir>/*.normalized.csv（多编码回退）
# 2) 将 year/period 映射到时间桶（默认：1901-1920,...,2021-2025）
# 3) 计算每个 (source, region, period_bucket) 的 strategy 占比
# 4) 对每个 (source, region) 的三条策略曲线进行变点检测（纯 numpy 实现）
# 5) 导出：
#    - strategy_share.csv
#    - timechunker_changepoint.json
#
# 依赖：pandas, numpy

import os, sys, json, argparse, glob, re
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

# ---------------- 默认参数 ----------------
DEFAULT_INDIR  = r"C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\normalize_alias_plus-output"
DEFAULT_OUTDIR = r"C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\timechunker_changepoint-out"
DEFAULT_BUCKETS = "1901-1920,1921-1940,1941-1960,1961-1980,1981-2000,2001-2020,2021-2025"

# 策略归一（尽量兼容）
STRATEGY_MAP = {
    "音": "音", "音译": "音", "phonetic": "音",
    "意": "意", "意译": "意", "semantic": "意",
    "并存": "并存", "并行": "并存", "mixed": "并存",
}

# 读取 CSV 的编码回退顺序
READ_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk", "cp936", "big5", "cp950", "latin1"]

# ---------------- 工具函数 ----------------
def ensure_outdir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def parse_buckets(buckets_str: str) -> List[Tuple[int,int,str]]:
    """
    "1901-1920,1921-1940" -> [(1901,1920,'1901–1920'), ...]
    """
    out = []
    for seg in [s.strip() for s in buckets_str.split(",") if s.strip()]:
        if "-" not in seg:
            continue
        a, b = seg.split("-", 1)
        try:
            start, end = int(a), int(b)
            out.append((start, end, f"{start}–{end}"))
        except ValueError:
            pass
    return out

def year_to_bucket(y: int, buckets: List[Tuple[int,int,str]]) -> str:
    for a, b, label in buckets:
        if a <= y <= b:
            return label
    return "Other"

def safe_int(x):
    try:
        return int(x)
    except:
        return None

def coerce_strategy(x) -> str:
    if pd.isna(x):
        return "其他"
    v = str(x).strip()
    return STRATEGY_MAP.get(v, v) if v else "其他"

def infer_year_and_bucket(df: pd.DataFrame, buckets) -> pd.DataFrame:
    # 优先 year，其次从 period 推断起始年
    df = df.copy()
    if "year" in df.columns:
        df["year_int"] = df["year"].apply(safe_int)
    else:
        df["year_int"] = None

    if "period" in df.columns:
        def _from_period(p):
            if pd.isna(p): return None
            s = str(p)
            for sep in ["–", "-"]:
                if sep in s:
                    return safe_int(s.split(sep)[0].strip())
            return safe_int(s)
        df["year_from_period"] = df["period"].apply(_from_period)
        df["year_int"] = df["year_int"].fillna(df["year_from_period"])

    df["period_bucket"] = df["year_int"].apply(lambda y: year_to_bucket(y, buckets) if y is not None else "Other")
    return df

def read_csv_with_fallback(path: str) -> pd.DataFrame:
    last_err = None
    for enc in READ_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read CSV with common encodings: {path}\nLast error: {last_err}")

def load_concat_normalized(indir: str) -> pd.DataFrame:
    files = glob.glob(str(Path(indir) / "*.normalized.csv"))
    if not files:
        raise FileNotFoundError(f"No *.normalized.csv found in: {indir}")
    dfs = []
    for f in files:
        df = read_csv_with_fallback(f)
        df["__srcfile"] = os.path.basename(f)
        dfs.append(df)
    big = pd.concat(dfs, ignore_index=True)
    return big

def compute_strategy_share(df: pd.DataFrame, buckets) -> pd.DataFrame:
    need_cols = ["source", "region", "strategy", "frequency"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan

    df["strategy"] = df["strategy"].apply(coerce_strategy)

    def _freq(v):
        try:
            return int(v)
        except:
            return 1
    df["frequency"] = df["frequency"].apply(_freq)

    df = infer_year_and_bucket(df, buckets)

    # 只保留三大策略，其余并入“其他”
    core = ["音", "意", "并存"]
    df.loc[~df["strategy"].isin(core), "strategy"] = "其他"

    g = (df.groupby(["source","region","period_bucket","strategy"], dropna=False)["frequency"]
            .sum().reset_index(name="count"))
    tot = (g.groupby(["source","region","period_bucket"], dropna=False)["count"]
             .sum().reset_index(name="total"))
    out = g.merge(tot, on=["source","region","period_bucket"], how="left")
    out["share"] = (out["count"] / out["total"].replace({0: np.nan})).fillna(0.0).clip(0.0, 1.0)
    return out

# ---------- 更准确的时期排序 ----------
def period_sort_key(label: str) -> Tuple[int, int]:
    """
    将 '1901–1920' / '1901-1920' / '1901-20' / '未知' / 'Other' 等转为排序键 (start,end)；
    无法解析的标签排到最后。
    """
    s = str(label)
    ys = re.findall(r"(18|19|20)\d{2}", s)
    if len(ys) >= 2:
        return (int(ys[0]), int(ys[1]))
    if len(ys) == 1:
        m2 = re.search(r"[–-](\d{2,4})", s)
        start = int(ys[0])
        if m2:
            r = m2.group(1)
            if len(r) == 2:
                end = (start // 100) * 100 + int(r)
            else:
                end = int(r)
        else:
            end = start
        return (start, end)
    return (10**9, 10**9)

# ---------- 变点检测（纯 numpy 简洁法） ----------
def detect_changepoints_1d(series: np.ndarray, sensitivity: float = 0.25) -> List[int]:
    """
    简洁的 1D 序列变点：基于相邻差分的标准化与累计均值漂移的综合阈值。
    - sensitivity: (0,1]；越大越敏感，默认 0.25
    返回切割点索引（1..n-1）
    """
    n = len(series)
    if n < 4:
        return []
    diffs = np.diff(series)
    if np.allclose(diffs, 0):
        return []

    sd = np.std(diffs)
    if sd < 1e-9:
        return []
    z = np.abs(diffs / sd)

    cum = np.cumsum(series - np.mean(series))
    cum = np.abs(cum[:-1])  # 对齐到边界

    z_norm = z / (np.max(z) + 1e-9)
    c_norm = cum / (np.max(cum) + 1e-9)
    score = 0.6 * z_norm + 0.4 * c_norm

    thr = np.quantile(score, 1 - max(0.05, min(0.5, sensitivity)))
    candidates = np.where(score >= thr)[0] + 1

    cps = []
    for i in sorted(candidates.tolist()):
        if not cps or i - cps[-1] > 1:
            cps.append(i)
    return cps

def build_json_for_panel(shares: pd.DataFrame) -> List[Dict]:
    """
    面板直读结构：
    [
      {
        "source": "...",
        "region": "...",
        "series": [
          {"period": "1901–1920", "strategy_share": {"音":0.62,"意":0.30,"并存":0.08,"其他":0.00}, "total": 123}
        ],
        "changepoints": {
          "音": [{"between": ["1901–1920","1921–1940"], "index": 1}],
          "意": [...],
          "并存": [...]
        }
      },
      ...
    ]
    """
    results = []
    keys = shares.groupby(["source","region"]).size().reset_index()[["source","region"]].values.tolist()

    for source, region in keys:
        sub = shares[(shares["source"]==source) & (shares["region"]==region)]
        periods = sorted(sub["period_bucket"].unique(), key=period_sort_key)

        # 每期策略占比 + total
        series = []
        for p in periods:
            row = {"period": p, "strategy_share": {}, "total": 0}
            t = sub[sub["period_bucket"]==p]["total"].drop_duplicates()
            row["total"] = int(t.iloc[0]) if len(t)>0 and not pd.isna(t.iloc[0]) else 0
            for st in ["音","意","并存","其他"]:
                tmp = sub[(sub["period_bucket"]==p) & (sub["strategy"]==st)]
                row["strategy_share"][st] = float(tmp["share"].iloc[0]) if len(tmp)>0 else 0.0
            series.append(row)

        # 变点（仅对三条主曲线）
        cps_dict = {}
        for st in ["音","意","并存"]:
            vec = np.array([r["strategy_share"][st] for r in series], dtype=float)
            cps = detect_changepoints_1d(vec, sensitivity=0.25)
            labels = []
            for i in cps:
                if 0 < i < len(periods):
                    labels.append({"between": [periods[i-1], periods[i]], "index": i})
            cps_dict[st] = labels

        results.append({
            "source": None if pd.isna(source) else str(source),
            "region": None if pd.isna(region) else str(region),
            "series": series,
            "changepoints": cps_dict
        })
    return results

# === 在文件末尾 main() 之前增加 ===
def export_viz_friendly(shares: pd.DataFrame, outdir: str):
    """
    导出前端友好 CSV：period, region, source, strategy, frequency, total, share
    frequency 取 count；period 用 period_bucket
    """
    from pathlib import Path
    Path(outdir).mkdir(parents=True, exist_ok=True)
    out = shares.rename(columns={
        "period_bucket": "period",
        "count": "frequency"
    })[["period","region","source","strategy","frequency","total","share"]]
    out.to_csv(str(Path(outdir)/"viz_strategy_share.csv"), index=False, encoding="utf-8-sig")

# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="TimeChunker + ChangePoint (minimal deps, robust CSV reading)")
    ap.add_argument("--indir", default=DEFAULT_INDIR, help="输入目录（*.normalized.csv）")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="输出目录")
    ap.add_argument("--buckets", default=DEFAULT_BUCKETS, help="时间桶，例如 \"1901-1920,1921-1940,...\"")
    ap.add_argument("--sensitivity", type=float, default=0.25, help="变点敏感度 (0,1]，越大越敏感；默认 0.25")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    buckets = parse_buckets(args.buckets)
    if not buckets:
        print("[WARN] Invalid --buckets, fallback to default.", file=sys.stderr)
        buckets = parse_buckets(DEFAULT_BUCKETS)

    print(f"[INFO] Loading normalized CSVs from: {args.indir}")
    df = load_concat_normalized(args.indir)

    print(f"[INFO] Computing strategy shares ...")
    shares = compute_strategy_share(df, buckets)

    # 保存长表 CSV（便于检验或在别的工具复用）
    share_csv = str(Path(args.outdir) / "strategy_share.csv")
    shares.to_csv(share_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] strategy_share.csv saved to: {share_csv}")

    # === 在 main() 里 shares 计算完成后追加 ===
    if args is not None:
        export_viz_friendly(shares, args.outdir)
        print(f"[OK] viz_strategy_share.csv saved to: {Path(args.outdir) / 'viz_strategy_share.csv'}")

    print(f"[INFO] Building panel JSON ...")
    # 先用默认敏感度构建
    panel_json = build_json_for_panel(shares)

    # 如果用户调整了灵敏度，则按该值重新检测 changepoints（shares 不变）
    if abs(args.sensitivity - 0.25) > 1e-9:
        results = []
        keys = shares.groupby(["source","region"]).size().reset_index()[["source","region"]].values.tolist()
        for source, region in keys:
            sub = shares[(shares["source"]==source) & (shares["region"]==region)]
            periods = sorted(sub["period_bucket"].unique(), key=period_sort_key)
            series = []
            for p in periods:
                row = {"period": p, "strategy_share": {}, "total": 0}
                t = sub[sub["period_bucket"]==p]["total"].drop_duplicates()
                row["total"] = int(t.iloc[0]) if len(t)>0 and not pd.isna(t.iloc[0]) else 0
                for st in ["音","意","并存","其他"]:
                    tmp = sub[(sub["period_bucket"]==p) & (sub["strategy"]==st)]
                    row["strategy_share"][st] = float(tmp["share"].iloc[0]) if len(tmp)>0 else 0.0
                series.append(row)
            cps_dict = {}
            for st in ["音","意","并存"]:
                vec = np.array([r["strategy_share"][st] for r in series], dtype=float)
                cps = detect_changepoints_1d(vec, sensitivity=args.sensitivity)
                labels = []
                for i in cps:
                    if 0 < i < len(periods):
                        labels.append({"between": [periods[i-1], periods[i]], "index": i})
                cps_dict[st] = labels
            results.append({
                "source": None if pd.isna(source) else str(source),
                "region": None if pd.isna(region) else str(region),
                "series": series,
                "changepoints": cps_dict
            })
        panel_json = results

    json_path = str(Path(args.outdir) / "timechunker_changepoint.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(panel_json, f, ensure_ascii=False, indent=2)
    print(f"[OK] timechunker_changepoint.json saved to: {json_path}")

if __name__ == "__main__":
    main()



# # 用默认目录（即 normalize_alias_plus-output → timechunker_changepoint-out）
# python .\timechunker_changepoint.py

# # 指定更敏感的变点阈值（例如 0.35）
# python .\timechunker_changepoint.py --sensitivity 0.35

# # 自定义时间桶
# python .\timechunker_changepoint.py --buckets "1890-1900,1901-1920,1921-1940,1941-1960,1961-1980,1981-2000,2001-2020,2021-2025"