#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# case_tracker_from_metrics.py — 主形时间线 & 切换/分歧（稳健读取版）
# ================================================================
# 输入（默认）：
#   C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\unity_meter-output
#   - 优先读取 unity_metrics.csv；否则匹配 *_metrics.csv
#
# 输出（默认）：
#   C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\case_tracker_from_metrics-outpt
#   - case_dominant_timeline.csv
#   - switches_by_region.csv
#   - divergences_by_period.csv
#   - case_tracker_report.txt



import os, re, csv, argparse, glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ---------- 默认目录 ----------
DEFAULT_INDIR  = r"C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\unity_meter-output"
DEFAULT_OUTDIR = r"C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\case_tracker_from_metrics-outpt"

# ---------- 多编码回退 ----------
READ_ENCODINGS = ["utf-8-sig","utf-8","gb18030","gbk","cp936","big5","cp950","latin1"]

def open_csv_reader(path: str):
    last_err = None
    for enc in READ_ENCODINGS:
        try:
            f = open(path, "r", encoding=enc, newline="")
            rdr = csv.DictReader(f)
            _ = rdr.fieldnames  # 触发行读取
            return f, rdr, enc
        except Exception as e:
            last_err = e
            try:
                f.close()
            except Exception:
                pass
            continue
    raise RuntimeError(f"Failed to read CSV with common encodings: {path}\nLast error: {last_err}")

# ---------- 实用函数 ----------
def sortkey_period(p: str) -> Tuple[int,int]:
    """
    将 '1901–1920' / '1901-1920' / '1901-20' / '未知' 等转为排序键 (start,end)
    """
    if not p:
        return (10**9, 10**9)
    s = str(p)
    m = re.findall(r"(18|19|20)\d{2}", s)
    if len(m) >= 1:
        start = int(m[0] + ("" if len(m[0])==4 else ""))
        end = start
        if len(m) >= 2:
            end = int(m[1])
        else:
            # 试着抓第二个年份（形如 1901–1920 的右侧）
            m2 = re.search(r"[–-](\d{2,4})", s)
            if m2:
                r = m2.group(1)
                if len(r)==2:  # '20' -> 1920 (跟随起始世纪)
                    end = (start//100)*100 + int(r)
                else:
                    end = int(r)
        return (start, end)
    # 单年份/其它
    m1 = re.search(r"\d{4}", s)
    if m1:
        v = int(m1.group(0))
        return (v, v)
    return (10**9, 10**9)

def detect_columns(fieldnames: List[str]) -> Tuple[str,str,str]:
    """
    返回 (period_col, region_col, dominant_col, share_col)
    主形列别名：dominant_canonical / top_canonical / canonical
    占比列别名：dominant_share / top_ratio / share
    """
    fset = { (x or "").strip().lower() for x in (fieldnames or []) }
    period_col = "period" if "period" in fset else None
    region_col = "region" if "region" in fset else None

    d_candidates = ["dominant_canonical","top_canonical","canonical"]
    s_candidates = ["dominant_share","top_ratio","share"]

    dominant_col = next((c for c in d_candidates if c in fset), None)
    share_col    = next((c for c in s_candidates if c in fset), None)

    if not period_col or not region_col or not dominant_col:
        raise ValueError(f"Missing required columns. Have={sorted(fset)} "
                         f"Need period/region and one of {d_candidates}")
    return period_col, region_col, dominant_col, share_col or ""

def pick_metrics_file(indir: str) -> str:
    p = Path(indir)
    if not p.exists():
        raise FileNotFoundError(f"Input dir not found: {indir}")
    cand1 = p / "unity_metrics.csv"
    if cand1.exists():
        return str(cand1)
    # 回退：任意 *_metrics.csv
    cands = sorted(Path(indir).glob("*_metrics.csv"))
    if cands:
        return str(cands[0])
    raise FileNotFoundError(f"No metrics CSV found in {indir} (expected unity_metrics.csv or *_metrics.csv)")

# ---------- 输出 ----------
def ensure_outdir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def write_csv(rows: List[List], headers: List[str], outpath: str):
    ensure_outdir(os.path.dirname(outpath))
    with open(outpath, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

# ---------- 主逻辑 ----------
def build_timeline_and_switches(metrics_path: str, outdir: str):
    f, rdr, enc = open_csv_reader(metrics_path)
    with f:
        period_col, region_col, dom_col, share_col = detect_columns(rdr.fieldnames or [])
        # 收集 -> {(region): [(period, dominant, share)]}
        reg_map: Dict[str, List[Tuple[str,str,Optional[float]]]] = {}
        period_set = set()
        for row in rdr:
            period = (row.get(period_col) or "").strip()
            region = (row.get(region_col) or "").strip()
            dominant = (row.get(dom_col) or "").strip()
            share = None
            if share_col:
                try:
                    share = float((row.get(share_col) or "").strip())
                except Exception:
                    share = None
            if not period or not region or not dominant:
                continue
            period_set.add(period)
            reg_map.setdefault(region, []).append((period, dominant, share))

    # 统一 period 顺序
    period_order = sorted(period_set, key=sortkey_period)

    # 1) 时间线表（逐 region, 逐 period）
    tl_rows: List[List] = []
    for region, items in reg_map.items():
        # 按 period 排序并补齐缺失期（share 为空记空）
        d = {p: (dom, sh) for p, dom, sh in items}
        last_dom = None
        for p in period_order:
            dom, sh = d.get(p, (last_dom, None))
            if dom is None:
                # 无历史主形，留空
                tl_rows.append([p, region, "", ""])
            else:
                tl_rows.append([p, region, dom, ("" if sh is None else f"{sh:.4f}")])
                last_dom = dom

    timeline_path = str(Path(outdir) / "case_dominant_timeline.csv")
    write_csv(tl_rows, ["period","region","dominant","share"], timeline_path)

    # 2) 切换统计
    sw_rows: List[List] = []
    for region, items in reg_map.items():
        seq = sorted(items, key=lambda x: sortkey_period(x[0]))
        # 只依据显式期的主形序列计算切换
        switches = 0
        last = None
        segments = []
        for p, dom, _ in seq:
            if not dom:
                continue
            if last is None:
                segments = [[p, p, dom]]  # start, end, dom
                last = dom
            else:
                if dom == last:
                    segments[-1][1] = p  # extend end
                else:
                    switches += 1
                    segments.append([p, p, dom])
                    last = dom
        seg_str = ";".join([f"{a}→{b}:{d}" for a,b,d in segments]) if segments else ""
        first_dom = segments[0][2] if segments else ""
        last_dom  = segments[-1][2] if segments else ""
        sw_rows.append([region, switches, first_dom, last_dom, seg_str, len(seq)])

    switches_path = str(Path(outdir) / "switches_by_region.csv")
    write_csv(sw_rows, ["region","switches","first_dominant","last_dominant","segments","observed_periods"], switches_path)

    # 3) 分歧表（同一 period 不同地区主形是否一致）
    div_rows: List[List] = []
    # 反向索引：period -> {region: dominant}
    period_map: Dict[str, Dict[str,str]] = {}
    for region, items in reg_map.items():
        for p, dom, _ in items:
            if not p or not dom:
                continue
            period_map.setdefault(p, {})[region] = dom
    for p in sorted(period_map.keys(), key=sortkey_period):
        m = period_map[p]
        doms = list(m.values())
        uniq = len(set(doms))
        is_div = "Y" if uniq > 1 else "N"
        regions_join = ";".join(sorted(m.keys()))
        doms_join = ";".join([f"{r}:{m[r]}" for r in sorted(m.keys())])
        div_rows.append([p, regions_join, doms_join, uniq, is_div])

    diverg_path = str(Path(outdir) / "divergences_by_period.csv")
    write_csv(div_rows, ["period","regions","dominants","unique_dominant_count","is_divergent"], diverg_path)

    # 4) 报告
    report_path = str(Path(outdir) / "case_tracker_report.txt")
    with open(report_path, "w", encoding="utf-8") as fr:
        fr.write("== Case Tracker Report ==\n")
        fr.write(f"Input metrics : {metrics_path}\n")
        fr.write(f"Output dir    : {outdir}\n")
        fr.write(f"Regions       : {len(reg_map)}\n")
        fr.write(f"Periods(total): {len(period_order)}\n")
        fr.write(f"Files enc     : auto-fallback\n")
        fr.write("Outputs       : case_dominant_timeline.csv / switches_by_region.csv / divergences_by_period.csv / report\n")

    print(f"[OK] timeline  -> {timeline_path}")
    print(f"[OK] switches  -> {switches_path}")
    print(f"[OK] divergent -> {diverg_path}")
    print(f"[OK] report    -> {report_path}")

def main():
    ap = argparse.ArgumentParser(description="Build dominant timeline & switches from unity_metrics (robust I/O).")
    ap.add_argument("--indir",  default=DEFAULT_INDIR,  help="输入目录（默认 unity_meter-output）")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="输出目录（默认 case_tracker_from_metrics-outpt）")
    ap.add_argument("-i","--input", help="直接指定 metrics CSV 文件（可选）")
    args = ap.parse_args()

    indir  = args.indir or DEFAULT_INDIR
    outdir = args.outdir or DEFAULT_OUTDIR
    ensure_outdir(outdir)

    if args.input:
        metrics_path = args.input
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Input file not found: {metrics_path}")
    else:
        metrics_path = pick_metrics_file(indir)

    build_timeline_and_switches(metrics_path, outdir)
    print("[DONE] Case Tracker finished.")

if __name__ == "__main__":
    main()


# # 直接零参数（按你的默认目录）
# python .\case_tracker_from_metrics.py

# # 或者手动指定输入/输出（可选）
# python .\case_tracker_from_metrics.py -i ".\unity_meter-output\unity_metrics.csv" --outdir ".\case_tracker_from_metrics-outpt"
