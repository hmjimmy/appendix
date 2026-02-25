#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# unity_meter_plus.py — 跨地区×时间段用形统一性评估器（零参数可跑版）
# ================================================================
# 输入（默认）：
#   C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\normalize_alias_plus-output
#   （读取其中的 *.normalized.csv）
#
# 输出（默认）：
#   C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\unity_meter-output
#   产物：
#     - unity_metrics.csv
#     - unity_pairs.csv
#     - unity_discord_top.csv
#     - unity_report.txt
#
# 可选参数：--indir / --outdir / 列名与文件名推断参数等

import os, csv, argparse, math
from collections import Counter, defaultdict
from itertools import combinations
from glob import glob

# ---------------- 默认目录 ----------------
DEFAULT_INDIR  = r"C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\normalize_alias_plus-output"
DEFAULT_OUTDIR = r"C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\unity_meter-output"

# ---------------- 读取 CSV（多编码回退） ----------------
READ_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk", "cp936", "big5", "cp950", "latin1"]

def open_csv_reader(path):
    last_err = None
    for enc in READ_ENCODINGS:
        try:
            f = open(path, "r", encoding=enc, newline="")
            rdr = csv.DictReader(f)
            _ = rdr.fieldnames  # 触发一次字段名读取
            return f, rdr, enc
        except Exception as e:
            last_err = e
            try:
                f.close()
            except Exception:
                pass
            continue
    raise RuntimeError(f"Failed to read CSV with common encodings: {path}\nLast error: {last_err}")

# ---------------- Wilson 置信区间 ----------------
def wilson_ci(k, n, z=1.96):
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z*z)/n
    centre = (p + (z*z)/(2*n)) / denom
    margin = z * math.sqrt( (p*(1-p)/n) + (z*z)/(4*n*n) ) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))

# ---------------- 文件名解析辅助（缺列兜底） ----------------
def infer_from_filename(fname_no_ext, period_idx=None, region_idx=None, sep="."):
    parts = fname_no_ext.split(sep)
    period = parts[period_idx] if (period_idx is not None and -len(parts) <= period_idx < len(parts)) else None
    region = parts[region_idx] if (region_idx is not None and -len(parts) <= region_idx < len(parts)) else None
    return period, region

# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="Period×Region 用形统一性评估器")
    ap.add_argument("--indir", default=DEFAULT_INDIR, help="输入目录（包含 *.normalized.csv）")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="输出目录（默认 unity_meter-output）")
    ap.add_argument("--period-col", default="period", help="period 列名（默认 period）")
    ap.add_argument("--region-col", default="region", help="region 列名（默认 region）")
    ap.add_argument("--freq-col",   default="frequency", help="频次列名（默认 frequency）")
    ap.add_argument("--canon-col",  default="canonical", help="规范形列名（默认 canonical）")
    ap.add_argument("--filename-period-idx", type=int, default=None,
                    help="若 CSV 内无 period 列，可从文件名按分隔符提取 period 的索引（例如 1）")
    ap.add_argument("--filename-region-idx", type=int, default=None,
                    help="若 CSV 内无 region 列，可从文件名按分隔符提取 region 的索引")
    ap.add_argument("--filename-sep", default=".", help="文件名分隔符（默认 .）")
    ap.add_argument("--min-total", type=int, default=10, help="period×region 的最小总频次，小于则跳过（默认 10）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 收集 normalized CSV
    files = sorted(glob(os.path.join(args.indir, "*.normalized.csv")))
    if not files:
        raise SystemExit(f"目录下未找到 *.normalized.csv：{args.indir}")

    # period×region 统计
    freq_counter = defaultdict(Counter)         # {(period,region): Counter(canonical->freq)}
    row_counter  = defaultdict(Counter)         # （备用：行计数）
    per_period_region_canonset = defaultdict(lambda: defaultdict(set))  # {period: {region: {canon,...}}}

    total_rows_read = 0
    files_read = 0

    for fpath in files:
        try:
            f, rdr, used_enc = open_csv_reader(fpath)
        except Exception as e:
            print(f"[跳过] 打开失败：{os.path.basename(fpath)} | {e}")
            continue

        with f:
            fields = rdr.fieldnames or []
            has_period = args.period_col in fields
            has_region = args.region_col in fields
            has_freq   = args.freq_col   in fields
            has_canon  = args.canon_col  in fields

            if not has_canon:
                print(f"[跳过] 缺少列 {args.canon_col} ：{os.path.basename(fpath)} (enc={used_enc})")
                continue

            base_no_ext = os.path.splitext(os.path.basename(fpath))[0]
            inferred_p, inferred_r = infer_from_filename(
                base_no_ext,
                args.filename_period_idx,
                args.filename_region_idx,
                sep=args.filename_sep
            )

            for row in rdr:
                total_rows_read += 1
                period = (row.get(args.period_col) if has_period else None) or inferred_p
                region = (row.get(args.region_col) if has_region else None) or inferred_r
                if not period or not region:
                    continue

                canon = (row.get(args.canon_col) or "").strip()
                if not canon:
                    continue

                freq = 1
                if has_freq:
                    try:
                        freq = int(row.get(args.freq_col, 1) or 1)
                    except Exception:
                        freq = 1

                key = (period, region)
                freq_counter[key][canon] += freq
                row_counter[key][canon]  += 1
                per_period_region_canonset[period][region].add(canon)

            files_read += 1
            print(f"[读入] {os.path.basename(fpath)} (enc={used_enc})")

    # 输出 1：metrics
    metrics_path = os.path.join(args.outdir, "unity_metrics.csv")
    with open(metrics_path, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["period","region","total_freq","unique_canonical",
                    "top_canonical","top_ratio","top_ratio_ci_low","top_ratio_ci_high"])
        for (period, region), c2f in sorted(freq_counter.items()):
            total = sum(c2f.values())
            if total < args.min_total:
                continue
            uniq = len(c2f)
            top_canon, top_freq = max(c2f.items(), key=lambda kv: kv[1])
            ratio = top_freq / total
            lo, hi = wilson_ci(top_freq, total)
            w.writerow([period, region, total, uniq, top_canon,
                        f"{ratio:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    print(f"[OK] metrics -> {metrics_path}")

    # 输出 2：pairs（Jaccard）
    pairs_path = os.path.join(args.outdir, "unity_pairs.csv")
    with open(pairs_path, "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["period","region_a","region_b","jaccard","|A|","|B|","|A∩B|","|A∪B|"])
        for period, reg2set in sorted(per_period_region_canonset.items()):
            regs = sorted(reg2set.keys())
            for ra, rb in combinations(regs, 2):
                A, B = reg2set[ra], reg2set[rb]
                inter = len(A & B)
                union = len(A | B) if (A or B) else 0
                j = inter/union if union else 0.0
                w.writerow([period, ra, rb, f"{j:.4f}", len(A), len(B), inter, union])
    print(f"[OK] pairs -> {pairs_path}")

    # 输出 3：discord（主形分歧）
    discord_path = os.path.join(args.outdir, "unity_discord_top.csv")
    with open(discord_path, "w", encoding="utf-8-sig", newline="") as fd:
        w = csv.writer(fd)
        w.writerow(["period","regions","top_canonicals","is_discord"])
        per_period_top = defaultdict(dict)
        for (period, region), c2f in freq_counter.items():
            if sum(c2f.values()) < args.min_total:
                continue
            top_canon, top_freq = max(c2f.items(), key=lambda kv: kv[1])
            ratio = top_freq / sum(c2f.values())
            per_period_top[period][region] = (top_canon, ratio)
        for period in sorted(per_period_top.keys()):
            reg_map = per_period_top[period]
            tops = [v[0] for v in reg_map.values()]
            is_discord = len(set(tops)) > 1
            regions_join = ";".join(sorted(reg_map.keys()))
            tops_join = ";".join(f"{r}:{reg_map[r][0]}({reg_map[r][1]:.2f})" for r in sorted(reg_map.keys()))
            w.writerow([period, regions_join, tops_join, "Y" if is_discord else "N"])
    print(f"[OK] discord -> {discord_path}")

    # 输出 4：report
    report_path = os.path.join(args.outdir, "unity_report.txt")
    with open(report_path, "w", encoding="utf-8") as fr:
        fr.write("== Unity Meter Report ==\n")
        fr.write(f"Input dir : {args.indir}\n")
        fr.write(f"Output dir: {args.outdir}\n")
        fr.write(f"Files read: {files_read} / found: {len(files)}\n")
        fr.write(f"Rows read : {total_rows_read}\n")
        fr.write(f"Min total : {args.min_total}\n")
        fr.write("Outputs   : metrics / pairs / discord / report\n")
    print(f"[OK] report  -> {report_path}")

    print("[DONE] Unity Meter finished.")

if __name__ == "__main__":
    main()

# 使用示例（PowerShell）
# cd "C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1"
# python .\unity_meter_plus.py
# 或手动指定输入输出：
# python .\unity_meter_plus.py --indir ".\normalize_alias_plus-output" --outdir ".\unity_meter-output"
