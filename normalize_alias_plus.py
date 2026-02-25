#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# normalize_alias_plus.py — 批量清洗与标准化（固定 I/O 目录, 兼容 frequency/total_freq）
# v2025-09-18T23:40  不直接读取 total_freq；frequency 兜底；别名候选空表安全

import os, re, csv, argparse, unicodedata, glob, sys, traceback
from pathlib import Path
from typing import Dict
import pandas as pd

BANNER = "normalize_alias_plus.py v2025-09-18T23:40 (no direct read of total_freq; frequency fallback + safe candidates)"

# ---------- 固定目录 ----------
DEFAULT_INDIR  = r"C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\ccl2ann-output"
DEFAULT_OUTDIR = r"C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\normalize_alias_plus-output"

# ---------- 读写辅助 ----------
READ_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk", "cp936", "big5", "cp950", "latin1"]

def read_csv_auto(path: str) -> pd.DataFrame:
    last_err = None
    for enc in READ_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read CSV with common encodings: {path}\nLast error: {last_err}")

def write_csv(df: pd.DataFrame, path: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

# ---------- 文本规范化 ----------
ZERO_WIDTH = re.compile(r'[\u200B\u200C\u200D\uFEFF]')
DOTS = [("•","·"), ("‧","·"), ("∙","·"), ("⋅","·"), (".","·")]
DASHES = [("—","-"), ("–","-"), ("－","-"), ("―","-")]
SPACES = re.compile(r'\s+')

def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def fullwidth_to_halfwidth(s: str) -> str:
    out = []
    for ch in s:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)

def normalize_punct(s: str) -> str:
    if not s: return s
    x = s
    x = ZERO_WIDTH.sub("", x)
    x = nfkc(x)
    x = fullwidth_to_halfwidth(x)
    for a,b in DOTS: x = x.replace(a,b)
    for a,b in DASHES: x = x.replace(a,b)
    x = x.replace("，", ",").replace("、", ",").replace("；",";").replace("：",":")
    x = x.replace("（","(").replace("）",")").replace("【","[").replace("】","]")
    x = x.replace("“","\"").replace("”","\"").replace("‘","'").replace("’","'")
    x = SPACES.sub(" ", x).strip()
    x = re.sub(r'\s*·\s*', '·', x)
    return x

# ---------- 别名映射 ----------
def load_alias_map() -> Dict[str,str]:
    candidates = [
        os.path.join(os.getcwd(), "alias_map.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "alias_map.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            last_err = None
            # 先尝试按有表头（variant/canonical）读取
            try:
                with open(p, "r", encoding="utf-8-sig", newline="") as f:
                    rdr = csv.DictReader(f)
                    cols = [c.strip().lower() for c in (rdr.fieldnames or [])]
                    if {"variant","canonical"} <= set(cols):
                        mp = {}
                        for row in rdr:
                            v = normalize_punct(str(row.get("variant","")))
                            c = normalize_punct(str(row.get("canonical","")))
                            if v: mp[v] = c
                        return mp
            except Exception as e:
                last_err = e
            # 再尝试无表头两列
            try:
                mp = {}
                with open(p, "r", encoding="utf-8-sig", newline="") as f:
                    for row in csv.reader(f):
                        if len(row) >= 2:
                            v = normalize_punct(str(row[0])); c = normalize_punct(str(row[1]))
                            if v: mp[v] = c
                return mp
            except Exception as e:
                last_err = e
                print(f"[WARN] alias_map.csv found but failed to read: {p} ({last_err})")
    return {}

def to_canonical(x: str, amap: Dict[str,str]) -> str:
    if not x: return x
    norm = normalize_punct(x)
    return amap.get(norm, norm)

# ---------- 别名候选（安全版） ----------
def suggest_alias_candidates(df: pd.DataFrame, min_freq: int = 2) -> pd.DataFrame:
    # translation 列缺失时返回空表（带标准列，避免 sort_values 报错）
    if "translation" not in df.columns:
        return pd.DataFrame(columns=["key","forms","total_freq","forms_detail"])

    tmp = df.copy()
    tmp["__norm_key"] = (
        tmp["translation"].astype(str)
        .apply(normalize_punct)
        .str.lower()
        .str.replace(r"[·\-\s,_]", "", regex=True)
    )

    # 统一频次口径：只用 frequency，没有则默认 1
    tmp["frequency"] = pd.to_numeric(tmp.get("frequency", 1), errors="coerce").fillna(1).astype(int)

    rows = []
    for k, sub in tmp.groupby("__norm_key", dropna=False):
        forms = sub["translation"].apply(normalize_punct).value_counts().to_dict()
        total = int(sub["frequency"].sum())
        # 只有当存在多个变体且总频次达到阈值时，才生成候选
        if len(forms) >= 2 and total >= min_freq:
            rows.append({
                "key": k,
                "forms": ";".join(list(forms.keys())[:10]),
                "total_freq": total,
                "forms_detail": ";".join([f"{f}({n})" for f,n in forms.items()])
            })

    # rows 可能为空：返回带标准列的空表；否则按 total_freq 排序
    if not rows:
        return pd.DataFrame(columns=["key","forms","total_freq","forms_detail"])

    out = pd.DataFrame(rows)
    if "total_freq" in out.columns:
        return out.sort_values("total_freq", ascending=False)
    else:
        return out  # 双保险

# ---------- 核心处理 ----------
def process_one_csv(in_csv: str, outdir: str, amap: Dict[str,str]):
    name = os.path.splitext(os.path.basename(in_csv))[0]
    base = os.path.join(outdir, name)

    df = read_csv_auto(in_csv)

    # —— 关键兼容：只用 frequency；如果没有，就由 total_freq 迁移或默认 1 ——
    has_freq = "frequency" in df.columns
    has_total = "total_freq" in df.columns
    if not has_freq:
        if has_total:
            df["frequency"] = df["total_freq"]
        else:
            df["frequency"] = 1

    if "translation" not in df.columns:
        df["translation"] = ""

    # === NEW === 在任何清洗/规范化之前，保留原始译形
    df["translation_raw"] = df["translation"].astype(str)

    # 规范化（作用于 translation，用于后续聚合与别名映射）
    df["translation"] = df["translation"].astype(str).apply(normalize_punct)
    df["frequency"]   = pd.to_numeric(df["frequency"], errors="coerce").fillna(1).astype(int)

    # 应用别名映射（不要覆盖 translation，仅生成 canonical）
    amap = amap or {}
    # === NEW === 只写 canonical，不覆盖 translation
    df["canonical"] = df["translation"].apply(lambda s: to_canonical(s, amap))

    # 产物1：逐行 normalized（现在包含 translation_raw 与 canonical）
    normalized_path = base + ".normalized.csv"
    write_csv(df, normalized_path)

    # 产物2：canonical 聚合（输出列名叫 total_freq，但值来自 frequency 求和）
    agg = (
        df.groupby("canonical", dropna=False)
          .agg(total_freq=("frequency","sum"),
               forms=("translation", lambda x: ";".join(sorted(set(x)))))
          .reset_index()
          .sort_values("total_freq", ascending=False)
    )
    alias_stats_path = base + ".alias_stats.csv"
    write_csv(agg, alias_stats_path)

    # 产物3：启发式别名候选（安全版）
    cand_df = suggest_alias_candidates(df, min_freq=2)
    alias_cand_path = base + ".alias_candidates.csv"
    write_csv(cand_df, alias_cand_path)

    # 产物4：报告
    report_path = base + ".normalize_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("== Normalize & Alias Report ==\n")
        f.write(f"Script     : {os.path.abspath(__file__)}\n")
        f.write(f"Input CSV  : {in_csv}\n")
        f.write(f"Output dir : {outdir}\n")
        f.write(f"Rows       : {len(df)}\n")
        f.write(f"Unique forms    : {df['translation'].nunique(dropna=False)}\n")
        f.write(f"Unique canonical: {df['canonical'].nunique(dropna=False)}\n")
        f.write(f"Has alias_map.csv: {'Y' if amap else 'N'}\n")

    return normalized_path

# ---------- 主程序 ----------
def main():
    print(BANNER)
    print("Running file:", os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Batch normalize translations and apply alias mapping (fixed I/O).")
    parser.add_argument("--indir",  default=DEFAULT_INDIR,  help="输入目录（默认 ccl2ann-output）")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="输出目录（默认 normalize_alias_plus-output）")
    args = parser.parse_args()

    indir  = args.indir  or DEFAULT_INDIR
    outdir = args.outdir or DEFAULT_OUTDIR
    Path(outdir).mkdir(parents=True, exist_ok=True)

    amap = load_alias_map()
    if not amap:
        print("[INFO] alias_map.csv not found; only use normalized forms as canonical.")

    files = sorted(glob.glob(str(Path(indir) / "*.csv")))
    if not files:
        print(f"[WARN] No .csv found in: {indir}")
        sys.exit(1)

    total, ok = 0, 0
    for fp in files:
        total += 1
        try:
            process_one_csv(fp, outdir, amap)
            print(f"[OK] {os.path.basename(fp)}")
            ok += 1
        except Exception as e:
            print(f"[ERR] {fp}: {e}\n{traceback.format_exc()}")

    print(f"[SUMMARY] OK={ok}/{total} | IN={indir} | OUT={outdir}")

if __name__ == "__main__":
    main()

# cd "C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1"
# python ".\normalize_alias_plus.py"
