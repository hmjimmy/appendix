                                                                                                                                                                    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
ccl2ann.py — CCL → Annotation Studio converter（批处理、固定入出目录，避免 \U 转义）

输入目录（默认批处理来源）：
  C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\GOGOGO

输出目录（统一放产物 .json / .csv）：
  C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\ccl2ann-output

使用示例（PowerShell）：
  # 批处理整个文件夹
  python .\ccl2ann.py

  # 指定单个文件
  python .\ccl2ann.py -i ".\GOGOGO\corpus_德律风.txt"
"""

import os, re, csv, json, argparse, sys, traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

# 可选编码探测
try:
    import chardet
except Exception:
    chardet = None

# ---------- 配置：固定路径与默认分桶 ----------
DEFAULT_INDIR  = r"C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\GOGOGO"
DEFAULT_OUTDIR = r"C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1\ccl2ann-output"
DEFAULT_BUCKETS = [(1900,1919),(1920,1939),(1940,1959),(1960,1979),(1980,1999),(2000,2010),(2011,2020)]

# ---------- 推断辅助 ----------
def infer_region(path: str, default_region: str="大陆")->str:
    s = path
    if any(k in s for k in ["台湾","臺灣","Taiwan","台刊","台湾作家","台灣作家"]):
        return "台湾"
    if any(k in s for k in ["香港","HongKong","HK","香江","港报","港刊","香港文學","香港作家"]):
        return "香港"
    if any(k in s for k in ["澳门","Macau","澳門"]):
        return "澳门"
    return default_region

def infer_domain(path: str)->str:
    for key in ["文学","报刊","网络语料","社区问答","新闻","期刊","学术","科技","历史","杂志","小说","微博"]:
        if key in path:
            return key
    parts = re.split(r"[\\/]+", path)
    for seg in reversed(parts[:-1]):
        if seg and not re.search(r"\d{4}", seg):
            return seg
    return "未知"

def infer_year(path: str)->int:
    m = re.search(r"(18|19|20)\d{2}", path)
    return int(m.group(0)) if m else 0

def bucket_period(year: int, use_decade: bool, buckets):
    if year == 0: return "未知"
    if use_decade:
        decade = (year // 10) * 10
        return f"{decade}s"
    for a,b in buckets:
        if a <= year <= b:
            return f"{a}-{b}"
    lo = (year//20)*20
    return f"{lo}-{lo+19}"

def parse_header(line: str)->Dict[str,Any]:
    # 示例： 【 查询关键词：德律风  总结果数：88 】
    m = re.search(r"【\s*查询关键词[:：]\s*([^\s，,】]+).*?总结果数[:：]\s*(\d+)\s*】", line)
    return {"query": m.group(1), "total": int(m.group(2))} if m else {"query": "", "total": 0}

def parse_entry(line: str)->Dict[str,Any]:
    # 示例行：
    # 1: ……[德律风]……【 文件名：...\path\to\doc.txt  文章标题：某文  作者：某某 】
    m = re.match(
        r"\s*\d+:\s*(.+?)\s*【\s*文件名[:：]\s*(.*?)\s*文章标题[:：]\s*(.*?)\s*作者[:：]\s*(.*?)\s*】\s*$",
        line
    )
    if not m: return {}
    snippet, filepath, title, author = m.groups()
    hits = re.findall(r"\[([^\[\]]+)\]", snippet)
    return {
        "snippet": snippet,
        "filepath": filepath.strip(),
        "title": None if title.strip().lower()=="null" else title.strip(),
        "author": None if author.strip().lower()=="null" else author.strip(),
        "hits": hits
    }

# ---------- 读写 ----------
def read_lines(infile: str, encoding: str=None, auto: bool=False)->Tuple[List[str], str]:
    tried = []
    if auto and chardet is not None:
        with open(infile, "rb") as fb:
            raw = fb.read(65536)
        det = chardet.detect(raw or b"")
        enc = (det.get("encoding") or "").lower()
        if enc: tried.append(enc)
    if encoding: tried.insert(0, encoding)
    tried.extend(["utf-8","utf-8-sig","gb18030","gbk","cp936","big5","cp950","latin1"])

    last_err = None
    for enc in tried:
        try:
            with open(infile, "r", encoding=enc, errors="strict") as f:
                return [ln.rstrip("\n") for ln in f if ln.strip()], enc
        except UnicodeDecodeError as err:
            last_err = err
            continue
    raise UnicodeDecodeError(f"All encodings failed. Last error: {last_err}")

def rows_from_lines(lines: List[str], use_decade: bool, buckets, default_region: str, strict: bool):
    header = parse_header(lines[0]) if lines else {"query":"", "total":0}
    query = header.get("query","")
    rows = []
    for ln in lines[1:]:
        entry = parse_entry(ln)
        if not entry:
            continue
        year = infer_year(entry["filepath"])
        period = bucket_period(year, use_decade, buckets)
        region = infer_region(entry["filepath"], default_region=default_region)
        domain = infer_domain(entry["filepath"])
        source = query
        hits = [h for h in (entry["hits"] or []) if (not strict or h==query)] or ([query] if query else [])
        for hit in hits:
            ev = entry["snippet"]
            try:
                idx = ev.index(f"[{hit}]")
                kwic = ev[max(0, idx-20): idx+len(hit)+22]
            except ValueError:
                kwic = ev[:80]
            rows.append({
                "period": period, "year": year, "region": region,
                "source": source, "translation": hit, "strategy": "",
                "frequency": 1, "domain": domain, "F": "", "U": "",
                "evidence": f"{kwic} —— {entry['filepath']}",
                "alias_of": "", "notes": f"title={entry['title']}; author={entry['author']}; CCL"
            })
    return rows, query

def write_outputs(rows: List[Dict[str,Any]], outbase: str):
    Path(DEFAULT_OUTDIR).mkdir(parents=True, exist_ok=True)
    json_path = outbase + ".json"
    csv_path  = outbase + ".csv"
    with open(json_path, "w", encoding="utf-8") as g:
        json.dump(rows, g, ensure_ascii=False, indent=2)

    # 注意：这里把 total_freq 一并写出
    keys = ["period","year","region","source","translation","strategy","frequency","total_freq",
            "domain","F","U","evidence","alias_of","notes"]
    with open(csv_path, "w", encoding="utf-8", newline="") as g:
        w = csv.DictWriter(g, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    return json_path, csv_path

def parse_buckets(s: str):
    spans = []
    for token in (s or "").split(","):
        m = re.match(r"(\d{4})-(\d{4})", token.strip())
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            spans.append((a, b))
    return spans or DEFAULT_BUCKETS

# ---------- 核心处理 ----------
def process_one_file(inpath: str, encoding=None, auto_encoding=False,
                     use_decade=False, buckets=DEFAULT_BUCKETS,
                     region_default="大陆", strict=False):
    try:
        if not os.path.exists(inpath):
            return False, f"[SKIP] Not found: {inpath}"
        name_wo_ext = os.path.splitext(os.path.basename(inpath))[0]
        outbase = os.path.join(DEFAULT_OUTDIR, name_wo_ext)
        lines, used_enc = read_lines(inpath, encoding=encoding, auto=auto_encoding)
        rows, query = rows_from_lines(lines, use_decade, buckets, region_default, strict)

        # 去重（同一 translation/year/region/evidence 认为同条）
        seen = set()
        uniq = []
        for r in rows:
            key = (r["translation"], r["year"], r["region"], r["evidence"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)

        # ===== 新增：计算 total_freq（按 translation 汇总）=====
        counter = Counter(r["translation"] for r in uniq)
        for r in uniq:
            r["total_freq"] = int(counter.get(r["translation"], 0))  # 每行写入该译式的总频次

        jpath, cpath = write_outputs(uniq, outbase)
        return True, f"[OK] {len(uniq)} rows | Query='{query}' | Enc='{used_enc}' | JSON: {jpath} | CSV: {cpath}"
    except Exception:
        return False, f"[ERR] {inpath}\n{traceback.format_exc()}"

def iter_txt_files(root: str)->List[str]:
    p = Path(root)
    return [str(x) for x in p.glob("*.txt")] if p.exists() else []

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser(description="Convert CCL .txt to JSON/CSV (fixed indir/outdir)")
    ap.add_argument("-i", "--input", help="single file path (optional)")
    ap.add_argument("--indir", default=DEFAULT_INDIR, help="input folder (default=GOGOGO)")
    ap.add_argument("--strict", action="store_true", help="only keep bracket hits equal to query")
    ap.add_argument("--decade", action="store_true", help="use decade buckets")
    ap.add_argument("--buckets", default="1900-1919,1920-1939,1940-1959,1960-1979,1980-1999,2000-2010,2011-2020")
    ap.add_argument("--encoding", default=None)
    ap.add_argument("--auto-encoding", action="store_true")
    ap.add_argument("--region-default", default="大陆")
    args = ap.parse_args()

    buckets = parse_buckets(args.buckets)

    # 单文件模式
    if args.input:
        ok, msg = process_one_file(
            args.input, args.encoding, args.auto_encoding,
            args.decade, buckets, args.region_default, args.strict
        )
        print(msg)
        sys.exit(0 if ok else 1)

    # 批处理模式
    files = iter_txt_files(args.indir or DEFAULT_INDIR)
    if not files:
        print(f"[WARN] No .txt found in {args.indir}")
        sys.exit(1)

    print(f"[INFO] Found {len(files)} file(s) in {args.indir}")
    ok_count = 0
    fail_count = 0
    for fp in files:
        ok, msg = process_one_file(
            fp, args.encoding, args.auto_encoding,
            args.decade, buckets, args.region_default, args.strict
        )
        print(msg)
        ok_count += int(ok)
        fail_count += (0 if ok else 1)
    print(f"[SUMMARY] OK={ok_count}, FAIL={fail_count}, OutDir={DEFAULT_OUTDIR}")

if __name__ == "__main__":
    main()


# cd "C:\Users\18955\Desktop\MZJ的学术垃圾\计算机\python文件保存\CCL1"
# python .\ccl2ann.py


# # 批处理（扫描 GOGOGO 下所有 .txt）
# python .\ccl2ann.py

# # 处理单个文件
# python .\ccl2ann.py -i ".\GOGOGO\corpus_德律风.txt"

# 常用可选参数
# --strict            只保留与检索词完全相同的方括号命中
# --decade            使用年代桶（如 1970s），默认用区间桶
# --encoding ENC      强制输入编码（否则自动探测+回退）
# --auto-encoding     打开自动编码探测（需要 chardet）
