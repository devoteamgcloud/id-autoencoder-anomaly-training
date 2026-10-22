"""
fds_eval.py — shared, model-agnostic evaluation for the FDS retrain.

Works for ANY model (Autoencoder, Isolation Forest, aggregator statistical
model) because it evaluates a SCORE, not a model. Each trainer produces a
scored file; this script joins it to the analyst ground truth (the
MonitoringResult Excel) and reports ranking metrics.

────────────────────────────────────────────────────────────────────────────
INPUT CONTRACT — the scored file (.parquet or .csv) must contain:

  grain = transaction   (commerce: Autoencoder / Isolation Forest)
      columns:  merchant_id, cust_key, trx_date, score
  grain = merchant_day  (aggregator model)
      columns:  merchant_id, trx_date, score

  `score` = anomaly score, HIGHER = more anomalous / more fraud-like.

How to produce it from your model (score your TEST features, then save keys+score):

  # Isolation Forest (sklearn) — higher = more anomalous
  s = -clf.score_samples(X)                       # negate: sklearn gives higher=normal
  out = test_df[['merchant_id','cust_key','trx_date']].copy(); out['score'] = s
  out.to_parquet('iso_scores.parquet', index=False)

  # Autoencoder — reconstruction error is already the score
  s = np.sqrt(np.mean((X - model.predict(X))**2, axis=1))

  # Aggregator (merchant-hour rows) — use DATE(hour_bucket) as trx_date
  out = test_df[['merchant_id','trx_date']].copy(); out['score'] = s
────────────────────────────────────────────────────────────────────────────

USAGE:
  python fds_eval.py --scores iso_scores.parquet \
      --xlsx "MonitoringResult_2026 (2).xlsx" --grain transaction --period-split 2026-09-01

  python fds_eval.py --scores agg_scores.parquet \
      --xlsx "MonitoringResult_2026 (2).xlsx" --grain merchant_day

Reads parquet/csv + Excel only. No BigQuery writes.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

# Analyst Action values that count as a confirmed true positive (fraud).
TP_ACTIONS = {"Fraud", "Blacklist", "EDD/CDD"}

# Join keys per grain.
GRAIN_KEYS = {
    "transaction": ["merchant_id", "cust_key", "trx_date"],
    "merchant_day": ["merchant_id", "trx_date"],
}

_KET = re.compile(r'"([^"]+)"\s*:\s*"([^"]*)"')


# ── analyst labels (Excel) ───────────────────────────────────────────────────

def _excel_cust_key(msisdn, cust_id):
    """Replicate the v2 SQL cust_key: valid cust_id (id:) else valid msisdn (ph:)."""
    cid = (cust_id or "").strip()
    if cid and cid not in ("", "0", "01") and not re.fullmatch(r"0+", cid):
        return "id:" + cid
    m = (msisdn or "").strip()
    mm = m.replace("+", "")
    if m and m != "081224567890" and mm != "" and not re.fullmatch(r"6?2?0{4,}", mm):
        return "ph:" + m
    return None


def load_alerts(xlsx_path, grain):
    """Parse analyst Excel into labels at the requested grain. PG rows only."""
    df = pd.read_excel(xlsx_path, sheet_name="Suspicious Transaction Checking")
    df = df[df["Service Transaction"] == "Payment Gateway"].copy()
    p = df["Keterangan"].apply(lambda s: dict(_KET.findall(s)) if isinstance(s, str) else {})

    out = pd.DataFrame({
        "merchant_id": p.apply(lambda d: (d.get("merchant_id") or "").strip()),
        "cust_key":    p.apply(lambda d: _excel_cust_key(d.get("cust_msisdn"), d.get("customer_id"))),
        "trx_date":    pd.to_datetime(p.apply(lambda d: d.get("date_trx")), errors="coerce").dt.strftime("%Y-%m-%d"),
    })
    out["gt_label"] = df["Action"].isin(TP_ACTIONS).astype(int).values

    keys = GRAIN_KEYS[grain]
    bad = out["merchant_id"].eq("") | out["trx_date"].isna()
    if "cust_key" in keys:
        bad = bad | out["cust_key"].isna()
    out = out[~bad]

    # Collapse to one label per key (a key is fraud if ANY of its alerts is TP).
    labels = out.groupby(keys, as_index=False)["gt_label"].max()
    print(f"[labels] grain={grain}  {len(labels)} keys  "
          f"({int(labels.gt_label.sum())} fraud, {int((labels.gt_label==0).sum())} clean)")
    return labels


# ── scored file ──────────────────────────────────────────────────────────────

def load_scores(path, keys, score_col):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = [c for c in keys + [score_col] if c not in df.columns]
    if missing:
        raise SystemExit(f"scored file is missing required columns: {missing}\n"
                         f"  has: {list(df.columns)}")
    df = df[keys + [score_col]].copy()
    df["merchant_id"] = df["merchant_id"].astype(str).str.strip()
    if "cust_key" in keys:
        df["cust_key"] = df["cust_key"].astype(str).str.strip()
    df["trx_date"] = pd.to_datetime(df["trx_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df.rename(columns={score_col: "score"})


# ── metrics ──────────────────────────────────────────────────────────────────

def report(df, threshold, title):
    """df has columns: score, gt_label. Print ranking metrics."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  precision_recall_fscore_support, confusion_matrix)
    y, s = df.gt_label.values, df.score.values
    n, n_pos = len(y), int(y.sum())
    print("\n" + "-" * 66)
    print(f"[{title}]  n={n:,}  fraud={n_pos:,}  clean={n - n_pos:,}")
    if n == 0:
        print("  (no rows joined)"); return
    if n_pos in (0, n):
        print("  only one class present — ranking metrics undefined"); return

    print(f"  ROC-AUC           : {roc_auc_score(y, s):.3f}")
    print(f"  PR-AUC (avg prec) : {average_precision_score(y, s):.3f}   (base rate {y.mean():.3f})")
    order = np.argsort(s)[::-1]
    for k in [100, 500, n_pos]:
        if k <= n:
            tag = " (=#pos → recall@P)" if k == n_pos else ""
            print(f"  precision@{k:<6}  : {y[order[:k]].mean():.3f}{tag}")
    if threshold is not None:
        pred = (s >= threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        cm = confusion_matrix(y, pred)
        print(f"  @threshold {threshold:g}: precision {p:.3f}  recall {r:.3f}  f1 {f1:.3f}  "
              f"flagged {pred.mean():.1%}")
        print(f"  confusion [[TN FP][FN TP]]: {cm.tolist()}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Model-agnostic FDS evaluation vs analyst labels")
    ap.add_argument("--scores", required=True, help="scored file (.parquet/.csv): keys + score")
    ap.add_argument("--xlsx", required=True, help="MonitoringResult_*.xlsx")
    ap.add_argument("--grain", choices=list(GRAIN_KEYS), default="transaction")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--threshold", type=float, default=None, help="optional decision threshold on score")
    ap.add_argument("--period-split", default=None, help="also report rows with trx_date >= this (held-out)")
    args = ap.parse_args()

    keys = GRAIN_KEYS[args.grain]
    scores = load_scores(args.scores, keys, args.score_col)
    labels = load_alerts(args.xlsx, args.grain)

    merged = scores.merge(labels, on=keys, how="inner")
    print("\n" + "=" * 66)
    print(f"FDS EVAL  (grain={args.grain})")
    print("=" * 66)
    print("NOTE: analyst-reviewed rows are alerts (biased subset). AUC reads as\n"
          "'among alerts, does the score rank real fraud above false positives?'")
    if merged.empty:
        print(f"\nNo rows joined on {keys}. Check key formats in the scored file.")
        return

    report(merged, args.threshold, "ALL labelled")
    if args.period_split:
        report(merged[merged.trx_date >= args.period_split], args.threshold,
               f"HELD-OUT  trx_date >= {args.period_split}")
        report(merged[merged.trx_date <  args.period_split], args.threshold,
               f"TRAIN-period  trx_date < {args.period_split}")
    print("\ndone.")


if __name__ == "__main__":
    main()
