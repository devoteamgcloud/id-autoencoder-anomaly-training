"""
evaluate.py — Evaluate a trained anomaly-detection autoencoder against the
analyst monitoring results.

The question this answers is NOT "is val loss low?" but "does reconstruction
error actually separate fraud from false alarms?". It scores transactions by
reconstruction error and checks whether that error ranks analyst-confirmed
fraud above analyst-cleared alerts.

GROUND TRUTH — from the analyst Excel (MonitoringResult_*.xlsx), sheet
"Suspicious Transaction Checking", Payment Gateway rows only:
    Action in {Fraud, Blacklist, EDD/CDD}  -> gt_label = 1  (true positive)
    everything else (False Positive, Watchlist, Case Management) -> 0
Each alert is a customer-merchant-day; every transaction matching
(merchant_id, cust_key, trx_date) of a TP alert is labeled fraud. Transactions
matching an FP alert are the negatives.

Reconstruction error == RMSE per row (same formula find_threshold uses).
Preprocessing reuses the trainer's preprocess() + saved precomputed.json /
columns_info.json so the feature matrix matches training exactly.

Periods are reported separately because most TP alerts (16/20) fall in the
May-Aug TRAINING window; only ~4 are in the September val window. Training-period
error is optimistically biased low — read those numbers with that in mind.

No data is written to BigQuery or anywhere external. Reads parquet + Excel only.

Run from the repo root (same place you run `python -m trainer.task`):

    python -m trainer.evaluate \
        --model-path  model/autoencoder-pg-retrain2026_20261015_retrain2026.keras \
        --xlsx        "MonitoringResult_2026 (2).xlsx" \
        --data-paths  /home/jupyter/retraining/data/train /home/jupyter/retraining/data/test \
        --artifacts   model
"""

import argparse
import json
import logging
import os
import re
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

import trainer.config as config
from trainer.utils.data_loader import preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Analyst "Action" values that count as a confirmed true positive (fraud).
TP_ACTIONS = {"Fraud", "Blacklist", "EDD/CDD"}

# In-data proxy labels (weak signals that live in the val parquet itself).
PROXY_LABELS = ["is_decline", "is_new_user_flag", "c_new_merchant_flag"]

# Join granularity: all transactions sharing this key with an alert are labeled.
JOIN_KEYS = ["merchant_id", "cust_key", "trx_date"]

# Transactions on/after this date are treated as the held-out val period.
DEFAULT_PERIOD_SPLIT = "2026-09-01"

_KET_PAIR = re.compile(r'"([^"]+)"\s*:\s*"([^"]*)"')


# ── artifact loading ─────────────────────────────────────────────────────────

def load_artifacts(artifacts_dir: str):
    """Load the feature spec, preprocessing mappings, and deployed threshold."""
    with open(os.path.join(artifacts_dir, "columns_info.json")) as f:
        cols = json.load(f)
    with open(os.path.join(artifacts_dir, "precomputed.json")) as f:
        precomp = json.load(f)

    threshold = None
    hp_path = os.path.join(artifacts_dir, "hyperparameters.json")
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            threshold = json.load(f).get("threshold")

    # StandardScaler fitted at train time — must be applied identically here.
    scaler = None
    scaler_path = os.path.join(artifacts_dir, "scaler.json")
    if os.path.exists(scaler_path):
        with open(scaler_path) as f:
            scaler = json.load(f)
        log.info("loaded scaler.json (features standardized)")
    else:
        log.warning("no scaler.json found — scoring UNSCALED (results won't match a scaled model)")

    kwargs = {
        "features":              cols["features"],
        "id_columns":            cols["id-columns"],
        "drop_columns":          cols["drop-columns"],
        "impute_columns":        cols["impute-columns"],
        "log_scale_columns":     cols["log-scale-columns"],
        "signed_log_columns":    cols.get("signed-log-columns", []),
        "stat_encoding_columns": cols["stat-encoding-columns"],
        "periodic_columns":      [tuple(x) for x in cols.get("periodic-columns", [])],
        "ohe_columns":           [tuple(x) for x in cols.get("ohe-columns", [])],
        "stat_mapping":          precomp["stat_mapping"],
        "ohe_class_names":       precomp["ohe_class_names"],
        "scaler":                scaler,
    }
    return cols["features"], kwargs, threshold


def load_model(model_path: str):
    from tensorflow import keras
    # compile=False: only the forward pass is needed, and this avoids having to
    # re-supply the custom `mape` metric used at train time.
    return keras.models.load_model(model_path, compile=False)


# ── analyst labels (Excel) ───────────────────────────────────────────────────

def _parse_keterangan(s: str) -> Dict[str, str]:
    if not isinstance(s, str):
        return {}
    return dict(_KET_PAIR.findall(s))


def _norm_merchant(x) -> str:
    return "" if x is None else str(x).strip()


def _excel_cust_key(msisdn: str, cust_id: str) -> str:
    """Replicate the SQL: COALESCE(NULLIF(TRIM(cust_msisdn),''), cust_id)."""
    m = (msisdn or "").strip()
    if m and m.lower() != "nan":
        return m
    return (cust_id or "").strip()


def load_alerts(xlsx_path: str) -> pd.DataFrame:
    """Parse the analyst Excel into one row per alert with join keys + gt_label.

    Returns columns: merchant_id, cust_key, trx_date (YYYY-MM-DD str),
    gt_label (0/1), action. Payment Gateway alerts only.
    """
    df = pd.read_excel(xlsx_path, sheet_name="Suspicious Transaction Checking")
    df = df[df["Service Transaction"] == "Payment Gateway"].copy()

    parsed = df["Keterangan"].apply(_parse_keterangan)
    out = pd.DataFrame({
        "merchant_id": parsed.apply(lambda d: _norm_merchant(d.get("merchant_id"))),
        "cust_key":    parsed.apply(lambda d: _excel_cust_key(d.get("cust_msisdn"),
                                                              d.get("customer_id"))),
        "trx_date":    pd.to_datetime(parsed.apply(lambda d: d.get("date_trx")),
                                      errors="coerce").dt.strftime("%Y-%m-%d"),
        "action":      df["Action"].values,
    })
    out["gt_label"] = out["action"].isin(TP_ACTIONS).astype(int)

    # Drop alerts we cannot join (missing merchant or date).
    bad = out["merchant_id"].eq("") | out["trx_date"].isna()
    if bad.any():
        log.warning("dropping %d alerts with missing merchant/date", int(bad.sum()))
    out = out[~bad].reset_index(drop=True)

    log.info("loaded %d PG alerts  (%d TP, %d FP)",
             len(out), int(out.gt_label.sum()), int((out.gt_label == 0).sum()))
    return out


# ── scoring ──────────────────────────────────────────────────────────────────

def reconstruction_error(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    pred = model.predict(X, batch_size=batch_size, verbose=0)
    if isinstance(pred, list):  # tolerate an old multi-head model
        pred = np.concatenate([p if isinstance(p, np.ndarray) else p.numpy()
                               for p in pred], axis=1)
    return np.sqrt(np.mean(np.square(X - pred), axis=1))


def _norm_keys_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["merchant_id"] = df["merchant_id"].astype(str).str.strip()
    df["cust_key"]    = df["cust_key"].astype(str).str.strip()
    df["trx_date"]    = pd.to_datetime(df["trx_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def score_labeled(model, data_paths: List[str], features: List[str], kwargs: Dict,
                  alerts: pd.DataFrame) -> pd.DataFrame:
    """Score only transactions whose (merchant_id, trx_date) matches an alert.

    Pre-filtering on (merchant, date) keeps this fast — we never score the full
    50M-row dataset, only the few alert-related transactions. Final label uses
    the full (merchant, cust_key, date) key.
    """
    # Candidate coarse filter and the exact TP/FP key sets.
    cand_md: Set[Tuple[str, str]] = set(zip(alerts.merchant_id, alerts.trx_date))
    tp_keys: Set[Tuple[str, str, str]] = set(
        zip(alerts[alerts.gt_label == 1].merchant_id,
            alerts[alerts.gt_label == 1].cust_key,
            alerts[alerts.gt_label == 1].trx_date))
    fp_keys: Set[Tuple[str, str, str]] = set(
        zip(alerts[alerts.gt_label == 0].merchant_id,
            alerts[alerts.gt_label == 0].cust_key,
            alerts[alerts.gt_label == 0].trx_date))

    parts = []
    for path in data_paths:
        shards = sorted(f for f in os.listdir(path) if f.endswith(".parquet"))
        log.info("scanning %d shards in %s", len(shards), path)
        for n, fname in enumerate(shards, 1):
            raw = pd.read_parquet(os.path.join(path, fname))
            keyed = _norm_keys_feature(raw[JOIN_KEYS])
            md = list(zip(keyed.merchant_id, keyed.trx_date))
            mask = np.array([t in cand_md for t in md])
            if not mask.any():
                continue
            sub = raw[mask].copy()
            subk = keyed[mask]
            full = list(zip(subk.merchant_id, subk.cust_key, subk.trx_date))
            label = np.array([1 if k in tp_keys else (0 if k in fp_keys else -1)
                              for k in full])
            keep = label != -1          # drop rows matching merchant+date but no alert cust
            if not keep.any():
                continue
            sub, subk, label = sub[keep], subk[keep], label[keep]

            X = preprocess(sub.copy(), **kwargs).values  # kwargs already holds `features`
            meta = subk.reset_index(drop=True)
            meta["trx_id"] = sub["trx_id"].values if "trx_id" in sub else np.nan
            meta["gt_label"] = label
            meta["recon_error"] = reconstruction_error(model, X)
            parts.append(meta)
            if n % 20 == 0 or n == len(shards):
                log.info("  %s: %d/%d shards  (%d labelled rows so far)",
                         os.path.basename(path), n, len(shards),
                         sum(len(p) for p in parts))

    if not parts:
        return pd.DataFrame(columns=JOIN_KEYS + ["trx_id", "gt_label", "recon_error"])
    scored = pd.concat(parts, ignore_index=True)
    log.info("scored %d labelled transactions (%d fraud, %d clean)",
             len(scored), int((scored.gt_label == 1).sum()),
             int((scored.gt_label == 0).sum()))
    return scored


def score_proxy_sample(model, val_path: str, features: List[str], kwargs: Dict,
                       max_shards: int) -> pd.DataFrame:
    """Score a sample of val shards (full population) for the proxy-label tier."""
    shards = sorted(f for f in os.listdir(val_path) if f.endswith(".parquet"))[:max_shards]
    log.info("proxy tier: scoring %d val shards (full population)", len(shards))
    parts = []
    for fname in shards:
        raw = pd.read_parquet(os.path.join(val_path, fname))
        meta = pd.DataFrame({c: raw[c].values for c in PROXY_LABELS if c in raw.columns})
        X = preprocess(raw.copy(), **kwargs).values  # kwargs already holds `features`
        meta["recon_error"] = reconstruction_error(model, X)
        parts.append(meta)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ── reports ──────────────────────────────────────────────────────────────────

def _auc(labels, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def gt_metrics(df: pd.DataFrame, threshold, title: str) -> None:
    from sklearn.metrics import (average_precision_score,
                                  precision_recall_fscore_support, confusion_matrix)
    print("\n" + "-" * 68)
    print(f"[{title}]  labelled trx: {len(df):,}  "
          f"fraud: {int((df.gt_label == 1).sum()):,}  "
          f"clean: {int((df.gt_label == 0).sum()):,}")
    y, err = df.gt_label.values, df.recon_error.values
    n_pos = int(y.sum())
    if len(df) == 0:
        print("  (no rows)")
        return
    if n_pos == 0 or n_pos == len(y):
        print("  only one class present — ranking metrics undefined")
    else:
        print(f"  ROC-AUC           : {_auc(y, err):.3f}")
        print(f"  PR-AUC (avg prec) : {average_precision_score(y, err):.3f}")
        order = np.argsort(err)[::-1]
        for k in [100, 500, n_pos]:
            if k <= len(y):
                tag = " (=#pos -> recall@P)" if k == n_pos else ""
                print(f"  precision@{k:<6}  : {y[order[:k]].mean():.3f}{tag}")
    if threshold is not None and n_pos not in (0, len(y)):
        pred = (err >= threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary",
                                                       zero_division=0)
        cm = confusion_matrix(y, pred)
        print(f"  @threshold {threshold:.4f}: precision {p:.3f}  recall {r:.3f}  f1 {f1:.3f}")
        print(f"  confusion [[TN FP][FN TP]]: {cm.tolist()}")


def proxy_report(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print("\n" + "=" * 68)
    print("TIER 1 — PROXY LABELS (weak in-data signals, full population)")
    print("=" * 68)
    print(f"{'label':<22}{'n_pos':>10}{'AUC':>8}{'mean+':>10}{'mean-':>10}{'ratio':>8}")
    err = df["recon_error"].values
    for col in PROXY_LABELS:
        if col not in df.columns:
            continue
        y = df[col].fillna(0).astype(int).values
        if y.sum() in (0, len(y)):
            print(f"{col:<22}{'(no variation)':>26}")
            continue
        mp, mn = err[y == 1].mean(), err[y == 0].mean()
        print(f"{col:<22}{int(y.sum()):>10}{_auc(y, err):>8.3f}"
              f"{mp:>10.3f}{mn:>10.3f}{(mp / mn if mn else float('nan')):>8.2f}")


def coverage_report(alerts: pd.DataFrame, scored: pd.DataFrame) -> None:
    """How many analyst alerts actually matched a transaction (join health)."""
    print("\n" + "=" * 68)
    print("JOIN COVERAGE (analyst alert -> transactions)")
    print("=" * 68)
    matched = set(zip(scored.merchant_id, scored.cust_key, scored.trx_date))
    for lbl, name in [(1, "TP (fraud)"), (0, "FP (clean)")]:
        a = alerts[alerts.gt_label == lbl]
        keys = set(zip(a.merchant_id, a.cust_key, a.trx_date))
        hit = sum(1 for k in keys if k in matched)
        print(f"  {name:<12}: {hit}/{len(keys)} alert-keys matched a transaction")
    print("  (unmatched alerts are usually masked/mismatched msisdn — expected "
          "for\n   non-placeholder customers; see notes.)")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate autoencoder vs analyst labels")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--xlsx", required=True, help="Path to MonitoringResult_*.xlsx")
    ap.add_argument("--data-paths", nargs="+",
                    default=[config.TRAIN_PATH, config.VAL_PATH],
                    help="Dirs of parquet shards to search for labelled transactions")
    ap.add_argument("--artifacts", default=config.MODEL_PATH)
    ap.add_argument("--period-split", default=DEFAULT_PERIOD_SPLIT,
                    help="trx_date >= this is the held-out val period")
    ap.add_argument("--proxy-val-path", default=config.VAL_PATH,
                    help="Dir to sample for the proxy tier")
    ap.add_argument("--proxy-shards", type=int, default=10,
                    help="val shards to sample for proxy tier (0 to skip)")
    ap.add_argument("--out", default="eval_out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    features, kwargs, threshold = load_artifacts(args.artifacts)
    model = load_model(args.model_path)
    log.info("model input %s, %d features", model.input_shape, len(features))

    alerts = load_alerts(args.xlsx)

    # ── Tier 2: analyst ground truth ─────────────────────────────────────────
    scored = score_labeled(model, args.data_paths, features, kwargs, alerts)
    if not scored.empty:
        scored.to_parquet(os.path.join(args.out, "labelled_scores.parquet"), index=False)

    print("\n" + "=" * 68)
    print("TIER 2 — ANALYST GROUND TRUTH (reconstruction error vs gt_label)")
    print("=" * 68)
    print("NOTE: analyst-reviewed rows are alerts (biased subset). AUC reads as\n"
          "'among alerts, does error rank real fraud above false positives?'")
    if scored.empty:
        print("\nNo labelled transactions joined. Check merchant/date/msisdn formats.")
    else:
        coverage_report(alerts, scored)
        val = scored[scored.trx_date >= args.period_split]
        trn = scored[scored.trx_date <  args.period_split]
        gt_metrics(scored, threshold, "FULL May-Oct  (16/20 TP in train window — biased low)")
        gt_metrics(val,    threshold, f"VAL  trx_date >= {args.period_split}  (held-out)")
        gt_metrics(trn,    threshold, f"TRAIN-period  trx_date < {args.period_split}")

    # ── Tier 1: proxy labels ─────────────────────────────────────────────────
    if args.proxy_shards > 0:
        proxy = score_proxy_sample(model, args.proxy_val_path, features, kwargs,
                                   args.proxy_shards)
        proxy_report(proxy)

    print(f"\noutputs in {args.out}/ (labelled_scores.parquet)")
    print("done.")


if __name__ == "__main__":
    main()
