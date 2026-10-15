"""
evaluate.py — Evaluate a trained anomaly-detection autoencoder.

The question this answers is NOT "is val loss low?" but "does reconstruction
error actually separate fraud from normal?". It scores every val transaction by
reconstruction error and checks whether that error ranks true fraud above
non-fraud, using two label sources:

  Tier 1 — Proxy labels (always available, live inside the val parquet):
            is_decline, is_new_user_flag, c_new_merchant_flag.
            Weak signals, but need no external data.

  Tier 2 — Analyst ground truth (BigQuery), joined on
            (merchant_id, cust_key, trx_date). This is the real metric.

Preprocessing is done with the trainer's OWN preprocess() + the saved
precomputed.json / columns_info.json, so the feature matrix is byte-identical
to what the model trained on. Any drift here would invalidate the errors.

SECURITY: BigQuery access is READ-ONLY. The script only runs a SELECT against
the analyst table. It never creates, updates, or deletes anything.

Run from the repo root (same place you run `python -m trainer.task`):

    python -m trainer.evaluate \
        --model-path  model/autoencoder-pg-retrain2026_20261015_retrain2026.keras \
        --val-path    /home/jupyter/retraining/data/test \
        --artifacts   model \
        --project-id  finnet-data-platform \
        --max-shards  20
"""

import argparse
import json
import logging
import os
from typing import Dict, List

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

# ── Proxy label columns (must exist in the val parquet) ──────────────────────
# Each is a 0/1 column where 1 is weakly correlated with "more suspicious".
PROXY_LABELS = ["is_decline", "is_new_user_flag", "c_new_merchant_flag"]

# ── Analyst ground-truth source ──────────────────────────────────────────────
# NOTE: confirm this table + column layout before trusting Tier-2 numbers.
# Mirrors the GT join in bq_materialize_features.py. Action='False Positive'
# means the analyst cleared it (not fraud) → gt_label=0; anything else → 1.
DEFAULT_GT_TABLE = "finnet-data-platform.gold_data.result_dashboard_paymentgateway"
GT_QUERY_TEMPLATE = """
SELECT
  CAST(JSON_EXTRACT_SCALAR(Keterangan, '$.merchant_id') AS INT64) AS merchant_id,
  JSON_EXTRACT_SCALAR(Keterangan, '$.msisdn')                      AS cust_key,
  DATE(JSON_EXTRACT_SCALAR(Keterangan, '$.date'))                  AS trx_date,
  CASE WHEN Action IN ('False Positive') THEN 0 ELSE 1 END         AS gt_label,
  Action                                                           AS gt_action
FROM `{gt_table}`
WHERE Status = 'Complete' AND Action IS NOT NULL AND Action != ''
"""

# Keys used to join reconstruction errors back to analyst labels.
JOIN_KEYS = ["merchant_id", "cust_key", "trx_date"]


# ── artifact loading ─────────────────────────────────────────────────────────

def load_artifacts(artifacts_dir: str):
    """Load the saved feature spec, preprocessing mappings, and threshold."""
    with open(os.path.join(artifacts_dir, "columns_info.json")) as f:
        cols = json.load(f)
    with open(os.path.join(artifacts_dir, "precomputed.json")) as f:
        precomp = json.load(f)

    threshold = None
    hp_path = os.path.join(artifacts_dir, "hyperparameters.json")
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            threshold = json.load(f).get("threshold")

    # preprocess() expects periodic/ohe as (col, period) tuples. JSON stored
    # them as [col, period] lists — unpacking works either way, but normalise.
    periodic = [tuple(x) for x in cols.get("periodic-columns", [])]
    ohe      = [tuple(x) for x in cols.get("ohe-columns", [])]

    kwargs = {
        "features":              cols["features"],
        "id_columns":            cols["id-columns"],
        "drop_columns":          cols["drop-columns"],
        "impute_columns":        cols["impute-columns"],
        "log_scale_columns":     cols["log-scale-columns"],
        "signed_log_columns":    cols.get("signed-log-columns", []),
        "stat_encoding_columns": cols["stat-encoding-columns"],
        "periodic_columns":      periodic,
        "ohe_columns":           ohe,
        "stat_mapping":          precomp["stat_mapping"],
        "ohe_class_names":       precomp["ohe_class_names"],
    }
    return cols["features"], kwargs, threshold


def load_model(model_path: str):
    from tensorflow import keras
    # compile=False: we only need the forward pass, and this avoids having to
    # supply the custom `mape` metric used at train time.
    return keras.models.load_model(model_path, compile=False)


# ── scoring ──────────────────────────────────────────────────────────────────

def reconstruction_error(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """RMSE per row — same formula find_threshold() uses to set the threshold."""
    pred = model.predict(X, batch_size=batch_size, verbose=0)
    if isinstance(pred, list):  # tolerate an old multi-head model
        pred = np.concatenate([p if isinstance(p, np.ndarray) else p.numpy()
                               for p in pred], axis=1)
    return np.sqrt(np.mean(np.square(X - pred), axis=1))


def score_val(model, val_path: str, features: List[str], kwargs: Dict,
              max_shards: int = None) -> pd.DataFrame:
    """Score every val shard. Returns ids + proxy labels + reconstruction error."""
    shards = sorted(f for f in os.listdir(val_path) if f.endswith(".parquet"))
    if max_shards is not None:
        shards = shards[:max_shards]
    log.info("scoring %d val shards from %s", len(shards), val_path)

    keep_cols = JOIN_KEYS + ["trx_id"] + PROXY_LABELS
    parts = []
    for n, fname in enumerate(shards, 1):
        raw = pd.read_parquet(os.path.join(val_path, fname))

        # Capture ids + proxies BEFORE preprocess() drops the id columns.
        meta = pd.DataFrame({c: raw[c].values for c in keep_cols if c in raw.columns})

        X = preprocess(raw.copy(), features=features, **kwargs).values
        meta["recon_error"] = reconstruction_error(model, X)
        parts.append(meta)

        if n % 5 == 0 or n == len(shards):
            log.info("  %d/%d shards  (%d rows so far)", n, len(shards),
                     sum(len(p) for p in parts))

    scored = pd.concat(parts, ignore_index=True)
    log.info("scored %d transactions", len(scored))
    return scored


# ── metrics ──────────────────────────────────────────────────────────────────

def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def proxy_report(scored: pd.DataFrame) -> None:
    """Tier 1: does error rank each proxy-positive group above the rest?"""
    print("\n" + "=" * 70)
    print("TIER 1 — PROXY LABELS (weak, in-data signals)")
    print("=" * 70)
    print(f"{'label':<22}{'n_pos':>10}{'AUC':>8}{'mean+':>10}{'mean-':>10}{'ratio':>8}")
    print("-" * 70)
    err = scored["recon_error"].values
    for col in PROXY_LABELS:
        if col not in scored.columns:
            continue
        y = scored[col].fillna(0).astype(int).values
        if y.sum() == 0 or y.sum() == len(y):
            print(f"{col:<22}{'(no variation — skipped)':>48}")
            continue
        mp, mn = err[y == 1].mean(), err[y == 0].mean()
        ratio = mp / mn if mn > 0 else float("nan")
        print(f"{col:<22}{int(y.sum()):>10}{_auc(y, err):>8.3f}"
              f"{mp:>10.3f}{mn:>10.3f}{ratio:>8.2f}")
    print("AUC > 0.5 means higher error does flag these groups. These are weak\n"
          "proxies, so 0.55–0.70 is already a healthy sign.")


def gt_report(scored: pd.DataFrame, gt: pd.DataFrame, threshold: float) -> None:
    """Tier 2: the real metric — error vs analyst ground truth."""
    from sklearn.metrics import (average_precision_score,
                                  precision_recall_fscore_support,
                                  confusion_matrix)

    merged = scored.merge(gt, on=JOIN_KEYS, how="inner")
    print("\n" + "=" * 70)
    print("TIER 2 — ANALYST GROUND TRUTH")
    print("=" * 70)
    if merged.empty:
        print("No rows matched the analyst labels on "
              f"{JOIN_KEYS}. Nothing to score.\n"
              "Check the GT table/columns and join-key dtypes.")
        return

    y = merged["gt_label"].astype(int).values
    err = merged["recon_error"].values
    n_pos, n = int(y.sum()), len(y)
    print(f"matched labelled transactions : {n:,}")
    print(f"  fraud (gt_label=1)          : {n_pos:,}")
    print(f"  cleared (gt_label=0)        : {n - n_pos:,}")
    print("NOTE: analyst-reviewed rows are a biased subset (they were flagged\n"
          "for review). Read AUC as 'among alerts, does error rank real fraud\n"
          "above false positives?'\n")

    if n_pos == 0 or n_pos == n:
        print("Only one class present — cannot compute ranking metrics.")
    else:
        print(f"ROC-AUC            : {_auc(y, err):.3f}")
        print(f"PR-AUC (avg prec)  : {average_precision_score(y, err):.3f}")
        # precision@k — the alert-queue view
        order = np.argsort(err)[::-1]
        for k in [100, 500, 1000, n_pos]:
            if k > n:
                continue
            topk = y[order[:k]]
            tag = " (=#positives → recall@P)" if k == n_pos else ""
            print(f"precision@{k:<6}    : {topk.mean():.3f}{tag}")

    if threshold is not None:
        pred = (err >= threshold).astype(int)
        print(f"\n-- At deployed threshold {threshold:.4f} (train q-{'95'}) --")
        if n_pos not in (0, n):
            p, r, f1, _ = precision_recall_fscore_support(
                y, pred, average="binary", zero_division=0)
            cm = confusion_matrix(y, pred)
            print(f"precision {p:.3f}   recall {r:.3f}   f1 {f1:.3f}")
            print(f"confusion [[TN FP][FN TP]]:\n{cm}")
        flagged = pred.mean()
        print(f"fraction flagged   : {flagged:.3f}")
    else:
        print("\n(no threshold in hyperparameters.json — skipping threshold eval)")


def fetch_gt(project_id: str, gt_table: str) -> pd.DataFrame:
    """READ-ONLY SELECT of analyst labels. Returns empty df on any failure."""
    from google.cloud import bigquery
    client = bigquery.Client(project=project_id, location="asia-southeast2")
    query = GT_QUERY_TEMPLATE.format(gt_table=gt_table)
    log.info("fetching analyst GT from %s (read-only)", gt_table)
    try:
        gt = client.query(query).result().to_dataframe()
    except Exception as e:  # table/columns may not exist yet
        log.warning("GT fetch failed: %s", e)
        return pd.DataFrame(columns=JOIN_KEYS + ["gt_label", "gt_action"])
    log.info("fetched %d analyst rows", len(gt))
    return gt


# ── join-key normalisation ───────────────────────────────────────────────────

def normalise_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Align merge-key dtypes so the inner join actually matches."""
    df = df.copy()
    if "merchant_id" in df:
        # int → clean string ("123.0" and NaN would silently break the join)
        df["merchant_id"] = pd.to_numeric(df["merchant_id"], errors="coerce")
        df["merchant_id"] = df["merchant_id"].astype("Int64").astype(str)
    if "cust_key" in df:
        df["cust_key"] = df["cust_key"].astype(str).str.strip()
    if "trx_date" in df:
        df["trx_date"] = pd.to_datetime(df["trx_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate anomaly-detection autoencoder")
    ap.add_argument("--model-path", required=True, help="Path to the .keras model file")
    ap.add_argument("--val-path", default=config.VAL_PATH, help="Dir of val parquet shards")
    ap.add_argument("--artifacts", default=config.MODEL_PATH,
                    help="Dir holding columns_info.json / precomputed.json / hyperparameters.json")
    ap.add_argument("--project-id", default="finnet-data-platform")
    ap.add_argument("--gt-table", default=DEFAULT_GT_TABLE,
                    help="BigQuery analyst table (read-only SELECT)")
    ap.add_argument("--max-shards", type=int, default=None,
                    help="Cap val shards scored (default: all)")
    ap.add_argument("--out", default="eval_out", help="Dir for CSV/PNG outputs")
    ap.add_argument("--no-gt", action="store_true", help="Skip the BigQuery GT tier")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    features, kwargs, threshold = load_artifacts(args.artifacts)
    model = load_model(args.model_path)
    log.info("loaded model: %s inputs, %s features expected",
             model.input_shape, len(features))

    scored = score_val(model, args.val_path, features, kwargs, args.max_shards)

    # Persist scores + the top anomalies for analyst follow-up.
    scored.to_parquet(os.path.join(args.out, "val_scores.parquet"), index=False)
    top = scored.sort_values("recon_error", ascending=False).head(1000)
    top.to_csv(os.path.join(args.out, "top1000_anomalies.csv"), index=False)
    log.info("wrote val_scores.parquet and top1000_anomalies.csv to %s/", args.out)

    # Error distribution summary.
    q = scored["recon_error"].quantile([0.5, 0.9, 0.95, 0.99, 0.999]).round(4)
    print("\nreconstruction-error quantiles:")
    print(q.to_string())
    if threshold is not None:
        above = (scored["recon_error"] >= threshold).mean()
        print(f"deployed threshold {threshold:.4f} flags {above:.2%} of val")

    proxy_report(scored)

    if not args.no_gt:
        gt = fetch_gt(args.project_id, args.gt_table)
        if not gt.empty:
            gt_report(normalise_keys(scored), normalise_keys(gt), threshold)
        else:
            print("\n(TIER 2 skipped — no analyst rows returned)")

    print("\ndone.")


if __name__ == "__main__":
    main()
