"""T1: train the citation ranker. Predict whether an AI answer cites a page for a
query, from raw rank + structural SEO features + query-page semantic similarity.

The thesis lives in the eval: does content signal beat "just trust the search
rank"? So every metric is reported as LIFT over the rank-only baseline. Split by
query_id (never by pair). Registers every run with a card, metrics JSON, and plots.

Reads the FGs directly and joins in pandas (small data); the no-skew guarantee is
the shared cited_features.feature_row(). Runs on cited-content-env (needs no
fastembed itself, but pandas/sklearn suffice). Deploy: python deploy_train.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

import hopsworks
from cited_features import FEATURES, feature_row

ASSETS = Path(__file__).resolve().parent / "assets"
MODEL_NAME = "cited_ranker"


def load_joined(fs) -> pd.DataFrame:
    serp = fs.get_feature_group("serp_capture", version=1).read()
    serp = serp.sort_values("captured_at").drop_duplicates(["query_id", "url"], keep="last")
    pages = fs.get_feature_group("page_features", version=1).read()
    pages = pages.sort_values("fetched_at").drop_duplicates("url", keep="last")
    queries = fs.get_feature_group("query_features", version=1).read()
    queries = queries.sort_values("fetched_at").drop_duplicates("query_id", keep="last")
    df = serp.merge(pages, on="url", how="left").merge(queries, on="query_id", how="left")
    print(f"joined {len(df)} rows, {df['query_id'].nunique()} queries, "
          f"cited rate {df['cited'].mean():.3f}, "
          f"page_features hit {df['fetch_ok'].notna().mean():.2f}", flush=True)
    return df


def build_matrix(df: pd.DataFrame):
    rows = [feature_row(rec) for rec in df.to_dict("records")]
    X = pd.DataFrame(rows, columns=list(FEATURES)).astype("float32")
    y = df["cited"].astype(int).to_numpy()
    groups = df["query_id"].to_numpy()
    return X, y, groups


def ranking_metrics(df: pd.DataFrame, score_col: str, k: int = 3) -> dict:
    """Per-query precision@k and NDCG@k for a scoring column (higher = cited).
    Averaged over queries that have >= 1 candidate."""
    precs, ndcgs, hits = [], [], []
    for _, g in df.groupby("query_id"):
        g = g.sort_values(score_col, ascending=False)
        rel = g["cited"].to_numpy()
        if rel.sum() == 0:
            continue
        topk = rel[:k]
        precs.append(topk.mean())
        dcg = (topk / np.log2(np.arange(2, len(topk) + 2))).sum()
        ideal = np.sort(rel)[::-1][:k]
        idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        hits.append(1.0 if topk.sum() > 0 else 0.0)
    return {
        f"precision_at_{k}": float(np.mean(precs)),
        f"ndcg_at_{k}": float(np.mean(ndcgs)),
        f"hit_at_{k}": float(np.mean(hits)),
        "queries_scored": len(precs),
    }


def plots(y_true, p_model, holdout: pd.DataFrame, imp, feat_names) -> None:
    ASSETS.mkdir(exist_ok=True)
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y_true, p_model)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUROC {roc_auc_score(y_true, p_model):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("false positive rate"); plt.ylabel("true positive rate")
    plt.title("Cited-or-buried: ROC"); plt.legend(); plt.tight_layout()
    plt.savefig(ASSETS / "roc.png", dpi=120); plt.close()

    prec, rec, _ = precision_recall_curve(y_true, p_model)
    plt.figure(figsize=(5, 5))
    plt.plot(rec, prec, lw=2, label=f"AP {average_precision_score(y_true, p_model):.3f}")
    plt.axhline(y_true.mean(), ls="--", color="gray", label=f"base rate {y_true.mean():.3f}")
    plt.xlabel("recall"); plt.ylabel("precision")
    plt.title("Cited-or-buried: PR"); plt.legend(); plt.tight_layout()
    plt.savefig(ASSETS / "pr.png", dpi=120); plt.close()

    order = np.argsort(imp)[::-1]
    plt.figure(figsize=(6, 5))
    plt.barh([feat_names[i] for i in order][::-1], imp[order][::-1])
    plt.title("Permutation importance (holdout)"); plt.tight_layout()
    plt.savefig(ASSETS / "importance.png", dpi=120); plt.close()

    # the divergence money-shot: citation rate by raw search rank
    by_rank = holdout.groupby("rank")["cited"].mean()
    plt.figure(figsize=(6, 4))
    plt.bar(by_rank.index, by_rank.values, color="#c0392b")
    plt.xlabel("raw search rank"); plt.ylabel("citation rate")
    plt.title("Rank does not decide citation (the gap is the product)")
    plt.tight_layout(); plt.savefig(ASSETS / "rank_vs_cited.png", dpi=120); plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-register", action="store_true")
    args = ap.parse_args()

    project = hopsworks.login()
    fs = project.get_feature_store()
    df = load_joined(fs)
    X, y, groups = build_matrix(df)

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(X, y, groups))
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, random_state=args.seed,
    )
    clf.fit(X.iloc[tr], y[tr])
    p = clf.predict_proba(X.iloc[te])[:, 1]

    hold = df.iloc[te].copy()
    hold["model_score"] = p
    hold["rank_score"] = -hold["rank"].astype(float)  # lower rank = higher score

    auroc = roc_auc_score(y[te], p)
    ap_score = average_precision_score(y[te], p)
    m_model = ranking_metrics(hold, "model_score", k=3)
    m_rank = ranking_metrics(hold, "rank_score", k=3)
    rank_corr = float(np.corrcoef(hold["rank"], hold["cited"])[0, 1])

    metrics = {
        "auroc": float(auroc),
        "ap": float(ap_score),
        "base_rate": float(y[te].mean()),
        "precision_at_3": m_model["precision_at_3"],
        "ndcg_at_3": m_model["ndcg_at_3"],
        "hit_at_3": m_model["hit_at_3"],
        "baseline_precision_at_3": m_rank["precision_at_3"],
        "baseline_ndcg_at_3": m_rank["ndcg_at_3"],
        "lift_precision_at_3": m_model["precision_at_3"] - m_rank["precision_at_3"],
        "lift_ndcg_at_3": m_model["ndcg_at_3"] - m_rank["ndcg_at_3"],
        "rank_citation_corr": rank_corr,
        "n_train": int(len(tr)), "n_holdout": int(len(te)),
        "holdout_queries": int(hold["query_id"].nunique()),
    }
    print(json.dumps(metrics, indent=2), flush=True)

    imp = permutation_importance(clf, X.iloc[te], y[te], n_repeats=8, random_state=args.seed).importances_mean
    plots(y[te], p, hold, imp, list(FEATURES))

    if args.no_register:
        return

    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    art = Path(__file__).resolve().parent / "model_out"
    art.mkdir(exist_ok=True)
    joblib.dump(clf, art / "model.joblib")
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for f in ("cited_features.py", "encoder.py"):
        (art / f).write_text((Path(__file__).resolve().parent / f).read_text())
    for p_img in ("roc.png", "pr.png", "importance.png", "rank_vs_cited.png"):
        (art / p_img).write_bytes((ASSETS / p_img).read_bytes())

    mr = project.get_model_registry()
    model = mr.python.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        description="Predicts AI-answer citation of a page for a query (GEO ranker). "
                    "Headline is lift over the rank-only baseline.",
    )
    model.save(str(art))
    print(f"registered {MODEL_NAME} v{model.version}: AUROC {auroc:.3f}, "
          f"P@3 {m_model['precision_at_3']:.3f} vs rank {m_rank['precision_at_3']:.3f} "
          f"(lift {metrics['lift_precision_at_3']:+.3f})", flush=True)


if __name__ == "__main__":
    main()
