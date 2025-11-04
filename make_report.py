#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

METRIC_BLOCK = ["TOTAL","SmoothL1","MAE","FL","SSIM"]
DERIVED = ["Precision","Recall","F1","CSI","FAR","HSS","ETS"]

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _nice(s):
    return s.replace("_"," ").replace("dBZ"," dBZ")

def load_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")
    df = pd.read_csv(csv_path)
    # epoch potrebbe essere int/str
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").fillna(0).astype(int)
    df["split"] = df["split"].astype(str)
    return df

def infer_thresholds(df):
    thr = []
    for c in df.columns:
        if c.endswith("_TP") and "dBZ_" in c:
            thr.append(int(c.split("dBZ_")[0]))
    thr = sorted(set(thr))
    return thr

def plot_train_val(ax, df, metric, title=None):
    # df columns: epoch, split, metric
    piv = df.pivot(index="epoch", columns="split", values=metric)
    piv = piv.sort_index()
    if "train" in piv.columns: ax.plot(piv.index, piv["train"], label="train")
    if "val" in piv.columns:   ax.plot(piv.index, piv["val"],   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    if title: ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

def extract_thr_block(df, thr, split):
    # ritorna DataFrame con epoch + DERIVED metriche per soglia/split
    cols = {}
    for k in ["TP","FP","TN","FN"] + DERIVED:
        col = f"{thr}dBZ_{k}"
        if col not in df.columns:
            continue
        cols[k] = col
    out = df.loc[df["split"]==split, ["epoch"] + list(cols.values())].copy()
    # rinomina a nomi corti
    out = out.rename(columns={v:k for k,v in cols.items()})
    return out.sort_values("epoch")

def best_epoch_by_val_total(df):
    sub = df[df["split"]=="val"][["epoch","TOTAL"]].sort_values(["TOTAL","epoch"])
    if sub.empty:
        return None
    best_row = sub.iloc[0]
    return int(best_row["epoch"])

def make_overview_plots(df, outdir, pdf=None, title_prefix=""):
    for m in METRIC_BLOCK:
        fig, ax = plt.subplots(figsize=(7,4))
        plot_train_val(ax, df, m, title=f"{title_prefix}{m}")
        fig.tight_layout()
        png = os.path.join(outdir, f"curve_{m}.png")
        fig.savefig(png, dpi=150)
        if pdf is not None: pdf.savefig(fig)
        plt.close(fig)

def make_threshold_plots(df, thresholds, outdir, pdf=None):
    for thr in thresholds:
        for met in DERIVED:
            fig, ax = plt.subplots(figsize=(7,4))
            # costruisci df “compattato” con colonne epoch/split/met
            rows = []
            for split in ["train","val"]:
                blk = extract_thr_block(df, thr, split)
                if met in blk.columns:
                    for _, r in blk.iterrows():
                        rows.append({"epoch": r["epoch"], "split": split, met: r[met]})
            if not rows:
                plt.close(fig); continue
            dmet = pd.DataFrame(rows)
            plot_train_val(ax, dmet, met, title=f"{thr} dBZ — {met}")
            fig.tight_layout()
            png = os.path.join(outdir, f"{thr}dBZ_{met}.png")
            fig.savefig(png, dpi=150)
            if pdf is not None: pdf.savefig(fig)
            plt.close(fig)

def make_best_epoch_table(df, thresholds, best_epoch):
    # Raccoglie: per ogni soglia, TN/FP/FN/TP + DERIVED su train/val al best_epoch
    rows = []
    for thr in thresholds:
        rec = {"epoch": best_epoch, "threshold_dBZ": thr}
        for split in ["train","val"]:
            row = df[(df["epoch"]==best_epoch) & (df["split"]==split)]
            if row.empty: 
                for k in ["TP","FP","TN","FN"]+DERIVED:
                    rec[f"{split}_{k}"] = np.nan
                continue
            for k in ["TP","FP","TN","FN"]+DERIVED:
                col = f"{thr}dBZ_{k}"
                rec[f"{split}_{k}"] = float(row.iloc[0][col]) if col in row.columns else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

def to_markdown_table(df, floatfmt="{:.4f}"):
    def fmt(x):
        if pd.isna(x): return ""
        if isinstance(x, (int, np.integer)): return str(int(x))
        if isinstance(x, (float, np.floating)): return floatfmt.format(x)
        return str(x)
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep  = "| " + " | ".join(["---"]*len(cols)) + " |"
    lines = [head, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="Cartella run (contiene metrics_per_epoch.csv)")
    ap.add_argument("--csv", default=None, help="Override path CSV (se diverso)")
    ap.add_argument("--outdir", default=None, help="Dove salvare report (default: <logdir>/report)")
    ap.add_argument("--title", default="Training Report", help="Titolo per i grafici/PDF")
    args = ap.parse_args()

    csv_path = args.csv or os.path.join(args.logdir, "metrics_per_epoch.csv")
    outdir = args.outdir or os.path.join(args.logdir, "report")
    _ensure_dir(outdir)

    df = load_csv(csv_path)
    thresholds = infer_thresholds(df)
    if not thresholds:
        print("Nessuna soglia dBZ trovata nel CSV (colonne tipo '<thr>dBZ_TP'). Esco.")
        return

    # PDF multipagina
    pdf_path = os.path.join(outdir, "report_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        # 1) curve globali
        make_overview_plots(df, outdir, pdf=pdf, title_prefix=f"{args.title} — ")

        # 2) curve per soglia
        make_threshold_plots(df, thresholds, outdir, pdf=pdf)

    # 3) Best epoch su validation (TOTAL minimo)
    be = best_epoch_by_val_total(df)
    if be is None:
        print("Impossibile determinare il best epoch (nessuna riga 'val').")
        return

    best_tbl = make_best_epoch_table(df, thresholds, be)

    # salva CSV & Markdown riassuntivi
    best_csv = os.path.join(outdir, f"best_epoch_{be}_summary.csv")
    best_tbl.to_csv(best_csv, index=False)

    md = []
    md.append(f"# {args.title}\n")
    md.append(f"- **Logdir**: `{args.logdir}`")
    md.append(f"- **Best epoch (val TOTAL min)**: `{be}`\n")
    md.append("## Metriche per soglia (train/val) al best epoch\n")
    md.append(to_markdown_table(best_tbl))
    md_path = os.path.join(outdir, "report_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("Report creato:")
    print(" - PDF:", pdf_path)
    print(" - Best-epoch CSV:", best_csv)
    print(" - Markdown:", md_path)

if __name__ == "__main__":
    main()
