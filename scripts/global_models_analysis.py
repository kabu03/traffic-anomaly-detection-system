import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.config import TRAIN_LEN

# --- Config ---
ITERATION = 1
STREAM_PARTITIONS_PATH = Path("data/stream_partitions.json")
OUT_DIR = Path("visualizations/global_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_SMOOTHED = True           # use *_smoothed columns if available
SAMPLE_N = 10_000            
MAX_STATIONS_OVERLAY = 10     # limit for per-station overlays (legibility)
USE_TSNE = False              # optional: set True to also run t-SNE (requires sklearn>=1.2)

plt.rcParams.update({
    "figure.dpi": 150, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
})

def _save(fig, name):
    p = OUT_DIR / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(p, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

def load_iteration_tuning_paths(iteration_id=ITERATION):
    with open(STREAM_PARTITIONS_PATH, "r") as f:
        data = json.load(f)
    return data[f"iteration_{iteration_id}"]["tuning_streams"]

def parse_stream_id(path_str):
    m = re.search(r"incident_(\d+)\.csv", path_str)
    return int(m.group(1)) if m else -1

def load_concat_tuning(tuning_paths):
    records = []
    for p in tuning_paths:
        df = pd.read_csv(p)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        stream_id = parse_stream_id(p)
        df['part'] = np.where(df.index < TRAIN_LEN, 'TRAIN', 'HOLDOUT')
        df['stream_id'] = stream_id
        df['stream_path'] = p
        needed = ['timestamp','speed','speed_smoothed','occ','occ_smoothed','station_id','is_incident','part','stream_id','stream_path']
        present = [c for c in needed if c in df.columns]
        records.append(df[present])
    return pd.concat(records, ignore_index=True)

def _choose_cols(df):
    spd_col = 'speed_smoothed' if (USE_SMOOTHED and 'speed_smoothed' in df.columns) else 'speed'
    occ_col = 'occ_smoothed' if (USE_SMOOTHED and 'occ_smoothed' in df.columns) else 'occ'
    if spd_col not in df.columns or occ_col not in df.columns:
        raise ValueError(f"Missing required columns: {spd_col}, {occ_col}")
    return spd_col, occ_col

def _sample(df, n=SAMPLE_N, random_state=42):
    return df.sample(n=min(n, len(df)), replace=False, random_state=random_state) if len(df) > n else df

def _station_palette(stations):
    uniq = pd.Index(stations).astype(str).unique()
    # use husl to accommodate many stations
    pal = sns.color_palette("husl", len(uniq))
    return {sid: pal[i] for i, sid in enumerate(uniq)}

def option_A_scatter(train, spd_col, occ_col):
    # raw scatter
    df_plot = train.dropna(subset=[spd_col, occ_col, 'station_id']).copy()
    df_plot['station_id'] = df_plot['station_id'].astype(str)
    df_s = _sample(df_plot, SAMPLE_N)
    pal = _station_palette(df_s['station_id'])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.scatterplot(data=df_s, x=spd_col, y=occ_col, hue='station_id', s=8, alpha=0.5, palette=pal, ax=ax, linewidth=0)
    ax.set_title('Speed vs Occupancy (raw) — TRAIN pooled')
    ax.set_xlabel('Speed (mph)')
    ax.set_ylabel('Occupancy (%)' if 'occ' in occ_col else 'Occupancy')
    ax.legend(title='Station', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, ncol=1)
    ax.grid(alpha=0.3, linestyle='--')
    _save(fig, "A_scatter_speed_vs_occ_raw")

    # scaled scatter
    scaler = StandardScaler()
    X = df_plot[[spd_col, occ_col]].to_numpy(dtype=float)
    Xz = scaler.fit_transform(X)
    df_plot['speed_z'] = Xz[:, 0]
    df_plot['occ_z'] = Xz[:, 1]
    df_s2 = _sample(df_plot, SAMPLE_N)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.scatterplot(data=df_s2, x='speed_z', y='occ_z', hue='station_id', s=8, alpha=0.5, palette=pal, ax=ax, linewidth=0)
    ax.set_title('Speed vs Occupancy (standardized) — TRAIN pooled')
    ax.set_xlabel('Speed (z-score)')
    ax.set_ylabel('Occupancy (z-score)')
    ax.legend(title='Station', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, ncol=1)
    ax.grid(alpha=0.3, linestyle='--')
    _save(fig, "A_scatter_speed_vs_occ_scaled")

def option_B_pca(train, spd_col, occ_col):
    # add hour-of-day cyclic enc (optional third signal)
    dfp = train.dropna(subset=[spd_col, occ_col]).copy()
    if 'timestamp' in dfp.columns and dfp['timestamp'].notna().any():
        dfp['hour'] = dfp['timestamp'].dt.hour
        dfp['h_sin'] = np.sin(2*np.pi*dfp['hour']/24.0)
        dfp['h_cos'] = np.cos(2*np.pi*dfp['hour']/24.0)
        feats = [spd_col, occ_col, 'h_sin', 'h_cos']
    else:
        feats = [spd_col, occ_col]

    X = dfp[feats].astype(float).to_numpy()
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xz)

    dfp['pc1'] = Z[:, 0]
    dfp['pc2'] = Z[:, 1]
    dfp['station_id'] = dfp['station_id'].astype(str)
    dfp_s = _sample(dfp, SAMPLE_N)
    pal = _station_palette(dfp_s['station_id'])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.scatterplot(data=dfp_s, x='pc1', y='pc2', hue='station_id', s=8, alpha=0.5, palette=pal, ax=ax, linewidth=0)
    ax.set_title('PCA: pooled TRAIN (color = station)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(title='Station', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, ncol=1)
    ax.grid(alpha=0.3, linestyle='--')
    _save(fig, "B_pca_station_clusters")

    if USE_TSNE:
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=30, random_state=42)
            Zt = tsne.fit_transform(Xz)
            dfp_s = _sample(dfp.assign(tsne1=Zt[:, 0], tsne2=Zt[:, 1]), SAMPLE_N)
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            sns.scatterplot(data=dfp_s, x='tsne1', y='tsne2', hue='station_id', s=8, alpha=0.5, palette=pal, ax=ax, linewidth=0)
            ax.set_title('t-SNE: pooled TRAIN (color = station)')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.legend(title='Station', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, ncol=1)
            ax.grid(alpha=0.3, linestyle='--')
            _save(fig, "B_tsne_station_clusters")
        except Exception as e:
            print(f"Skipping t-SNE due to error: {e}")

def option_C_kde2d(train, spd_col, occ_col):
    dfp = train.dropna(subset=[spd_col, occ_col]).copy()
    dfp_s = _sample(dfp, SAMPLE_N)

    # pooled 2D KDE
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.kdeplot(data=dfp_s, x=spd_col, y=occ_col, fill=True, thresh=0.02, levels=30, cmap="mako", ax=ax)
    ax.set_title('2D KDE (pooled TRAIN)')
    ax.set_xlabel('Speed (mph)')
    ax.set_ylabel('Occupancy (%)' if 'occ' in occ_col else 'Occupancy')
    ax.grid(alpha=0.3, linestyle='--')
    _save(fig, "C_kde2d_pooled")

    # per-station overlays (contours) for top stations
    top_stations = (
        dfp.groupby('station_id').size().sort_values(ascending=False).head(MAX_STATIONS_OVERLAY).index
    )
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    pal = _station_palette(top_stations.astype(str))
    for sid in top_stations:
        sub = _sample(dfp[dfp['station_id'] == sid], min(SAMPLE_N // MAX_STATIONS_OVERLAY, 3000))
        if len(sub) < 50:
            continue
        sns.kdeplot(data=sub, x=spd_col, y=occ_col, fill=False, levels=8, linewidths=1.0,
                    ax=ax, color=pal[str(sid)], label=str(sid))
    ax.set_title('2D KDE contours by station (TRAIN, top stations)')
    ax.set_xlabel('Speed (mph)')
    ax.set_ylabel('Occupancy (%)' if 'occ' in occ_col else 'Occupancy')
    ax.legend(title='Station', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, ncol=1)
    ax.grid(alpha=0.3, linestyle='--')
    _save(fig, "C_kde2d_per_station_contours")

def option_D_1d_overlays(train, spd_col, occ_col):
    # Speed KDE overlays by station
    top = train.groupby('station_id').size().sort_values(ascending=False).head(MAX_STATIONS_OVERLAY).index
    pal = _station_palette(top.astype(str))

    fig, ax = plt.subplots(figsize=(7, 4))
    for sid in top:
        sub = train[train['station_id'] == sid]
        sns.kdeplot(sub[spd_col].dropna(), ax=ax, lw=1.6, label=str(sid), color=pal[str(sid)])
    ax.set_title('Speed KDE by station (TRAIN, top stations)')
    ax.set_xlabel('Speed (mph)')
    ax.set_ylabel('Density')
    ax.legend(title='Station', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, ncol=1)
    ax.grid(alpha=0.3, linestyle='--')
    _save(fig, "D_kde_speed_per_station")

    # Occupancy KDE overlays by station
    fig, ax = plt.subplots(figsize=(7, 4))
    for sid in top:
        sub = train[train['station_id'] == sid]
        sns.kdeplot(sub[occ_col].dropna(), ax=ax, lw=1.6, label=str(sid), color=pal[str(sid)])
    ax.set_title('Occupancy KDE by station (TRAIN, top stations)')
    ax.set_xlabel('Occupancy (%)' if 'occ' in occ_col else 'Occupancy')
    ax.set_ylabel('Density')
    ax.legend(title='Station', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, ncol=1)
    ax.grid(alpha=0.3, linestyle='--')
    _save(fig, "D_kde_occ_per_station")

def keep_violin(train, spd_col):
    per_station = train.groupby('station_id')[spd_col].agg(['median','mean','std','count']).reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.violinplot(data=train, x='station_id', y=spd_col, inner=None, ax=ax, color='#4C72B0')
    sns.boxplot(data=train, x='station_id', y=spd_col, ax=ax, width=0.15, showcaps=True,
                boxprops={'facecolor':'white','alpha':0.7}, showfliers=False)
    ax.set_title('Per-Station Speed Distribution (TRAIN)')
    ax.set_xlabel('Station ID')
    ax.set_ylabel('Speed (mph)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    _save(fig, "violin_speed_per_station_train")
    return per_station

def write_summary_csv(train, spd_col, occ_col):
    per_station = train.groupby('station_id').agg(
        mean_speed=(spd_col,'mean'),
        median_speed=(spd_col,'median'),
        mean_occ=(occ_col,'mean'),
        median_occ=(occ_col,'median'),
        n_rows=('station_id','size')
    ).reset_index()
    out_csv = OUT_DIR / "per_station_speed_occ_train_stats.csv"
    per_station.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

def main():
    tuning_paths = load_iteration_tuning_paths()
    print(f"Loaded {len(tuning_paths)} tuning streams (Iteration {ITERATION}).")
    df = load_concat_tuning(tuning_paths)

    spd_col, occ_col = _choose_cols(df)
    # Use TRAIN rows to mirror the global scaler/model training step
    train = df[df['part'] == 'TRAIN'].copy()
    # Basic cleaning + labels
    train[spd_col] = pd.to_numeric(train[spd_col], errors='coerce')
    train[occ_col] = pd.to_numeric(train[occ_col], errors='coerce')
    train = train.dropna(subset=[spd_col, occ_col, 'station_id'])

    # Keep the per-station speed distribution visualization
    keep_violin(train, spd_col)

    # Option A — 2D scatter raw and standardized
    option_A_scatter(train, spd_col, occ_col)

    # Option B — PCA (and optional t-SNE)
    option_B_pca(train, spd_col, occ_col)

    # Option C — 2D KDE pooled + per-station contour overlays
    option_C_kde2d(train, spd_col, occ_col)

    # Option D — 1D KDE overlays for speed and occupancy
    option_D_1d_overlays(train, spd_col, occ_col)

    # Output 5 — per-station summary CSV
    write_summary_csv(train, spd_col, occ_col)

if __name__ == "__main__":
    main()