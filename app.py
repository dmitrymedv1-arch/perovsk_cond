# ============================================
# app.py - Ceramic Conductivity Data Explorer v4 (FULL)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.lines as mlines
import io, warnings, hashlib, itertools
from collections import Counter
from scipy import stats, interpolate, optimize, spatial
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

try:
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.decomposition import PCA
    from sklearn.linear_model import ElasticNetCV, Ridge, LinearRegression
    from sklearn.cross_decomposition import PLSRegression, CCA
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.inspection import permutation_importance, partial_dependence
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.manifold import TSNE
    from sklearn.feature_selection import mutual_info_regression
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

try:
    from umap import UMAP
    UMAP_OK = True
except Exception:
    UMAP_OK = False

try:
    import prince
    PRINCE_OK = True
except Exception:
    PRINCE_OK = False

try:
    from pygam import LinearGAM, s as gam_s
    PYGAM_OK = True
except Exception:
    PYGAM_OK = False

try:
    from SALib.sample import saltelli as sobol_saltelli
    from SALib.analyze import sobol as sobol_analyze
    SALIB_OK = True
except Exception:
    SALIB_OK = False

try:
    from skfda.preprocessing.dim_reduction import FPCA as skfda_FPCA
    from skfda.representation import FDataGrid
    SKFDA_OK = True
except Exception:
    SKFDA_OK = False

try:
    import pwlf
    PWLE_OK = True
except Exception:
    PWLE_OK = False

try:
    import dcor
    DCOR_OK = True
except Exception:
    DCOR_OK = False

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ============================================
# CONFIGURATION
# ============================================

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.labelweight': 'bold',
    'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.edgecolor': 'black', 'axes.linewidth': 1.0,
    'xtick.color': 'black', 'ytick.color': 'black',
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'legend.frameon': True,
    'legend.framealpha': 0.5, 'legend.edgecolor': 'black',
    'figure.dpi': 600, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1
})

COLOR_PALETTES = {
    'Viridis': 'viridis', 'Plasma': 'plasma', 'Inferno': 'inferno',
    'Magma': 'magma', 'Cividis': 'cividis', 'Turbo': 'turbo',
    'Jet': 'jet', 'Rainbow': 'rainbow', 'Spectral': 'Spectral',
    'Coolwarm': 'coolwarm',
    'Blues': 'Blues', 'Greens': 'Greens', 'Oranges': 'Oranges',
    'Purples': 'Purples', 'Reds': 'Reds',
    'YlGnBu': 'YlGnBu', 'YlOrRd': 'YlOrRd', 'PuBuGn': 'PuBuGn',
    'RdYlBu_r': 'RdYlBu_r', 'RdYlGn_r': 'RdYlGn_r'
}

COLUMN_LABELS = {
    'dop_cont': 'Dopant content',
    'x, wt%': 'Sintering additive concentration',
    'ρ, %': 'Density', 'd, mkm': 'Grain size',
    'tolerance_factor': 'Tolerance factor',
    'chi_B_avg': 'Average electronegativity (B-site)',
    'chi_ratio': 'Electronegativity ratio',
    'molar_mass': 'Molar mass', 'porosity': 'Porosity',
    'grain_boundary_area': 'Grain boundary area',
    'Ea_final': 'Activation energy',
    'Ea_calculated': 'Activation energy (calculated)',
    'T sin': 'Sintering temperature', 'T sin, °C': 'Sintering temperature',
}

FILTER_LABELS = {
    'Sintering additive': 'Sintering additive',
    'Atmospheres': 'Atmosphere', 'Humidity': 'Humidity',
    'Structure': 'Structure', 'Method': 'Synthesis method',
}

COMMON_MARKERS = ['o', 's', 'D', '*', '^', 'v', '<', '>', 'p', 'P', 'X']

MARKER_MAP = {
    'Sintering additive': {'Pure': 'o', 'Cu': 's', 'Ni': 'D', 'Zn': '*', 'Fe': '^',
                            'Co': 'v', 'Mn': '<', 'Cr': '>', 'Ag': 'p', 'Au': 'P', 'Pt': 'X'},
    'Atmospheres': {'Ox': 'o', 'Redox': 's', 'Inert': '^', 'Reducing': 'v', 'Air': 'D'},
    'Humidity': {'wet': 'D', 'dry': '*', 'humid': 's', 'Wet': 'D', 'Dry': '*'},
    'Structure': {'Cubic': 'o', 'orthorhombic': 's', 'hexagonal': '^',
                  'monoclinic': 'D', 'tetragonal': '*', 'rhombohedral': 'v'},
}

IONIC_RADII = {'Ba': 1.61, 'Zr': 0.72, 'Ce': 0.87, 'Sn': 0.69, 'Y': 0.90,
               'Gd': 0.94, 'Yb': 0.87, 'Sm': 0.96, 'O': 1.40}
ELECTRONEGATIVITY = {'Ba': 0.89, 'Zr': 1.33, 'Ce': 1.12, 'Sn': 1.96, 'Y': 1.22,
                     'Gd': 1.20, 'Yb': 1.10, 'Sm': 1.17, 'O': 3.44}
MOLAR_MASS = {'Ba': 137.33, 'Zr': 91.22, 'Ce': 140.12, 'Sn': 118.71, 'Y': 88.91,
              'Gd': 157.25, 'Yb': 173.05, 'Sm': 150.36, 'O': 16.00}

TEMP_COLS = [f'σ total, {t}' for t in [200, 250, 300, 350, 400, 450, 500, 550,
                                      600, 650, 700, 750, 800, 850, 900]]
TEMP_VALUES = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
NUMERIC_DESCRIPTORS = ['dop_cont', 'x, wt%', 'ρ, %', 'd, mkm',
                       'tolerance_factor', 'chi_B_avg', 'chi_ratio',
                       'molar_mass', 'porosity', 'grain_boundary_area', 'T sin']
DERIVED_DESCRIPTORS = ['x_B1', 'x_B2', 'x_dop', 'porosity', 'grain_boundary_area']

# ============================================
# CACHED COMPUTATIONS
# ============================================

@st.cache_data
def compute_descriptors_cached(df_hash, df):
    desc = pd.DataFrame(index=df.index)
    for idx, row in df.iterrows():
        x_B2 = row.get('B2_cont', 0) if pd.notna(row.get('B2_cont', np.nan)) else 0.0
        x_dop = row.get('dop_cont', 0) if pd.notna(row.get('dop_cont', np.nan)) else 0.0
        has_B2 = pd.notna(row.get('B2 cation', None)) and row.get('B2 cation', '') != ''
        x_B1 = (1.0 - x_B2 - x_dop) if has_B2 else (1.0 - x_dop)
        if not has_B2: x_B2 = 0.0
        b1 = row.get('B1 cation'); b2 = row.get('B2 cation'); dop = row.get('dopant')
        r_B = 0.0
        if x_B1 > 0 and b1 in IONIC_RADII: r_B += x_B1 * IONIC_RADII[b1]
        if x_B2 > 0 and has_B2 and b2 in IONIC_RADII: r_B += x_B2 * IONIC_RADII[b2]
        if x_dop > 0 and dop in IONIC_RADII: r_B += x_dop * IONIC_RADII[dop]
        r_O = IONIC_RADII['O']
        t = (IONIC_RADII['Ba'] + r_O) / (np.sqrt(2) * (r_B + r_O)) if r_B > 0 else np.nan
        chi = 0.0
        if x_B1 > 0 and b1 in ELECTRONEGATIVITY: chi += x_B1 * ELECTRONEGATIVITY[b1]
        if x_B2 > 0 and has_B2 and b2 in ELECTRONEGATIVITY: chi += x_B2 * ELECTRONEGATIVITY[b2]
        if x_dop > 0 and dop in ELECTRONEGATIVITY: chi += x_dop * ELECTRONEGATIVITY[dop]
        chi_ratio = chi / ELECTRONEGATIVITY['Ba'] if chi > 0 else np.nan
        mm = 0.0
        if x_B1 > 0 and b1 in MOLAR_MASS: mm += x_B1 * MOLAR_MASS[b1]
        if x_B2 > 0 and has_B2 and b2 in MOLAR_MASS: mm += x_B2 * MOLAR_MASS[b2]
        if x_dop > 0 and dop in MOLAR_MASS: mm += x_dop * MOLAR_MASS[dop]
        mm += 3 * MOLAR_MASS['O']
        rho = row.get('ρ, %', np.nan) if pd.notna(row.get('ρ, %', np.nan)) else np.nan
        d = row.get('d, mkm', np.nan) if pd.notna(row.get('d, mkm', np.nan)) else np.nan
        porosity = 100 - rho if pd.notna(rho) else np.nan
        gb = 3.722 / d if pd.notna(d) and d > 0 else np.nan
        desc.loc[idx, 'tolerance_factor'] = t
        desc.loc[idx, 'chi_B_avg'] = chi if chi > 0 else np.nan
        desc.loc[idx, 'chi_ratio'] = chi_ratio
        desc.loc[idx, 'molar_mass'] = mm
        desc.loc[idx, 'porosity'] = porosity
        desc.loc[idx, 'grain_boundary_area'] = gb
        desc.loc[idx, 'x_B1'] = x_B1
        desc.loc[idx, 'x_B2'] = x_B2
        desc.loc[idx, 'x_dop'] = x_dop
    return desc

@st.cache_data
def calculate_ea_cached(df_hash, df):
    out = []
    for _, row in df.iterrows():
        v = row.get('Ea (eV)', np.nan)
        if pd.notna(v) and v != '': out.append(float(v)); continue
        temps, sigmas = [], []
        for col, T in zip(TEMP_COLS, TEMP_VALUES):
            if col in row.index and pd.notna(row[col]) and row[col] > 0:
                temps.append(T + 273.15); sigmas.append(row[col] * 1e-3)
        if len(temps) < 2: out.append(np.nan); continue
        try:
            y = np.log(np.array(sigmas) * np.array(temps))
            x = 1000 / np.array(temps)
            slope, _ = np.polyfit(x, y, 1)
            out.append(-slope * 8.314 / 96485)
        except: out.append(np.nan)
    return out

@st.cache_data
def compute_curve_features_cached(df_hash, df):
    feats = pd.DataFrame(index=df.index)
    for idx, row in df.iterrows():
        vals = np.array([row.get(c, np.nan) for c in TEMP_COLS], dtype=float)
        T = np.array(TEMP_VALUES, dtype=float)
        mask = np.isfinite(vals) & (vals > 0)
        if mask.sum() < 3:
            feats.loc[idx, 'curve_valid'] = False
            continue
        feats.loc[idx, 'curve_valid'] = True
        v, t = vals[mask], T[mask]
        lt = t <= 500; ht = t >= 650
        if lt.sum() >= 2:
            try:
                s_, _ = np.polyfit(1000/(t[lt]+273.15), np.log(v[lt]*(t[lt]+273.15)), 1)
                feats.loc[idx, 'Ea_LT'] = -s_ * 8.314 / 96485
            except: pass
        if ht.sum() >= 2:
            try:
                s_, _ = np.polyfit(1000/(t[ht]+273.15), np.log(v[ht]*(t[ht]+273.15)), 1)
                feats.loc[idx, 'Ea_HT'] = -s_ * 8.314 / 96485
            except: pass
        if 'Ea_LT' in feats.columns and 'Ea_HT' in feats.columns:
            feats.loc[idx, 'Ea_diff'] = feats.loc[idx, 'Ea_HT'] - feats.loc[idx, 'Ea_LT']
        try:
            feats.loc[idx, 'sigma_500'] = np.interp(500, t, v)
            feats.loc[idx, 'sigma_max'] = np.max(v)
            feats.loc[idx, 'sigma_ratio'] = v[-1] / v[0] if v[0] > 0 else np.nan
        except: pass
        try:
            kernel = interpolate.interp1d(t, np.log(v), kind='linear', fill_value='extrapolate')
            tg = np.linspace(t.min(), t.max(), 200)
            d1 = np.gradient(kernel(tg), tg)
            feats.loc[idx, 'knee_T'] = tg[np.argmax(np.abs(d1))]
        except: pass
        feats.loc[idx, 'curve_area'] = np.trapezoid(np.log(v), t) if len(v) > 1 else np.nan
    return feats

# ============================================
# UTILITIES
# ============================================

def remove_outliers(df, col, n):
    if n is None or n == 0 or col not in df.columns: return df
    vals = df[col].dropna().sort_values().values
    if len(vals) <= 2 * n: return df
    low, cnt = [], 0
    for v in vals:
        if cnt >= n: break
        c = (df[col] == v).sum()
        if c <= (n - cnt): low.extend(df[df[col] == v].index.tolist()); cnt += c
        else: break
    high, cnt = [], 0
    for v in reversed(vals):
        if cnt >= n: break
        c = (df[col] == v).sum()
        if c <= (n - cnt): high.extend(df[df[col] == v].index.tolist()); cnt += c
        else: break
    return df[~df.index.isin(set(low + high))]

def get_filter_markers(df, active_count):
    if active_count != 1: return 'o', False, None
    for fname in FILTER_LABELS.keys():
        if fname in df.columns:
            vals = [v for v in df[fname].dropna().unique() if v != '']
            if len(vals) > 1:
                return {v: COMMON_MARKERS[i % len(COMMON_MARKERS)] for i, v in enumerate(vals)}, True, fname
    return 'o', False, None

def humanize_size(s): return COLUMN_LABELS.get(s, s)
def humanize_filter(f): return FILTER_LABELS.get(f, f)

def download_plot(fig, filename="plot.png"):
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
    buf.seek(0); return buf

def style_axis(ax, xlabel, ylabel, grid=True):
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    if grid: ax.grid(True, alpha=0.3, linestyle='--')

def plot_kde_marginal(ax, groups, ylabel, normalize, title):
    all_vals = np.concatenate([np.asarray(v)[np.isfinite(np.asarray(v, dtype=float))] for _, v in groups if len(v) > 0]) if groups else np.array([])
    if len(all_vals) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=10, fontweight='bold'); return
    lo, hi = np.min(all_vals), np.max(all_vals); span = hi - lo if hi > lo else 1.0
    grid = np.linspace(lo - 0.1*span, hi + 0.1*span, 200)
    cmap = plt.get_cmap('viridis'); n = max(1, len(groups))
    for i, (label, vals) in enumerate(groups):
        vals = np.asarray(vals, dtype=float); vals = vals[np.isfinite(vals)]
        if len(vals) < 2: continue
        try:
            from scipy.stats import gaussian_kde
            dens = gaussian_kde(vals)(grid)
        except: continue
        if normalize and dens.max() > 0: dens = dens / dens.max()
        c = cmap(i / max(1, n - 1)) if n > 1 else cmap(0.0)
        ax.fill_betweenx(grid, 0, dens, color=c, alpha=0.35, label=label)
        ax.plot(dens, grid, color=c, linewidth=1.5)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Normalized density' if normalize else 'Density', fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    if len(groups) > 0: ax.legend(fontsize=8, loc='best', frameon=True, framealpha=0.5)

def build_derived_features(df):
    df = df.copy()
    if 'Sintering additive' in df.columns:
        df['is_additive'] = (df['Sintering additive'] != 'Pure').astype(int)
    if 'ρ, %' in df.columns:
        df['density_norm'] = (df['ρ, %'] - df['ρ, %'].mean()) / (df['ρ, %'].std() + 1e-9)
    if 'T sin' in df.columns:
        df['Tsin_norm'] = (df['T sin'] - df['T sin'].mean()) / (df['T sin'].std() + 1e-9)
    if all(c in df.columns for c in ['x_B1', 'x_B2', 'x_dop']):
        for c in ['x_B1', 'x_B2', 'x_dop']:
            df[f'clr_{c}'] = np.log(np.clip(df[c], 1e-9, None) + 1e-6)
        if all(c in df.columns for c in ['clr_x_B1', 'clr_x_B2', 'clr_x_dop']):
            mean_clr = df[['clr_x_B1', 'clr_x_B2', 'clr_x_dop']].mean(axis=1)
            for c in ['x_B1', 'x_B2', 'x_dop']:
                df[f'clr_{c}'] = df[f'clr_{c}'] - mean_clr
        # canonical ILR basis for 3-part composition
        try:
            V = np.array([
                [1/np.sqrt(2), -1/np.sqrt(2), 0],
                [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)]
            ])
            clr = df[['clr_x_B1', 'clr_x_B2', 'clr_x_dop']].values
            ilr = clr @ V.T
            df['ilr_1'] = ilr[:, 0]
            df['ilr_2'] = ilr[:, 1]
        except: pass
    return df

# ============================================
# BASIC PLOTS
# ============================================

def create_scatter_heatmap(df, x_col, y_col, z_col, x_log, y_log, z_log,
                           palette, xlabel, ylabel, zlabel,
                           marker_style='o', show_legend=False, filter_name=None):
    d = df.dropna(subset=[x_col, y_col, z_col])
    if len(d) == 0: return None
    x, y, z = d[x_col].values, d[y_col].values, d[z_col].values
    if x_log:
        m = x > 0
        if m.sum() == 0: return None
        x, y, z, d = x[m], y[m], z[m], d[m]; xlabel = f'log10({xlabel})'; x = np.log10(x)
    if y_log:
        m = y > 0
        if m.sum() == 0: return None
        x, y, z, d = x[m], y[m], z[m], d[m]; ylabel = f'log10({ylabel})'; y = np.log10(y)
    if z_log:
        m = z > 0
        if m.sum() == 0: return None
        x, y, z, d = x[m], y[m], z[m], d[m]; zlabel = f'log10({zlabel})'; z = np.log10(z)
    fig, ax = plt.subplots(figsize=(8, 6))
    if show_legend and isinstance(marker_style, dict) and filter_name is not None:
        for cat in d[filter_name].unique():
            m = d[filter_name] == cat
            if m.sum() > 0:
                ax.scatter(x[m], y[m], c=z[m], cmap=palette, s=50, marker=marker_style.get(cat, 'o'),
                           edgecolors='black', linewidth=0.5, alpha=0.8, label=str(cat))
        sc = ax.scatter(x, y, c=z, cmap=palette, s=0, visible=False)
        cbar = plt.colorbar(sc, ax=ax); cbar.set_label(zlabel, fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', frameon=True, framealpha=0.5)
    else:
        sc = ax.scatter(x, y, c=z, cmap=palette, s=50,
                        marker=marker_style if isinstance(marker_style, str) else 'o',
                        edgecolors='black', linewidth=0.5, alpha=0.8)
        cbar = plt.colorbar(sc, ax=ax); cbar.set_label(zlabel, fontsize=11, fontweight='bold')
    style_axis(ax, xlabel, ylabel); plt.tight_layout(); return fig

def create_contour_heatmap(df, x_col, y_col, z_col, x_log, y_log, z_log,
                           palette, xlabel, ylabel, zlabel,
                           grid_resolution=50, show_lines=True, show_labels=True):
    d = df.dropna(subset=[x_col, y_col, z_col])
    if len(d) < 4: return None
    x, y, z = d[x_col].values, d[y_col].values, d[z_col].values
    if x_log:
        m = x > 0
        if m.sum() < 4: return None
        x, y, z = x[m], y[m], z[m]; x = np.log10(x); xlabel = f'log10({xlabel})'
    if y_log:
        m = y > 0
        if m.sum() < 4: return None
        x, y, z = x[m], y[m], z[m]; y = np.log10(y); ylabel = f'log10({ylabel})'
    if z_log:
        m = z > 0
        if m.sum() < 4: return None
        x, y, z = x[m], y[m], z[m]; z = np.log10(z); zlabel = f'log10({zlabel})'
    xi = np.linspace(x.min(), x.max(), grid_resolution)
    yi = np.linspace(y.min(), y.max(), grid_resolution)
    xi, yi = np.meshgrid(xi, yi)
    try: zi = interpolate.griddata((x, y), z, (xi, yi), method='cubic')
    except:
        try: zi = interpolate.griddata((x, y), z, (xi, yi), method='linear')
        except: return None
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xi, yi, zi, levels=30, cmap=palette, alpha=0.85)
    if show_lines:
        cl = ax.contour(xi, yi, zi, levels=15, colors='black', linewidths=0.5, alpha=0.4)
        if show_labels: ax.clabel(cl, inline=True, fontsize=8, fmt='%1.2f')
    ax.scatter(x, y, c='red', s=30, edgecolors='white', linewidth=1, alpha=0.7, label='Data')
    cbar = plt.colorbar(cf, ax=ax); cbar.set_label(zlabel, fontsize=11, fontweight='bold')
    style_axis(ax, xlabel, ylabel); ax.legend(loc='upper right'); plt.tight_layout(); return fig

def create_3d_surface(df, x_col, y_col, z_col, x_log, y_log, z_log,
                      palette, xlabel, ylabel, zlabel, n_bootstrap=200):
    d = df.dropna(subset=[x_col, y_col, z_col])
    if len(d) < 4: return None
    x, y, z = d[x_col].values, d[y_col].values, d[z_col].values
    if x_log:
        m = x > 0
        if m.sum() < 4: return None
        x, y, z = x[m], y[m], z[m]; x = np.log10(x); xlabel = f'log10({xlabel})'
    if y_log:
        m = y > 0
        if m.sum() < 4: return None
        x, y, z = x[m], y[m], z[m]; y = np.log10(y); ylabel = f'log10({ylabel})'
    if z_log:
        m = z > 0
        if m.sum() < 4: return None
        x, y, z = x[m], y[m], z[m]; z = np.log10(z); zlabel = f'log10({zlabel})'
    xi = np.linspace(x.min(), x.max(), 30); yi = np.linspace(y.min(), y.max(), 30)
    xi_g, yi_g = np.meshgrid(xi, yi)
    try: zi = interpolate.griddata((x, y), z, (xi_g, yi_g), method='cubic')
    except:
        try: zi = interpolate.griddata((x, y), z, (xi_g, yi_g), method='linear')
        except: return None
    bands = []
    n = len(x)
    for _ in range(min(n_bootstrap, 100)):
        idx = np.random.choice(n, n, replace=True)
        try:
            zb = interpolate.griddata((x[idx], y[idx]), z[idx], (xi_g, yi_g), method='linear')
            bands.append(zb)
        except: pass
    std = np.nanstd(np.stack(bands, axis=0), axis=0) if len(bands) > 10 else None
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi_g, yi_g, zi, cmap=palette, alpha=0.8, linewidth=0, antialiased=True)
    if std is not None:
        try:
            ax.plot_surface(xi_g, yi_g, zi + std, alpha=0.1, color='blue', linewidth=0)
            ax.plot_surface(xi_g, yi_g, zi - std, alpha=0.1, color='blue', linewidth=0)
        except: pass
    ax.scatter(x, y, z, c='red', s=30, alpha=0.7)
    ax.set_xlabel(xlabel, fontweight='bold'); ax.set_ylabel(ylabel, fontweight='bold'); ax.set_zlabel(zlabel, fontweight='bold')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10).set_label(zlabel, fontweight='bold')
    plt.tight_layout(); return fig

def create_contour_with_bands(df, x_col, y_col, z_col, x_log, y_log, z_log,
                              palette, xlabel, ylabel, zlabel,
                              grid_resolution=50, n_bootstrap=200):
    d = df.dropna(subset=[x_col, y_col, z_col])
    if len(d) < 5: return None
    x, y, z = d[x_col].values, d[y_col].values, d[z_col].values
    if x_log:
        m = x > 0
        if m.sum() < 5: return None
        x, y, z = x[m], y[m], z[m]; x = np.log10(x); xlabel = f'log10({xlabel})'
    if y_log:
        m = y > 0
        if m.sum() < 5: return None
        x, y, z = x[m], y[m], z[m]; y = np.log10(y); ylabel = f'log10({ylabel})'
    if z_log:
        m = z > 0
        if m.sum() < 5: return None
        x, y, z = x[m], y[m], z[m]; z = np.log10(z); zlabel = f'log10({zlabel})'
    xi = np.linspace(x.min(), x.max(), grid_resolution)
    yi = np.linspace(y.min(), y.max(), grid_resolution)
    xi_g, yi_g = np.meshgrid(xi, yi)
    try: zi = interpolate.griddata((x, y), z, (xi_g, yi_g), method='cubic')
    except:
        try: zi = interpolate.griddata((x, y), z, (xi_g, yi_g), method='linear')
        except: return None
    bands = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            zb = interpolate.griddata((x[idx], y[idx]), z[idx], (xi_g, yi_g), method='linear')
            bands.append(zb)
        except: pass
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xi_g, yi_g, zi, levels=30, cmap=palette, alpha=0.85)
    if len(bands) > 10:
        bands_arr = np.stack(bands, axis=0)
        std = np.nanstd(bands_arr, axis=0)
        try: ax.contour(xi_g, yi_g, std, levels=5, colors='blue', linewidths=1.0, linestyles='--')
        except: pass
    ax.scatter(x, y, c='red', s=30, edgecolors='white', linewidth=1, alpha=0.7, label='Data')
    cbar = plt.colorbar(cf, ax=ax); cbar.set_label(zlabel, fontsize=11, fontweight='bold')
    style_axis(ax, xlabel, ylabel); ax.legend(loc='upper right'); plt.tight_layout(); return fig

def create_bubble_chart(df, x_col, y_col, color_col, size_col, x_log, y_log, color_log, size_log,
                        palette, xlabel, ylabel, marker_style='o', show_legend=False,
                        filter_name=None, size_col_name=None, filter_display_name=None,
                        show_marginals=False, normalize_marginals=False,
                        size_col_raw=None, y_col_raw=None):
    d = df.dropna(subset=[x_col, y_col, color_col, size_col]).copy()
    if len(d) == 0: return None
    raw_s = d[size_col_raw].copy() if size_col_raw and size_col_raw in d.columns else d[size_col].copy()
    raw_y = d[y_col_raw].copy() if y_col_raw and y_col_raw in d.columns else d[y_col].copy()
    x, y, c, s = d[x_col].values, d[y_col].values, d[color_col].values, d[size_col].values
    if x_log:
        m = x > 0
        if m.sum() == 0: return None
        x, y, c, s = x[m], y[m], c[m], s[m]; d = d[m]; raw_s = raw_s[m]; raw_y = raw_y[m]
        x = np.log10(x); xlabel = f'log10({xlabel})'
    if y_log:
        m = y > 0
        if m.sum() == 0: return None
        x, y, c, s = x[m], y[m], c[m], s[m]; d = d[m]; raw_s = raw_s[m]; raw_y = raw_y[m]
        y = np.log10(y); ylabel = f'log10({ylabel})'
    if color_log:
        m = c > 0
        if m.sum() == 0: return None
        x, y, c, s = x[m], y[m], c[m], s[m]; d = d[m]; raw_s = raw_s[m]; raw_y = raw_y[m]
        c = np.log10(c)
    if size_log:
        m = s > 0
        if m.sum() == 0: return None
        x, y, c, s = x[m], y[m], c[m], s[m]; d = d[m]; raw_s = raw_s[m]; raw_y = raw_y[m]
        s = np.log10(s)
    if len(s) > 0:
        smin, smax = s.min(), s.max()
        s_sc = 20 + 180*(s - smin)/(smax - smin) if smax > smin else np.ones_like(s)*50
    else: s_sc = np.ones_like(s)*50

    build_marg = show_marginals and ((size_col_name is not None and size_col_raw is not None) or
                                     (filter_display_name is not None and filter_name is not None and show_legend))
    if build_marg:
        n_m = int(size_col_name is not None and size_col_raw is not None) + \
              int(filter_display_name is not None and filter_name is not None and show_legend)
        fig = plt.figure(figsize=(8 + 3*n_m, 9))
        gs = fig.add_gridspec(2, n_m, height_ratios=[3, 1.2], hspace=0.35, wspace=0.35)
        ax = fig.add_subplot(gs[0, :])
        marg_axes = [fig.add_subplot(gs[1, i]) for i in range(n_m)]
    else:
        fig, ax = plt.subplots(figsize=(8, 6)); marg_axes = []

    if show_legend and isinstance(marker_style, dict) and filter_name is not None:
        for cat in d[filter_name].unique():
            m = d[filter_name] == cat
            if m.sum() > 0:
                ax.scatter(x[m], y[m], c=c[m], s=s_sc[m], cmap=palette, marker=marker_style.get(cat, 'o'),
                           alpha=0.7, edgecolors='black', linewidth=0.5, label=str(cat))
        sc = ax.scatter(x, y, c=c, s=s_sc, cmap=palette, alpha=0.7, edgecolors='black', linewidth=0.5, visible=False)
        plt.colorbar(sc, ax=ax).set_label(d[color_col].name, fontsize=11, fontweight='bold')
        leg1 = ax.legend(loc='upper left', bbox_to_anchor=(0, 0.6), frameon=True, framealpha=0.5,
                         title=humanize_filter(filter_display_name) + ':' if filter_display_name else 'Category:')
        ax.add_artist(leg1)
    else:
        sc = ax.scatter(x, y, c=c, s=s_sc, cmap=palette, alpha=0.7, edgecolors='black', linewidth=0.5,
                        marker=marker_style if isinstance(marker_style, str) else 'o')
        plt.colorbar(sc, ax=ax).set_label(d[color_col].name, fontsize=11, fontweight='bold')
    if size_col_name is not None:
        size_title = humanize_size(size_col_name) + ':'
        if len(s) > 0: lo, med, hi = s.min(), np.percentile(s, 50), s.max()
        else: lo, med, hi = 1, 2, 3
        handles = [mlines.Line2D([0], [0], marker='o', color='w', markersize=np.sqrt(ms/2),
                                  markerfacecolor='gray', markeredgecolor='black',
                                  label=f'{lbl}: {v:.3f}')
                   for ms, lbl, v in [(20, 'Min', lo), (60, 'Median', med), (150, 'Max', hi)]]
        ax.legend(handles=handles, title=size_title, loc='upper right', frameon=True, framealpha=0.5)
    style_axis(ax, xlabel, ylabel)

    if build_marg and len(marg_axes) > 0:
        y_raw_arr = np.asarray(raw_y.values, dtype=float)
        ax_i = 0
        if size_col_name is not None and size_col_raw is not None:
            s_raw = np.asarray(raw_s.values, dtype=float)
            mask = np.isfinite(s_raw) & np.isfinite(y_raw_arr)
            sr, yr = s_raw[mask], y_raw_arr[mask]
            if len(sr) >= 3:
                q33, q66 = np.percentile(sr, 33.333), np.percentile(sr, 66.667)
                groups = [(n_, yr[m]) for n_, m in [('0–33%', sr <= q33), ('33–66%', (sr > q33) & (sr <= q66)), ('66–100%', sr > q66)] if m.sum() > 0]
                plot_kde_marginal(marg_axes[ax_i], groups, ylabel, normalize_marginals,
                                  f'Y by {humanize_size(size_col_name)} terciles')
            ax_i += 1
        if filter_display_name is not None and filter_name is not None and show_legend:
            cats = d[filter_name].values
            groups = []
            for cat in pd.unique(cats):
                vals = y_raw_arr[cats == cat]; vals = vals[np.isfinite(vals)]
                if len(vals) > 0: groups.append((str(cat), vals))
            plot_kde_marginal(marg_axes[ax_i], groups, ylabel, normalize_marginals,
                              f'Y by {humanize_filter(filter_display_name)}')
    plt.tight_layout(); return fig

# ============================================
# NEW PLOTS
# ============================================

def plot_pairplot(df, cols, hue_col=None, palette='viridis'):
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2: return None
    d = df[cols + ([hue_col] if hue_col and hue_col in df.columns else [])].dropna()
    if len(d) < 3: return None
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))
    if hue_col and hue_col in d.columns:
        cats = d[hue_col].unique(); cmap = plt.get_cmap(palette)
        colors = {c: cmap(i/max(1, len(cats)-1)) for i, c in enumerate(cats)}
    else: colors = None
    for i in range(n):
        for j in range(n):
            ax = axes[i, j] if n > 1 else axes
            if i == j: ax.hist(d[cols[i]], bins=15, alpha=0.6, edgecolor='black')
            else:
                if colors:
                    for c in d[hue_col].unique():
                        sub = d[d[hue_col] == c]
                        ax.scatter(sub[cols[j]], sub[cols[i]], s=15, alpha=0.6, c=[colors[c]],
                                   edgecolors='black', linewidth=0.3)
                else:
                    ax.scatter(d[cols[j]], d[cols[i]], s=15, alpha=0.6, c='steelblue',
                               edgecolors='black', linewidth=0.3)
                try:
                    z = np.polyfit(d[cols[j]], d[cols[i]], 1)
                    xx = np.linspace(d[cols[j]].min(), d[cols[j]].max(), 10)
                    ax.plot(xx, np.polyval(z, xx), 'r--', linewidth=0.8, alpha=0.7)
                except: pass
            if i == n - 1: ax.set_xlabel(cols[j], fontsize=8, fontweight='bold')
            if j == 0: ax.set_ylabel(cols[i], fontsize=8, fontweight='bold')
            ax.tick_params(labelsize=7)
    plt.tight_layout(); return fig

def plot_violin_box(df, y_col, group_col, log_y=False):
    d = df[[y_col, group_col]].dropna()
    if len(d) < 3: return None
    grouped = list(d.groupby(group_col))
    groups = [g[y_col].values for _, g in grouped if len(g) > 0]
    labels = [str(name) for name, g in grouped if len(g) > 0]
    if len(groups) < 1: return None
    fig, ax = plt.subplots(figsize=(max(6, 1.2*len(groups)), 5))
    if log_y:
        groups_plot = [np.log10(np.clip(g, 1e-10, None)) for g in groups]; ylab = f'log10({y_col})'
    else:
        groups_plot = groups; ylab = y_col
    parts = ax.violinplot(groups_plot, showmeans=False, showmedians=True, widths=0.7)
    for pc in parts['bodies']: pc.set_facecolor('steelblue'); pc.set_alpha(0.5); pc.set_edgecolor('black')
    bp = ax.boxplot(groups_plot, widths=0.15, patch_artist=True, showfliers=False)
    for box in bp['boxes']: box.set_facecolor('white'); box.set_edgecolor('black')
    for i, g in enumerate(groups_plot):
        jitter = np.random.normal(0, 0.05, len(g))
        ax.scatter(np.ones_like(g)*(i+1)+jitter, g, s=8, alpha=0.4, color='black')
    ax.set_xticks(range(1, len(labels)+1)); ax.set_xticklabels(labels, rotation=30, ha='right')
    style_axis(ax, group_col, ylab); plt.tight_layout(); return fig

def plot_parallel_coordinates(df, cols, hue_col=None, palette='viridis'):
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2: return None
    d = df[cols + ([hue_col] if hue_col and hue_col in df.columns else [])].dropna()
    if len(d) < 2: return None
    if SKLEARN_OK:
        norm = StandardScaler().fit_transform(d[cols])
    else:
        norm = (d[cols] - d[cols].mean()) / (d[cols].std() + 1e-9)
    fig, ax = plt.subplots(figsize=(10, 5))
    if hue_col and hue_col in d.columns:
        cats = d[hue_col].unique(); cmap = plt.get_cmap(palette)
        for i, cat in enumerate(cats):
            m = (d[hue_col] == cat).values
            for row in norm[m]:
                ax.plot(range(len(cols)), row, alpha=0.6,
                        color=cmap(i/max(1, len(cats)-1)), label=str(cat) if i < 10 else None)
        ax.legend(fontsize=8, loc='best')
    else:
        for row in norm: ax.plot(range(len(cols)), row, alpha=0.5, color='steelblue')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha='right')
    ax.set_ylabel('Normalized value', fontweight='bold'); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); return fig

def plot_radar(df, additives, cols):
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 3: return None
    group_means = df.groupby('Sintering additive')[cols].mean()
    group_means = group_means.loc[[a for a in additives if a in group_means.index]].dropna(how='all')
    if len(group_means) < 1: return None
    norm = (group_means - group_means.min()) / (group_means.max() - group_means.min() + 1e-9)
    angles = np.linspace(0, 2*np.pi, len(cols), endpoint=False).tolist(); angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    cmap = plt.get_cmap('viridis')
    for i, (name, row) in enumerate(norm.iterrows()):
        vals = row.tolist() + [row.iloc[0]]
        c = cmap(i/max(1, len(norm)-1))
        ax.plot(angles, vals, label=name, color=c, linewidth=1.5)
        ax.fill(angles, vals, color=c, alpha=0.15)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cols, fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
    plt.tight_layout(); return fig

def ternary_to_xy(a, b, c):
    total = a + b + c
    if total == 0: return 0, 0
    return 0.5*(2*b + c)/total, (np.sqrt(3)/2)*c/total

def plot_ternary(df, comp_cols, color_col, palette='viridis', log_color=False):
    comp_cols = [c for c in comp_cols if c in df.columns]
    if len(comp_cols) != 3: return None
    d = df[comp_cols + [color_col]].dropna()
    if len(d) < 2: return None
    fig, ax = plt.subplots(figsize=(7, 6.5))
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], 'k-', linewidth=1.5)
    xs, ys, cvals = [], [], []
    for _, row in d.iterrows():
        a, b, c = row[comp_cols[0]], row[comp_cols[1]], row[comp_cols[2]]
        tot = a + b + c
        if tot == 0: continue
        x, y = ternary_to_xy(a, b, c)
        xs.append(x); ys.append(y); cvals.append(row[color_col])
    cvals = np.array(cvals)
    if log_color: cvals = np.log10(np.clip(cvals, 1e-10, None))
    sc = ax.scatter(xs, ys, c=cvals, cmap=palette, s=50, edgecolors='black', linewidth=0.5)
    plt.colorbar(sc, ax=ax).set_label(('log10(' if log_color else '') + color_col + (')' if log_color else ''), fontweight='bold')
    ax.text(-0.05, -0.05, comp_cols[0], ha='right', fontsize=10, fontweight='bold')
    ax.text(1.05, -0.05, comp_cols[1], ha='left', fontsize=10, fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, comp_cols[2], ha='center', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.0); ax.set_aspect('equal'); ax.axis('off')
    plt.tight_layout(); return fig

def plot_correlation_heatmap(corr_df, title):
    fig, ax = plt.subplots(figsize=(max(6, 0.5*len(corr_df)), max(5, 0.5*len(corr_df))))
    im = ax.imshow(corr_df.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(corr_df.columns))); ax.set_xticklabels(corr_df.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(corr_df.index))); ax.set_yticklabels(corr_df.index, fontsize=9)
    for i in range(len(corr_df)):
        for j in range(len(corr_df.columns)):
            v = corr_df.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if abs(v) > 0.5 else 'black')
    plt.colorbar(im, ax=ax); ax.set_title(title, fontweight='bold')
    plt.tight_layout(); return fig

def plot_clustermap(corr_df):
    if len(corr_df) < 2: return None
    dist = 1 - corr_df.values
    try:
        Z = linkage(dist[np.triu_indices_from(dist, k=1)], method='average')
        order = leaves_list(Z)
    except: return None
    ordered = corr_df.iloc[order, order]
    fig, ax = plt.subplots(figsize=(max(6, 0.6*len(ordered)), max(5, 0.6*len(ordered))))
    im = ax.imshow(ordered.values, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(ordered.columns))); ax.set_xticklabels(ordered.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(ordered.index))); ax.set_yticklabels(ordered.index, fontsize=8)
    plt.colorbar(im, ax=ax); ax.set_title('Hierarchically clustered correlation', fontweight='bold')
    plt.tight_layout(); return fig

def plot_pca_biplot(scores, loadings, explained, labels, color_vals=None, palette='viridis'):
    fig, ax = plt.subplots(figsize=(9, 7))
    if color_vals is not None:
        sc = ax.scatter(scores[:, 0], scores[:, 1], c=color_vals, cmap=palette, s=60,
                        edgecolors='black', alpha=0.8)
        plt.colorbar(sc, ax=ax, label='Color value')
    else:
        ax.scatter(scores[:, 0], scores[:, 1], s=60, c='steelblue', edgecolors='black', alpha=0.8)
    scale = max(np.abs(scores[:, :2]).max(), 1e-6) * 0.7
    for i, lab in enumerate(labels):
        x, y = loadings[i, 0]*scale, loadings[i, 1]*scale
        ax.arrow(0, 0, x, y, color='red', alpha=0.6, head_width=scale*0.02)
        ax.text(x*1.1, y*1.1, lab, color='red', fontsize=8, fontweight='bold')
    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)', fontweight='bold')
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)' if len(explained) > 1 else 'PC2', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    plt.tight_layout(); return fig

def plot_embedding_2d(emb, title, color_vals=None, palette='viridis'):
    fig, ax = plt.subplots(figsize=(8, 6))
    if color_vals is not None:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=color_vals, cmap=palette, s=50,
                        edgecolors='black', alpha=0.8)
        plt.colorbar(sc, ax=ax)
    else:
        ax.scatter(emb[:, 0], emb[:, 1], s=50, c='steelblue', edgecolors='black', alpha=0.8)
    style_axis(ax, f'{title} 1', f'{title} 2'); ax.set_title(title, fontweight='bold')
    plt.tight_layout(); return fig

def plot_interaction(df, x_col, y_col, group_col, log_y=False):
    d = df[[x_col, y_col, group_col]].dropna()
    if len(d) < 4: return None
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = d[group_col].unique(); cmap = plt.get_cmap('viridis')
    for i, cat in enumerate(cats):
        sub = d[d[group_col] == cat].sort_values(x_col)
        if len(sub) < 2: continue
        c = cmap(i/max(1, len(cats)-1))
        y_plot = np.log10(np.clip(sub[y_col].values, 1e-10, None)) if log_y else sub[y_col].values
        ax.plot(sub[x_col], y_plot, 'o-', color=c, alpha=0.7, label=str(cat), markersize=5)
        try:
            z = np.polyfit(sub[x_col], y_plot, 1)
            xx = np.linspace(sub[x_col].min(), sub[x_col].max(), 20)
            ax.plot(xx, np.polyval(z, xx), '--', color=c, linewidth=1, alpha=0.5)
        except: pass
    style_axis(ax, x_col, ('log10(' if log_y else '') + y_col + (')' if log_y else ''))
    if len(cats) <= 12: ax.legend(fontsize=8, loc='best')
    plt.tight_layout(); return fig

def plot_forest(effects, labels, cis, title='Effect sizes'):
    if len(effects) == 0: return None
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4*len(effects))))
    y_pos = np.arange(len(effects))
    for i, (e, ci) in enumerate(zip(effects, cis)):
        ax.plot([ci[0], ci[1]], [i, i], 'k-', linewidth=1.5)
        ax.plot(e, i, 'o', color='steelblue', markersize=8, markeredgecolor='black')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Effect (mean difference)', fontweight='bold'); ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x'); plt.tight_layout(); return fig

def plot_pdp_1d(model, X, feature_idx, feature_name, grid_size=20):
    try:
        res = partial_dependence(model, X, [feature_idx], grid_resolution=grid_size)
        grid = res['grid_values'][0]; avg = res['average'][0]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(grid, avg, 'o-', color='steelblue', linewidth=2, markersize=4)
        style_axis(ax, feature_name, 'Predicted value'); plt.tight_layout(); return fig
    except: return None

def plot_pdp_2d(model, X, feat_idx_pair, feat_names, grid_size=20):
    try:
        res = partial_dependence(model, X, [feat_idx_pair], grid_resolution=grid_size)
        grid = res['grid_values']; avg = res['average'][0]
        fig, ax = plt.subplots(figsize=(8, 6))
        cf = ax.contourf(grid[0], grid[1], avg, levels=30, cmap='viridis')
        plt.colorbar(cf, ax=ax, label='Predicted value')
        style_axis(ax, feat_names[0], feat_names[1])
        ax.set_title(f'2D PDP: {feat_names[0]} × {feat_names[1]}', fontweight='bold')
        plt.tight_layout(); return fig
    except: return None

def plot_ale(feature_values, ale_values, feature_name):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(feature_values, ale_values, 'o-', color='steelblue', linewidth=2)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    style_axis(ax, feature_name, 'ALE (centered)')
    plt.tight_layout(); return fig

def plot_friedman_H(pairs, h_vals, title='Friedman H (interaction strength)'):
    if len(pairs) == 0: return None
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4*len(pairs))))
    labels = [' × '.join(p) for p in pairs]
    ax.barh(labels, h_vals, color='steelblue', edgecolor='black')
    style_axis(ax, 'H-statistic', 'Feature pair'); ax.set_title(title, fontweight='bold')
    plt.tight_layout(); return fig

def plot_curve_clusters(curves, labels, centroids):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap('viridis')
    unique = np.unique(labels)
    T = np.array(TEMP_VALUES, dtype=float)
    for i, cid in enumerate(unique):
        color = cmap(i/max(1, len(unique)-1))
        for c in curves[labels == cid]:
            if np.isfinite(c).sum() >= 3:
                ax.plot(T[np.isfinite(c)], 10**c[np.isfinite(c)], alpha=0.3, color=color, linewidth=0.8)
        cen = centroids[i]
        if np.isfinite(cen).sum() >= 3:
            ax.plot(T[np.isfinite(cen)], 10**cen[np.isfinite(cen)], color=color, linewidth=2.5,
                    label=f'Cluster {cid} (n={np.sum(labels==cid)})')
    ax.set_yscale('log'); style_axis(ax, 'Temperature (°C)', 'Conductivity σ (mS/cm)')
    ax.legend(fontsize=8); plt.tight_layout(); return fig

def plot_fpca_components(components, explained, T_vals):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap('viridis')
    for i, comp in enumerate(components[:5]):
        c = cmap(i/max(1, len(components)-1))
        ax.plot(T_vals, comp, color=c, lw=2, label=f'FPC {i+1} ({explained[i]*100:.1f}%)')
    style_axis(ax, 'Temperature (°C)', 'FPC loading'); ax.legend(fontsize=8)
    ax.set_title('FPCA components of σ(T) curves', fontweight='bold')
    plt.tight_layout(); return fig

def plot_segmented_regression(x, y, breakpoint, seg1_params, seg2_params, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.5, s=30, edgecolor='black')
    x1 = np.linspace(x.min(), breakpoint, 50); x2 = np.linspace(breakpoint, x.max(), 50)
    ax.plot(x1, np.polyval(seg1_params, x1), 'r-', lw=2, label='Segment 1')
    ax.plot(x2, np.polyval(seg2_params, x2), 'b-', lw=2, label='Segment 2')
    ax.axvline(breakpoint, color='black', ls='--', alpha=0.7, label=f'Breakpoint = {breakpoint:.3f}')
    style_axis(ax, xlabel, ylabel); ax.legend(fontsize=9)
    plt.tight_layout(); return fig

def plot_meditation_diagram(total, direct, indirect, a_path, b_path, p_total, p_direct):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.1, 0.5, 'Additive\n(X)', ha='center', va='center', bbox=dict(boxstyle='round', fc='lightblue'), fontsize=10)
    ax.text(0.5, 0.75, 'Density\n(M)', ha='center', va='center', bbox=dict(boxstyle='round', fc='lightgreen'), fontsize=10)
    ax.text(0.9, 0.5, 'log(σ)\n(Y)', ha='center', va='center', bbox=dict(boxstyle='round', fc='lightyellow'), fontsize=10)
    ax.annotate('', xy=(0.85, 0.55), xytext=(0.15, 0.55), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(0.5, 0.58, f'total = {total:.3f} (p={p_total:.3g})', ha='center', fontsize=9, fontweight='bold')
    ax.annotate('', xy=(0.47, 0.7), xytext=(0.13, 0.55), arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax.text(0.25, 0.65, f'a = {a_path:.3f}', color='blue', fontsize=9, fontweight='bold')
    ax.annotate('', xy=(0.85, 0.58), xytext=(0.53, 0.72), arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax.text(0.68, 0.7, f'b = {b_path:.3f}', color='blue', fontsize=9, fontweight='bold')
    ax.text(0.5, 0.42, f'direct = {direct:.3f} (p={p_direct:.3g})\nindirect = {indirect:.3f}',
            ha='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(0.3, 0.9); ax.axis('off')
    plt.tight_layout(); return fig

def plot_coverage_heatmap(df, col1, col2):
    d = df[[col1, col2]].dropna()
    if len(d) < 2: return None
    fig, ax = plt.subplots(figsize=(7, 5))
    h = ax.hexbin(d[col1], d[col2], gridsize=15, cmap='viridis', mincnt=1,
                  edgecolors='black', linewidth=0.3)
    plt.colorbar(h, ax=ax, label='Count'); style_axis(ax, col1, col2)
    plt.tight_layout(); return fig

def plot_convex_hull(df, col1, col2):
    d = df[[col1, col2]].dropna()
    if len(d) < 3: return None
    try: hull = ConvexHull(d.values)
    except: return None
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d[col1], d[col2], s=30, alpha=0.6, c='steelblue', edgecolor='black')
    for simplex in hull.simplices:
        ax.plot(d.values[simplex, 0], d.values[simplex, 1], 'k-', lw=1)
    ax.fill(d.values[hull.vertices, 0], d.values[hull.vertices, 1], alpha=0.1, color='green')
    style_axis(ax, col1, col2); ax.set_title('Convex hull (coverage)', fontweight='bold')
    plt.tight_layout(); return fig

def plot_convex_hull_pca(df, cols):
    if not SKLEARN_OK: return None
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2: return None
    d = df[cols].dropna()
    if len(d) < 4: return None
    try:
        X = StandardScaler().fit_transform(d)
        pca = PCA(n_components=2); scores = pca.fit_transform(X)
        hull = ConvexHull(scores)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(scores[:, 0], scores[:, 1], s=30, alpha=0.6, c='steelblue', edgecolor='black')
        for simplex in hull.simplices:
            ax.plot(scores[simplex, 0], scores[simplex, 1], 'k-', lw=1)
        ax.fill(scores[hull.vertices, 0], scores[hull.vertices, 1], alpha=0.1, color='green')
        style_axis(ax, f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                   f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Convex hull in PCA space (coverage)', fontweight='bold')
        plt.tight_layout(); return fig
    except: return None

def plot_missingness(df):
    miss = df.isna().astype(int)
    fig, ax = plt.subplots(figsize=(max(6, 0.3*len(df.columns)), max(3, 0.05*len(df))))
    ax.imshow(miss.values, aspect='auto', cmap='gray_r', interpolation='nearest')
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(0, len(df), max(1, len(df)//20)))
    ax.set_title('Missingness (black = missing)', fontweight='bold')
    plt.tight_layout(); return fig

def plot_sankey(df, source_col, mid_col, target_col, bins=2):
    d = df[[source_col, mid_col, target_col]].dropna().copy() if all(c in df.columns for c in [source_col, mid_col, target_col]) else None
    if d is None or len(d) < 3: return None
    try:
        d['_target_bin'] = pd.qcut(d[target_col], q=bins, labels=[f'Low {target_col}', f'High {target_col}'], duplicates='drop')
    except: return None
    if d['_target_bin'].isna().all(): return None
    labels = list(pd.unique(d[source_col].astype(str)))
    for v in pd.unique(d[mid_col].astype(str)):
        if v not in labels: labels.append(v)
    for v in pd.unique(d['_target_bin'].astype(str)):
        if v not in labels: labels.append(v)
    idx = {v: i for i, v in enumerate(labels)}
    links = []
    for _, row in d.iterrows():
        links.append((idx[str(row[source_col])], idx[str(row[mid_col])]))
        links.append((idx[str(row[mid_col])], idx[str(row['_target_bin'])]))
    cnt = Counter(links)
    src, tgt, val = zip(*[(s, t, v) for (s, t), v in cnt.items()])
    if PLOTLY_OK:
        try:
            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=labels),
                link=dict(source=src, target=tgt, value=val))])
            fig.update_layout(title='Sankey flow', font=dict(size=10))
            return fig
        except: pass
    # fallback to matplotlib-only Sankey
    try:
        from matplotlib.sankey import Sankey
        fig, ax = plt.subplots(figsize=(10, 5))
        n_layers = 3
        for lyr, name in enumerate([source_col, mid_col, '_target_bin']):
            unique_vals = list(pd.unique(d[name].astype(str)))
            for j, v in enumerate(unique_vals):
                y = (j + 1) / (len(unique_vals) + 1)
                ax.text(lyr / (n_layers - 1), y, v, ha='center', va='center',
                        bbox=dict(boxstyle='round', fc='lightblue'), fontsize=8)
        for _, row in d.iterrows():
            try:
                y_src = (list(pd.unique(d[source_col].astype(str))).index(str(row[source_col])) + 1) / (len(pd.unique(d[source_col])) + 1)
                y_mid = (list(pd.unique(d[mid_col].astype(str))).index(str(row[mid_col])) + 1) / (len(pd.unique(d[mid_col])) + 1)
                y_tgt = (list(pd.unique(d['_target_bin'].astype(str))).index(str(row['_target_bin'])) + 1) / (len(pd.unique(d['_target_bin'])) + 1)
                ax.plot([0, 0.5], [y_src, y_mid], alpha=0.15, color='gray')
                ax.plot([0.5, 1], [y_mid, y_tgt], alpha=0.15, color='gray')
            except: pass
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(0, 1); ax.axis('off')
        ax.set_title('Sankey flow (fallback)', fontweight='bold')
        plt.tight_layout(); return fig
    except: return None

def plot_gam_fit(X, y, gam_model, feature_idx, feature_name):
    try:
        grid = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 100)
        X_dummy = np.tile(X.mean(axis=0), (100, 1)); X_dummy[:, feature_idx] = grid
        y_pred = gam_model.predict(X_dummy)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(X[:, feature_idx], y, alpha=0.4, s=20, c='gray')
        ax.plot(grid, y_pred, 'r-', lw=2)
        style_axis(ax, feature_name, 'Predicted value'); ax.set_title(f'GAM partial effect: {feature_name}', fontweight='bold')
        plt.tight_layout(); return fig
    except: return None

def plot_cca_scores(x_scores, y_scores):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_scores[:, 0], y_scores[:, 0], s=40, edgecolors='black', alpha=0.7, c='steelblue')
    try:
        r, _ = stats.pearsonr(x_scores[:, 0], y_scores[:, 0])
        z = np.polyfit(x_scores[:, 0], y_scores[:, 0], 1)
        xx = np.linspace(x_scores[:, 0].min(), x_scores[:, 0].max(), 20)
        ax.plot(xx, np.polyval(z, xx), 'r--', lw=1.5, label=f'r = {r:.3f}')
    except: pass
    style_axis(ax, 'CCA X-score 1', 'CCA Y-score 1'); ax.legend(); ax.set_title('CCA canonical scores', fontweight='bold')
    plt.tight_layout(); return fig

def plot_dunn_posthoc(group_labels, z_values, p_values):
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4*len(group_labels))))
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    ax.barh(group_labels, z_values, color=colors, edgecolor='black')
    ax.axvline(0, color='black', lw=0.5)
    style_axis(ax, 'z-statistic', 'Pair'); ax.set_title('Dunn post-hoc (red = p < 0.05)', fontweight='bold')
    plt.tight_layout(); return fig

def plot_meta_forest(df_effects, df_ci, labels, title='Meta-forest by Reference'):
    if len(df_effects) == 0: return None
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4*len(df_effects))))
    for i, (e, ci) in enumerate(zip(df_effects, df_ci)):
        ax.plot([ci[0], ci[1]], [i, i], 'k-', linewidth=1.5)
        ax.plot(e, i, 'o', color='steelblue', markersize=8, markeredgecolor='black')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_yticks(range(len(df_effects))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Effect (mean ΔEa)', fontweight='bold'); ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x'); plt.tight_layout(); return fig

def plot_additive_vs_pure(pairs_df, y_scale='log10', heat_metric='abs_delta_rho',
                          palette='viridis', clip_range=None, facet_by=None,
                          temperature_col='σ total, 600', facet_groups=None):
    """
    Main plot: x = x, wt%; y = ratio or log10(ratio);
    marker shape encodes same_Tsin (circle vs square);
    marker fill encodes heat_metric value (empty if heat_value is NaN or zero);
    background gradient: green above no-effect line, red below.
    """
    if pairs_df is None or len(pairs_df) == 0:
        return None

    d = pairs_df.dropna(subset=['x_wt', 'ratio']).copy()
    if y_scale == 'log10':
        d = d[d['ratio'] > 0].copy()
        d['y'] = d['log10_ratio']
        y_label = 'log10(σ_with / σ_without)'
        no_effect = 0.0
    else:
        d['y'] = d['ratio']
        y_label = 'σ_with / σ_without'
        no_effect = 1.0
    if len(d) == 0:
        return None

    facets = [None]
    if facet_by and facet_groups:
        facets = facet_groups if isinstance(facet_groups, list) else [facet_groups]

    n_facets = max(1, len(facets))
    if n_facets == 1:
        fig, axes = plt.subplots(figsize=(9, 6))
        axes_list = [axes]
    else:
        fig, axes = plt.subplots(1, n_facets, figsize=(7 * n_facets, 6), squeeze=False)
        axes_list = axes[0]

    heat_label = {'abs_delta_rho': '|Δρ| (%)',
                  'abs_delta_d': '|Δd| (µm)',
                  'd_ratio': 'd_additive / d_pure'}.get(heat_metric, heat_metric)

    heat_valid = d['heat_value'].dropna()
    has_heat = len(heat_valid) > 0 and heat_valid.max() > heat_valid.min()
    if has_heat:
        norm = Normalize(vmin=heat_valid.min(), vmax=heat_valid.max())
    else:
        norm = Normalize(vmin=0, vmax=1)

    for ax, facet_val in zip(axes_list, facets):
        if facet_val is None:
            sub = d.copy()
            title = ''
        else:
            if facet_by not in d.columns:
                continue
            sub = d[d[facet_by] == facet_val].copy()
            title = f'{facet_by}: {facet_val}'

        if clip_range is not None:
            x_lo = max(d['x_wt'].min(), 0) * 0.95
            x_hi = d['x_wt'].max() * 1.05
            y_lo, y_hi = clip_range
        else:
            x_lo = max(d['x_wt'].min(), 0) * 0.95 if d['x_wt'].min() >= 0 else d['x_wt'].min() * 1.05
            x_hi = d['x_wt'].max() * 1.05
            y_lo = d['y'].min()
            y_hi = d['y'].max()
            pad = max(abs(y_lo - no_effect), abs(y_hi - no_effect)) * 0.15
            y_lo -= pad; y_hi += pad

        # background gradient
        n_grad = 200
        y_grad = np.linspace(y_lo, y_hi, n_grad).reshape(-1, 1)
        dist = np.abs(y_grad - no_effect)
        max_dist = max(np.abs(y_hi - no_effect), np.abs(y_lo - no_effect))
        if max_dist <= 0:
            max_dist = 1.0
        alpha = (dist / max_dist) * 0.35
        sign = np.sign(y_grad - no_effect)
        bg = np.zeros((n_grad, 1, 4))
        bg[sign[:, 0] > 0, 0, :] = [0.1, 0.7, 0.1, 0.0]
        bg[sign[:, 0] < 0, 0, :] = [0.85, 0.15, 0.15, 0.0]
        bg[:, :, 3] = alpha
        ax.imshow(bg, extent=[x_lo, x_hi, y_lo, y_hi], aspect='auto',
                  origin='lower', zorder=0, interpolation='bilinear')

        ax.axhline(no_effect, color='black', linestyle='--', linewidth=1.0, alpha=0.7, zorder=1)

        same_mask = sub['same_Tsin'].astype(bool)
        for grp_mask, marker, label in [(same_mask, 'o', 'Same T_sin'),
                                         (~same_mask, 's', 'Different T_sin')]:
            g = sub[grp_mask]
            if len(g) == 0: continue
            filled = g[g['heat_value'].notna() & (g['heat_value'] != 0)]
            empty = g[g['heat_value'].isna() | (g['heat_value'] == 0)]
            if len(filled) > 0:
                ax.scatter(filled['x_wt'], filled['y'],
                           c=filled['heat_value'], cmap=palette, norm=norm,
                           marker=marker, s=110, edgecolors='black',
                           linewidth=0.8, zorder=3, label=label)
            if len(empty) > 0:
                ax.scatter(empty['x_wt'], empty['y'],
                           facecolors='none', edgecolors='black',
                           marker=marker, s=110, linewidth=1.2, zorder=3,
                           label=f'{label} (no/zero Δ)')

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        style_axis(ax, 'Sintering additive concentration, wt%', y_label)
        if title: ax.set_title(title, fontweight='bold')

    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        axes_list[0].legend(handles, labels, loc='upper right', fontsize=10,
                            frameon=True, framealpha=0.7)

    if has_heat:
        sm = ScalarMappable(norm=norm, cmap=palette)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes_list if n_facets > 1 else axes_list[0],
                            fraction=0.03, pad=0.02)
        cbar.set_label(heat_label, fontsize=10, fontweight='bold')

    fig.suptitle(f'Additive vs Pure — temperature: {temperature_col}', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_ratio_vs_delta(pairs_df, delta_col='d_rho', y_scale='log10', palette='viridis'):
    """Scatter of delta (d_rho or d_d) vs ratio (or log10 ratio)."""
    if pairs_df is None or len(pairs_df) == 0:
        return None
    d = pairs_df.dropna(subset=[delta_col, 'ratio']).copy()
    if y_scale == 'log10':
        d = d[d['ratio'] > 0].copy()
        d['y'] = d['log10_ratio']
        y_label = 'log10(σ_with / σ_without)'
        no_effect = 0.0
    else:
        d['y'] = d['ratio']
        y_label = 'σ_with / σ_without'
        no_effect = 1.0
    if len(d) < 2:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    color_by_same = d['same_Tsin'].astype(bool)
    ax.scatter(d.loc[color_by_same, delta_col], d.loc[color_by_same, 'y'],
               s=90, marker='o', c='steelblue', edgecolors='black',
               label='Same T_sin', alpha=0.85)
    ax.scatter(d.loc[~color_by_same, delta_col], d.loc[~color_by_same, 'y'],
               s=90, marker='s', c='darkorange', edgecolors='black',
               label='Different T_sin', alpha=0.85)
    ax.axhline(no_effect, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    try:
        r_s, p_s = stats.spearmanr(d[delta_col], d['y'])
        z = np.polyfit(d[delta_col], d['y'], 1)
        xx = np.linspace(d[delta_col].min(), d[delta_col].max(), 20)
        ax.plot(xx, np.polyval(z, xx), 'k-', linewidth=1.2, alpha=0.6)
        ax.set_title(f'Spearman ρ = {r_s:.3f}, p = {p_s:.3g}, n = {len(d)}', fontweight='bold')
    except Exception:
        pass
    x_label = {'d_rho': 'Δρ = ρ_additive − ρ_Pure (%)',
               'd_d': 'Δd = d_additive − d_Pure (µm)'}.get(delta_col, delta_col)
    style_axis(ax, x_label, y_label)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig

def plot_ratio_boxplot(pairs_df, group_col='Additive', y_scale='log10'):
    """Box + scatter of ratio (or log10 ratio) grouped by Additive or by same_Tsin."""
    if pairs_df is None or len(pairs_df) == 0:
        return None
    d = pairs_df.dropna(subset=[group_col, 'ratio']).copy()
    if y_scale == 'log10':
        d = d[d['ratio'] > 0].copy()
        d['y'] = d['log10_ratio']
        y_label = 'log10(σ_with / σ_without)'
        no_effect = 0.0
    else:
        d['y'] = d['ratio']
        y_label = 'σ_with / σ_without'
        no_effect = 1.0
    if len(d) < 2:
        return None
    groups = [g['y'].values for _, g in d.groupby(group_col) if len(g) > 0]
    labels = [str(k) for k, g in d.groupby(group_col) if len(g) > 0]
    if len(groups) == 0:
        return None
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(groups)), 5))
    bp = ax.boxplot(groups, widths=0.5, patch_artist=True, showfliers=False)
    for box in bp['boxes']:
        box.set_facecolor('lightblue'); box.set_edgecolor('black')
    for i, g in enumerate(groups):
        jitter = np.random.normal(0, 0.05, len(g))
        ax.scatter(np.ones_like(g) * (i + 1) + jitter, g, s=30, alpha=0.6,
                   c='steelblue', edgecolors='black', linewidth=0.5)
    ax.axhline(no_effect, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    style_axis(ax, group_col, y_label)
    plt.tight_layout()
    return fig

def summarize_pairs(pairs_df, y_scale='log10'):
    """Return a dict with numeric summary for same/diff Tsin groups + Spearman correlations."""
    if pairs_df is None or len(pairs_df) == 0:
        return None
    d = pairs_df.dropna(subset=['ratio']).copy()
    if y_scale == 'log10':
        d = d[d['ratio'] > 0].copy()
        d['y'] = d['log10_ratio']
        no_effect = 0.0
    else:
        d['y'] = d['ratio']
        no_effect = 1.0
    out = {}
    for grp_name, mask in [('same', d['same_Tsin'].astype(bool)),
                           ('diff', ~d['same_Tsin'].astype(bool))]:
        g = d[mask]
        out[f'n_{grp_name}'] = len(g)
        if len(g) > 0:
            out[f'mean_ratio_{grp_name}'] = g['ratio'].mean()
            out[f'median_ratio_{grp_name}'] = g['ratio'].median()
            out[f'frac_above_1_{grp_name}'] = (g['ratio'] > 1).mean()
            try:
                w, pw = stats.wilcoxon(g['y'] - no_effect)
                out[f'wilcoxon_p_{grp_name}'] = pw
            except Exception:
                out[f'wilcoxon_p_{grp_name}'] = np.nan
        else:
            out[f'mean_ratio_{grp_name}'] = np.nan
            out[f'median_ratio_{grp_name}'] = np.nan
            out[f'frac_above_1_{grp_name}'] = np.nan
            out[f'wilcoxon_p_{grp_name}'] = np.nan
    try:
        r_rho, p_rho = stats.spearmanr(d['d_rho'].dropna(),
                                       d.loc[d['d_rho'].notna(), 'y'])
        out['spearman_rho'] = r_rho
        out['spearman_rho_p'] = p_rho
    except Exception:
        out['spearman_rho'] = np.nan; out['spearman_rho_p'] = np.nan
    try:
        r_d, p_d = stats.spearmanr(d['d_d'].dropna(),
                                   d.loc[d['d_d'].notna(), 'y'])
        out['spearman_d'] = r_d
        out['spearman_d_p'] = p_d
    except Exception:
        out['spearman_d'] = np.nan; out['spearman_d_p'] = np.nan
    return out

def plot_bootstrap_coefs(coefs_samples, coef_names):
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4*len(coef_names))))
    for i, (name, vals) in enumerate(zip(coef_names, coefs_samples.T)):
        ax.boxplot([vals], positions=[i], vert=False, widths=0.6, patch_artist=True)
    ax.set_yticks(range(len(coef_names))); ax.set_yticklabels(coef_names)
    ax.axvline(0, color='red', ls='--', alpha=0.7)
    style_axis(ax, 'Coefficient (bootstrap distribution)', 'Feature')
    ax.set_title('Bootstrap CI of coefficients', fontweight='bold')
    plt.tight_layout(); return fig

# ============================================
# ANALYSIS FUNCTIONS
# ============================================

def run_anova(df, y_col, group_col):
    d = df[[y_col, group_col]].dropna()
    if len(d) < 4: return None
    groups = [g[y_col].values for _, g in d.groupby(group_col) if len(g) > 0]
    if len(groups) < 2: return None
    try:
        f, p = stats.f_oneway(*groups)
        all_v = np.concatenate(groups); gm = np.mean(all_v)
        ss_b = sum(len(g) * (np.mean(g) - gm)**2 for g in groups)
        ss_t = np.sum((all_v - gm)**2)
        eta = ss_b / ss_t if ss_t > 0 else 0
        return {'F': f, 'p': p, 'eta_squared': eta, 'n_groups': len(groups), 'n_total': len(all_v)}
    except: return None

def run_kruskal(df, y_col, group_col):
    d = df[[y_col, group_col]].dropna()
    if len(d) < 4: return None
    groups = [g[y_col].values for _, g in d.groupby(group_col) if len(g) > 0]
    if len(groups) < 2: return None
    try:
        h, p = stats.kruskal(*groups)
        return {'H': h, 'p': p, 'n_groups': len(groups), 'n_total': len(d)}
    except: return None

def run_dunn_posthoc(df, y_col, group_col):
    d = df[[y_col, group_col]].dropna()
    if len(d) < 6: return None
    groups = list(d.groupby(group_col))
    if len(groups) < 2: return None
    try:
        from scipy.stats import rankdata
        ranks = rankdata(d[y_col].values)
        d = d.copy(); d['_rank'] = ranks
        n = len(d)
        # pairwise
        pairs, zvals, pvals = [], [], []
        cats = [g for g, _ in groups]
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                gi = d[d[group_col] == cats[i]]['_rank'].values
                gj = d[d[group_col] == cats[j]]['_rank'].values
                if len(gi) < 1 or len(gj) < 1: continue
                ri = np.mean(gi); rj = np.mean(gj)
                ni = len(gi); nj = len(gj)
                sigma2 = (n * (n + 1)) / 12.0
                denom = np.sqrt(sigma2 * (1/ni + 1/nj))
                z = (ri - rj) / denom if denom > 0 else 0
                p = 2 * (1 - stats.norm.cdf(abs(z)))
                pairs.append(f'{cats[i]} vs {cats[j]}'); zvals.append(z); pvals.append(p)
        # Bonferroni
        pvals_adj = [min(p * len(pvals), 1.0) for p in pvals]
        return pd.DataFrame({'pair': pairs, 'z': zvals, 'p_raw': pvals, 'p_adj': pvals_adj})
    except: return None

def run_tukey(df, y_col, group_col):
    if not STATSMODELS_OK: return None
    d = df[[y_col, group_col]].dropna()
    if len(d) < 6: return None
    try:
        res = pairwise_tukeyhsd(d[y_col].values, d[group_col].values, alpha=0.05)
        return res
    except: return None

def run_pairwise_comparisons(df, y_col, group_col, reference_group='Pure'):
    d = df[[y_col, group_col]].dropna()
    if reference_group not in d[group_col].values: return None
    ref = d[d[group_col] == reference_group][y_col].values
    results = []
    for cat in d[group_col].unique():
        if cat == reference_group: continue
        other = d[d[group_col] == cat][y_col].values
        if len(other) < 2 or len(ref) < 2: continue
        try:
            _, p = stats.mannwhitneyu(ref, other, alternative='two-sided')
            diff = np.mean(other) - np.mean(ref)
            se = np.sqrt(np.var(ref)/len(ref) + np.var(other)/len(other))
            ci = (diff - 1.96*se, diff + 1.96*se)
            d_cohen = diff / np.sqrt((np.var(ref) + np.var(other))/2) if (np.var(ref) + np.var(other)) > 0 else np.nan
            results.append({'group': cat, 'n': len(other), 'mean': np.mean(other),
                           'diff': diff, 'ci_low': ci[0], 'ci_high': ci[1],
                           'p': p, 'cohen_d': d_cohen})
        except: continue
    return pd.DataFrame(results) if results else None

def run_mixed_effects(df, y_col, fixed_cols, group_col='References'):
    if not STATSMODELS_OK: return None
    cols = [y_col] + fixed_cols + [group_col]
    d = df[cols].dropna()
    if len(d) < 10: return None
    try:
        formula = f'Q("{y_col}") ~ ' + ' + '.join([f'Q("{c}")' for c in fixed_cols])
        model = smf.mixedlm(formula, data=d, groups=d[group_col])
        return model.fit(disp=False)
    except: return None

def run_partial_correlation(df, x_col, y_col, control_cols):
    cols = [x_col, y_col] + control_cols
    d = df[cols].dropna()
    if len(d) < len(cols) + 3: return None
    try:
        if not SKLEARN_OK: return None
        Z = StandardScaler().fit_transform(d[control_cols])
        X = d[[x_col, y_col]].values
        Z_with_int = np.column_stack([np.ones(len(Z)), Z])
        beta = np.linalg.lstsq(Z_with_int, X, rcond=None)[0]
        resid = X - Z_with_int @ beta
        r, p = stats.pearsonr(resid[:, 0], resid[:, 1])
        return {'partial_r': r, 'p': p, 'n': len(d)}
    except: return None

def run_distance_correlation(df, x_col, y_col):
    d = df[[x_col, y_col]].dropna()
    if len(d) < 5: return None
    try:
        if DCOR_OK:
            val = dcor.distance_correlation(d[x_col].values, d[y_col].values)
            return {'dcor': float(val), 'n': len(d)}
        else:
            x = d[x_col].values; y = d[y_col].values
            a = np.abs(x[:, None] - x[None, :])
            b = np.abs(y[:, None] - y[None, :])
            A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
            B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
            dcov2 = np.mean(A * B)
            dvar_x = np.mean(A * A); dvar_y = np.mean(B * B)
            dcor_val = np.sqrt(max(dcov2, 0) / np.sqrt(max(dvar_x * dvar_y, 1e-12)))
            return {'dcor': float(dcor_val), 'n': len(d)}
    except: return None

def run_pca(df, cols):
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2 or not SKLEARN_OK: return None
    d = df[cols].dropna()
    if len(d) < 3: return None
    X = StandardScaler().fit_transform(d)
    pca = PCA(n_components=min(5, len(cols), len(d)-1))
    scores = pca.fit_transform(X)
    return {'scores': scores, 'loadings': pca.components_.T,
            'explained': pca.explained_variance_ratio_,
            'n': len(d), 'cols': cols, 'index': d.index}

def run_famd(df, num_cols, cat_cols):
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    cols = num_cols + cat_cols
    if len(cols) < 2: return None
    d = df[cols].dropna()
    if len(d) < 3: return None
    if PRINCE_OK and len(cat_cols) > 0:
        try:
            famd = prince.FAMD(n_components=min(5, len(cols)), random_state=42)
            scores = famd.fit_transform(d).values
            return {'scores': scores, 'explained': np.array(famd.explained_inertia_),
                    'n': len(d), 'cols': cols, 'index': d.index, 'method': 'prince.FAMD'}
        except: pass
    try:
        if not SKLEARN_OK: return None
        X_num = StandardScaler().fit_transform(d[num_cols]) if num_cols else np.zeros((len(d), 0))
        if cat_cols:
            enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_cat = enc.fit_transform(d[cat_cols])
        else:
            X_cat = np.zeros((len(d), 0))
        X = np.hstack([X_num, X_cat])
        pca = PCA(n_components=min(5, X.shape[1], len(d)-1))
        scores = pca.fit_transform(X)
        return {'scores': scores, 'loadings': pca.components_[:2].T if pca.components_.shape[0] >= 2 else pca.components_.T,
                'explained': pca.explained_variance_ratio_,
                'n': len(d), 'cols': num_cols + [f'cat_{c}' for c in cat_cols],
                'index': d.index, 'method': 'PCA (fallback)'}
    except: return None

def run_umap(df, cols, n_neighbors=15):
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2 or not UMAP_OK or not SKLEARN_OK: return None
    d = df[cols].dropna()
    if len(d) < 10: return None
    try:
        X = StandardScaler().fit_transform(d)
        reducer = UMAP(n_neighbors=min(n_neighbors, len(d)-1), n_components=2, random_state=42)
        emb = reducer.fit_transform(X)
        return {'embedding': emb, 'index': d.index, 'n': len(d)}
    except: return None

def run_tsne(df, cols, perplexity=15):
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2 or not SKLEARN_OK: return None
    d = df[cols].dropna()
    if len(d) < 5: return None
    try:
        X = StandardScaler().fit_transform(d)
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(d)-1), random_state=42, init='pca', learning_rate='auto')
        emb = tsne.fit_transform(X)
        return {'embedding': emb, 'index': d.index, 'n': len(d)}
    except: return None

def run_elastic_net(df, target_col, feature_cols, alpha_cv=5):
    if not SKLEARN_OK: return None
    cols = [c for c in feature_cols if c in df.columns]
    d = df[cols + [target_col]].dropna()
    if len(d) < max(10, 3*len(cols)): return None
    X = StandardScaler().fit_transform(d[cols]); y = d[target_col].values
    try:
        model = ElasticNetCV(cv=min(alpha_cv, len(d)), l1_ratio=[0.1, 0.5, 0.9, 1.0], max_iter=10000)
        model.fit(X, y)
        return {'model': model, 'cols': cols, 'coefs': model.coef_, 'intercept': model.intercept_,
                'alpha': model.alpha_, 'l1_ratio': model.l1_ratio_, 'X': X, 'y': y, 'n': len(d), 'index': d.index}
    except: return None

def bootstrap_en_coefs(df, target_col, feature_cols, n_bootstrap=500, seed=42):
    if not SKLEARN_OK: return None
    cols = [c for c in feature_cols if c in df.columns]
    d = df[cols + [target_col]].dropna()
    if len(d) < max(10, 3*len(cols)): return None
    rng = np.random.default_rng(seed)
    coefs_samples = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(d), len(d), replace=True)
        X_b = StandardScaler().fit_transform(d[cols].values[idx])
        y_b = d[target_col].values[idx]
        try:
            m = ElasticNetCV(cv=3, l1_ratio=[0.5, 0.9, 1.0], max_iter=5000)
            m.fit(X_b, y_b)
            coefs_samples.append(m.coef_)
        except: continue
    if len(coefs_samples) < 10: return None
    coefs_arr = np.array(coefs_samples)
    return {'samples': coefs_arr, 'mean': coefs_arr.mean(axis=0),
            'ci_lower': np.percentile(coefs_arr, 2.5, axis=0),
            'ci_upper': np.percentile(coefs_arr, 97.5, axis=0),
            'cols': cols}

def run_gam(df, target_col, feature_cols):
    if not PYGAM_OK or not SKLEARN_OK: return None
    cols = [c for c in feature_cols if c in df.columns]
    d = df[cols + [target_col]].dropna()
    if len(d) < 20: return None
    try:
        X = StandardScaler().fit_transform(d[cols]); y = d[target_col].values
        terms = gam_s(0)
        for i in range(1, len(cols)): terms = terms + gam_s(i)
        gam = LinearGAM(terms).gridsearch(X, y)
        return {'model': gam, 'cols': cols, 'X': X, 'y': y, 'n': len(d)}
    except: return None

def run_gam_statsmodels(df, target_col, feature_cols):
    if not STATSMODELS_OK: return None
    cols = [c for c in feature_cols if c in df.columns]
    d = df[cols + [target_col]].dropna()
    if len(d) < 20: return None
    try:
        from statsmodels.gam.api import GLMGam, BSplines
        X = d[cols].values; y = d[target_col].values
        bs = BSplines(X, df=[5]*len(cols), degree=[3]*len(cols))
        gam = GLMGam(y, exog=np.ones((len(X), 1)), smoother=bs).fit()
        return {'model': gam, 'cols': cols, 'X': X, 'y': y, 'n': len(d), 'smoother': bs}
    except: return None

def run_cca(df, x_cols, y_cols):
    if not SKLEARN_OK: return None
    x_cols = [c for c in x_cols if c in df.columns]
    y_cols = [c for c in y_cols if c in df.columns]
    if len(x_cols) < 2 or len(y_cols) < 2: return None
    d = df[x_cols + y_cols].dropna()
    if len(d) < 6: return None
    try:
        X = StandardScaler().fit_transform(d[x_cols]); Y = StandardScaler().fit_transform(d[y_cols])
        cca = CCA(n_components=min(2, len(x_cols), len(y_cols)))
        x_s, y_s = cca.fit_transform(X, Y)
        return {'x_scores': x_s, 'y_scores': y_s, 'x_cols': x_cols, 'y_cols': y_cols, 'n': len(d)}
    except: return None

def run_permutation_importance(model, X, y, feature_names, n_repeats=10):
    if not SKLEARN_OK: return None
    try:
        res = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
        return pd.DataFrame({'feature': feature_names, 'importance': res.importances_mean,
                             'std': res.importances_std}).sort_values('importance', ascending=False)
    except: return None

def run_shap_linear(model, X, feature_names):
    if SHAP_OK:
        try:
            explainer = shap.LinearExplainer(model, X)
            vals = explainer.shap_values(X)
            return {'shap_values': vals, 'features': feature_names, 'method': 'shap'}
        except: pass
    coefs = model.coef_ if hasattr(model, 'coef_') else None
    if coefs is not None:
        X_centered = X - X.mean(axis=0)
        vals = X_centered * coefs
        return {'shap_values': vals, 'features': feature_names, 'method': 'linear_approx'}
    return None

def run_pls(df, target_col, feature_cols):
    if not SKLEARN_OK: return None
    cols = [c for c in feature_cols if c in df.columns]
    d = df[cols + [target_col]].dropna()
    if len(d) < 5: return None
    try:
        X = StandardScaler().fit_transform(d[cols]); y = d[target_col].values
        pls = PLSRegression(n_components=min(3, len(cols)))
        pls.fit(X, y)
        vip = np.sqrt((pls.x_weights_**2 * pls.x_scores_.var(axis=0)).sum(axis=1))
        return {'model': pls, 'cols': cols, 'vip': vip, 'n': len(d), 'X': X, 'y': y}
    except: return None

def compute_ale(model, X, feature_idx, n_bins=20):
    try:
        x = X[:, feature_idx]
        qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        qs = np.unique(qs)
        if len(qs) < 3: return None
        ale_vals = [0.0]
        for i in range(len(qs) - 1):
            lo, hi = qs[i], qs[i+1]
            mask = (x >= lo) & (x <= hi)
            if mask.sum() == 0: ale_vals.append(ale_vals[-1]); continue
            X_lo = X[mask].copy(); X_hi = X[mask].copy()
            X_lo[:, feature_idx] = lo; X_hi[:, feature_idx] = hi
            diff = np.mean(model.predict(X_hi) - model.predict(X_lo))
            ale_vals.append(ale_vals[-1] + diff)
        ale_vals = np.array(ale_vals)
        ale_vals = ale_vals - ale_vals.mean()
        centers = (qs[:-1] + qs[1:]) / 2 if len(qs) - 1 == len(ale_vals) else np.linspace(qs[0], qs[-1], len(ale_vals))
        return {'grid': centers, 'ale': ale_vals}
    except: return None

def compute_ale_2d(model, X, feature_idx_pair, n_bins=15):
    try:
        i, j = feature_idx_pair
        xi = np.quantile(X[:, i], np.linspace(0, 1, n_bins + 1))
        yi = np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1))
        ale_grid = np.zeros((n_bins, n_bins))
        for a in range(n_bins):
            for b in range(n_bins):
                lo_i, hi_i = xi[a], xi[a+1]
                lo_j, hi_j = yi[b], yi[b+1]
                mask = (X[:, i] >= lo_i) & (X[:, i] <= hi_i) & (X[:, j] >= lo_j) & (X[:, j] <= hi_j)
                if mask.sum() == 0: continue
                X_lo = X[mask].copy(); X_hi = X[mask].copy()
                X_lo[:, i] = lo_i; X_lo[:, j] = lo_j
                X_hi[:, i] = hi_i; X_hi[:, j] = hi_j
                try:
                    ale_grid[b, a] = np.mean(model.predict(X_hi) - model.predict(X_lo))
                except: pass
        ale_grid = ale_grid - np.mean(ale_grid)
        fig, ax = plt.subplots(figsize=(8, 6))
        cf = ax.contourf((xi[:-1] + xi[1:]) / 2, (yi[:-1] + yi[1:]) / 2, ale_grid, levels=30, cmap='RdBu_r')
        plt.colorbar(cf, ax=ax, label='ALE (centered)')
        style_axis(ax, f'Feature {i}', f'Feature {j}')
        ax.set_title(f'2D ALE: {i} × {j}', fontweight='bold')
        plt.tight_layout(); return fig
    except: return None

def compute_friedman_H(model, X, feat_idx_pair, grid_size=15):
    try:
        res = partial_dependence(model, X, [feat_idx_pair], grid_resolution=grid_size)
        pd_vals = res['average'][0]
        # canonical H: sqrt(sum((PD - row_mean - col_mean + grand_mean)^2) / sum(PD^2))
        pd_c = pd_vals - pd_vals.mean()
        denom = np.sum(pd_vals**2)
        if denom <= 0: return 0.0
        return float(np.sqrt(np.sum(pd_c**2) / denom))
    except: return None

def run_sobol(df, target_col, feature_cols, n_samples=512):
    cols = [c for c in feature_cols if c in df.columns]
    d = df[cols + [target_col]].dropna()
    if len(d) < 20: return None
    if SALIB_OK and SKLEARN_OK:
        try:
            problem = {'num_vars': len(cols),
                       'names': cols,
                       'bounds': [[float(d[c].min()), float(d[c].max())] for c in cols]}
            X_s = sobol_saltelli(problem, n_samples, calc_second_order=False)
            model = GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=3)
            model.fit(d[cols].values, d[target_col].values)
            Y_s = model.predict(X_s)
            Si = sobol_analyze.analyze(problem, Y_s, print_to_console=False)
            return {'df': pd.DataFrame({'Variable': cols, 'S1': Si['S1'], 'ST': Si['ST'],
                                        'S1_conf': Si['S1_conf'], 'ST_conf': Si['ST_conf']}).sort_values('ST', ascending=False),
                    'method': 'SALib (Sobol)'}
        except: pass
    # canonical fallback: variance-based first-order via binning
    try:
        if not SKLEARN_OK: return None
        model = GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=3)
        X = d[cols].values; y = d[target_col].values
        model.fit(X, y)
        y_pred = model.predict(X)
        total_var = np.var(y_pred)
        res = []
        for i, c in enumerate(cols):
            n_bins = min(10, len(np.unique(X[:, i])))
            if n_bins < 3:
                res.append({'Variable': c, 'S1_approx': 0.0, 'ST_approx': 0.0}); continue
            try:
                bins = pd.qcut(X[:, i], q=n_bins, labels=False, duplicates='drop')
            except: 
                res.append({'Variable': c, 'S1_approx': 0.0, 'ST_approx': 0.0}); continue
            # first-order: var(E[y|x])
            cond_means = []
            for b in range(int(bins.max()) + 1):
                mask = bins == b
                if mask.sum() > 0: cond_means.append(np.mean(y_pred[mask]))
            s1 = np.var(cond_means) / total_var if total_var > 0 else 0
            # total-order: via permutation of feature values
            X_perm = X.copy(); X_perm[:, i] = np.random.permutation(X_perm[:, i])
            y_perm = model.predict(X_perm)
            st_val = (total_var - np.var(y_perm)) / total_var if total_var > 0 else 0
            res.append({'Variable': c, 'S1_approx': max(0, s1), 'ST_approx': max(0, st_val)})
        return {'df': pd.DataFrame(res).sort_values('ST_approx', ascending=False),
                'method': 'Variance decomposition (binning + permutation)'}
    except: return None

def run_curve_clustering(df, n_clusters=3):
    curves = df[TEMP_COLS].values.astype(float)
    valid = np.array([np.isfinite(c).sum() >= 3 and np.all(np.where(np.isfinite(c), c > 0, True)) for c in curves])
    if valid.sum() < n_clusters: return None
    vc = curves[valid]
    log_curves = np.log10(np.clip(vc, 1e-10, None))
    filled = np.array([np.interp(TEMP_VALUES, np.array(TEMP_VALUES)[np.isfinite(c)], c[np.isfinite(c)]) for c in log_curves])
    try:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(filled)
        return {'labels': labels, 'centroids': km.cluster_centers_, 'curves': filled,
                'valid_mask': valid, 'valid_index': df.index[valid]}
    except: return None

def run_fpca(df, n_components=3):
    curves = df[TEMP_COLS].values.astype(float)
    valid = np.array([np.isfinite(c).sum() >= 3 and np.all(np.where(np.isfinite(c), c > 0, True)) for c in curves])
    if valid.sum() < max(n_components + 2, 5): return None
    vc = curves[valid]
    log_curves = np.log10(np.clip(vc, 1e-10, None))
    filled = np.array([np.interp(TEMP_VALUES, np.array(TEMP_VALUES)[np.isfinite(c)], c[np.isfinite(c)]) for c in log_curves])
    if SKFDA_OK:
        try:
            fd = FDataGrid(filled, grid_points=TEMP_VALUES)
            fpca = skfda_FPCA(n_components=n_components)
            fpca.fit(fd)
            scores = fpca.transform(fd)
            comps = np.array([fpca.components_.data_matrix[i, 0, :] for i in range(n_components)])
            return {'scores': scores, 'components': comps,
                    'explained': fpca.explained_variance_ratio_, 'n': valid.sum(),
                    'valid_index': df.index[valid], 'T_vals': np.array(TEMP_VALUES),
                    'method': 'scikit-fda FPCA'}
        except: pass
    # canonical fallback: PCA on B-spline smoothed curves
    try:
        if not SKLEARN_OK: return None
        from scipy.interpolate import splrep, splev
        smoothed = []
        for c in filled:
            try:
                tck = splrep(TEMP_VALUES, c, k=3, s=0.01)
                smoothed.append(splev(TEMP_VALUES, tck))
            except:
                smoothed.append(c)
        smoothed = np.array(smoothed)
        X = StandardScaler().fit_transform(smoothed)
        pca = PCA(n_components=min(n_components, smoothed.shape[1], smoothed.shape[0]-1))
        scores = pca.fit_transform(X)
        return {'scores': scores, 'components': pca.components_,
                'explained': pca.explained_variance_ratio_, 'n': valid.sum(),
                'valid_index': df.index[valid], 'T_vals': np.array(TEMP_VALUES),
                'method': 'Spline-PCA (fallback)'}
    except: return None

def run_segmented_regression(x, y, n_segments=2):
    if len(x) < n_segments + 2: return None
    if PWLE_OK:
        try:
            model = pwlf.PiecewiseLinFit(x, y)
            breaks = model.fit(n_segments)
            return {'breakpoints': breaks[1:-1], 'model': model,
                    'n_segments': n_segments, 'method': 'pwlf'}
        except: pass
    # canonical fallback: exhaustive grid-search over breakpoints with continuous optimization
    try:
        sorted_idx = np.argsort(x)
        xs, ys = x[sorted_idx], y[sorted_idx]
        best_bp, best_sse = None, np.inf
        for bp_idx in range(3, len(xs) - 3):
            x1, y1 = xs[:bp_idx], ys[:bp_idx]
            x2, y2 = xs[bp_idx:], ys[bp_idx:]
            try:
                p1 = np.polyfit(x1, y1, 1); p2 = np.polyfit(x2, y2, 1)
                sse = np.sum((y1 - np.polyval(p1, x1))**2) + np.sum((y2 - np.polyval(p2, x2))**2)
                if sse < best_sse: best_sse = sse; best_bp = xs[bp_idx]; best_p1 = p1; best_p2 = p2
            except: continue
        if best_bp is not None:
            return {'breakpoints': np.array([best_bp]), 'seg1_params': best_p1, 'seg2_params': best_p2, 'method': 'grid-search'}
        return None
    except: return None

def run_segmented_regression_batch(df, n_segments=2):
    results = []
    for idx, row in df.iterrows():
        vals = np.array([row.get(c, np.nan) for c in TEMP_COLS], dtype=float)
        T = np.array(TEMP_VALUES)
        mask = np.isfinite(vals) & (vals > 0)
        if mask.sum() < 4: continue
        x = 1000 / (T[mask] + 273.15)
        y = np.log(vals[mask] * (T[mask] + 273.15))
        res = run_segmented_regression(x, y, n_segments)
        if res and 'breakpoints' in res:
            results.append({'index': idx, 'breakpoint_1000T': res['breakpoints'][0],
                            'T_C': 1000 / res['breakpoints'][0] - 273.15 if res['breakpoints'][0] > 0 else np.nan})
    return pd.DataFrame(results) if results else None

def run_mediation(df, treatment_col, mediator_col, outcome_col):
    d = df[[treatment_col, mediator_col, outcome_col]].dropna()
    if len(d) < 10: return None
    X = d[treatment_col].values.astype(float); M = d[mediator_col].values.astype(float); Y = d[outcome_col].values.astype(float)
    try:
        a_coef, _, _, _, _ = stats.linregress(X, M)
        c_coef, _, _, c_p, _ = stats.linregress(X, Y)
        X_design = np.column_stack([np.ones_like(X), X, M])
        coef, _, _, _ = np.linalg.lstsq(X_design, Y, rcond=None)
        c_prime = coef[1]; b = coef[2]
        y_pred = X_design @ coef; resid = Y - y_pred
        dof = len(Y) - 3
        sse = np.sum(resid**2)
        cov = sse / dof * np.linalg.inv(X_design.T @ X_design)
        se = np.sqrt(np.diag(cov))
        t_stat = c_prime / se[1] if se[1] > 0 else 0
        p_direct = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
        indirect = a_coef * b
        return {'total': c_coef, 'direct': c_prime, 'indirect': indirect,
                'a_path': a_coef, 'b_path': b, 'p_total': c_p, 'p_direct': p_direct, 'n': len(d)}
    except: return None

def match_pairs_by(df, ref_col, treatment_col='is_additive', match_on=('ρ, %', 'T sin')):
    rows = []
    for ref, g in df.groupby(ref_col):
        if 'Sintering additive' not in g.columns: continue
        pure = g[g['Sintering additive'] == 'Pure']
        additive = g[g['Sintering additive'] != 'Pure']
        if len(pure) == 0 or len(additive) == 0: continue
        for _, a in additive.iterrows():
            best, best_d = None, np.inf
            for _, p in pure.iterrows():
                dist = 0
                for col in match_on:
                    if col in df.columns and pd.notna(a.get(col)) and pd.notna(p.get(col)):
                        dist += abs(a[col] - p[col]) / (np.nanstd(df[col]) + 1e-9)
                if dist < best_d: best_d = dist; best = p
            if best is not None:
                rows.append({'Reference': ref, 'Additive': a.get('Sintering additive'),
                             'ref_idx': best.name, 'add_idx': a.name, 'match_dist': best_d})
    return pd.DataFrame(rows) if rows else None

def run_cem_matching(df, ref_col, match_on=('ρ, %', 'T sin'), n_bins=5):
    """Canonical Coarsened Exact Matching with quantile binning."""
    if 'Sintering additive' not in df.columns: return None
    rows = []
    for ref, g in df.groupby(ref_col):
        pure = g[g['Sintering additive'] == 'Pure']
        additive = g[g['Sintering additive'] != 'Pure']
        if len(pure) == 0 or len(additive) == 0: continue
        # create coarse bins per variable
        bin_edges = {}
        for col in match_on:
            if col not in df.columns: continue
            vals = g[col].dropna()
            if len(vals) < 2: continue
            try:
                qs = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
                qs = np.unique(qs)
                bin_edges[col] = qs
            except: continue
        if not bin_edges: continue
        def get_bins(row):
            key = []
            for col, edges in bin_edges.items():
                v = row.get(col, np.nan)
                if pd.isna(v): return None
                b = np.searchsorted(edges, v, side='right') - 1
                b = min(max(b, 0), len(edges) - 2)
                key.append(b)
            return tuple(key)
        # bin all samples in this reference
        pure_bins = {}
        for _, p in pure.iterrows():
            b = get_bins(p)
            if b is None: continue
            pure_bins.setdefault(b, []).append(p.name)
        for _, a in additive.iterrows():
            b = get_bins(a)
            if b is None or b not in pure_bins or not pure_bins[b]: continue
            match_idx = pure_bins[b].pop(0)
            rows.append({'Reference': ref, 'Additive': a.get('Sintering additive'),
                         'ref_idx': match_idx, 'add_idx': a.name, 'match_dist': 0.0, 'bin': b})
    return pd.DataFrame(rows) if rows else None

def build_pairs_within_reference(df, temperature_col, heat_metric='delta_rho'):
    """
    Build (additive, Pure) pairs within each Reference.
    - For each Reference, pick ONE Pure sample:
      * if only one Pure -> use it
      * if several Pure -> pick the one with the LOWEST sigma at temperature_col
        (among those with valid sigma > 0 at this temperature)
    - If no Pure with valid sigma -> skip the whole Reference
    - For each additive sample in the same Reference with valid sigma > 0 and non-null x, wt%:
      * compute ratio = sigma_additive / sigma_pure
      * compute d_rho, d_d, d_ratio, same_Tsin
    Returns a pandas DataFrame with one row per valid pair.
    """
    if temperature_col not in df.columns:
        return None
    if 'Sintering additive' not in df.columns or 'References' not in df.columns:
        return None
    if 'x, wt%' not in df.columns:
        return None

    rows = []
    for ref, g in df.groupby('References'):
        pure_mask = g['Sintering additive'] == 'Pure'
        pure_g = g[pure_mask]
        if len(pure_g) == 0:
            continue
        pure_valid = pure_g[pure_g[temperature_col].notna() & (pure_g[temperature_col] > 0)]
        if len(pure_valid) == 0:
            continue
        # choose Pure with LOWEST sigma at temperature_col
        best_pure = pure_valid.sort_values(temperature_col, ascending=True).iloc[0]
        sigma_pure = best_pure[temperature_col]
        if pd.isna(sigma_pure) or sigma_pure <= 0:
            continue
        rho_pure = best_pure.get('ρ, %', np.nan) if 'ρ, %' in g.columns else np.nan
        d_pure = best_pure.get('d, mkm', np.nan) if 'd, mkm' in g.columns else np.nan
        tsin_pure = best_pure.get('T sin', np.nan) if 'T sin' in g.columns else np.nan

        additive_g = g[~pure_mask]
        for _, a in additive_g.iterrows():
            sigma_a = a.get(temperature_col, np.nan)
            x_wt = a.get('x, wt%', np.nan)
            if pd.isna(sigma_a) or sigma_a <= 0:
                continue
            if pd.isna(x_wt):
                continue
            ratio = sigma_a / sigma_pure
            if not np.isfinite(ratio) or ratio <= 0:
                continue
            log10_ratio = np.log10(ratio)

            rho_a = a.get('ρ, %', np.nan) if 'ρ, %' in g.columns else np.nan
            d_a = a.get('d, mkm', np.nan) if 'd, mkm' in g.columns else np.nan
            tsin_a = a.get('T sin', np.nan) if 'T sin' in g.columns else np.nan

            d_rho = (rho_a - rho_pure) if pd.notna(rho_a) and pd.notna(rho_pure) else np.nan
            d_d = (d_a - d_pure) if pd.notna(d_a) and pd.notna(d_pure) else np.nan
            d_ratio = (d_a / d_pure) if pd.notna(d_a) and pd.notna(d_pure) and d_pure > 0 else np.nan

            same_tsin = False
            if pd.notna(tsin_a) and pd.notna(tsin_pure):
                same_tsin = abs(float(tsin_a) - float(tsin_pure)) <= 1.0

            if heat_metric == 'abs_delta_rho':
                heat_val = abs(d_rho) if pd.notna(d_rho) else np.nan
            elif heat_metric == 'abs_delta_d':
                heat_val = abs(d_d) if pd.notna(d_d) else np.nan
            elif heat_metric == 'd_ratio':
                heat_val = d_ratio if pd.notna(d_ratio) else np.nan
            else:
                heat_val = abs(d_rho) if pd.notna(d_rho) else np.nan

            rows.append({
                'Reference': ref,
                'Additive': a.get('Sintering additive'),
                'x_wt': float(x_wt),
                'sigma_with': float(sigma_a),
                'sigma_without': float(sigma_pure),
                'ratio': float(ratio),
                'log10_ratio': float(log10_ratio),
                'rho_additive': rho_a,
                'rho_pure': rho_pure,
                'd_rho': d_rho,
                'd_additive': d_a,
                'd_pure': d_pure,
                'd_d': d_d,
                'd_ratio': d_ratio,
                'Tsin_additive': tsin_a,
                'Tsin_pure': tsin_pure,
                'same_Tsin': bool(same_tsin),
                'heat_value': heat_val,
                'temperature_col': temperature_col,
            })

    if not rows:
        return None
    return pd.DataFrame(rows)

def run_meta_forest(df, ref_col, y_col='σ total, 600', treatment_group='Pure'):
    """Per-reference effect of each additive vs Pure (mean diff + CI)."""
    if 'Sintering additive' not in df.columns: return None
    rows = []
    for ref, g in df.groupby(ref_col):
        pure_vals = g[g['Sintering additive'] == treatment_group][y_col].dropna().values
        if len(pure_vals) < 1: continue
        for additive, sub in g[g['Sintering additive'] != treatment_group].groupby('Sintering additive'):
            vals = sub[y_col].dropna().values
            if len(vals) < 1: continue
            diff = np.mean(vals) - np.mean(pure_vals)
            se = np.sqrt(np.var(pure_vals)/len(pure_vals) + np.var(vals)/len(vals)) if len(pure_vals) > 1 and len(vals) > 1 else 0
            rows.append({'Reference': ref, 'Additive': additive, 'diff': diff,
                         'ci_low': diff - 1.96*se, 'ci_high': diff + 1.96*se,
                         'n_additive': len(vals), 'n_pure': len(pure_vals)})
    return pd.DataFrame(rows) if rows else None

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    st.set_page_config(page_title="Ceramic Conductivity Data Explorer", page_icon="🔬", layout="wide")
    st.title("🔬 Interactive Analysis of Ceramic Conductivity")
    st.markdown("---")

    st.header("📊 Data Loading")
    st.markdown("**Paste data (TSV or CSV):**")
    data_input = st.text_area("Paste data here:", height=180, key="data_input")

    if 'df_loaded' not in st.session_state: st.session_state.df_loaded = None
    if 'df_hash' not in st.session_state: st.session_state.df_hash = None

    if not data_input:
        st.info("⏳ Paste data to begin"); st.stop()

    current_hash = hashlib.md5(data_input.encode()).hexdigest()
    if st.session_state.df_hash != current_hash or st.session_state.df_loaded is None:
        try:
            lines = data_input.strip().split('\n')
            sep = '\t' if '\t' in lines[0] else ','
            df = pd.read_csv(io.StringIO(data_input), sep=sep)
            df.columns = df.columns.str.strip()
            for c in TEMP_COLS + ['ρ, %', 'd, mkm', 'T sin', 'dop_cont', 'x, wt%', 'Ea (eV)']:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            st.session_state.df_loaded = df; st.session_state.df_hash = current_hash
            st.success(f"✅ Loaded: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Parse error: {e}"); st.stop()
    else:
        df = st.session_state.df_loaded
        st.success(f"✅ Cached: {len(df)} rows, {len(df.columns)} columns")

    with st.spinner("Computing descriptors..."):
        df_hash_desc = hashlib.md5(df.to_csv().encode()).hexdigest()
        desc = compute_descriptors_cached(df_hash_desc, df)
        for c in desc.columns: df[c] = desc[c]
        df['Ea_calculated'] = calculate_ea_cached(df_hash_desc, df)
        df['Ea_final'] = df.get('Ea (eV)', pd.Series(index=df.index)).fillna(df['Ea_calculated'])
        curve_feats = compute_curve_features_cached(df_hash_desc, df)
        for c in curve_feats.columns: df[c] = curve_feats[c]
        df = build_derived_features(df)

    st.sidebar.header("🔍 Filters")
    df_f = df.copy()
    active_count = 0
    for fname in FILTER_LABELS.keys():
        if fname in df_f.columns:
            opts = sorted([x for x in df_f[fname].dropna().unique() if x != ''])
            if opts:
                sel = st.sidebar.multiselect(fname, options=['All'] + opts, default=['All'], key=f'filt_{fname}')
                if 'All' not in sel:
                    df_f = df_f[df_f[fname].isin(sel)]
                    active_count += 1
    st.sidebar.markdown("---")
    if 'T sin' in df_f.columns and df_f['T sin'].notna().any():
        tmin, tmax = float(df_f['T sin'].min()), float(df_f['T sin'].max())
        if tmax > tmin:
            r = st.sidebar.slider("T sin (°C)", tmin, tmax, (tmin, tmax), key='tsin')
            df_f = df_f[(df_f['T sin'] >= r[0]) & (df_f['T sin'] <= r[1])]
    if 'dop_cont' in df_f.columns and df_f['dop_cont'].notna().any():
        dmin, dmax = float(df_f['dop_cont'].min()), float(df_f['dop_cont'].max())
        if dmax > dmin:
            r = st.sidebar.slider("Dopant content", dmin, dmax, (dmin, dmax), key='dopc')
            df_f = df_f[(df_f['dop_cont'] >= r[0]) & (df_f['dop_cont'] <= r[1])]
    if 'x, wt%' in df_f.columns and df_f['x, wt%'].notna().any():
        xmin, xmax = float(df_f['x, wt%'].min()), float(df_f['x, wt%'].max())
        if xmax > xmin:
            r = st.sidebar.slider("Additive concentration", xmin, xmax, (xmin, xmax), key='xw')
            df_f = df_f[(df_f['x, wt%'] >= r[0]) & (df_f['x, wt%'] <= r[1])]
    st.sidebar.markdown(f"**Rows: {len(df_f)}**")

    tabs = st.tabs(["📊 Data & QC", "🔍 Exploratory Plots", "🔗 Correlations",
                    "📈 Statistical Comparisons", "🎯 Multivariate",
                    "🤖 Regression & Attribution", "🔬 Interactions & Sensitivity",
                    "🌡️ Curve Analysis", "🎯 Additive vs Pure", "📋 Export"])

    with tabs[0]:
        st.header("📊 Data & Quality Control")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", len(df_f))
        col2.metric("References", df_f['References'].nunique() if 'References' in df_f else 0)
        col3.metric("Additives", df_f['Sintering additive'].nunique() if 'Sintering additive' in df_f else 0)
        col4.metric("Valid curves", int(df_f['curve_valid'].sum()) if 'curve_valid' in df_f else 0)
        st.subheader("Data preview"); st.dataframe(df_f.head(15))
        st.subheader("Missingness")
        try: st.pyplot(plot_missingness(df_f))
        except: st.info("Unavailable")
        st.subheader("Balance (Additive × Atmosphere)")
        if 'Sintering additive' in df_f.columns and 'Atmospheres' in df_f.columns:
            st.dataframe(pd.crosstab(df_f['Sintering additive'], df_f['Atmospheres']))
        st.subheader("Coverage of descriptor space")
        c1, c2 = st.columns(2)
        cov_x = c1.selectbox("X", NUMERIC_DESCRIPTORS, index=2, key='qc_x')
        cov_y = c2.selectbox("Y", NUMERIC_DESCRIPTORS, index=0, key='qc_y')
        try:
            st.pyplot(plot_coverage_heatmap(df_f, cov_x, cov_y))
            st.pyplot(plot_convex_hull(df_f, cov_x, cov_y))
            st.pyplot(plot_convex_hull_pca(df_f, NUMERIC_DESCRIPTORS))
        except: pass
        st.subheader("Descriptor summary"); st.dataframe(df_f[NUMERIC_DESCRIPTORS + ['Ea_final']].describe())

    with tabs[1]:
        st.header("🔍 Exploratory Plots")
        plot_kind = st.selectbox("Plot type",
            ["Heat map (scatter)", "Heat map (contour)", "Heat map (3D)",
             "Heat map (contour + CI bands)", "Bubble chart", "Pairplot",
             "Violin + Box", "Parallel coordinates", "Radar",
             "Ternary (composition)", "Sankey"], key='expl_kind')
        marker_style, show_leg, filt_name = get_filter_markers(df_f, active_count)

        if plot_kind.startswith("Heat map"):
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X", NUMERIC_DESCRIPTORS, index=0, key='ex_x')
            y_col = c2.selectbox("Y", NUMERIC_DESCRIPTORS, index=1, key='ex_y')
            z_opts = [c for c in TEMP_COLS if c in df_f.columns] + ['Ea_final']
            z_col = c3.selectbox("Z", z_opts, index=0, key='ex_z')
            d_plot = remove_outliers(df_f, x_col, st.number_input("Outliers X", 0, 5, 0, key='ex_ox'))
            d_plot = remove_outliers(d_plot, y_col, st.number_input("Outliers Y", 0, 5, 0, key='ex_oy'))
            c1, c2, c3, c4 = st.columns(4)
            logx = c1.checkbox("log X", key='ex_lx'); logy = c2.checkbox("log Y", key='ex_ly')
            logz = c3.checkbox("log Z", key='ex_lz'); pal = c4.selectbox("Palette", list(COLOR_PALETTES.keys()), key='ex_pal')
            if st.button("Generate", key='ex_btn'):
                zlab = z_col.replace('σ total, ', 'σ at ') + ' °C (mS/cm)' if z_col in TEMP_COLS else 'Ea (eV)'
                try:
                    if "scatter" in plot_kind:
                        fig = create_scatter_heatmap(d_plot, x_col, y_col, z_col, logx, logy, logz,
                                                     COLOR_PALETTES[pal], x_col, y_col, zlab, marker_style, show_leg, filt_name)
                    elif "CI bands" in plot_kind:
                        fig = create_contour_with_bands(d_plot, x_col, y_col, z_col, logx, logy, logz,
                                                        COLOR_PALETTES[pal], x_col, y_col, zlab)
                    elif "contour" in plot_kind:
                        fig = create_contour_heatmap(d_plot, x_col, y_col, z_col, logx, logy, logz,
                                                     COLOR_PALETTES[pal], x_col, y_col, zlab)
                    else:
                        fig = create_3d_surface(d_plot, x_col, y_col, z_col, logx, logy, logz,
                                                COLOR_PALETTES[pal], x_col, y_col, zlab)
                    if fig: st.pyplot(fig); plt.close(fig)
                except Exception as e: st.error(f"Error: {e}")

        elif plot_kind == "Bubble chart":
            y_opts = [c for c in TEMP_COLS if c in df_f.columns] + ['Ea_final']
            c1, c2, c3, c4 = st.columns(4)
            y_col = c1.selectbox("Y", y_opts, index=0, key='bb_y')
            x_col = c2.selectbox("X", NUMERIC_DESCRIPTORS, index=0, key='bb_x')
            color_col = c3.selectbox("Color", ['None'] + NUMERIC_DESCRIPTORS + list(FILTER_LABELS.keys()), index=0, key='bb_c')
            size_col = c4.selectbox("Size", ['None'] + NUMERIC_DESCRIPTORS, index=0, key='bb_s')
            c1, c2, c3, c4 = st.columns(4)
            lx = c1.checkbox("log X", key='bb_lx'); ly = c2.checkbox("log Y", key='bb_ly')
            lc = c3.checkbox("log C", key='bb_lc'); ls = c4.checkbox("log S", key='bb_ls')
            show_marg = st.checkbox("Marginal distributions", key='bb_marg')
            if st.button("Generate bubble", key='bb_btn'):
                try:
                    d_plot = df_f.copy()
                    if color_col == 'None': d_plot['_c'] = 1; color_col = '_c'
                    if size_col == 'None': d_plot['_s'] = 1; size_col = '_s'
                    ylab = y_col.replace('σ total, ', 'σ at ') + ' °C (mS/cm)' if y_col in TEMP_COLS else 'Ea (eV)'
                    fig = create_bubble_chart(d_plot, x_col, y_col, color_col, size_col, lx, ly, lc, ls,
                                              COLOR_PALETTES['Viridis'], x_col, ylab, marker_style, show_leg, filt_name,
                                              size_col if size_col != '_s' else None, filt_name if show_leg else None,
                                              show_marg, False, size_col if size_col != '_s' else None, y_col)
                    if fig: st.pyplot(fig); plt.close(fig)
                except Exception as e: st.error(f"Error: {e}")

        elif plot_kind == "Pairplot":
            cols = st.multiselect("Features", NUMERIC_DESCRIPTORS + ['Ea_final'],
                                   default=['ρ, %', 'd, mkm', 'Ea_final'], key='pp_cols')
            hue = st.selectbox("Hue", ['None'] + list(FILTER_LABELS.keys()), key='pp_hue')
            if st.button("Generate", key='pp_btn') and len(cols) >= 2:
                try:
                    fig = plot_pairplot(df_f, cols, None if hue == 'None' else hue)
                    if fig: st.pyplot(fig); plt.close(fig)
                except Exception as e: st.error(f"Error: {e}")

        elif plot_kind == "Violin + Box":
            y_col = st.selectbox("Y", ['Ea_final'] + [c for c in TEMP_COLS if c in df_f.columns], key='vb_y')
            g_col = st.selectbox("Group by", list(FILTER_LABELS.keys()), key='vb_g')
            log_y = st.checkbox("log Y", key='vb_log')
            if st.button("Generate", key='vb_btn'):
                try:
                    fig = plot_violin_box(df_f, y_col, g_col, log_y)
                    if fig: st.pyplot(fig); plt.close(fig)
                except Exception as e: st.error(f"Error: {e}")

        elif plot_kind == "Parallel coordinates":
            cols = st.multiselect("Features", NUMERIC_DESCRIPTORS + ['Ea_final'],
                                   default=['ρ, %', 'd, mkm', 'Ea_final'], key='pc_cols')
            hue = st.selectbox("Hue", ['None'] + list(FILTER_LABELS.keys()), key='pc_hue')
            if st.button("Generate", key='pc_btn') and len(cols) >= 2:
                try:
                    fig = plot_parallel_coordinates(df_f, cols, None if hue == 'None' else hue)
                    if fig: st.pyplot(fig); plt.close(fig)
                except Exception as e: st.error(f"Error: {e}")

        elif plot_kind == "Radar":
            if 'Sintering additive' in df_f.columns:
                additives = st.multiselect("Additives", sorted(df_f['Sintering additive'].dropna().unique()),
                                            default=sorted(df_f['Sintering additive'].dropna().unique())[:4], key='rd_a')
                cols = st.multiselect("Axes", NUMERIC_DESCRIPTORS + ['Ea_final'],
                                       default=['ρ, %', 'd, mkm', 'Ea_final', 'porosity'], key='rd_c')
                if st.button("Generate", key='rd_btn') and len(cols) >= 3:
                    try:
                        fig = plot_radar(df_f, additives, cols)
                        if fig: st.pyplot(fig); plt.close(fig)
                    except Exception as e: st.error(f"Error: {e}")

        elif plot_kind == "Ternary (composition)":
            comp = st.multiselect("Composition columns (3)", ['x_B1', 'x_B2', 'x_dop'],
                                   default=['x_B1', 'x_B2', 'x_dop'], key='tr_c')
            color_col = st.selectbox("Color", ['Ea_final'] + [c for c in TEMP_COLS if c in df_f.columns], key='tr_z')
            log_c = st.checkbox("log Color", key='tr_logc')
            if st.button("Generate", key='tr_btn') and len(comp) == 3:
                try:
                    fig = plot_ternary(df_f, comp, color_col, log_col=log_c)
                    if fig: st.pyplot(fig); plt.close(fig)
                except Exception as e: st.error(f"Error: {e}")

        elif plot_kind == "Sankey":
            c1, c2, c3 = st.columns(3)
            src = c1.selectbox("Source", list(FILTER_LABELS.keys()), key='sk_src')
            mid = c2.selectbox("Middle", list(FILTER_LABELS.keys()), key='sk_mid')
            tgt = c3.selectbox("Target (numeric)", ['Ea_final'], key='sk_tgt')
            if st.button("Generate Sankey", key='sk_btn'):
                try:
                    fig = plot_sankey(df_f, src, mid, tgt)
                    if fig is not None:
                        if PLOTLY_OK and hasattr(fig, 'to_html'):
                            st.plotly_chart(fig, use_container_width=True)
                        else: st.pyplot(fig); plt.close(fig)
                except Exception as e: st.error(f"Error: {e}")

    with tabs[2]:
        st.header("🔗 Correlation Analysis")
        cols = st.multiselect("Features", NUMERIC_DESCRIPTORS + ['Ea_final'] + [c for c in TEMP_COLS if c in df_f.columns],
                              default=['ρ, %', 'd, mkm', 'T sin', 'dop_cont', 'Ea_final'], key='corr_cols')
        method = st.selectbox("Method", ["pearson", "spearman", "kendall"], key='corr_m')
        if st.button("Compute matrix", key='corr_btn') and len(cols) >= 2:
            try:
                d = df_f[cols].dropna()
                corr = d.corr(method=method)
                st.pyplot(plot_correlation_heatmap(corr, f"{method.capitalize()} correlation (n={len(d)})"))
                st.pyplot(plot_clustermap(corr))
                st.subheader("Mutual information")
                if SKLEARN_OK:
                    mi = np.zeros((len(cols), len(cols)))
                    for i, ci in enumerate(cols):
                        for j, cj in enumerate(cols):
                            if i == j: mi[i, j] = 1.0
                            else:
                                sub = df_f[[ci, cj]].dropna()
                                if len(sub) > 5:
                                    mi[i, j] = mutual_info_regression(sub[[ci]], sub[cj], random_state=42)[0]
                    st.dataframe(pd.DataFrame(mi, index=cols, columns=cols).style.format("{:.3f}").background_gradient(cmap='viridis'))
                st.subheader("Distance correlation")
                dcor_results = []
                for ci in cols:
                    for cj in cols:
                        if ci >= cj: continue
                        r = run_distance_correlation(df_f, ci, cj)
                        if r: dcor_results.append({'X': ci, 'Y': cj, 'dCor': r['dcor'], 'n': r['n']})
                if dcor_results:
                    st.dataframe(pd.DataFrame(dcor_results).sort_values('dCor', ascending=False).style.format({'dCor': '{:.3f}'}))
                st.download_button("📥 Download matrix (TSV)", corr.to_csv(sep='\t'),
                                    file_name="correlations.tsv", mime="text/tab-separated-values")
            except Exception as e: st.error(f"Error: {e}")
        st.markdown("---"); st.subheader("Partial correlation")
        c1, c2 = st.columns(2)
        x_p = c1.selectbox("X", NUMERIC_DESCRIPTORS, key='pc_x')
        y_p = c2.selectbox("Y", ['Ea_final'] + [c for c in TEMP_COLS if c in df_f.columns], key='pc_y')
        controls = st.multiselect("Control for", NUMERIC_DESCRIPTORS, default=['T sin'], key='pc_ctrl')
        if st.button("Compute partial", key='pc_btn') and controls:
            res = run_partial_correlation(df_f, x_p, y_p, controls)
            if res: st.success(f"Partial r = {res['partial_r']:.3f}, p = {res['p']:.3g}, n = {res['n']}")
            else: st.warning("Not enough data")

    with tabs[3]:
        st.header("📈 Statistical Comparisons")
        y_col = st.selectbox("Outcome", ['Ea_final'] + [c for c in TEMP_COLS if c in df_f.columns], key='sc_y')
        group_col = st.selectbox("Group by", list(FILTER_LABELS.keys()), key='sc_g')
        c1, c2 = st.columns(2)
        if c1.button("Run ANOVA", key='sc_a'):
            res = run_anova(df_f, y_col, group_col)
            if res: st.success(f"F = {res['F']:.3f}, p = {res['p']:.3g}, η² = {res['eta_squared']:.3f}, n = {res['n_total']}")
            else: st.warning("Insufficient data")
        if c2.button("Run Kruskal-Wallis", key='sc_k'):
            res = run_kruskal(df_f, y_col, group_col)
            if res: st.success(f"H = {res['H']:.3f}, p = {res['p']:.3g}, n = {res['n_total']}")
            else: st.warning("Insufficient data")
        st.markdown("---"); st.subheader("Post-hoc tests")
        c1, c2 = st.columns(2)
        if c1.button("Dunn post-hoc", key='sc_dunn'):
            res = run_dunn_posthoc(df_f, y_col, group_col)
            if res is not None and len(res) > 0:
                st.dataframe(res.style.format({'z': '{:.3f}', 'p_raw': '{:.3g}', 'p_adj': '{:.3g}'}))
                fig = plot_dunn_posthoc(res['pair'].tolist(), res['z'].tolist(), res['p_adj'].tolist())
                if fig: st.pyplot(fig); plt.close(fig)
            else: st.warning("Dunn failed")
        if c2.button("Tukey HSD", key='sc_tukey'):
            res = run_tukey(df_f, y_col, group_col)
            if res is not None: st.text(str(res.summary()))
            else: st.warning("Tukey failed or statsmodels missing")
        st.markdown("---"); st.subheader("Pairwise vs Pure")
        if 'Sintering additive' in df_f.columns:
            if st.button("Compute pairwise", key='sc_p'):
                res = run_pairwise_comparisons(df_f, y_col, 'Sintering additive', 'Pure')
                if res is not None and len(res) > 0:
                    st.dataframe(res.style.format({c: '{:.4g}' for c in ['mean','diff','ci_low','ci_high','p','cohen_d']}))
                    try:
                        fig = plot_forest(res['diff'].values, res['group'].tolist(),
                                          list(zip(res['ci_low'], res['ci_high'])), f'Effect on {y_col}')
                        if fig: st.pyplot(fig); plt.close(fig)
                    except: pass
                    st.download_button("📥 Download (TSV)", res.to_csv(sep='\t', index=False),
                                        file_name="pairwise.tsv", mime="text/tab-separated-values")
                else: st.warning("Insufficient data")
        st.markdown("---"); st.subheader("Meta-forest by Reference")
        if 'References' in df_f.columns and st.button("Build meta-forest", key='sc_meta'):
            res = run_meta_forest(df_f, 'References', y_col, 'Pure')
            if res is not None and len(res) > 0:
                st.dataframe(res)
                fig = plot_meta_forest(res['diff'].values, list(zip(res['ci_low'], res['ci_high'])),
                                       [f"{r['Reference']}/{r['Additive']}" for _, r in res.iterrows()])
                if fig: st.pyplot(fig); plt.close(fig)
                st.download_button("📥 Download meta-forest (TSV)", res.to_csv(sep='\t', index=False),
                                    file_name="meta_forest.tsv", mime="text/tab-separated-values")
            else: st.warning("Not enough references")
        st.markdown("---"); st.subheader("Mixed-effects model")
        if STATSMODELS_OK:
            fixed = st.multiselect("Fixed effects", NUMERIC_DESCRIPTORS + ['is_additive'],
                                    default=['is_additive', 'ρ, %', 'T sin'], key='me_fx')
            if st.button("Fit mixed model", key='me_btn') and fixed:
                res = run_mixed_effects(df_f, y_col, fixed)
                if res is not None: st.text(str(res.summary()))
                else: st.warning("Model failed or insufficient data")
        else: st.info("statsmodels not installed")

    with tabs[4]:
        st.header("🎯 Multivariate Analysis")
        method = st.selectbox("Method", ["PCA", "FAMD", "UMAP", "t-SNE", "Cluster (k-means)"], key='mv_m')
        cols = st.multiselect("Numeric features", NUMERIC_DESCRIPTORS + ['Ea_final'],
                              default=['ρ, %', 'd, mkm', 'T sin', 'dop_cont', 'Ea_final'], key='mv_c')
        if method == "PCA" and st.button("Run PCA", key='mv_pca') and len(cols) >= 2:
            res = run_pca(df_f, cols)
            if res:
                color_col = st.selectbox("Color by", ['None'] + NUMERIC_DESCRIPTORS, key='mv_pca_c')
                color_vals = df_f.loc[res['index'], color_col].values if color_col != 'None' else None
                st.pyplot(plot_pca_biplot(res['scores'], res['loadings'], res['explained'], res['cols'], color_vals))
                st.info(f"Explained variance: PC1={res['explained'][0]*100:.1f}%, PC2={res['explained'][1]*100:.1f}%")
                st.dataframe(pd.DataFrame(res['loadings'][:, :2], columns=['PC1', 'PC2'], index=res['cols']))
            else: st.warning("PCA failed")
        elif method == "FAMD" and st.button("Run FAMD", key='mv_famd'):
            cat_cols = st.multiselect("Categorical features", list(FILTER_LABELS.keys()),
                                       default=list(FILTER_LABELS.keys()), key='mv_famd_cat')
            res = run_famd(df_f, cols, cat_cols)
            if res:
                st.info(f"Method: {res['method']}")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(res['scores'][:, 0], res['scores'][:, 1], s=50, c='steelblue', edgecolors='black', alpha=0.8)
                style_axis(ax, f'Dim 1 ({res["explained"][0]*100:.1f}%)', f'Dim 2 ({res["explained"][1]*100:.1f}%)')
                st.pyplot(fig); plt.close(fig)
        elif method == "UMAP" and st.button("Run UMAP", key='mv_umap'):
            if not UMAP_OK: st.warning("umap-learn not installed")
            else:
                res = run_umap(df_f, cols)
                if res:
                    color_col = st.selectbox("Color by", ['None'] + NUMERIC_DESCRIPTORS, key='mv_umap_c')
                    color_vals = df_f.loc[res['index'], color_col].values if color_col != 'None' else None
                    st.pyplot(plot_embedding_2d(res['embedding'], 'UMAP', color_vals))
                else: st.warning("UMAP failed")
        elif method == "t-SNE" and st.button("Run t-SNE", key='mv_tsne'):
            if not SKLEARN_OK: st.warning("sklearn not installed")
            else:
                perplexity = st.slider("Perplexity", 5, 50, 15, key='mv_tsne_p')
                res = run_tsne(df_f, cols, perplexity)
                if res:
                    color_col = st.selectbox("Color by", ['None'] + NUMERIC_DESCRIPTORS, key='mv_tsne_c')
                    color_vals = df_f.loc[res['index'], color_col].values if color_col != 'None' else None
                    st.pyplot(plot_embedding_2d(res['embedding'], 't-SNE', color_vals))
                else: st.warning("t-SNE failed")
        elif method == "Cluster (k-means)" and st.button("Run clustering", key='mv_clust') and len(cols) >= 2 and SKLEARN_OK:
            n_cl = st.slider("Clusters", 2, 8, 3, key='mv_ncl')
            try:
                d = df_f[cols].dropna()
                X = StandardScaler().fit_transform(d)
                km = KMeans(n_clusters=n_cl, random_state=42, n_init=10)
                labels = km.fit_predict(X)
                st.dataframe(pd.Series(labels, index=d.index, name='cluster').value_counts().rename('count'))
                st.dataframe(d.groupby(labels).mean())
            except Exception as e: st.error(f"Error: {e}")

    with tabs[5]:
        st.header("🤖 Regression & Attribution")
        target = st.selectbox("Target", ['Ea_final'] + [c for c in TEMP_COLS if c in df_f.columns], key='rg_t')
        features = st.multiselect("Features", NUMERIC_DESCRIPTORS + ['is_additive'],
                                   default=['is_additive', 'ρ, %', 'T sin', 'd, mkm'], key='rg_f')
        if st.button("Fit Elastic Net", key='rg_btn') and features:
            res = run_elastic_net(df_f, target, features)
            if res:
                st.success(f"alpha = {res['alpha']:.4g}, l1_ratio = {res['l1_ratio']:.2f}, n = {res['n']}")
                coef_df = pd.DataFrame({'Feature': res['cols'], 'Coefficient': res['coefs']}).sort_values('Coefficient', key=abs, ascending=False)
                st.dataframe(coef_df)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh(coef_df['Feature'], coef_df['Coefficient'],
                        color=['red' if v < 0 else 'green' for v in coef_df['Coefficient']], edgecolor='black')
                ax.axvline(0, color='black', lw=0.8); style_axis(ax, 'Coefficient', 'Feature')
                st.pyplot(fig); plt.close(fig)
                perm = run_permutation_importance(res['model'], res['X'], res['y'], res['cols'])
                if perm is not None:
                    st.subheader("Permutation importance"); st.dataframe(perm)
                shap_res = run_shap_linear(res['model'], res['X'], res['cols'])
                if shap_res is not None:
                    st.subheader(f"SHAP ({shap_res['method']})")
                    mean_abs = np.abs(shap_res['shap_values']).mean(axis=0)
                    shap_df = pd.DataFrame({'Feature': shap_res['features'], 'Mean |SHAP|': mean_abs}).sort_values('Mean |SHAP|', ascending=False)
                    st.dataframe(shap_df)
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.barh(shap_df['Feature'], shap_df['Mean |SHAP|'], color='steelblue', edgecolor='black')
                    style_axis(ax, 'Mean |SHAP|', 'Feature'); st.pyplot(fig); plt.close(fig)
                st.subheader("Residual diagnostics")
                try:
                    y_pred = res['model'].predict(res['X']); resid = res['y'] - y_pred
                    c1, c2 = st.columns(2)
                    fig1, ax1 = plt.subplots(figsize=(5, 4))
                    ax1.scatter(y_pred, resid, alpha=0.6, s=30, edgecolors='black'); ax1.axhline(0, color='red', ls='--')
                    style_axis(ax1, 'Fitted', 'Residual'); c1.pyplot(fig1); plt.close(fig1)
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    stats.probplot(resid, dist='norm', plot=ax2); ax2.set_title('QQ plot')
                    c2.pyplot(fig2); plt.close(fig2)
                except: pass
            else: st.warning("Elastic Net failed")
        st.markdown("---"); st.subheader("Bootstrap CI for Elastic Net coefficients")
        if st.button("Bootstrap coefficients", key='boot_btn') and features:
            with st.spinner("Bootstrapping (500 iterations)..."):
                res = bootstrap_en_coefs(df_f, target, features)
            if res:
                fig = plot_bootstrap_coefs(res['samples'], res['cols'])
                if fig: st.pyplot(fig); plt.close(fig)
                ci_df = pd.DataFrame({'Feature': res['cols'], 'Mean': res['mean'],
                                      'CI_lower': res['ci_lower'], 'CI_upper': res['ci_upper']})
                st.dataframe(ci_df.style.format(precision=4))
            else: st.warning("Bootstrap failed (need n >= 3*features)")
        st.markdown("---"); st.subheader("GAM / splines")
        if PYGAM_OK and st.button("Fit GAM (pygam)", key='gam_btn') and features:
            res = run_gam(df_f, target, features)
            if res:
                for i, f in enumerate(res['cols']):
                    try:
                        fig = plot_gam_fit(res['X'], res['y'], res['model'], i, f)
                        if fig: st.pyplot(fig); plt.close(fig)
                    except: pass
            else: st.warning("GAM failed")
        elif not PYGAM_OK and st.button("Fit GAM (statsmodels)", key='gam_sm_btn') and features:
            if not STATSMODELS_OK: st.warning("Neither pygam nor statsmodels available")
            else:
                res = run_gam_statsmodels(df_f, target, features)
                if res: st.success(f"GAM fitted with statsmodels. n = {res['n']}")
                else: st.warning("statsmodels GAM failed")
        st.markdown("---"); st.subheader("PLS regression")
        if SKLEARN_OK and st.button("Run PLS", key='pls_btn') and features:
            res = run_pls(df_f, target, features)
            if res:
                vip_df = pd.DataFrame({'Feature': res['cols'], 'VIP': res['vip']}).sort_values('VIP', ascending=False)
                st.dataframe(vip_df)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh(vip_df['Feature'], vip_df['VIP'], color='steelblue', edgecolor='black'); ax.axvline(1, color='red', ls='--')
                style_axis(ax, 'VIP score', 'Feature'); st.pyplot(fig); plt.close(fig)
            else: st.warning("PLS failed")
        st.markdown("---"); st.subheader("CCA")
        if SKLEARN_OK:
            c1, c2 = st.columns(2)
            x_cca = c1.multiselect("X block", NUMERIC_DESCRIPTORS, default=['dop_cont', 'ρ, %', 'T sin'], key='cca_x')
            y_cca = c2.multiselect("Y block", [c for c in TEMP_COLS if c in df_f.columns],
                                    default=[c for c in TEMP_COLS if c in df_f.columns][:3], key='cca_y')
            if st.button("Run CCA", key='cca_btn') and len(x_cca) >= 2 and len(y_cca) >= 2:
                res = run_cca(df_f, x_cca, y_cca)
                if res: st.pyplot(plot_cca_scores(res['x_scores'], res['y_scores']))
                else: st.warning("CCA failed")

    with tabs[6]:
        st.header("🔬 Interactions & Sensitivity")
        st.subheader("Interaction plot")
        c1, c2, c3 = st.columns(3)
        x_int = c1.selectbox("X", ['dop_cont', 'x, wt%', 'T sin', 'ρ, %', 'd, mkm'], key='int_x')
        y_int = c2.selectbox("Y", ['Ea_final'] + [c for c in TEMP_COLS if c in df_f.columns], key='int_y')
        g_int = c3.selectbox("Group", list(FILTER_LABELS.keys()), key='int_g')
        log_y = st.checkbox("log Y", key='int_logy')
        if st.button("Generate", key='int_btn'):
            try:
                fig = plot_interaction(df_f, x_int, y_int, g_int, log_y)
                if fig: st.pyplot(fig); plt.close(fig)
            except Exception as e: st.error(f"Error: {e}")
        st.markdown("---"); st.subheader("PDP / ALE")
        if SKLEARN_OK:
            features_pdp = st.multiselect("Model features", NUMERIC_DESCRIPTORS + ['is_additive'],
                                           default=['is_additive', 'ρ, %', 'T sin'], key='pdp_f')
            target_pdp = st.selectbox("Target", ['Ea_final'] + [c for c in TEMP_COLS if c in df_f.columns], key='pdp_t')
            if st.button("Compute 1D PDP/ALE", key='pdp_btn') and features_pdp:
                try:
                    d = df_f[features_pdp + [target_pdp]].dropna()
                    if len(d) >= 20:
                        m = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
                        m.fit(d[features_pdp], d[target_pdp])
                        for i, f in enumerate(features_pdp):
                            fig = plot_pdp_1d(m, d[features_pdp].values, i, f)
                            if fig: st.pyplot(fig); plt.close(fig)
                            ale = compute_ale(m, d[features_pdp].values, i)
                            if ale:
                                fig = plot_ale(ale['grid'], ale['ale'], f)
                                if fig: st.pyplot(fig); plt.close(fig)
                    else: st.warning("Need n >= 20")
                except Exception as e: st.error(f"Error: {e}")
            c1, c2 = st.columns(2)
            if features_pdp and len(features_pdp) >= 2:
                f1 = c1.selectbox("Feature 1", features_pdp, key='pdp2_f1')
                f2 = c2.selectbox("Feature 2", features_pdp, index=min(1, len(features_pdp)-1), key='pdp2_f2')
                if st.button("Compute 2D PDP/ALE", key='pdp2_btn') and f1 != f2:
                    try:
                        d = df_f[features_pdp + [target_pdp]].dropna()
                        if len(d) >= 20:
                            m = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
                            m.fit(d[features_pdp], d[target_pdp])
                            i1 = features_pdp.index(f1); i2 = features_pdp.index(f2)
                            fig = plot_pdp_2d(m, d[features_pdp].values, (i1, i2), [f1, f2])
                            if fig: st.pyplot(fig); plt.close(fig)
                            fig = compute_ale_2d(m, d[features_pdp].values, (i1, i2))
                            if fig: st.pyplot(fig); plt.close(fig)
                        else: st.warning("Need n >= 20")
                    except Exception as e: st.error(f"Error: {e}")
        st.markdown("---"); st.subheader("Friedman H (pairwise interactions)")
        if SKLEARN_OK and st.button("Compute Friedman H", key='fh_btn'):
            try:
                d = df_f[NUMERIC_DESCRIPTORS + ['Ea_final']].dropna()
                if len(d) >= 20:
                    m = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
                    m.fit(d[NUMERIC_DESCRIPTORS], d['Ea_final'])
                    pairs, hs = [], []
                    for i in range(min(4, len(NUMERIC_DESCRIPTORS))):
                        for j in range(i+1, min(4, len(NUMERIC_DESCRIPTORS))):
                            h = compute_friedman_H(m, d[NUMERIC_DESCRIPTORS].values, (i, j))
                            if h is not None:
                                pairs.append((NUMERIC_DESCRIPTORS[i], NUMERIC_DESCRIPTORS[j]))
                                hs.append(h)
                    if pairs:
                        fig = plot_friedman_H(pairs, hs)
                        if fig: st.pyplot(fig); plt.close(fig)
                    else: st.warning("Friedman H computation failed")
                else: st.warning("Need n >= 20")
            except Exception as e: st.error(f"Error: {e}")
        st.markdown("---"); st.subheader("Sobol variance decomposition")
        if st.button("Compute Sobol", key='sb_btn'):
            with st.spinner("Computing Sobol indices..."):
                res = run_sobol(df_f, 'Ea_final', NUMERIC_DESCRIPTORS)
            if res is not None:
                st.info(f"Method: {res['method']}")
                df_res = res['df']
                st.dataframe(df_res.style.format(precision=4))
                col = 'ST' if 'ST' in df_res.columns else 'ST_approx'
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(df_res['Variable'], df_res[col], color='steelblue', edgecolor='black')
                style_axis(ax, f'{col} index', 'Variable'); st.pyplot(fig); plt.close(fig)
            else: st.warning("Sobol computation failed")

    with tabs[7]:
        st.header("🌡️ σ(T) Curve Analysis")
        st.subheader("Clustering")
        n_cl = st.slider("Clusters", 2, 6, 3, key='cv_ncl')
        if st.button("Cluster curves", key='cv_btn'):
            res = run_curve_clustering(df_f, n_cl)
            if res:
                st.pyplot(plot_curve_clusters(res['curves'], res['labels'], res['centroids']))
                cluster_df = pd.DataFrame({'cluster': res['labels']}, index=res['valid_index'])
                merged = df_f.loc[res['valid_index']].join(cluster_df)
                if 'Sintering additive' in merged.columns:
                    st.subheader("Additive distribution per cluster")
                    st.dataframe(pd.crosstab(merged['cluster'], merged['Sintering additive']))
                mean_cols = [c for c in ['Ea_LT', 'Ea_HT', 'Ea_diff', 'knee_T', 'sigma_500'] if c in merged.columns]
                if mean_cols: st.dataframe(merged.groupby('cluster')[mean_cols].mean())
            else: st.warning("Clustering failed")
        st.markdown("---"); st.subheader("FPCA")
        if st.button("Run FPCA", key='fpca_btn'):
            res = run_fpca(df_f, n_components=3)
            if res:
                st.info(f"Method: {res['method']}")
                st.pyplot(plot_fpca_components(res['components'], res['explained'], res['T_vals']))
                st.info("Explained variance: " + ", ".join([f"FPC{i+1}={r*100:.1f}%" for i, r in enumerate(res['explained'][:3])]))
                scores_df = pd.DataFrame(res['scores'], index=res['valid_index'])
                st.dataframe(scores_df.head(20))
            else: st.warning("FPCA failed")
        st.markdown("---"); st.subheader("Segmented regression (Ea per segment)")
        c1, c2 = st.columns(2)
        if c1.button("Batch segmented regression (all curves)", key='seg_batch_btn'):
            try:
                res = run_segmented_regression_batch(df_f, n_segments=2)
                if res is not None and len(res) > 0:
                    st.dataframe(res)
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.hist(res['T_C'].dropna(), bins=20, edgecolor='black', color='steelblue')
                    style_axis(ax, 'Breakpoint T (°C)', 'Count'); ax.set_title('Breakpoint distribution')
                    st.pyplot(fig); plt.close(fig)
                    st.download_button("📥 Download breakpoints (TSV)", res.to_csv(sep='\t', index=False),
                                        file_name="breakpoints.tsv", mime="text/tab-separated-values")
                else: st.warning("Batch failed")
            except Exception as e: st.error(f"Error: {e}")
        if c2.button("Single-curve segmented regression", key='seg_btn'):
            try:
                curves = df_f[TEMP_COLS].values.astype(float)
                valid = np.array([np.isfinite(c).sum() >= 3 and np.all(np.where(np.isfinite(c), c > 0, True)) for c in curves])
                if valid.sum() >= 5:
                    vc = curves[valid][0]
                    T_vals = np.array(TEMP_VALUES)
                    mask = np.isfinite(vc) & (vc > 0)
                    x_arr = 1000 / (T_vals[mask] + 273.15)
                    y_arr = np.log(vc[mask] * (T_vals[mask] + 273.15))
                    res = run_segmented_regression(x_arr, y_arr)
                    if res and 'breakpoints' in res:
                        bp = res['breakpoints'][0]
                        seg1 = res.get('seg1_params'); seg2 = res.get('seg2_params')
                        if seg1 is not None and seg2 is not None:
                            fig = plot_segmented_regression(x_arr, y_arr, bp, seg1, seg2, '1000/T', 'log(σT)')
                            if fig: st.pyplot(fig); plt.close(fig)
                            st.info(f"Breakpoint at 1000/T = {bp:.4f} (T ≈ {1000/bp - 273.15:.0f} °C)")
                        else: st.info(f"Breakpoint at 1000/T = {bp:.4f}")
                else: st.warning("Insufficient valid curves")
            except Exception as e: st.error(f"Error: {e}")
        st.markdown("---"); st.subheader("Scalar curve features")
        feature_cols = [c for c in ['Ea_LT', 'Ea_HT', 'Ea_diff', 'sigma_500', 'sigma_ratio', 'knee_T', 'curve_area'] if c in df_f.columns]
        if feature_cols:
            st.dataframe(df_f[feature_cols].describe())
            if 'Sintering additive' in df_f.columns:
                st.dataframe(df_f.groupby('Sintering additive')[feature_cols].mean())
            st.download_button("📥 Download curve features (TSV)", df_f[feature_cols].to_csv(sep='\t'),
                                file_name="curve_features.tsv", mime="text/tab-separated-values")

    with tabs[8]:
        st.header("🎯 Additive vs Pure — ratio-based analysis")
        if 'Sintering additive' not in df_f.columns or 'References' not in df_f.columns or 'x, wt%' not in df_f.columns:
            st.warning("Need 'Sintering additive', 'References', and 'x, wt%' columns")
        else:
            _temp_opts = [c for c in TEMP_COLS if c in df_f.columns]
            _default_temp = 'σ total, 600' if 'σ total, 600' in df_f.columns else (_temp_opts[0] if _temp_opts else None)
            c1, c2, c3 = st.columns(3)
            temp_col = c1.selectbox("Temperature for σ", _temp_opts,
                                     index=_temp_opts.index(_default_temp) if _default_temp in _temp_opts else 0,
                                     key='ap_temp')
            y_scale = c2.selectbox("Y-scale", ['log10', 'ratio'], index=0, key='ap_yscale')
            heat_metric = c3.selectbox("Heat scale metric",
                                        ['abs_delta_rho', 'abs_delta_d', 'd_ratio'],
                                        format_func=lambda x: {'abs_delta_rho': '|Δρ|',
                                                                'abs_delta_d': '|Δd|',
                                                                'd_ratio': 'd_additive/d_Pure'}[x],
                                        index=0, key='ap_heat')
            c1, c2, c3 = st.columns(3)
            _ap_pal_default = 'YlOrRd' if 'YlOrRd' in COLOR_PALETTES else 'Viridis'
            _ap_pal_idx = list(COLOR_PALETTES.keys()).index(_ap_pal_default)
            pal_ap = c1.selectbox("Palette", list(COLOR_PALETTES.keys()),
                                   index=_ap_pal_idx, key='ap_pal',
                                   help="Sequential or 3-gradient schemes are recommended — small |Δ| values will appear light, not dark.")
            additive_filter = c2.multiselect("Filter additives", sorted([a for a in df_f['Sintering additive'].dropna().unique() if a != 'Pure']),
                                              default=[], key='ap_addfilter',
                                              help="Leave empty to show all")
            facet_by = c3.selectbox("Facet by", ['None', 'Atmospheres', 'Humidity', 'Structure', 'Method'],
                                     key='ap_facet')
            clip_on = st.checkbox("Clip Y axis (limit visible range)", value=False, key='ap_clip')
            if clip_on:
                if y_scale == 'log10':
                    lo, hi = st.slider("Y visible range", -4.0, 4.0, (-2.0, 2.0), 0.1, key='ap_clip_range')
                else:
                    lo, hi = st.slider("Y visible range", 0.01, 200.0, (0.01, 100.0), key='ap_clip_range')
                clip_range = (lo, hi)
            else:
                clip_range = None

            # rebuild pairs when any control that affects pairing changes
            _pair_key = (temp_col, heat_metric, tuple(sorted(additive_filter)))
            if st.session_state.get('ap_pairs_key') != _pair_key or 'ap_pairs_df' not in st.session_state:
                pairs_df = build_pairs_within_reference(df_f, temperature_col=temp_col, heat_metric=heat_metric)
                if pairs_df is not None and len(additive_filter) > 0:
                    pairs_df = pairs_df[pairs_df['Additive'].isin(additive_filter)].copy()
                st.session_state['ap_pairs_df'] = pairs_df
                st.session_state['ap_pairs_key'] = _pair_key
            else:
                pairs_df = st.session_state['ap_pairs_df']

            if pairs_df is None or len(pairs_df) == 0:
                st.warning("No valid pairs found. Check that Pure and additive samples have valid σ at chosen temperature.")
            else:
                st.success(f"✅ {len(pairs_df)} pairs from {pairs_df['Reference'].nunique()} references "
                           f"({pairs_df['Additive'].nunique()} additives)")
                st.caption(f"Pure selection rule: for each reference, the Pure sample with the LOWEST σ at {temp_col} is used as baseline.")

                # main plot
                st.subheader("σ_with / σ_without vs additive concentration")
                facet_groups = None
                if facet_by != 'None' and facet_by in pairs_df.columns:
                    facet_groups = sorted(pairs_df[facet_by].dropna().unique())
                try:
                    fig = plot_additive_vs_pure(pairs_df, y_scale=y_scale, heat_metric=heat_metric,
                                                 palette=COLOR_PALETTES[pal_ap], clip_range=clip_range,
                                                 facet_by=(facet_by if facet_by != 'None' else None),
                                                 facet_groups=facet_groups,
                                                 temperature_col=temp_col)
                    if fig is not None:
                        st.pyplot(fig)
                        buf = download_plot(fig, "additive_vs_pure.png")
                        st.download_button("📥 Download plot (PNG)", data=buf,
                                            file_name="additive_vs_pure.png", mime="image/png",
                                            key='ap_dl_plot')
                        plt.close(fig)
                except Exception as e:
                    st.error(f"Plot failed: {e}")

                # numeric summary
                st.subheader("Numeric summary")
                summary = summarize_pairs(pairs_df, y_scale=y_scale)
                if summary:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Same T_sin (additive and Pure sintered at same temperature)**")
                        st.markdown(f"- n pairs: **{summary['n_same']}**")
                        if summary['n_same'] > 0:
                            st.markdown(f"- mean ratio: **{summary['mean_ratio_same']:.3f}**")
                            st.markdown(f"- median ratio: **{summary['median_ratio_same']:.3f}**")
                            st.markdown(f"- fraction with ratio > 1: **{summary['frac_above_1_same']*100:.1f}%**")
                            st.markdown(f"- Wilcoxon (y vs no-effect) p = **{summary['wilcoxon_p_same']:.3g}**")
                    with c2:
                        st.markdown("**Different T_sin**")
                        st.markdown(f"- n pairs: **{summary['n_diff']}**")
                        if summary['n_diff'] > 0:
                            st.markdown(f"- mean ratio: **{summary['mean_ratio_diff']:.3f}**")
                            st.markdown(f"- median ratio: **{summary['median_ratio_diff']:.3f}**")
                            st.markdown(f"- fraction with ratio > 1: **{summary['frac_above_1_diff']*100:.1f}%**")
                            st.markdown(f"- Wilcoxon (y vs no-effect) p = **{summary['wilcoxon_p_diff']:.3g}**")
                    st.markdown("**Spearman correlations (all pairs)**")
                    st.markdown(f"- |Δρ| ↔ ratio: ρ = **{summary['spearman_rho']:.3f}**, p = **{summary['spearman_rho_p']:.3g}**")
                    st.markdown(f"- |Δd| ↔ ratio: ρ = **{summary['spearman_d']:.3f}**, p = **{summary['spearman_d_p']:.3g}**")

                # pair table
                st.subheader("Pair table")
                st.dataframe(pairs_df)
                st.download_button("📥 Download pairs (TSV)", pairs_df.to_csv(sep='\t', index=False),
                                    file_name="additive_vs_pure_pairs.tsv",
                                    mime="text/tab-separated-values", key='ap_dl_pairs')

                # additional plots
                st.subheader("Additional diagnostics")
                c1, c2 = st.columns(2)
                if c1.button("Δρ vs ratio", key='ap_plot_drho'):
                    try:
                        fig = plot_ratio_vs_delta(pairs_df, delta_col='d_rho', y_scale=y_scale)
                        if fig is not None:
                            st.pyplot(fig); plt.close(fig)
                    except Exception as e: st.error(f"Failed: {e}")
                if c2.button("Δd vs ratio", key='ap_plot_dd'):
                    try:
                        fig = plot_ratio_vs_delta(pairs_df, delta_col='d_d', y_scale=y_scale)
                        if fig is not None:
                            st.pyplot(fig); plt.close(fig)
                    except Exception as e: st.error(f"Failed: {e}")
                c1, c2 = st.columns(2)
                if c1.button("Box by additive", key='ap_plot_boxa'):
                    try:
                        fig = plot_ratio_boxplot(pairs_df, group_col='Additive', y_scale=y_scale)
                        if fig is not None:
                            st.pyplot(fig); plt.close(fig)
                    except Exception as e: st.error(f"Failed: {e}")
                if c2.button("Box by same/different T_sin", key='ap_plot_boxt'):
                    try:
                        _tmp = pairs_df.copy()
                        _tmp['Tsin_group'] = _tmp['same_Tsin'].map({True: 'Same Tsin', False: 'Different Tsin'})
                        fig = plot_ratio_boxplot(_tmp, group_col='Tsin_group', y_scale=y_scale)
                        if fig is not None:
                            st.pyplot(fig); plt.close(fig)
                    except Exception as e: st.error(f"Failed: {e}")

    with tabs[9]:
        st.header("📋 Data & Export")
        st.subheader("Filtered data with all derived features")
        st.dataframe(df_f)
        st.download_button("📥 Download full filtered data (TSV)", df_f.to_csv(sep='\t', index=False),
                            file_name="filtered_data.tsv", mime="text/tab-separated-values")
        st.subheader("Summary statistics")
        st.dataframe(df_f.select_dtypes(include=[np.number]).describe())

if __name__ == "__main__":
    main()
