# ============================================
# app.py - Полный код Streamlit приложения
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
import io
import warnings
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION AND INITIALIZATION
# ============================================

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.5,
    'legend.edgecolor': 'black',
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

COLOR_PALETTES = {
    'Viridis': 'viridis',
    'Plasma': 'plasma',
    'Inferno': 'inferno',
    'Magma': 'magma',
    'Cividis': 'cividis',
    'Turbo': 'turbo',
    'Jet': 'jet',
    'Rainbow': 'rainbow',
    'Spectral': 'Spectral',
    'Coolwarm': 'coolwarm'
}

# Human-readable labels for numeric descriptors (used in legend titles)
COLUMN_LABELS = {
    'dop_cont': 'Dopant content',
    'x, wt%': 'Sintering additive concentration',
    'ρ, %': 'Density',
    'd, mkm': 'Grain size',
    'tolerance_factor': 'Tolerance factor',
    'chi_B_avg': 'Average electronegativity (B-site)',
    'chi_ratio': 'Electronegativity ratio',
    'molar_mass': 'Molar mass',
    'porosity': 'Porosity',
    'grain_boundary_area': 'Grain boundary area',
    'Ea_final': 'Activation energy',
    'Ea_calculated': 'Activation energy (calculated)',
    'T sin': 'Sintering temperature',
    'T sin, °C': 'Sintering temperature',
}

# Human-readable labels for filter columns (used in legend titles)
FILTER_LABELS = {
    'Sintering additive': 'Sintering additive',
    'Atmospheres': 'Atmosphere',
    'Humidity': 'Humidity',
    'Structure': 'Structure',
}

# Common marker set used for every active filter (order matters)
COMMON_MARKERS = ['o', 's', 'D', '*', '^']

# Marker dictionaries for different filters
MARKER_MAP = {
    'Sintering additive': {
        'Pure': 'o',
        'Cu': 's',
        'Ni': 'D',
        'Zn': '*',
        'Fe': '^',
        'Co': 'v',
        'Mn': '<',
        'Cr': '>',
        'Ag': 'p',
        'Au': 'P',
        'Pt': 'X'
    },
    'Atmospheres': {
        'Ox': 'o',
        'Redox': 's',
        'Inert': '^',
        'Reducing': 'v',
        'Air': 'D'
    },
    'Humidity': {
        'wet': 'D',
        'dry': '*',
        'humid': 's',
        'Wet': 'D',
        'Dry': '*'
    },
    'Structure': {
        'Cubic': 'o',
        'orthorhombic': 's',
        'hexagonal': '^',
        'monoclinic': 'D',
        'tetragonal': '*',
        'rhombohedral': 'v'
    }
}

IONIC_RADII = {
    'Ba': 1.61,
    'Zr': 0.72,
    'Ce': 0.87,
    'Sn': 0.69,
    'Y': 0.90,
    'Gd': 0.94,
    'Yb': 0.87,
    'Sm': 0.96,
    'O': 1.40
}

ELECTRONEGATIVITY = {
    'Ba': 0.89,
    'Zr': 1.33,
    'Ce': 1.12,
    'Sn': 1.96,
    'Y': 1.22,
    'Gd': 1.20,
    'Yb': 1.10,
    'Sm': 1.17,
    'O': 3.44
}

MOLAR_MASS = {
    'Ba': 137.33,
    'Zr': 91.22,
    'Ce': 140.12,
    'Sn': 118.71,
    'Y': 88.91,
    'Gd': 157.25,
    'Yb': 173.05,
    'Sm': 150.36,
    'O': 16.00
}

# ============================================
# CACHED FUNCTIONS FOR DESCRIPTOR CALCULATION
# ============================================

@st.cache_data
def compute_descriptors_cached(df_hash, df):
    """
    Cached version of descriptor calculation.
    df_hash is used as a cache key - when data changes, hash changes.
    """
    desc_df = pd.DataFrame(index=df.index)
    
    for idx, row in df.iterrows():
        x_B2 = row['B2_cont'] if pd.notna(row['B2_cont']) else 0.0
        x_dop = row['dop_cont'] if pd.notna(row['dop_cont']) else 0.0
        
        has_B2 = pd.notna(row['B2 cation']) and row['B2 cation'] != ''
        
        if has_B2:
            x_B1 = 1.0 - x_B2 - x_dop
        else:
            x_B1 = 1.0 - x_dop
            x_B2 = 0.0
        
        r_B = 0.0
        if x_B1 > 0:
            r_B += x_B1 * IONIC_RADII[row['B1 cation']]
        if x_B2 > 0 and has_B2:
            r_B += x_B2 * IONIC_RADII[row['B2 cation']]
        if x_dop > 0:
            r_B += x_dop * IONIC_RADII[row['dopant']]
        
        r_A = IONIC_RADII['Ba']
        r_O = IONIC_RADII['O']
        t_factor = (r_A + r_O) / (np.sqrt(2) * (r_B + r_O))
        
        chi_B = 0.0
        if x_B1 > 0:
            chi_B += x_B1 * ELECTRONEGATIVITY[row['B1 cation']]
        if x_B2 > 0 and has_B2:
            chi_B += x_B2 * ELECTRONEGATIVITY[row['B2 cation']]
        if x_dop > 0:
            chi_B += x_dop * ELECTRONEGATIVITY[row['dopant']]
        
        chi_ratio = chi_B / ELECTRONEGATIVITY['Ba']
        
        molar_mass = 0.0
        molar_mass += x_B1 * MOLAR_MASS[row['B1 cation']]
        if x_B2 > 0 and has_B2:
            molar_mass += x_B2 * MOLAR_MASS[row['B2 cation']]
        if x_dop > 0:
            molar_mass += x_dop * MOLAR_MASS[row['dopant']]
        molar_mass += 3 * MOLAR_MASS['O']
        
        rho = row['ρ, %'] if pd.notna(row['ρ, %']) else np.nan
        d = row['d, mkm'] if pd.notna(row['d, mkm']) else np.nan
        
        porosity = 100 - rho if pd.notna(rho) else np.nan
        gb_area = 3.722 / d if pd.notna(d) and d > 0 else np.nan
        
        desc_df.loc[idx, 'tolerance_factor'] = t_factor
        desc_df.loc[idx, 'chi_B_avg'] = chi_B
        desc_df.loc[idx, 'chi_ratio'] = chi_ratio
        desc_df.loc[idx, 'molar_mass'] = molar_mass
        desc_df.loc[idx, 'porosity'] = porosity
        desc_df.loc[idx, 'grain_boundary_area'] = gb_area
        desc_df.loc[idx, 'x_B1'] = x_B1
        desc_df.loc[idx, 'x_B2'] = x_B2
        desc_df.loc[idx, 'x_dop'] = x_dop
    
    return desc_df

@st.cache_data
def calculate_ea_cached(df_hash, df):
    """
    Cached version of Ea calculation.
    """
    result = []
    for idx, row in df.iterrows():
        if pd.notna(row['Ea (eV)']) and row['Ea (eV)'] != '':
            result.append(row['Ea (eV)'])
            continue
        
        temp_cols = ['σ total, 200', 'σ total, 250', 'σ total, 300', 'σ total, 350',
                     'σ total, 400', 'σ total, 450', 'σ total, 500', 'σ total, 550',
                     'σ total, 600', 'σ total, 650', 'σ total, 700', 'σ total, 750',
                     'σ total, 800', 'σ total, 850', 'σ total, 900']
        
        temps = []
        sigmas = []
        
        for col in temp_cols:
            if col in row.index and pd.notna(row[col]) and row[col] > 0:
                T = int(col.split(', ')[1])
                temps.append(T + 273.15)
                sigmas.append(row[col] * 1e-3)
        
        if len(temps) < 2:
            result.append(np.nan)
            continue
        
        y = np.log(np.array(sigmas) * np.array(temps))
        x = 1000 / np.array(temps)
        
        try:
            slope, intercept = np.polyfit(x, y, 1)
            ea = -slope * 8.314 / 96485
            result.append(ea)
        except:
            result.append(np.nan)
    
    return result

# ============================================
# FUNCTION FOR OUTLIER REMOVAL
# ============================================

def remove_outliers(df, col, n):
    """
    Remove n smallest and n largest values from the dataset for a given column.
    Logic:
    - If selected value is 1: remove the single smallest and single largest value
    - If the extreme value appears more than n times, none of those rows are removed
    - We remove individual rows, not unique values
    """
    if n is None or n == 0:
        return df
    
    if col not in df.columns:
        return df
    
    # Get all values sorted
    values = df[col].dropna().sort_values().values
    
    if len(values) <= 2 * n:
        return df
    
    # Find which rows to remove from the low end
    low_rows_to_remove = []
    removed_count = 0
    
    # Iterate from smallest to largest
    for val in values:
        if removed_count >= n:
            break
        # Count how many rows have this exact value
        count = (df[col] == val).sum()
        # If count <= remaining removals, remove all of them
        if count <= (n - removed_count):
            low_rows_to_remove.extend(df[df[col] == val].index.tolist())
            removed_count += count
        else:
            # If count > remaining removals, we cannot remove any of this value
            break
    
    # Find which rows to remove from the high end
    high_rows_to_remove = []
    removed_count = 0
    
    # Iterate from largest to smallest
    for val in reversed(values):
        if removed_count >= n:
            break
        # Count how many rows have this exact value
        count = (df[col] == val).sum()
        # If count <= remaining removals, remove all of them
        if count <= (n - removed_count):
            high_rows_to_remove.extend(df[df[col] == val].index.tolist())
            removed_count += count
        else:
            # If count > remaining removals, we cannot remove any of this value
            break
    
    # Create filter mask - keep rows that are NOT in removal lists
    rows_to_remove = set(low_rows_to_remove + high_rows_to_remove)
    mask = ~df.index.isin(rows_to_remove)
    
    return df[mask]

# ============================================
# FUNCTIONS FOR PLOTTING
# ============================================

def get_filter_markers(df, active_filters_count):
    """
    Determine if custom markers should be used based on active filters count.
    Returns: (marker_dict or 'o', show_legend (bool), filter_name (str or None))

    Only one filter is used to control marker shapes. When exactly one filter is
    active (i.e. its categories really split the data), that filter's categories
    receive markers from COMMON_MARKERS in order. If several filters are active,
    marker shapes are NOT shown (show_legend=False).
    """
    if active_filters_count != 1:
        return 'o', False, None

    # Find which filter is active (has more than one unique value)
    for filter_name in FILTER_LABELS.keys():
        if filter_name in df.columns:
            unique_vals = df[filter_name].dropna().unique()
            unique_vals = [v for v in unique_vals if v != '']
            if len(unique_vals) > 1:
                # Build marker dict using COMMON_MARKERS in the order of categories
                marker_dict = {}
                for i, cat in enumerate(unique_vals):
                    marker_dict[cat] = COMMON_MARKERS[i % len(COMMON_MARKERS)]
                return marker_dict, True, filter_name

    return 'o', False, None


def _humanize_size_label(size_col_name):
    """Return a human-readable label for a size descriptor column."""
    if size_col_name in COLUMN_LABELS:
        return COLUMN_LABELS[size_col_name]
    return size_col_name


def _humanize_filter_label(filter_name):
    """Return a human-readable label for a filter column."""
    if filter_name in FILTER_LABELS:
        return FILTER_LABELS[filter_name]
    return filter_name


def _plot_kde_marginal(ax, groups, ylabel, normalize, title):
    """
    Plot smoothed KDE marginals on the given axes.

    groups: list of (label, values_array) tuples.
    normalize: if True, each KDE is normalized to integrate to 1 inside its group
               (gaussian_kde is already normalized per-sample, so we still scale
               so that each group's density integrates to 1 over the plot range).
    """
    y_min = None
    y_max = None
    for _, vals in groups:
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        if y_min is None or vals.min() < y_min:
            y_min = vals.min()
        if y_max is None or vals.max() > y_max:
            y_max = vals.max()

    if y_min is None or y_max is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel(ylabel, fontsize=9, fontweight='bold')
        return

    span = y_max - y_min
    if span <= 0:
        span = 1.0
    y_grid = np.linspace(y_min - 0.1 * span, y_max + 0.1 * span, 200)

    cmap = plt.get_cmap('viridis')
    n_groups = max(1, len(groups))
    for i, (label, vals) in enumerate(groups):
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        try:
            kde = gaussian_kde(vals)
            dens = kde(y_grid)
        except Exception:
            continue
        if normalize:
            # Normalize density to integrate to 1 over y_grid
            area = np.trapezoid(dens, y_grid)
            if area > 0:
                dens = dens / area
        color = cmap(i / max(1, n_groups - 1)) if n_groups > 1 else cmap(0.0)
        ax.fill_betweenx(y_grid, 0, dens, color=color, alpha=0.35, label=label)
        ax.plot(dens, y_grid, color=color, linewidth=1.5)

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Density' if not normalize else 'Normalized density',
                  fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    if len(groups) > 0:
        ax.legend(fontsize=8, loc='best', frameon=True, framealpha=0.5)


def create_scatter_heatmap(df, x_col, y_col, z_col, x_log, y_log, z_log, 
                           palette, title, xlabel, ylabel, zlabel,
                           marker_style='o', show_legend=False, filter_name=None):
    plot_data = df.dropna(subset=[x_col, y_col, z_col])
    
    if len(plot_data) == 0:
        st.warning("No data available for plotting")
        return None
    
    x = plot_data[x_col].values
    y = plot_data[y_col].values
    z = plot_data[z_col].values
    
    if x_log:
        mask = x > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) == 0:
            st.warning("All x values ≤ 0, cannot create logarithmic plot")
            return None
        x = np.log10(x)
        xlabel = f'log10({xlabel})'
    
    if y_log:
        mask = y > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(y) == 0:
            st.warning("All y values ≤ 0, cannot create logarithmic plot")
            return None
        y = np.log10(y)
        ylabel = f'log10({ylabel})'
    
    if z_log:
        mask = z > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(z) == 0:
            st.warning("All z values ≤ 0, cannot create logarithmic plot")
            return None
        z = np.log10(z)
        zlabel = f'log10({zlabel})'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if show_legend and isinstance(marker_style, dict) and filter_name is not None:
        # Use different markers for each category
        categories = plot_data[filter_name].unique()
        for cat in categories:
            mask = plot_data[filter_name] == cat
            if mask.sum() > 0:
                marker = marker_style.get(cat, 'o')
                ax.scatter(x[mask], y[mask], c=z[mask], cmap=palette, 
                          s=50, marker=marker, edgecolors='black', 
                          linewidth=0.5, alpha=0.8, label=cat)
        
        # Create colorbar from all data
        scatter = ax.scatter(x, y, c=z, cmap=palette, s=50, 
                            edgecolors='black', linewidth=0.5, alpha=0.8, visible=False)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(zlabel, fontsize=11, fontweight='bold')
        
        ax.legend(loc='upper right', frameon=True, framealpha=0.5)
    else:
        scatter = ax.scatter(x, y, c=z, cmap=palette, s=50, 
                            marker=marker_style if isinstance(marker_style, str) else 'o',
                            edgecolors='black', linewidth=0.5, alpha=0.8)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(zlabel, fontsize=11, fontweight='bold')
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def create_contour_heatmap(df, x_col, y_col, z_col, x_log, y_log, z_log,
                           palette, title, xlabel, ylabel, zlabel, 
                           grid_resolution=50, show_contour_lines=True,
                           show_contour_labels=True):
    plot_data = df.dropna(subset=[x_col, y_col, z_col])
    
    if len(plot_data) < 4:
        st.warning("Not enough data for contour plot (minimum 4 points required)")
        return None
    
    x = plot_data[x_col].values
    y = plot_data[y_col].values
    z = plot_data[z_col].values
    
    if x_log:
        mask = x > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) < 4:
            st.warning("Not enough data after logarithmic transformation")
            return None
        x = np.log10(x)
        xlabel = f'log10({xlabel})'
    
    if y_log:
        mask = y > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) < 4:
            st.warning("Not enough data after logarithmic transformation")
            return None
        y = np.log10(y)
        ylabel = f'log10({ylabel})'
    
    if z_log:
        mask = z > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) < 4:
            st.warning("Not enough data after logarithmic transformation")
            return None
        z = np.log10(z)
        zlabel = f'log10({zlabel})'
    
    xi = np.linspace(x.min(), x.max(), grid_resolution)
    yi = np.linspace(y.min(), y.max(), grid_resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    try:
        zi = griddata((x, y), z, (xi, yi), method='cubic')
    except:
        try:
            zi = griddata((x, y), z, (xi, yi), method='linear')
        except:
            st.warning("Failed to interpolate data")
            return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create filled contour with smooth color transition
    contour = ax.contourf(xi, yi, zi, levels=30, cmap=palette, alpha=0.85)
    
    # Add contour lines if requested
    if show_contour_lines:
        contour_lines = ax.contour(xi, yi, zi, levels=15, colors='black', 
                                   linewidths=0.5, alpha=0.4)
        if show_contour_labels:
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.2f')
    
    # Plot original data points
    ax.scatter(x, y, color='red', s=30, edgecolors='white', 
               linewidth=1, alpha=0.7, label='Data points')
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(zlabel, fontsize=11, fontweight='bold')
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def create_3d_surface(df, x_col, y_col, z_col, x_log, y_log, z_log,
                      palette, title, xlabel, ylabel, zlabel):
    plot_data = df.dropna(subset=[x_col, y_col, z_col])
    
    if len(plot_data) < 4:
        st.warning("Not enough data for 3D surface plot (minimum 4 points required)")
        return None
    
    x = plot_data[x_col].values
    y = plot_data[y_col].values
    z = plot_data[z_col].values
    
    if x_log:
        mask = x > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) < 4:
            st.warning("Not enough data after logarithmic transformation")
            return None
        x = np.log10(x)
        xlabel = f'log10({xlabel})'
    
    if y_log:
        mask = y > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) < 4:
            st.warning("Not enough data after logarithmic transformation")
            return None
        y = np.log10(y)
        ylabel = f'log10({ylabel})'
    
    if z_log:
        mask = z > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) < 4:
            st.warning("Not enough data after logarithmic transformation")
            return None
        z = np.log10(z)
        zlabel = f'log10({zlabel})'
    
    xi = np.linspace(x.min(), x.max(), 30)
    yi = np.linspace(y.min(), y.max(), 30)
    xi, yi = np.meshgrid(xi, yi)
    
    try:
        zi = griddata((x, y), z, (xi, yi), method='cubic')
    except:
        try:
            zi = griddata((x, y), z, (xi, yi), method='linear')
        except:
            st.warning("Failed to interpolate data for 3D plot")
            return None
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(xi, yi, zi, cmap=palette, alpha=0.8, 
                           linewidth=0, antialiased=True)
    
    ax.scatter(x, y, z, color='red', s=30, alpha=0.7, label='Data points')
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_zlabel(zlabel, fontsize=11, fontweight='bold')
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(zlabel, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_bubble_chart(df, x_col, y_col, color_col, size_col, 
                        x_log, y_log, color_log, size_log,
                        palette, title, xlabel, ylabel,
                        marker_style='o', show_legend=False, filter_name=None,
                        size_col_name=None, filter_display_name=None,
                        show_marginals=False, normalize_marginals=False,
                        size_col_raw=None, y_col_raw=None):
    plot_data = df.dropna(subset=[x_col, y_col, color_col, size_col]).copy()
    
    if len(plot_data) == 0:
        st.warning("No data available for plotting")
        return None
    
    # Keep raw size for terciles (before log transform)
    if size_col_raw is not None and size_col_raw in plot_data.columns:
        raw_size_series = plot_data[size_col_raw].copy()
    else:
        raw_size_series = plot_data[size_col].copy()
    
    # Keep raw y for marginals (before log transform)
    if y_col_raw is not None and y_col_raw in plot_data.columns:
        raw_y_series = plot_data[y_col_raw].copy()
    else:
        raw_y_series = plot_data[y_col].copy()
    
    x = plot_data[x_col].values
    y = plot_data[y_col].values
    colors = plot_data[color_col].values
    sizes = plot_data[size_col].values
    
    if x_log:
        mask = x > 0
        x = x[mask]
        y = y[mask]
        colors = colors[mask]
        sizes = sizes[mask]
        plot_data = plot_data[mask]
        raw_size_series = raw_size_series[mask]
        raw_y_series = raw_y_series[mask]
        if len(x) == 0:
            st.warning("All x values ≤ 0, cannot create logarithmic plot")
            return None
        x = np.log10(x)
        xlabel = f'log10({xlabel})'
    
    if y_log:
        mask = y > 0
        x = x[mask]
        y = y[mask]
        colors = colors[mask]
        sizes = sizes[mask]
        plot_data = plot_data[mask]
        raw_size_series = raw_size_series[mask]
        raw_y_series = raw_y_series[mask]
        if len(y) == 0:
            st.warning("All y values ≤ 0, cannot create logarithmic plot")
            return None
        y = np.log10(y)
        ylabel = f'log10({ylabel})'
    
    if color_log:
        mask = colors > 0
        x = x[mask]
        y = y[mask]
        colors = colors[mask]
        sizes = sizes[mask]
        plot_data = plot_data[mask]
        raw_size_series = raw_size_series[mask]
        raw_y_series = raw_y_series[mask]
        if len(colors) == 0:
            st.warning("All color values ≤ 0, cannot create logarithmic plot")
            return None
        colors = np.log10(colors)
    
    if size_log:
        mask = sizes > 0
        x = x[mask]
        y = y[mask]
        colors = colors[mask]
        sizes = sizes[mask]
        plot_data = plot_data[mask]
        raw_size_series = raw_size_series[mask]
        raw_y_series = raw_y_series[mask]
        if len(sizes) == 0:
            st.warning("All size values ≤ 0, cannot create logarithmic plot")
            return None
        sizes = np.log10(sizes)
    
    # Scale bubble sizes for the plot
    if len(sizes) > 0:
        size_min, size_max = sizes.min(), sizes.max()
        if size_max > size_min:
            sizes_scaled = 20 + 180 * (sizes - size_min) / (size_max - size_min)
        else:
            sizes_scaled = np.ones_like(sizes) * 50
    else:
        sizes_scaled = np.ones_like(sizes) * 50
    
    # Determine whether we should build the marginal figure layout
    build_marginals = False
    if show_marginals:
        has_size_groups = (size_col_name is not None) and (size_col_raw is not None)
        has_filter_groups = (filter_display_name is not None) and (filter_name is not None) and show_legend
        build_marginals = has_size_groups or has_filter_groups
    
    if build_marginals:
        n_marginals = 0
        if (size_col_name is not None) and (size_col_raw is not None):
            n_marginals += 1
        if (filter_display_name is not None) and (filter_name is not None) and show_legend:
            n_marginals += 1
        # Layout: top = main chart (spans full width), bottom row = marginal plots
        fig = plt.figure(figsize=(8 + 3 * n_marginals, 9))
        gs = fig.add_gridspec(2, n_marginals,
                              height_ratios=[3, 1.2],
                              hspace=0.35, wspace=0.35)
        ax = fig.add_subplot(gs[0, :])
        marginal_axes = []
        for i in range(n_marginals):
            marginal_axes.append(fig.add_subplot(gs[1, i]))
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        marginal_axes = []
    
    import matplotlib.lines as mlines
    from matplotlib.legend_handler import HandlerLine2D
    
    # Human-readable legend titles
    if filter_display_name is not None:
        filter_title = _humanize_filter_label(filter_display_name) + ':'
    else:
        filter_title = 'Category:'
    
    if size_col_name is not None:
        size_title = _humanize_size_label(size_col_name) + ':'
    else:
        size_title = 'Size:'
    
    if show_legend and isinstance(marker_style, dict) and filter_name is not None:
        # Use different markers for each category
        categories = plot_data[filter_name].unique()
        for cat in categories:
            mask = plot_data[filter_name] == cat
            if mask.sum() > 0:
                marker = marker_style.get(cat, 'o')
                ax.scatter(x[mask], y[mask], c=colors[mask], 
                          s=sizes_scaled[mask], cmap=palette,
                          marker=marker, alpha=0.7, edgecolors='black', 
                          linewidth=0.5, label=cat)
        
        # Create colorbar from all data
        scatter = ax.scatter(x, y, c=colors, s=sizes_scaled, 
                            cmap=palette, alpha=0.7, edgecolors='black', 
                            linewidth=0.5, visible=False)
        cbar = plt.colorbar(scatter, ax=ax)
        if color_log:
            cbar.set_label(f'log10({plot_data[color_col].name})', 
                          fontsize=11, fontweight='bold')
        else:
            cbar.set_label(plot_data[color_col].name, 
                          fontsize=11, fontweight='bold')
        
        # Prepare size legend entries with FIXED visual sizes
        if size_log:
            size_label = f'log10({_humanize_size_label(size_col_name)})' if size_col_name is not None else 'log10(size)'
        else:
            size_label = _humanize_size_label(size_col_name) if size_col_name is not None else 'Size'
        
        # Calculate min, median, max values for the legend
        size_min_val = sizes.min() if len(sizes) > 0 else 1
        size_median_val = np.percentile(sizes, 50) if len(sizes) > 0 else 2
        size_max_val = sizes.max() if len(sizes) > 0 else 3
        
        # Fixed marker sizes for legend (always small, medium, large)
        # These are the actual marker sizes in points
        LEGEND_SMALL_SIZE = 20
        LEGEND_MEDIUM_SIZE = 60
        LEGEND_LARGE_SIZE = 150
        
        # Create FIRST LEGEND: category with FIXED marker size
        category_handles = []
        category_labels = []
        
        unique_categories = plot_data[filter_name].unique()
        
        for cat in unique_categories:
            marker = marker_style.get(cat, 'o')
            handle = mlines.Line2D([0], [0], marker=marker, color='w',
                                  label=cat,
                                  markersize=10,
                                  markerfacecolor='gray',
                                  markeredgecolor='black',
                                  linestyle='None')
            category_handles.append(handle)
            category_labels.append(cat)
        
        legend1 = ax.legend(category_handles, category_labels, title=filter_title,
                           loc='upper left', frameon=True, framealpha=0.5)
        
        ax.add_artist(legend1)
        
        # Create SECOND LEGEND: Size values with FIXED visual sizes
        # Always show small, medium, large markers regardless of data range
        size_handles = []
        size_labels = []
        
        # Small marker
        size_handles.append(mlines.Line2D([0], [0], marker='o', color='w',
                          label=f'Min: {size_min_val:.3f}',
                          markersize=np.sqrt(LEGEND_SMALL_SIZE/2),
                          markerfacecolor='gray', 
                          markeredgecolor='black'))
        size_labels.append(f'Min: {size_min_val:.3f}')
        
        # Medium marker
        size_handles.append(mlines.Line2D([0], [0], marker='o', color='w',
                          label=f'Median: {size_median_val:.3f}',
                          markersize=np.sqrt(LEGEND_MEDIUM_SIZE/2),
                          markerfacecolor='gray', 
                          markeredgecolor='black'))
        size_labels.append(f'Median: {size_median_val:.3f}')
        
        # Large marker
        size_handles.append(mlines.Line2D([0], [0], marker='o', color='w',
                          label=f'Max: {size_max_val:.3f}',
                          markersize=np.sqrt(LEGEND_LARGE_SIZE/2),
                          markerfacecolor='gray', 
                          markeredgecolor='black'))
        size_labels.append(f'Max: {size_max_val:.3f}')
        
        legend2 = ax.legend(size_handles, size_labels, title=size_title,
                           loc='upper right', frameon=True, framealpha=0.5)
        
        ax.add_artist(legend2)
        
    else:
        scatter = ax.scatter(x, y, c=colors, s=sizes_scaled, 
                            cmap=palette, alpha=0.7, edgecolors='black', 
                            linewidth=0.5, marker=marker_style if isinstance(marker_style, str) else 'o')
        
        cbar = plt.colorbar(scatter, ax=ax)
        if color_log:
            cbar.set_label(f'log10({plot_data[color_col].name})', 
                          fontsize=11, fontweight='bold')
        else:
            cbar.set_label(plot_data[color_col].name, 
                          fontsize=11, fontweight='bold')
        
        # Add size legend only with FIXED visual sizes
        if size_log:
            size_label = f'log10({_humanize_size_label(size_col_name)})' if size_col_name is not None else 'log10(size)'
        else:
            size_label = _humanize_size_label(size_col_name) if size_col_name is not None else 'Size'
        
        size_min_val = sizes.min() if len(sizes) > 0 else 1
        size_median_val = np.percentile(sizes, 50) if len(sizes) > 0 else 2
        size_max_val = sizes.max() if len(sizes) > 0 else 3
        
        # Fixed marker sizes for legend
        LEGEND_SMALL_SIZE = 20
        LEGEND_MEDIUM_SIZE = 60
        LEGEND_LARGE_SIZE = 150
        
        size_handles = []
        size_labels = []
        
        # Small marker
        size_handles.append(mlines.Line2D([0], [0], marker='o', color='w',
                          label=f'Min: {size_min_val:.3f}',
                          markersize=np.sqrt(LEGEND_SMALL_SIZE/2),
                          markerfacecolor='gray', 
                          markeredgecolor='black'))
        size_labels.append(f'Min: {size_min_val:.3f}')
        
        # Medium marker
        size_handles.append(mlines.Line2D([0], [0], marker='o', color='w',
                          label=f'Median: {size_median_val:.3f}',
                          markersize=np.sqrt(LEGEND_MEDIUM_SIZE/2),
                          markerfacecolor='gray', 
                          markeredgecolor='black'))
        size_labels.append(f'Median: {size_median_val:.3f}')
        
        # Large marker
        size_handles.append(mlines.Line2D([0], [0], marker='o', color='w',
                          label=f'Max: {size_max_val:.3f}',
                          markersize=np.sqrt(LEGEND_LARGE_SIZE/2),
                          markerfacecolor='gray', 
                          markeredgecolor='black'))
        size_labels.append(f'Max: {size_max_val:.3f}')
        
        ax.legend(size_handles, size_labels, title=size_title,
                  loc='upper right', frameon=True, framealpha=0.5)
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # ============================================
    # MARGINAL DISTRIBUTIONS (ggside style)
    # ============================================
    if build_marginals and len(marginal_axes) > 0:
        # Y values used for the marginal distributions (raw, before log transform)
        y_raw_vals = np.asarray(raw_y_series.values, dtype=float)
        
        y_marg_label = ylabel
        
        ax_idx = 0
        
        # 1) Marginal by size terciles
        if (size_col_name is not None) and (size_col_raw is not None):
            size_raw_vals = np.asarray(raw_size_series.values, dtype=float)
            mask_finite = np.isfinite(size_raw_vals) & np.isfinite(y_raw_vals)
            size_ok = size_raw_vals[mask_finite]
            y_ok = y_raw_vals[mask_finite]
            if len(size_ok) >= 3:
                q33 = np.percentile(size_ok, 33.333)
                q66 = np.percentile(size_ok, 66.667)
                g1_mask = size_ok <= q33
                g2_mask = (size_ok > q33) & (size_ok <= q66)
                g3_mask = size_ok > q66
                groups = []
                if g1_mask.sum() > 0:
                    groups.append(('0–33%', y_ok[g1_mask]))
                if g2_mask.sum() > 0:
                    groups.append(('33–66%', y_ok[g2_mask]))
                if g3_mask.sum() > 0:
                    groups.append(('66–100%', y_ok[g3_mask]))
                _plot_kde_marginal(
                    marginal_axes[ax_idx],
                    groups,
                    y_marg_label,
                    normalize_marginals,
                    f'Y distribution by {_humanize_size_label(size_col_name)} terciles'
                )
            else:
                marginal_axes[ax_idx].text(0.5, 0.5, 'Not enough data',
                                            ha='center', va='center',
                                            transform=marginal_axes[ax_idx].transAxes,
                                            fontsize=9)
                marginal_axes[ax_idx].set_title(
                    f'Y distribution by {_humanize_size_label(size_col_name)} terciles',
                    fontsize=10, fontweight='bold')
            ax_idx += 1
        
        # 2) Marginal by active filter categories
        if (filter_display_name is not None) and (filter_name is not None) and show_legend:
            cats = plot_data[filter_name].values
            mask_finite = pd.notna(cats) & np.isfinite(y_raw_vals)
            cats_ok = cats[mask_finite]
            y_ok = y_raw_vals[mask_finite]
            groups = []
            for cat in pd.unique(cats_ok):
                vals = y_ok[cats_ok == cat]
                if len(vals) > 0:
                    groups.append((str(cat), vals))
            _plot_kde_marginal(
                marginal_axes[ax_idx],
                groups,
                y_marg_label,
                normalize_marginals,
                f'Y distribution by {_humanize_filter_label(filter_display_name)}'
            )
            ax_idx += 1
    
    plt.tight_layout()
    return fig

# ============================================
# FUNCTION FOR DOWNLOADING PLOTS
# ============================================

def download_plot(fig, filename="plot.png"):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
    buf.seek(0)
    return buf

# ============================================
# MAIN APPLICATION FUNCTION
# ============================================

def main():
    st.set_page_config(
        page_title="Ceramic Conductivity Data Explorer",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Interactive Analysis of Ceramic Conductivity")
    st.markdown("---")
    
    # ============================================
    # SECTION A: Data Loading (text input)
    # ============================================
    st.header("📊 Data Loading")
    
    st.markdown("**Paste your data in TSV (tab-separated) or CSV (comma-separated) format:**")
    
    data_input = st.text_area(
        "Paste data here:",
        height=200,
        key="data_input"
    )
    
    # Initialize session state for data
    if 'df_loaded' not in st.session_state:
        st.session_state.df_loaded = None
    if 'df_hash' not in st.session_state:
        st.session_state.df_hash = None
    
    if not data_input:
        st.info("⏳ Please paste your data to begin")
        st.stop()
    
    # Generate hash of input data to detect changes
    import hashlib
    current_hash = hashlib.md5(data_input.encode()).hexdigest()
    
    # Only reload if data has changed
    if st.session_state.df_hash != current_hash or st.session_state.df_loaded is None:
        try:
            lines = data_input.strip().split('\n')
            if '\t' in lines[0]:
                sep = '\t'
            elif ',' in lines[0]:
                sep = ','
            else:
                st.error("Unable to detect separator. Please use tab or comma.")
                st.stop()
            
            df = pd.read_csv(io.StringIO(data_input), sep=sep)
            df.columns = df.columns.str.strip()
            
            # Store in session state
            st.session_state.df_loaded = df
            st.session_state.df_hash = current_hash
            
            st.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
            st.dataframe(df.head(10))
            
        except Exception as e:
            st.error(f"Error parsing data: {str(e)}")
            st.stop()
    else:
        # Use cached data
        df = st.session_state.df_loaded
        st.success(f"✅ Using cached data: {len(df)} rows, {len(df.columns)} columns")
        st.dataframe(df.head(10))
    
    # ============================================
    # DESCRIPTOR AND Ea CALCULATION (with caching)
    # ============================================
    st.markdown("---")
    st.header("🔄 Automatic Descriptor Calculation")
    
    # Generate hash for descriptor caching
    df_hash_desc = hashlib.md5(df.to_csv().encode()).hexdigest()
    
    with st.spinner("Computing structural and electronegativity descriptors..."):
        desc_df = compute_descriptors_cached(df_hash_desc, df)
        
        for col in desc_df.columns:
            df[col] = desc_df[col]
        
        st.success("✅ Descriptors calculated (cached)")
    
    with st.spinner("Calculating activation energy (Ea)..."):
        ea_values = calculate_ea_cached(df_hash_desc, df)
        df['Ea_calculated'] = ea_values
        df['Ea_final'] = df['Ea (eV)'].fillna(df['Ea_calculated'])
        
        st.success("✅ Ea calculated (cached)")
    
    st.dataframe(df[['References', 'tolerance_factor', 'chi_B_avg', 
                    'chi_ratio', 'molar_mass', 'porosity', 
                    'grain_boundary_area', 'Ea_final']].head(10))
    
    # ============================================
    # SECTION C: Filters (sidebar)
    # ============================================
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Data Filters")
    
    # Store filters in session state
    if 'active_filters_count' not in st.session_state:
        st.session_state.active_filters_count = 0
    if 'active_filter_name' not in st.session_state:
        st.session_state.active_filter_name = None
    
    active_filters_count = 0
    active_filter_name = None
    
    if 'Atmospheres' in df.columns:
        atmos_options = sorted([x for x in df['Atmospheres'].unique() 
                               if pd.notna(x) and x != ''])
        selected_atmos = st.sidebar.multiselect(
            "Atmosphere",
            options=['All'] + atmos_options,
            default=['All'],
            key="atmos_filter"
        )
        if 'All' not in selected_atmos:
            df_filtered = df[df['Atmospheres'].isin(selected_atmos)]
            active_filters_count += 1
            active_filter_name = 'Atmospheres'
        else:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
    
    if 'Humidity' in df.columns:
        humidity_options = sorted([x for x in df_filtered['Humidity'].unique() 
                                  if pd.notna(x) and x != ''])
        if humidity_options:
            selected_humidity = st.sidebar.multiselect(
                "Humidity",
                options=['All'] + humidity_options,
                default=['All'],
                key="humidity_filter"
            )
            if 'All' not in selected_humidity:
                df_filtered = df_filtered[df_filtered['Humidity'].isin(selected_humidity)]
                active_filters_count += 1
                active_filter_name = 'Humidity'
    
    if 'Structure' in df.columns:
        structure_options = sorted([x for x in df_filtered['Structure'].unique() 
                                   if pd.notna(x) and x != ''])
        if structure_options:
            selected_structure = st.sidebar.multiselect(
                "Structure",
                options=['All'] + structure_options,
                default=['All'],
                key="structure_filter"
            )
            if 'All' not in selected_structure:
                df_filtered['Structure_lower'] = df_filtered['Structure'].str.lower()
                selected_structure_lower = [s.lower() for s in selected_structure]
                df_filtered = df_filtered[df_filtered['Structure_lower'].isin(selected_structure_lower)]
                active_filters_count += 1
                active_filter_name = 'Structure'
    
    if 'Sintering additive' in df.columns:
        additive_options = sorted([x for x in df_filtered['Sintering additive'].unique() 
                                  if pd.notna(x) and x != ''])
        if additive_options:
            selected_additive = st.sidebar.multiselect(
                "Sintering additive",
                options=['All'] + additive_options,
                default=['All'],
                key="additive_filter"
            )
            if 'All' not in selected_additive:
                df_filtered = df_filtered[df_filtered['Sintering additive'].isin(selected_additive)]
                active_filters_count += 1
                active_filter_name = 'Sintering additive'
    
    # Store active filters in session state
    st.session_state.active_filters_count = active_filters_count
    st.session_state.active_filter_name = active_filter_name
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Value Ranges")
    
    if 'T sin' in df_filtered.columns:
        t_min = float(df_filtered['T sin'].min()) if not df_filtered['T sin'].isna().all() else 0
        t_max = float(df_filtered['T sin'].max()) if not df_filtered['T sin'].isna().all() else 1000
        if t_max > t_min:
            t_range = st.sidebar.slider(
                "Sintering temperature, °C",
                min_value=t_min,
                max_value=t_max,
                value=(t_min, t_max),
                key="tsin_range"
            )
            df_filtered = df_filtered[(df_filtered['T sin'] >= t_range[0]) & 
                                     (df_filtered['T sin'] <= t_range[1])]
    
    if 'dop_cont' in df_filtered.columns:
        d_min = float(df_filtered['dop_cont'].min()) if not df_filtered['dop_cont'].isna().all() else 0
        d_max = float(df_filtered['dop_cont'].max()) if not df_filtered['dop_cont'].isna().all() else 1
        if d_max > d_min:
            d_range = st.sidebar.slider(
                "Dopant content",
                min_value=d_min,
                max_value=d_max,
                value=(d_min, d_max),
                key="dopant_range"
            )
            df_filtered = df_filtered[(df_filtered['dop_cont'] >= d_range[0]) & 
                                     (df_filtered['dop_cont'] <= d_range[1])]
    
    if 'x, wt%' in df_filtered.columns:
        x_min = float(df_filtered['x, wt%'].min()) if not df_filtered['x, wt%'].isna().all() else 0
        x_max = float(df_filtered['x, wt%'].max()) if not df_filtered['x, wt%'].isna().all() else 10
        if x_max > x_min:
            x_range = st.sidebar.slider(
                "Sintering additive concentration, wt%",
                min_value=x_min,
                max_value=x_max,
                value=(x_min, x_max),
                key="additive_range"
            )
            df_filtered = df_filtered[(df_filtered['x, wt%'] >= x_range[0]) & 
                                     (df_filtered['x, wt%'] <= x_range[1])]
    
    st.sidebar.markdown(f"**Data after filtering: {len(df_filtered)}**")
    
    # ============================================
    # MAIN TABS
    # ============================================
    tab1, tab2, tab3 = st.tabs(["🌡️ Heat Maps", "🫧 Bubble Charts", "📋 Data"])
    
    # ============================================
    # TAB 1: HEAT MAPS
    # ============================================
    with tab1:
        st.header("🌡️ Heat Maps")
        
        numeric_cols = ['dop_cont', 'x, wt%', 'ρ, %', 'd, mkm', 
                       'tolerance_factor', 'chi_B_avg', 'chi_ratio', 
                       'molar_mass', 'porosity', 'grain_boundary_area']
        
        temp_cols = ['σ total, 200', 'σ total, 250', 'σ total, 300', 'σ total, 350',
                    'σ total, 400', 'σ total, 450', 'σ total, 500', 'σ total, 550',
                    'σ total, 600', 'σ total, 650', 'σ total, 700', 'σ total, 750',
                    'σ total, 800', 'σ total, 850', 'σ total, 900']
        
        available_temp_cols = [col for col in temp_cols if col in df_filtered.columns]
        
        temp_counts = {}
        for col in available_temp_cols:
            count = df_filtered[col].count()
            temp_counts[col] = count
        
        if temp_counts:
            default_temp = max(temp_counts, key=temp_counts.get)
            temp_options = available_temp_cols + ['Ea_final']
        else:
            default_temp = 'Ea_final'
            temp_options = ['Ea_final']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox(
                "X-axis (descriptor)",
                options=numeric_cols,
                index=0,
                key="heatmap_x"
            )
        
        with col2:
            y_axis = st.selectbox(
                "Y-axis (descriptor)",
                options=numeric_cols,
                index=1 if len(numeric_cols) > 1 else 0,
                key="heatmap_y"
            )
        
        with col3:
            z_axis = st.selectbox(
                "Color scale (Z)",
                options=temp_options,
                index=0,
                key="heatmap_z"
            )
        
        # Outlier removal controls for X and Y
        col1, col2 = st.columns(2)
        with col1:
            outlier_x = st.selectbox(
                "Remove outliers from X",
                options=[0, 1, 2, 3],
                index=0,
                format_func=lambda x: 'None' if x == 0 else str(x),
                key="heatmap_outlier_x"
            )
        with col2:
            outlier_y = st.selectbox(
                "Remove outliers from Y",
                options=[0, 1, 2, 3],
                index=0,
                format_func=lambda x: 'None' if x == 0 else str(x),
                key="heatmap_outlier_y"
            )
        
        # Apply outlier removal
        df_plot = df_filtered.copy()
        if outlier_x > 0:
            df_plot = remove_outliers(df_plot, x_axis, outlier_x)
        if outlier_y > 0:
            df_plot = remove_outliers(df_plot, y_axis, outlier_y)
        
        if z_axis in available_temp_cols:
            selected_temp = st.selectbox(
                "Select temperature for conductivity",
                options=available_temp_cols,
                index=available_temp_cols.index(default_temp) if default_temp in available_temp_cols else 0,
                key="heatmap_temp"
            )
            z_col = selected_temp
            z_label = selected_temp.replace('σ total, ', 'σ at ') + ' °C (mS/cm)'
        else:
            z_col = 'Ea_final'
            z_label = 'Ea (eV)'
        
        plot_type = st.radio(
            "Heat map type",
            options=['Scatter with color scale', 'Contour plot', '3D surface'],
            horizontal=True,
            key="heatmap_type"
        )
        
        palette_name = st.selectbox(
            "Color palette",
            options=list(COLOR_PALETTES.keys()),
            index=0,
            key="heatmap_palette"
        )
        palette = COLOR_PALETTES[palette_name]
        
        # Additional controls for contour plot
        if plot_type == 'Contour plot':
            col1, col2, col3 = st.columns(3)
            with col1:
                grid_resolution = st.slider(
                    "Smoothing (grid resolution)",
                    min_value=20,
                    max_value=100,
                    value=50,
                    step=10,
                    key="contour_resolution"
                )
            with col2:
                show_contour_lines = st.checkbox("Show contour lines", value=True, key="contour_lines")
            with col3:
                show_contour_labels = st.checkbox("Show contour labels", value=True, key="contour_labels")
        else:
            grid_resolution = 50
            show_contour_lines = True
            show_contour_labels = True
        
        col1, col2, col3 = st.columns(3)
        with col1:
            log_x = st.checkbox("log10(X)", value=False, key="heatmap_log_x")
        with col2:
            log_y = st.checkbox("log10(Y)", value=False, key="heatmap_log_y")
        with col3:
            log_z = st.checkbox("log10(Z)", value=False, key="heatmap_log_z")
        
        if st.button("Generate Heat Map", key="heatmap_btn"):
            if len(df_plot) == 0:
                st.warning("No data available after filtering")
            else:
                # Determine marker style based on active filters
                marker_style, show_legend, filter_name_active = get_filter_markers(
                    df_plot, st.session_state.active_filters_count)
                
                if plot_type == 'Scatter with color scale':
                    fig = create_scatter_heatmap(
                        df_plot, x_axis, y_axis, z_col,
                        log_x, log_y, log_z, palette,
                        '', x_axis, y_axis, z_label,
                        marker_style, show_legend, filter_name_active
                    )
                elif plot_type == 'Contour plot':
                    fig = create_contour_heatmap(
                        df_plot, x_axis, y_axis, z_col,
                        log_x, log_y, log_z, palette,
                        '', x_axis, y_axis, z_label,
                        grid_resolution, show_contour_lines, show_contour_labels
                    )
                else:
                    fig = create_3d_surface(
                        df_plot, x_axis, y_axis, z_col,
                        log_x, log_y, log_z, palette,
                        '', x_axis, y_axis, z_label
                    )
                
                if fig is not None:
                    st.pyplot(fig)
                    
                    buf = download_plot(fig, "heatmap.png")
                    st.download_button(
                        label="📥 Download Plot (PNG, 600 dpi)",
                        data=buf,
                        file_name="heatmap.png",
                        mime="image/png",
                        key="download_heatmap"
                    )
                    
                    plt.close(fig)
    
    # ============================================
    # TAB 2: BUBBLE CHARTS
    # ============================================
    with tab2:
        st.header("🫧 Bubble Charts")
        
        numeric_cols_bubble = ['dop_cont', 'x, wt%', 'ρ, %', 'd, mkm', 
                              'tolerance_factor', 'chi_B_avg', 'chi_ratio', 
                              'molar_mass', 'porosity', 'grain_boundary_area']
        
        y_options = available_temp_cols + ['Ea_final']
        
        if available_temp_cols:
            default_y = max(available_temp_cols, 
                           key=lambda c: df_filtered[c].count() if c in df_filtered.columns else 0)
            y_index = y_options.index(default_y) if default_y in y_options else 0
        else:
            default_y = 'Ea_final'
            y_index = y_options.index('Ea_final') if 'Ea_final' in y_options else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            y_axis_bubble = st.selectbox(
                "Y-axis (conductivity/Ea)",
                options=y_options,
                index=y_index,
                key="bubble_y"
            )
        
        with col2:
            x_axis_bubble = st.selectbox(
                "X-axis",
                options=numeric_cols_bubble,
                index=0,
                key="bubble_x"
            )
        
        with col3:
            color_axis = st.selectbox(
                "Bubble color",
                options=['None'] + numeric_cols_bubble + ['Sintering additive', 'Atmospheres', 'Humidity', 'Structure'],
                index=0,
                key="bubble_color"
            )
        
        with col4:
            size_axis = st.selectbox(
                "Bubble size",
                options=['None'] + numeric_cols_bubble,
                index=0,
                key="bubble_size"
            )
        
        # Outlier removal controls for X and Y in bubble charts
        col1, col2 = st.columns(2)
        with col1:
            outlier_x_bubble = st.selectbox(
                "Remove outliers from X",
                options=[0, 1, 2, 3],
                index=0,
                format_func=lambda x: 'None' if x == 0 else str(x),
                key="bubble_outlier_x"
            )
        with col2:
            outlier_y_bubble = st.selectbox(
                "Remove outliers from Y",
                options=[0, 1, 2, 3],
                index=0,
                format_func=lambda x: 'None' if x == 0 else str(x),
                key="bubble_outlier_y"
            )
        
        # Apply outlier removal
        df_plot_bubble = df_filtered.copy()
        if outlier_x_bubble > 0:
            df_plot_bubble = remove_outliers(df_plot_bubble, x_axis_bubble, outlier_x_bubble)
        if outlier_y_bubble > 0:
            df_plot_bubble = remove_outliers(df_plot_bubble, y_axis_bubble, outlier_y_bubble)
        
        if y_axis_bubble in available_temp_cols:
            y_label = y_axis_bubble.replace('σ total, ', 'σ at ') + ' °C (mS/cm)'
        else:
            y_label = 'Ea (eV)'
        
        palette_name_bubble = st.selectbox(
            "Color palette for bubbles",
            options=list(COLOR_PALETTES.keys()),
            index=0,
            key="bubble_palette"
        )
        palette_bubble = COLOR_PALETTES[palette_name_bubble]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            log_x_bubble = st.checkbox("log10(X)", value=False, key="bubble_log_x")
        with col2:
            log_y_bubble = st.checkbox("log10(Y)", value=False, key="bubble_log_y")
        with col3:
            log_color_bubble = st.checkbox("log10(Color)", value=False, key="bubble_log_color")
        with col4:
            log_size_bubble = st.checkbox("log10(Size)", value=False, key="bubble_log_size")
        
        # Marginal distributions controls
        col1, col2 = st.columns(2)
        with col1:
            show_marginals = st.checkbox(
                "Show marginal distributions (ggside)",
                value=False,
                key="bubble_show_marginals"
            )
        with col2:
            normalize_marginals = st.checkbox(
                "Normalize marginals (per group)",
                value=False,
                key="bubble_normalize_marginals",
                disabled=not show_marginals
            )
        
        if st.button("Generate Bubble Chart", key="bubble_btn"):
            if len(df_plot_bubble) == 0:
                st.warning("No data available after filtering")
            else:
                if color_axis == 'None':
                    df_temp = df_plot_bubble.copy()
                    df_temp['_color_temp'] = 1
                    color_col_bubble = '_color_temp'
                else:
                    color_col_bubble = color_axis
                
                if size_axis == 'None':
                    df_temp = df_plot_bubble.copy()
                    df_temp['_size_temp'] = 1
                    size_col_bubble = '_size_temp'
                else:
                    size_col_bubble = size_axis
                
                # Determine marker style based on active filters
                marker_style, show_legend, filter_name_active = get_filter_markers(
                    df_plot_bubble, st.session_state.active_filters_count)
                
                # Determine whether to pass size info for marginals
                size_col_name_for_marg = None
                size_col_raw_for_marg = None
                if size_axis != 'None':
                    size_col_name_for_marg = size_axis
                    size_col_raw_for_marg = size_axis
                
                filter_display_name_for_marg = None
                if show_legend and filter_name_active is not None:
                    filter_display_name_for_marg = filter_name_active
                
                fig = create_bubble_chart(
                    df_plot_bubble, x_axis_bubble, y_axis_bubble, 
                    color_col_bubble, size_col_bubble,
                    log_x_bubble, log_y_bubble, log_color_bubble, log_size_bubble,
                    palette_bubble, '', x_axis_bubble, y_label,
                    marker_style, show_legend, filter_name_active,
                    size_col_name=size_col_name_for_marg,
                    filter_display_name=filter_display_name_for_marg,
                    show_marginals=show_marginals,
                    normalize_marginals=normalize_marginals,
                    size_col_raw=size_col_raw_for_marg,
                    y_col_raw=y_axis_bubble
                )
                
                if fig is not None:
                    st.pyplot(fig)
                    
                    buf = download_plot(fig, "bubble_chart.png")
                    st.download_button(
                        label="📥 Download Plot (PNG, 600 dpi)",
                        data=buf,
                        file_name="bubble_chart.png",
                        mime="image/png",
                        key="download_bubble"
                    )
                    
                    plt.close(fig)
    
    # ============================================
    # TAB 3: DATA
    # ============================================
    with tab3:
        st.header("📋 Data with Calculated Descriptors")
        
        st.dataframe(df_filtered)
        
        csv_data = df_filtered.to_csv(index=False, sep='\t')
        st.download_button(
            label="📥 Download Data (TSV)",
            data=csv_data,
            file_name="filtered_data.tsv",
            mime="text/tab-separated-values",
            key="download_data"
        )
        
        st.subheader("📊 Data Statistics")
        numeric_display = df_filtered.select_dtypes(include=[np.number])
        if not numeric_display.empty:
            st.dataframe(numeric_display.describe())

# ============================================
# APPLICATION LAUNCH
# ============================================

if __name__ == "__main__":
    main()
