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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
import seaborn as sns
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
    'legend.framealpha': 0.9,
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
# FUNCTIONS FOR DESCRIPTOR CALCULATION
# ============================================

def compute_descriptors(df):
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

# ============================================
# FUNCTION FOR Ea CALCULATION
# ============================================

def calculate_ea(row):
    if pd.notna(row['Ea (eV)']) and row['Ea (eV)'] != '':
        return row['Ea (eV)']
    
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
        return np.nan
    
    y = np.log(np.array(sigmas) * np.array(temps))
    x = 1000 / np.array(temps)
    
    try:
        slope, intercept = np.polyfit(x, y, 1)
        ea = -slope * 8.314 / 96485
        return ea
    except:
        return np.nan

# ============================================
# FUNCTION FOR DELTA DESCRIPTORS CALCULATION
# ============================================

def calculate_delta_descriptors(df):
    """
    Calculate Δρ, Δd, ΔT, and Δσ for each additive compared to Pure
    Grouped by References and chemical composition
    """
    delta_rows = []
    
    # Define columns that define chemical composition
    composition_cols = ['A cation', 'B1 cation', 'B2 cation', 'B2_cont', 'dopant', 'dop_cont']
    
    # Define temperature columns
    temp_cols = ['σ total, 200', 'σ total, 250', 'σ total, 300', 'σ total, 350',
                 'σ total, 400', 'σ total, 450', 'σ total, 500', 'σ total, 550',
                 'σ total, 600', 'σ total, 650', 'σ total, 700', 'σ total, 750',
                 'σ total, 800', 'σ total, 850', 'σ total, 900']
    
    # Group by References
    for ref, ref_group in df.groupby('References'):
        # Find Pure samples in this group
        pure_mask = ref_group['Sintering additive'].apply(
            lambda x: pd.notna(x) and x.lower() == 'pure'
        )
        pure_samples = ref_group[pure_mask]
        
        if len(pure_samples) == 0:
            continue
        
        # For each pure sample, find matching additives
        for pure_idx, pure_row in pure_samples.iterrows():
            # Get composition of pure sample
            pure_composition = {col: pure_row[col] for col in composition_cols}
            
            # Find additives with same composition
            for add_idx, add_row in ref_group.iterrows():
                if pd.notna(add_row['Sintering additive']) and add_row['Sintering additive'].lower() == 'pure':
                    continue
                
                # Check if composition matches
                composition_match = True
                for col in composition_cols:
                    if pd.isna(pure_row[col]) and pd.isna(add_row[col]):
                        continue
                    if pure_row[col] != add_row[col]:
                        composition_match = False
                        break
                
                if not composition_match:
                    continue
                
                # Calculate delta values
                delta = {
                    'References': ref,
                    'Sintering additive': add_row['Sintering additive'],
                    'x, wt%': add_row['x, wt%'] if pd.notna(add_row['x, wt%']) else np.nan,
                    'T_sin_Pure': pure_row['T sin'] if pd.notna(pure_row['T sin']) else np.nan,
                    'T_sin_Additive': add_row['T sin'] if pd.notna(add_row['T sin']) else np.nan,
                }
                
                # ΔT
                if pd.notna(pure_row['T sin']) and pd.notna(add_row['T sin']):
                    delta['ΔT'] = pure_row['T sin'] - add_row['T sin']
                else:
                    delta['ΔT'] = np.nan
                
                # Δρ
                if pd.notna(pure_row['ρ, %']) and pd.notna(add_row['ρ, %']):
                    delta['Δρ'] = add_row['ρ, %'] - pure_row['ρ, %']
                    delta['Δρ_rel'] = (delta['Δρ'] / pure_row['ρ, %']) * 100
                else:
                    delta['Δρ'] = np.nan
                    delta['Δρ_rel'] = np.nan
                
                # Δd
                if pd.notna(pure_row['d, mkm']) and pd.notna(add_row['d, mkm']):
                    delta['Δd'] = add_row['d, mkm'] - pure_row['d, mkm']
                    delta['Δd_rel'] = (delta['Δd'] / pure_row['d, mkm']) * 100
                else:
                    delta['Δd'] = np.nan
                    delta['Δd_rel'] = np.nan
                
                # Δσ for each temperature
                for temp_col in temp_cols:
                    if temp_col in pure_row.index and temp_col in add_row.index:
                        if pd.notna(pure_row[temp_col]) and pd.notna(add_row[temp_col]):
                            delta[f'Δ{temp_col}'] = add_row[temp_col] - pure_row[temp_col]
                        else:
                            delta[f'Δ{temp_col}'] = np.nan
                    else:
                        delta[f'Δ{temp_col}'] = np.nan
                
                delta_rows.append(delta)
    
    if len(delta_rows) == 0:
        return pd.DataFrame()
    
    delta_df = pd.DataFrame(delta_rows)
    return delta_df

# ============================================
# FUNCTION FOR OUTLIER REMOVAL
# ============================================

def remove_outliers(df, col, n):
    """
    Remove n smallest and n largest values from the dataset for a given column.
    If the same extreme value appears more than n times, none of those rows are removed.
    """
    if n is None or n == 0:
        return df
    
    if col not in df.columns:
        return df
    
    unique_vals = sorted(df[col].dropna().unique())
    
    if len(unique_vals) <= 2 * n:
        return df
    
    low_vals_to_remove = []
    for i in range(min(n, len(unique_vals))):
        val = unique_vals[i]
        count = (df[col] == val).sum()
        if count <= n:
            low_vals_to_remove.append(val)
        else:
            break
    
    high_vals_to_remove = []
    for i in range(min(n, len(unique_vals))):
        val = unique_vals[-(i+1)]
        count = (df[col] == val).sum()
        if count <= n:
            high_vals_to_remove.append(val)
        else:
            break
    
    mask = pd.Series(True, index=df.index)
    if low_vals_to_remove:
        mask = mask & ~df[col].isin(low_vals_to_remove)
    if high_vals_to_remove:
        mask = mask & ~df[col].isin(high_vals_to_remove)
    
    return df[mask]

# ============================================
# FUNCTIONS FOR PLOTTING
# ============================================

def get_filter_markers(df, active_filters_count):
    """
    Determine if custom markers should be used based on active filters count.
    Returns: marker_style (string or dict), show_legend (bool)
    """
    if active_filters_count == 1:
        for filter_name, marker_dict in MARKER_MAP.items():
            if filter_name in df.columns:
                unique_vals = df[filter_name].dropna().unique()
                if len(unique_vals) > 1:
                    return marker_dict, True
        return 'o', False
    else:
        return 'o', False

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
        categories = plot_data[filter_name].unique()
        for cat in categories:
            mask = plot_data[filter_name] == cat
            if mask.sum() > 0:
                marker = marker_style.get(cat, 'o')
                ax.scatter(x[mask], y[mask], c=z[mask], cmap=palette, 
                          s=50, marker=marker, edgecolors='black', 
                          linewidth=0.5, alpha=0.8, label=cat)
        
        scatter = ax.scatter(x, y, c=z, cmap=palette, s=50, 
                            edgecolors='black', linewidth=0.5, alpha=0.8, visible=False)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(zlabel, fontsize=11, fontweight='bold')
        
        ax.legend(loc='upper right', frameon=True, framealpha=0.9)
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
    
    contour = ax.contourf(xi, yi, zi, levels=30, cmap=palette, alpha=0.85)
    
    if show_contour_lines:
        contour_lines = ax.contour(xi, yi, zi, levels=15, colors='black', 
                                   linewidths=0.5, alpha=0.4)
        if show_contour_labels:
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.2f')
    
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
                        marker_style='o', show_legend=False, filter_name=None):
    plot_data = df.dropna(subset=[x_col, y_col, color_col, size_col])
    
    if len(plot_data) == 0:
        st.warning("No data available for plotting")
        return None
    
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
        if len(sizes) == 0:
            st.warning("All size values ≤ 0, cannot create logarithmic plot")
            return None
        sizes = np.log10(sizes)
    
    if len(sizes) > 0:
        size_min, size_max = sizes.min(), sizes.max()
        if size_max > size_min:
            sizes_scaled = 20 + 180 * (sizes - size_min) / (size_max - size_min)
        else:
            sizes_scaled = np.ones_like(sizes) * 50
    else:
        sizes_scaled = np.ones_like(sizes) * 50
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    import matplotlib.lines as mlines
    
    if show_legend and isinstance(marker_style, dict) and filter_name is not None:
        categories = plot_data[filter_name].unique()
        for cat in categories:
            mask = plot_data[filter_name] == cat
            if mask.sum() > 0:
                marker = marker_style.get(cat, 'o')
                ax.scatter(x[mask], y[mask], c=colors[mask], 
                          s=sizes_scaled[mask], cmap=palette,
                          marker=marker, alpha=0.7, edgecolors='black', 
                          linewidth=0.5, label=cat)
        
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
        
        handles, labels = ax.get_legend_handles_labels()
        
        if size_log:
            size_label = f'log10({plot_data[size_col].name})'
        else:
            size_label = plot_data[size_col].name
        
        size_legend_values = [np.percentile(sizes, 25), np.percentile(sizes, 50), 
                              np.percentile(sizes, 75)] if len(sizes) > 0 else [1, 2, 3]
        size_legend_sizes = [20 + 180 * (v - sizes.min()) / (sizes.max() - sizes.min()) 
                             if sizes.max() > sizes.min() else 50 for v in size_legend_values]
        
        for val, size in zip(size_legend_values, size_legend_sizes):
            handles.append(mlines.Line2D([0], [0], marker='o', color='w',
                          label=f'Size: {val:.2f}',
                          markersize=np.sqrt(size/2),
                          markerfacecolor='gray', 
                          markeredgecolor='black'))
            labels.append(f'Size: {val:.2f}')
        
        ax.legend(handles, labels, title='',
                  loc='upper right', frameon=True, framealpha=0.9)
        
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
        
        if size_log:
            size_label = f'log10({plot_data[size_col].name})'
        else:
            size_label = plot_data[size_col].name
        
        size_legend_values = [np.percentile(sizes, 25), np.percentile(sizes, 50), 
                              np.percentile(sizes, 75)] if len(sizes) > 0 else [1, 2, 3]
        size_legend_sizes = [20 + 180 * (v - sizes.min()) / (sizes.max() - sizes.min()) 
                             if sizes.max() > sizes.min() else 50 for v in size_legend_values]
        
        legend_elements = []
        for val, size in zip(size_legend_values, size_legend_sizes):
            legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w',
                                  label=f'Size: {val:.2f}',
                                  markersize=np.sqrt(size/2),
                                  markerfacecolor='gray', 
                                  markeredgecolor='black'))
        
        ax.legend(handles=legend_elements, title='',
                  loc='upper right', frameon=True, framealpha=0.9)
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

# ============================================
# FUNCTION FOR CORRELATION MATRIX
# ============================================

def create_correlation_matrix(df, selected_temp, palette):
    """
    Create correlation matrix heatmap for selected variables
    """
    # Define columns for correlation
    corr_cols = ['dop_cont', 'tolerance_factor', 'chi_B_avg', 'chi_ratio', 
                 'molar_mass', 'x_B1', 'x_B2', 'x_dop']
    
    # Add microstructural descriptors if they exist
    micro_cols = ['ρ, %', 'd, mkm', 'porosity', 'grain_boundary_area']
    for col in micro_cols:
        if col in df.columns:
            corr_cols.append(col)
    
    # Add delta descriptors if they exist
    delta_cols = ['Δρ', 'Δd', 'ΔT', 'Δρ_rel', 'Δd_rel']
    for col in delta_cols:
        if col in df.columns:
            corr_cols.append(col)
    
    # Add x, wt% if it exists
    if 'x, wt%' in df.columns:
        corr_cols.append('x, wt%')
    
    # Add conductivity at selected temperature
    if selected_temp in df.columns:
        corr_cols.append(selected_temp)
    
    # Add Ea if it exists
    if 'Ea_final' in df.columns:
        corr_cols.append('Ea_final')
    
    # Filter to only columns that exist in df
    corr_cols = [col for col in corr_cols if col in df.columns]
    
    if len(corr_cols) < 2:
        st.warning("Not enough numerical columns for correlation matrix")
        return None
    
    # Calculate correlation matrix
    corr_df = df[corr_cols].dropna()
    
    if len(corr_df) < 3:
        st.warning("Not enough data for correlation matrix (need at least 3 rows)")
        return None
    
    corr_matrix = corr_df.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use seaborn for nicer heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=palette,
                vmin=-1, vmax=1, center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8}, ax=ax)
    
    ax.set_title('')
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
        height=200
    )
    
    if not data_input:
        st.info("⏳ Please paste your data to begin")
        st.stop()
    
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
        
        st.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
        st.dataframe(df.head(10))
        
    except Exception as e:
        st.error(f"Error parsing data: {str(e)}")
        st.stop()
    
    # ============================================
    # DESCRIPTOR AND Ea CALCULATION
    # ============================================
    st.markdown("---")
    st.header("🔄 Automatic Descriptor Calculation")
    
    with st.spinner("Computing structural and electronegativity descriptors..."):
        desc_df = compute_descriptors(df)
        
        for col in desc_df.columns:
            df[col] = desc_df[col]
        
        st.success("✅ Descriptors calculated")
    
    with st.spinner("Calculating activation energy (Ea)..."):
        df['Ea_calculated'] = df.apply(calculate_ea, axis=1)
        
        df['Ea_final'] = df['Ea (eV)'].fillna(df['Ea_calculated'])
        
        st.success("✅ Ea calculated")
    
    st.dataframe(df[['References', 'tolerance_factor', 'chi_B_avg', 
                    'chi_ratio', 'molar_mass', 'porosity', 
                    'grain_boundary_area', 'Ea_final']].head(10))
    
    # ============================================
    # DELTA DESCRIPTORS CALCULATION
    # ============================================
    st.markdown("---")
    st.header("📊 Delta Descriptors Calculation")
    
    with st.spinner("Calculating delta descriptors (Δρ, Δd, ΔT, Δσ)..."):
        delta_df = calculate_delta_descriptors(df)
        
        if len(delta_df) > 0:
            st.success(f"✅ Delta descriptors calculated: {len(delta_df)} additive pairs found")
            st.dataframe(delta_df.head(10))
            
            # Merge delta descriptors back to main dataframe for use in plots
            # We'll keep delta_df separate for dedicated analysis
        else:
            st.warning("No additive pairs found for delta descriptor calculation")
    
    # ============================================
    # SECTION C: Filters (sidebar)
    # ============================================
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Data Filters")
    
    active_filters_count = 0
    active_filter_name = None
    
    if 'Atmospheres' in df.columns:
        atmos_options = sorted([x for x in df['Atmospheres'].unique() 
                               if pd.notna(x) and x != ''])
        selected_atmos = st.sidebar.multiselect(
            "Atmosphere",
            options=['All'] + atmos_options,
            default=['All']
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
                default=['All']
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
                default=['All']
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
                default=['All']
            )
            if 'All' not in selected_additive:
                df_filtered = df_filtered[df_filtered['Sintering additive'].isin(selected_additive)]
                active_filters_count += 1
                active_filter_name = 'Sintering additive'
    
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
                value=(t_min, t_max)
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
                value=(d_min, d_max)
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
                value=(x_min, x_max)
            )
            df_filtered = df_filtered[(df_filtered['x, wt%'] >= x_range[0]) & 
                                     (df_filtered['x, wt%'] <= x_range[1])]
    
    st.sidebar.markdown(f"**Data after filtering: {len(df_filtered)}**")
    
    # ============================================
    # MAIN TABS
    # ============================================
    tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Heat Maps", "🫧 Bubble Charts", "📊 Additive Analysis", "📋 Data"])
    
    # ============================================
    # TAB 1: HEAT MAPS
    # ============================================
    with tab1:
        st.header("🌡️ Heat Maps")
        
        numeric_cols = ['dop_cont', 'x, wt%', 'ρ, %', 'd, mkm', 
                       'tolerance_factor', 'chi_B_avg', 'chi_ratio', 
                       'molar_mass', 'porosity', 'grain_boundary_area']
        
        # Add delta descriptors if they exist in df
        delta_cols_available = []
        for col in ['Δρ', 'Δd', 'ΔT', 'Δρ_rel', 'Δd_rel']:
            if col in df_filtered.columns:
                numeric_cols.append(col)
                delta_cols_available.append(col)
        
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
                index=0
            )
        
        with col2:
            y_axis = st.selectbox(
                "Y-axis (descriptor)",
                options=numeric_cols,
                index=1 if len(numeric_cols) > 1 else 0
            )
        
        with col3:
            z_axis = st.selectbox(
                "Color scale (Z)",
                options=temp_options,
                index=0
            )
        
        # Data type filter
        data_type = st.radio(
            "Data type",
            options=['All data', 'Pure only', 'Additives only'],
            horizontal=True
        )
        
        # Apply data type filter
        df_plot_type = df_filtered.copy()
        if data_type == 'Pure only':
            df_plot_type = df_plot_type[df_plot_type['Sintering additive'].apply(
                lambda x: pd.notna(x) and x.lower() == 'pure'
            )]
        elif data_type == 'Additives only':
            df_plot_type = df_plot_type[df_plot_type['Sintering additive'].apply(
                lambda x: pd.notna(x) and x.lower() != 'pure'
            )]
        
        # Outlier removal controls for X and Y
        col1, col2 = st.columns(2)
        with col1:
            outlier_x = st.selectbox(
                "Remove outliers from X",
                options=[0, 1, 2, 3],
                index=0,
                format_func=lambda x: 'None' if x == 0 else str(x)
            )
        with col2:
            outlier_y = st.selectbox(
                "Remove outliers from Y",
                options=[0, 1, 2, 3],
                index=0,
                format_func=lambda x: 'None' if x == 0 else str(x)
            )
        
        # Apply outlier removal
        df_plot = df_plot_type.copy()
        if outlier_x > 0:
            df_plot = remove_outliers(df_plot, x_axis, outlier_x)
        if outlier_y > 0:
            df_plot = remove_outliers(df_plot, y_axis, outlier_y)
        
        if z_axis in available_temp_cols:
            selected_temp = st.selectbox(
                "Select temperature for conductivity",
                options=available_temp_cols,
                index=available_temp_cols.index(default_temp) if default_temp in available_temp_cols else 0
            )
            z_col = selected_temp
            z_label = selected_temp.replace('σ total, ', 'σ at ') + ' °C (mS/cm)'
        else:
            z_col = 'Ea_final'
            z_label = 'Ea (eV)'
        
        plot_type = st.radio(
            "Heat map type",
            options=['Scatter with color scale', 'Contour plot', '3D surface'],
            horizontal=True
        )
        
        palette_name = st.selectbox(
            "Color palette",
            options=list(COLOR_PALETTES.keys()),
            index=0
        )
        palette = COLOR_PALETTES[palette_name]
        
        if plot_type == 'Contour plot':
            col1, col2, col3 = st.columns(3)
            with col1:
                grid_resolution = st.slider(
                    "Smoothing (grid resolution)",
                    min_value=20,
                    max_value=100,
                    value=50,
                    step=10
                )
            with col2:
                show_contour_lines = st.checkbox("Show contour lines", value=True)
            with col3:
                show_contour_labels = st.checkbox("Show contour labels", value=True)
        else:
            grid_resolution = 50
            show_contour_lines = True
            show_contour_labels = True
        
        col1, col2, col3 = st.columns(3)
        with col1:
            log_x = st.checkbox("log10(X)", value=False)
        with col2:
            log_y = st.checkbox("log10(Y)", value=False)
        with col3:
            log_z = st.checkbox("log10(Z)", value=False)
        
        if st.button("Generate Heat Map", key="heatmap_btn"):
            if len(df_plot) == 0:
                st.warning("No data available after filtering")
            else:
                marker_style, show_legend = get_filter_markers(df_plot, active_filters_count)
                
                if plot_type == 'Scatter with color scale':
                    fig = create_scatter_heatmap(
                        df_plot, x_axis, y_axis, z_col,
                        log_x, log_y, log_z, palette,
                        '', x_axis, y_axis, z_label,
                        marker_style, show_legend, active_filter_name
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
                        mime="image/png"
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
        
        # Add delta descriptors if they exist in df
        for col in ['Δρ', 'Δd', 'ΔT', 'Δρ_rel', 'Δd_rel']:
            if col in df_filtered.columns:
                numeric_cols_bubble.append(col)
        
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
                index=y_index
            )
        
        with col2:
            x_axis_bubble = st.selectbox(
                "X-axis",
                options=numeric_cols_bubble,
                index=0
            )
        
        with col3:
            color_axis = st.selectbox(
                "Bubble color",
                options=['None'] + numeric_cols_bubble + ['Sintering additive', 'Atmospheres', 'Humidity', 'Structure'],
                index=0
            )
        
        with col4:
            size_axis = st.selectbox(
                "Bubble size",
                options=['None'] + numeric_cols_bubble,
                index=0
            )
        
        # Data type filter
        data_type_bubble = st.radio(
            "Data type",
            options=['All data', 'Pure only', 'Additives only'],
            horizontal=True,
            key="data_type_bubble"
        )
        
        # Apply data type filter
        df_plot_type_bubble = df_filtered.copy()
        if data_type_bubble == 'Pure only':
            df_plot_type_bubble = df_plot_type_bubble[df_plot_type_bubble['Sintering additive'].str.lower() == 'pure']
        elif data_type_bubble == 'Additives only':
            df_plot_type_bubble = df_plot_type_bubble[df_plot_type_bubble['Sintering additive'].str.lower() != 'pure']
        
        # Outlier removal controls for X and Y in bubble charts
        col1, col2 = st.columns(2)
        with col1:
            outlier_x_bubble = st.selectbox(
                "Remove outliers from X",
                options=[0, 1, 2, 3],
                index=0,
                format_func=lambda x: 'None' if x == 0 else str(x),
                key="outlier_x_bubble"
            )
        with col2:
            outlier_y_bubble = st.selectbox(
                "Remove outliers from Y",
                options=[0, 1, 2, 3],
                index=0,
                format_func=lambda x: 'None' if x == 0 else str(x),
                key="outlier_y_bubble"
            )
        
        # Apply outlier removal
        df_plot_bubble = df_plot_type_bubble.copy()
        if outlier_x_bubble > 0:
            df_plot_bubble = remove_outliers(df_plot_bubble, x_axis_bubble, outlier_x_bubble)
        if outlier_y_bubble > 0:
            df_plot_bubble = remove_outliers(df_plot_bubble, y_axis_bubble, outlier_y_bubble)
        
        if y_axis_bubble in available_temp_cols:
            selected_temp_bubble = st.selectbox(
                "Select temperature for conductivity",
                options=available_temp_cols,
                index=available_temp_cols.index(default_temp) if default_temp in available_temp_cols else 0,
                key="temp_bubble"
            )
            y_label = selected_temp_bubble.replace('σ total, ', 'σ at ') + ' °C (mS/cm)'
            y_col_bubble = selected_temp_bubble
        else:
            y_label = 'Ea (eV)'
            y_col_bubble = 'Ea_final'
        
        palette_name_bubble = st.selectbox(
            "Color palette for bubbles",
            options=list(COLOR_PALETTES.keys()),
            index=0
        )
        palette_bubble = COLOR_PALETTES[palette_name_bubble]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            log_x_bubble = st.checkbox("log10(X)", value=False, key="log_x_bubble")
        with col2:
            log_y_bubble = st.checkbox("log10(Y)", value=False, key="log_y_bubble")
        with col3:
            log_color_bubble = st.checkbox("log10(Color)", value=False, key="log_color_bubble")
        with col4:
            log_size_bubble = st.checkbox("log10(Size)", value=False, key="log_size_bubble")
        
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
                
                marker_style, show_legend = get_filter_markers(df_plot_bubble, active_filters_count)
                
                fig = create_bubble_chart(
                    df_plot_bubble, x_axis_bubble, y_col_bubble, 
                    color_col_bubble, size_col_bubble,
                    log_x_bubble, log_y_bubble, log_color_bubble, log_size_bubble,
                    palette_bubble, '', x_axis_bubble, y_label,
                    marker_style, show_legend, active_filter_name
                )
                
                if fig is not None:
                    st.pyplot(fig)
                    
                    buf = download_plot(fig, "bubble_chart.png")
                    st.download_button(
                        label="📥 Download Plot (PNG, 600 dpi)",
                        data=buf,
                        file_name="bubble_chart.png",
                        mime="image/png"
                    )
                    
                    plt.close(fig)
    
    # ============================================
    # TAB 3: ADDITIVE ANALYSIS
    # ============================================
    with tab3:
        st.header("📊 Analysis of Sintering Additives")
        
        if len(delta_df) == 0:
            st.warning("No additive pairs found. Delta descriptors cannot be calculated.")
        else:
            # Filter delta_df using the same filters as main data
            # (We need to merge with main df to apply filters)
            delta_filtered = delta_df.copy()
            
            # Apply additive type filter if selected in sidebar
            if 'Sintering additive' in df_filtered.columns:
                available_additives = df_filtered['Sintering additive'].unique()
                delta_filtered = delta_filtered[delta_filtered['Sintering additive'].isin(available_additives)]
            
            if len(delta_filtered) == 0:
                st.warning("No data available after filtering")
            else:
                st.subheader("Delta Descriptors Table")
                st.dataframe(delta_filtered)
                
                # Download delta data
                csv_delta = delta_filtered.to_csv(index=False, sep='\t')
                st.download_button(
                    label="📥 Download Delta Data (TSV)",
                    data=csv_delta,
                    file_name="delta_descriptors.tsv",
                    mime="text/tab-separated-values"
                )
                
                st.markdown("---")
                st.subheader("Effectiveness of Sintering Additives")
                
                # Select temperature for Δσ
                delta_temp_cols = [col for col in delta_filtered.columns if col.startswith('Δσ total,')]
                
                if len(delta_temp_cols) > 0:
                    selected_delta_temp = st.selectbox(
                        "Select temperature for Δσ analysis",
                        options=delta_temp_cols,
                        index=0
                    )
                    
                    # Plotting options for delta analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        x_delta = st.selectbox(
                            "X-axis for delta analysis",
                            options=['Δρ', 'Δd', 'ΔT', 'Δρ_rel', 'Δd_rel', 'x, wt%'],
                            index=0
                        )
                    
                    with col2:
                        color_delta = st.selectbox(
                            "Color by",
                            options=['Sintering additive', 'Atmospheres', 'Humidity'],
                            index=0
                        )
                    
                    # Prepare data for plotting
                    plot_delta = delta_filtered.dropna(subset=[x_delta, selected_delta_temp, color_delta])
                    
                    if len(plot_delta) > 0:
                        # Scatter plot: Δρ/Δd vs Δσ
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        # Get unique categories for coloring
                        categories = plot_delta[color_delta].unique()
                        palette_colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
                        
                        for i, cat in enumerate(categories):
                            mask = plot_delta[color_delta] == cat
                            if mask.sum() > 0:
                                ax.scatter(
                                    plot_delta[mask][x_delta],
                                    plot_delta[mask][selected_delta_temp],
                                    label=cat,
                                    color=palette_colors[i],
                                    s=80,
                                    alpha=0.7,
                                    edgecolors='black',
                                    linewidth=0.5
                                )
                        
                        ax.set_xlabel(x_delta, fontsize=11, fontweight='bold')
                        ax.set_ylabel(selected_delta_temp.replace('Δσ total, ', 'Δσ at ') + ' °C (mS/cm)', 
                                     fontsize=11, fontweight='bold')
                        ax.grid(True, alpha=0.3, linestyle='--')
                        ax.legend(loc='upper right', frameon=True, framealpha=0.9)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        buf = download_plot(fig, "delta_analysis.png")
                        st.download_button(
                            label="📥 Download Plot (PNG, 600 dpi)",
                            data=buf,
                            file_name="delta_analysis.png",
                            mime="image/png"
                        )
                        plt.close(fig)
                    else:
                        st.warning("Not enough data for the selected parameters")
                
                # Correlation matrix for additives
                st.markdown("---")
                st.subheader("Correlation Matrix for Additives")
                
                corr_temp_options = [col for col in delta_filtered.columns if col.startswith('Δσ total,')]
                if len(corr_temp_options) == 0:
                    corr_temp_options = [col for col in available_temp_cols if col in df_filtered.columns]
                
                if len(corr_temp_options) > 0:
                    selected_corr_temp = st.selectbox(
                        "Select temperature for correlation matrix",
                        options=corr_temp_options,
                        index=0,
                        key="corr_temp_additive"
                    )
                    
                    if st.button("Generate Correlation Matrix (Additives)", key="corr_additive_btn"):
                        # Use delta_filtered for correlation
                        corr_df = delta_filtered.copy()
                        
                        # Rename Δσ columns to simpler names for correlation
                        if selected_corr_temp in corr_df.columns:
                            corr_df['Δσ_selected'] = corr_df[selected_corr_temp]
                        
                        # Select columns for correlation
                        corr_cols_add = ['Δρ', 'Δd', 'ΔT', 'Δρ_rel', 'Δd_rel', 'x, wt%']
                        corr_cols_add = [col for col in corr_cols_add if col in corr_df.columns]
                        
                        if 'Δσ_selected' in corr_df.columns:
                            corr_cols_add.append('Δσ_selected')
                        
                        if len(corr_cols_add) >= 2:
                            # Calculate correlation on numeric columns only
                            corr_data = corr_df[corr_cols_add].select_dtypes(include=[np.number])
                            corr_data = corr_data.dropna()
                            
                            if len(corr_data) > 0:
                                # Rename columns for better display
                                rename_dict = {}
                                for col in corr_data.columns:
                                    if col == 'Δσ_selected':
                                        rename_dict[col] = f'Δσ at {selected_corr_temp.replace("Δσ total, ", "")}'
                                    else:
                                        rename_dict[col] = col
                                corr_data = corr_data.rename(columns=rename_dict)
                                
                                corr_matrix = corr_data.corr()
                                
                                fig, ax = plt.subplots(figsize=(10, 8))
                                sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                                           cmap='coolwarm', vmin=-1, vmax=1, center=0,
                                           square=True, linewidths=0.5,
                                           annot_kws={"size": 9}, ax=ax)
                                ax.set_title('')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                buf = download_plot(fig, "correlation_additive.png")
                                st.download_button(
                                    label="📥 Download Correlation Matrix (PNG, 600 dpi)",
                                    data=buf,
                                    file_name="correlation_additive.png",
                                    mime="image/png"
                                )
                                plt.close(fig)
                            else:
                                st.warning("Not enough numeric data for correlation")
                        else:
                            st.warning("Not enough columns for correlation matrix")
    
    # ============================================
    # TAB 4: DATA
    # ============================================
    with tab4:
        st.header("📋 Data with Calculated Descriptors")
        
        st.dataframe(df_filtered)
        
        csv_data = df_filtered.to_csv(index=False, sep='\t')
        st.download_button(
            label="📥 Download Data (TSV)",
            data=csv_data,
            file_name="filtered_data.tsv",
            mime="text/tab-separated-values"
        )
        
        st.subheader("📊 Data Statistics")
        numeric_display = df_filtered.select_dtypes(include=[np.number])
        if not numeric_display.empty:
            st.dataframe(numeric_display.describe())
        
        # Correlation matrix for all data
        st.markdown("---")
        st.subheader("Correlation Matrix (All Data)")
        
        if len(available_temp_cols) > 0:
            corr_temp_all = st.selectbox(
                "Select temperature for correlation matrix",
                options=available_temp_cols,
                index=0,
                key="corr_temp_all"
            )
            
            palette_corr = st.selectbox(
                "Color palette for correlation",
                options=list(COLOR_PALETTES.keys()),
                index=0,
                key="corr_palette"
            )
            
            if st.button("Generate Correlation Matrix (All Data)", key="corr_all_btn"):
                fig = create_correlation_matrix(df_filtered, corr_temp_all, COLOR_PALETTES[palette_corr])
                
                if fig is not None:
                    st.pyplot(fig)
                    
                    buf = download_plot(fig, "correlation_all.png")
                    st.download_button(
                        label="📥 Download Correlation Matrix (PNG, 600 dpi)",
                        data=buf,
                        file_name="correlation_all.png",
                        mime="image/png"
                    )
                    plt.close(fig)

# ============================================
# APPLICATION LAUNCH
# ============================================

if __name__ == "__main__":
    main()
