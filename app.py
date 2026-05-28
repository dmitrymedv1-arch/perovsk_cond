import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata, RBFInterpolator
import re
from datetime import datetime
import openpyxl
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import xgboost as xgb
from itertools import combinations
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import linregress
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.spatial import cKDTree
import time
import umap.umap_ as umap
from sklearn.decomposition import KernelPCA
from scipy.spatial.distance import pdist, squareform
warnings.filterwarnings('ignore')

# ============================================================================
# CUSTOM CSS FOR MODERN SCIENTIFIC UI
# ============================================================================
def apply_custom_css():
    """Apply modern scientific styling to the Streamlit app"""
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #FFFFFF;
    }
    
    /* Block container styling */
    .stApp {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
    }
    
    /* Metric cards */
    .stMetric {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1E293B !important;
        font-weight: 600 !important;
    }
    
    h1 {
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 10px;
        display: inline-block;
    }
    
    h2 {
        border-left: 4px solid #3B82F6;
        padding-left: 15px;
        margin-top: 20px;
    }
    
    h3 {
        color: #475569 !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        background: #FFFFFF;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    .stAlert[data-baseweb="notification"] {
        background-color: #F8FAFC;
    }
    
    /* Select boxes */
    .stSelectbox label, .stMultiSelect label {
        color: #475569 !important;
        font-weight: 500 !important;
    }
    
    /* Sliders */
    .stSlider label {
        color: #475569 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        border-radius: 12px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #E2E8F0;
        border-radius: 8px;
        padding: 8px 16px;
        color: #475569;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #F8FAFC;
        border-radius: 8px;
        color: #1E293B;
        font-weight: 500;
    }
    
    /* Plot containers */
    .plot-container {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        margin: 16px 0;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-12ttj6m {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #94A3B8;
        font-size: 12px;
        border-top: 1px solid #E2E8F0;
        margin-top: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================
AVOGADRO_NUMBER = 6.02214076e23  # mol⁻¹
OXYGEN_RADIUS = 1.4  # Å
PREFACTOR_VOLUME = 16 * np.pi / 3  # 16π/3 for sphere volume calculation
GAS_CONSTANT = 8.314  # J/(mol·K)
BOLTZMANN_EV = 8.617333262145e-5  # eV/K

# Scientific plot style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#D1D5DB',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'xtick.color': '#000000',
    'ytick.color': '#000000',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#D1D5DB',
    'legend.fancybox': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.facecolor': '#FFFFFF',
    'axes.labelcolor': '#000000',
    'text.color': '#000000',
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'errorbar.capsize': 3,
})

# ============================================================================
# DATABASE OF IONIC RADII (Shannon)
# ============================================================================
IONIC_RADII = {
    # Format: (ion, charge, CN): (crystal radius, ionic radius)
    # For A-site use CN=12, for B-site CN=6, for O - fixed value
    ('Ba', 2, 12): 1.61,
    ('Sr', 2, 12): 1.44,
    ('O', -2, 6): 1.4,
    
    # B-cations (CN=6)
    ('Ce', 4, 6): 0.87,
    ('Zr', 4, 6): 0.72,
    ('Sn', 4, 6): 0.69,
    ('Ti', 4, 6): 0.605,
    ('Hf', 4, 6): 0.71,
    
    # D-dopants (acceptors, usually 3+, CN=6)
    ('Gd', 3, 6): 0.938,
    ('Sm', 3, 6): 0.958,
    ('Y', 3, 6): 0.9,
    ('In', 3, 6): 0.8,
    ('Sc', 3, 6): 0.745,
    ('Dy', 3, 6): 0.912,
    ('Ho', 3, 6): 0.901,
    ('Yb', 3, 6): 0.868,
    ('Eu', 3, 6): 0.947,
    ('Nd', 3, 6): 0.983,
    ('La', 3, 6): 1.032,
    ('Pr', 3, 6): 0.99,
    ('Tb', 3, 6): 0.923,
    ('Er', 3, 6): 0.89,
    ('Tm', 3, 6): 0.88,
    ('Lu', 3, 6): 0.861,
    ('Ca', 2, 6): 1.00,
    
    # Sintering additives (transition metals, CN=6 for simplicity)
    ('Cu', 2, 6): 0.73,
    ('Ni', 2, 6): 0.69,
    ('Zn', 2, 6): 0.74,
    ('Co', 2, 6): 0.65,
}

# ============================================================================
# DATABASE OF ELECTRONEGATIVITY (Pauling)
# ============================================================================
ELECTRONEGATIVITY = {
    'Ba': 0.89,
    'Sr': 0.95,
    'Ce': 1.12,
    'Zr': 1.33,
    'Sn': 1.96,
    'Ti': 1.54,
    'Hf': 1.3,
    'Gd': 1.20,
    'Sm': 1.17,
    'Y': 1.22,
    'In': 1.78,
    'Sc': 1.36,
    'Dy': 1.22,
    'Ho': 1.23,
    'Yb': 1.22,
    'Eu': 1.20,
    'Nd': 1.14,
    'La': 1.10,
    'Pr': 1.13,
    'Tb': 1.20,
    'Er': 1.24,
    'Tm': 1.25,
    'Lu': 1.27,
    'Ca': 1.00,
    'O': 3.44,
    'Cu': 1.90,
    'Ni': 1.91,
    'Zn': 1.65,
    'Co': 1.88,
}

# ============================================================================
# DATABASE OF IONIC CHARGES
# ============================================================================
IONIC_CHARGES = {
    'Ba': 2,
    'Sr': 2,
    'Ce': 4,
    'Zr': 4,
    'Sn': 4,
    'Ti': 4,
    'Hf': 4,
    'Gd': 3,
    'Sm': 3,
    'Y': 3,
    'In': 3,
    'Sc': 3,
    'Dy': 3,
    'Ho': 3,
    'Yb': 3,
    'Eu': 3,
    'Nd': 3,
    'La': 3,
    'Pr': 3,
    'Tb': 3,
    'Er': 3,
    'Tm': 3,
    'Lu': 3,
    'Ca': 2,
    'O': -2,
    'Cu': 2,
    'Ni': 2,
    'Zn': 2,
    'Co': 2,
}

# ============================================================================
# DATABASE OF BASIC STRUCTURE PROPERTIES
# ============================================================================
MATERIAL_PROPERTIES = {
    'BaCeO3': {
        'band_gap': 2.299,
        'E_form': -3.550,
        'density': 6.034,
        'M_molar': 341.36,
        'r_A': 1.61,
        'r_B': 0.87,
        'r_O': 1.4
    },
    'BaSnO3': {
        'band_gap': 0.372,
        'E_form': -2.587,
        'density': 7.097,
        'M_molar': 336.03,
        'r_A': 1.61,
        'r_B': 0.69,
        'r_O': 1.4
    },
    'BaHfO3': {
        'band_gap': 3.539,
        'E_form': -3.787,
        'density': 8.332,
        'M_molar': 428.72,
        'r_A': 1.61,
        'r_B': 0.71,
        'r_O': 1.4
    },
    'BaZrO3': {
        'band_gap': 3.116,
        'E_form': -3.639,
        'density': 6.148,
        'M_molar': 348.54,
        'r_A': 1.61,
        'r_B': 0.72,
        'r_O': 1.4
    },
    'BaTiO3': {
        'band_gap': None,
        'E_form': -1.685,
        'density': 4.547,
        'M_molar': 233.19,
        'r_A': 1.61,
        'r_B': 0.605,
        'r_O': 1.4
    },
    'SrSnO3': {
        'band_gap': 1.555,
        'E_form': -2.631,
        'density': 6.355,
        'M_molar': 302.34,
        'r_A': 1.44,
        'r_B': 0.69,
        'r_O': 1.4
    }
}

# ============================================================================
# ATOMIC MASSES (g/mol)
# ============================================================================
ATOMIC_MASSES = {
    'Ba': 137.33,
    'Sr': 87.62,
    'Ce': 140.12,
    'Zr': 91.22,
    'Sn': 118.71,
    'Ti': 47.87,
    'Hf': 178.49,
    'Gd': 157.25,
    'Sm': 150.36,
    'Y': 88.91,
    'In': 114.82,
    'Sc': 44.96,
    'Dy': 162.50,
    'Ho': 164.93,
    'Yb': 173.05,
    'Eu': 151.96,
    'Nd': 144.36,
    'La': 138.91,
    'Pr': 140.91,
    'Tb': 158.93,
    'Er': 167.26,
    'Tm': 168.93,
    'Lu': 174.97,
    'Ca': 40.08,
    'O': 16.00,
    'Cu': 63.55,
    'Ni': 58.69,
    'Zn': 65.38,
    'Co': 58.93,
}

# Color map for B-cations
B_COLORS = {
    'Ce': '#E41A1C',
    'Zr': '#377EB8',
    'Sn': '#4DAF4A',
    'Ti': '#984EA3',
    'Hf': '#FF7F00',
    'default': '#999999'
}

# Color map for sintering additives
SINTERING_ADDITIVE_COLORS = {
    'Pure': '#10B981',
    'Cu': '#EF4444',
    'Ni': '#3B82F6',
    'Zn': '#8B5CF6',
    'Co': '#F59E0B',
    'default': '#6B7280'
}

# Color map for dopant types
DOPANT_COLORS = {
    'Y': '#1E88E5',
    'Gd': '#DC143C',
    'Sm': '#FF8C00',
    'Yb': '#2E8B57',
    'Nd': '#8B008B',
    'Eu': '#FFD700',
    'Dy': '#00CED1',
    'Er': '#FF69B4',
    'Ho': '#7B68EE',
    'Lu': '#A0522D',
    'Sc': '#556B2F',
    'In': '#D2691E',
    'Ca': '#808080',
    'default': '#6B7280'
}

# Markers for sintering additives
SINTERING_ADDITIVE_MARKERS = {
    'Pure': 'o',
    'Cu': 's',
    'Ni': '^',
    'Zn': 'D',
    'Co': 'v',
    'default': 'o'
}

# Literature-based incorporation likelihood (True = dissolves in lattice, False = segregates to GB)
ADDITIVE_INCORPORATION = {
    'Zn': True,   # Zn often dissolves into B-site
    'Cu': False,  # Cu typically segregates to grain boundaries
    'Ni': False,  # Ni tends to segregate
    'Co': True,   # Co can incorporate depending on conditions
    'Pure': True  # No additive, N/A
}

# ============================================================================
# GLOBAL PLOT SETTINGS (will be populated from session state)
# ============================================================================
def init_plot_settings():
    """Initialize global plot settings in session state"""
    if 'plot_settings' not in st.session_state:
        st.session_state.plot_settings = {
            'default_cmap': 'viridis',
            'show_trendlines': True,
            'marker_size': 80,
            'font_scale': 1.0,
            'outlier_sensitivity': 'medium (IQR=1.5)',
            'contour_grid_resolution': 50,
            'bubble_alpha': 0.7,
            'show_contour_labels': True,
            'contour_levels': 20
        }

def get_outlier_iqr_multiplier():
    """Get IQR multiplier based on outlier sensitivity setting"""
    sensitivity = st.session_state.plot_settings.get('outlier_sensitivity', 'medium (IQR=1.5)')
    if 'low' in sensitivity:
        return 2.0
    elif 'high' in sensitivity:
        return 1.0
    else:
        return 1.5

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_float_converter(value):
    """
    Safely convert a value to float, handling NaN, None, and string values.
    
    Parameters
    ----------
    value : any
        Value to convert to float
        
    Returns
    -------
    float or None
        Converted float value or None if conversion fails
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        # Handle empty strings
        if value.strip() == '':
            return None
        # Handle percentage signs
        if '%' in value:
            value = value.replace('%', '')
        # Handle comma as decimal separator (European format)
        if ',' in value and '.' not in value:
            value = value.replace(',', '.')
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


class FlexibleColumnMapper:
    """
    Flexible column mapper for finding columns with different naming conventions.
    
    This class handles variations in column names like:
    - "σ total, mS" vs "sigma_total_mS" vs "Total conductivity (mS/cm)"
    - "600" vs "600C" vs "600°C" vs "600 C"
    """
    
    def __init__(self):
        """Initialize the column mapper with patterns for different measurement types."""
        self.patterns = {
            'sigma_total': [
                r'σ\s*total',
                r'sigma\s*_?\s*total',
                r'total\s*conductivity',
                r'σ\s*\(total\)',
                r'σ\s*,\s*mS',
                r'σ\s*total\s*,\s*mS',
                r'σ\s*total\s*mS',
                r'sigma_total_mS'
            ],
            'sigma_bulk': [
                r'σ\s*bulk',
                r'sigma\s*_?\s*bulk',
                r'bulk\s*conductivity',
                r'σ\s*\(bulk\)',
                r'σ\s*bulk\s*mS',
                r'sigma_bulk_mS'
            ],
            'sigma_gb': [
                r'σ\s*gb',
                r'sigma\s*_?\s*gb',
                r'grain\s*boundary',
                r'σ\s*\(gb\)',
                r'σ\s*gb\s*mS',
                r'sigma_gb_mS',
                r'σ\s*gb\s*conductivity'
            ]
        }
    
    def find_column(self, df, col_type, temperature):
        """
        Find a column in the dataframe by type and temperature.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe to search
        col_type : str
            Type of column ('sigma_total', 'sigma_bulk', 'sigma_gb')
        temperature : int
            Temperature in Celsius
            
        Returns
        -------
        str or None
            Column name if found, None otherwise
        """
        temp_patterns = [
            f'{temperature}',
            f'{temperature}C',
            f'{temperature}°C',
            f'{temperature} C',
            f'{temperature}° C',
            f'_{temperature}_',
            f'@{temperature}',
            f' {temperature} '
        ]
        
        for col in df.columns:
            col_str = str(col).lower()
            
            # Check if column matches the type pattern
            type_match = False
            for pattern in self.patterns.get(col_type, []):
                if re.search(pattern, col_str, re.IGNORECASE):
                    type_match = True
                    break
            
            if not type_match:
                continue
            
            # Check if column contains the temperature
            temp_match = False
            for t_pattern in temp_patterns:
                if t_pattern.lower() in col_str:
                    temp_match = True
                    break
            
            if temp_match:
                return col
        
        return None


# ============================================================================
# NEW FUNCTION: READ EXCEL WITH SINGLE ROW HEADER
# ============================================================================
def read_excel_simple(uploaded_file):
    """
    Read Excel file with single row header structure.
    
    The Excel file has:
    - Row 1: Column names (A cation, B1 cation, σ total, 200, σ bulk, 200, σ gb, 200, etc.)
    - Row 2 onwards: Data
    
    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object
        
    Returns
    -------
    pandas.DataFrame
        Processed dataframe with proper column names
    """
    # Read the Excel file with header on first row
    df = pd.read_excel(uploaded_file, engine='openpyxl', header=0)
    
    # Clean column names: strip whitespace
    df.columns = df.columns.str.strip()
    
    # Remove any completely empty rows
    df = df.dropna(how='all')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


# ============================================================================
# NEW FUNCTION: EXTRAPOLATE CONDUCTIVITY (without Ea)
# ============================================================================
def extrapolate_conductivity(sigma_data, target_temperature):
    """
    Extrapolate conductivity to a target temperature using Arrhenius law.
    
    Parameters
    ----------
    sigma_data : list of dict
        Conductivity data at different temperatures
    target_temperature : float
        Target temperature in Celsius
    
    Returns
    -------
    float or None
        Predicted conductivity at target temperature
    """
    if len(sigma_data) < 2:
        return None
    
    # Transform for Arrhenius: ln(σ) vs 1000/T
    temps_K = []
    ln_sigma = []
    
    for data in sigma_data:
        T_K = data.get('temperature_K')
        sigma = data.get('sigma_total_mS') or data.get('sigma_bulk_mS') or data.get('sigma_gb_mS')
        if sigma is not None and sigma > 0 and T_K is not None and T_K > 0:
            ln_sigma.append(np.log(sigma))
            temps_K.append(1000.0 / T_K)
    
    if len(temps_K) < 2:
        return None
    
    # Linear regression
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(temps_K, ln_sigma)
        
        # Extrapolation
        target_K = target_temperature + 273.15
        target_inv = 1000.0 / target_K
        ln_sigma_pred = intercept + slope * target_inv
        
        return np.exp(ln_sigma_pred)
    except (ValueError, TypeError, stats.LinAlgError):
        return None


def extrapolate_conductivity_for_sample(row_sigma_data, temperatures_to_extrapolate):
    """
    Extrapolate conductivity for a sample to multiple temperatures.
    
    Parameters
    ----------
    row_sigma_data : list of dict
        Existing conductivity data for the sample
    temperatures_to_extrapolate : list
        List of target temperatures in Celsius
    
    Returns
    -------
    dict
        Dictionary mapping temperature to extrapolated conductivity
    """
    results = {}
    
    if len(row_sigma_data) < 2:
        return results
    
    # Extract data for Arrhenius plot
    temps_K = []
    ln_sigma = []
    
    for data in row_sigma_data:
        T_K = data.get('temperature_K')
        sigma = data.get('sigma_total_mS') or data.get('sigma_bulk_mS') or data.get('sigma_gb_mS')
        if sigma is not None and sigma > 0 and T_K is not None and T_K > 0:
            ln_sigma.append(np.log(sigma))
            temps_K.append(1000.0 / T_K)
    
    if len(temps_K) < 2:
        return results
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(temps_K, ln_sigma)
        
        for T in temperatures_to_extrapolate:
            target_K = T + 273.15
            target_inv = 1000.0 / target_K
            ln_sigma_pred = intercept + slope * target_inv
            results[T] = np.exp(ln_sigma_pred)
    except (ValueError, TypeError, stats.LinAlgError):
        pass
    
    return results


# ============================================================================
# NEW FUNCTION: CALCULATE ACTIVATION ENERGY FROM ARRHENIUS
# ============================================================================

def calculate_activation_energy_from_data(temperatures_C, conductivities_mS):
    """
    Calculate activation energy (Ea) from Arrhenius plot.
    
    Uses the equation: ln(σ * T) = ln(A) - (Ea / kB) * (1000/T)
    where kB = 8.617e-5 eV/K
    
    Parameters
    ----------
    temperatures_C : list or array
        Temperatures in Celsius
    conductivities_mS : list or array
        Conductivities in mS/cm
    
    Returns
    -------
    dict
        Contains 'Ea_eV', 'R_squared', 'slope', 'intercept', 'n_points'
    """
    # Convert to numpy arrays
    temps_C = np.array(temperatures_C)
    sigmas = np.array(conductivities_mS)
    
    # Remove NaN and zero values
    valid_mask = ~(np.isnan(temps_C) | np.isnan(sigmas) | (sigmas <= 0))
    temps_C = temps_C[valid_mask]
    sigmas = sigmas[valid_mask]
    
    if len(temps_C) < 3:
        return {
            'Ea_eV': None,
            'R_squared': None,
            'slope': None,
            'intercept': None,
            'n_points': len(temps_C),
            'error': 'Insufficient points (need at least 3)'
        }
    
    # Convert to Kelvin
    temps_K = temps_C + 273.15
    
    # Calculate Arrhenius variables
    invT_1000 = 1000.0 / temps_K
    ln_sigmaT = np.log(sigmas * temps_K)
    
    # Linear regression
    try:
        slope, intercept, r_value, p_value, std_err = linregress(invT_1000, ln_sigmaT)
        
        # Ea = -slope * kB (where kB is in eV/K)
        Ea_eV = -slope * BOLTZMANN_EV
        
        return {
            'Ea_eV': Ea_eV,
            'R_squared': r_value ** 2,
            'slope': slope,
            'intercept': intercept,
            'n_points': len(temps_C),
            'error': None
        }
    except Exception as e:
        return {
            'Ea_eV': None,
            'R_squared': None,
            'slope': None,
            'intercept': None,
            'n_points': len(temps_C),
            'error': str(e)
        }


def compute_activation_energies_for_df(df_long):
    """
    Compute activation energies for all samples and conductivity types.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data with columns: sample_id, temperature_C, sigma_total_mS, sigma_bulk_mS, sigma_gb_mS
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with Ea values for each sample and conductivity type
    """
    Ea_results = []
    
    for sample_id in df_long['sample_id'].unique():
        sample_data = df_long[df_long['sample_id'] == sample_id]
        
        result = {
            'sample_id': sample_id,
            'A_cation': sample_data['A_cation'].iloc[0] if len(sample_data) > 0 else None,
            'B1_cation': sample_data['B1_cation'].iloc[0] if len(sample_data) > 0 else None,
            'dopant': sample_data['dopant'].iloc[0] if len(sample_data) > 0 else None,
            'additive_type': sample_data['additive_type'].iloc[0] if len(sample_data) > 0 else None,
        }
        
        # Total conductivity Ea
        total_data = sample_data.dropna(subset=['sigma_total_mS'])
        if len(total_data) >= 3:
            temps = total_data['temperature_C'].values
            sigmas = total_data['sigma_total_mS'].values
            ea_result = calculate_activation_energy_from_data(temps, sigmas)
            result['Ea_total_eV'] = ea_result['Ea_eV']
            result['Ea_total_R2'] = ea_result['R_squared']
            result['Ea_total_n_points'] = ea_result['n_points']
        else:
            result['Ea_total_eV'] = None
            result['Ea_total_R2'] = None
            result['Ea_total_n_points'] = 0
        
        # Bulk conductivity Ea
        bulk_data = sample_data.dropna(subset=['sigma_bulk_mS'])
        if len(bulk_data) >= 3:
            temps = bulk_data['temperature_C'].values
            sigmas = bulk_data['sigma_bulk_mS'].values
            ea_result = calculate_activation_energy_from_data(temps, sigmas)
            result['Ea_bulk_eV'] = ea_result['Ea_eV']
            result['Ea_bulk_R2'] = ea_result['R_squared']
            result['Ea_bulk_n_points'] = ea_result['n_points']
        else:
            result['Ea_bulk_eV'] = None
            result['Ea_bulk_R2'] = None
            result['Ea_bulk_n_points'] = 0
        
        # Grain boundary conductivity Ea
        gb_data = sample_data.dropna(subset=['sigma_gb_mS'])
        if len(gb_data) >= 3:
            temps = gb_data['temperature_C'].values
            sigmas = gb_data['sigma_gb_mS'].values
            ea_result = calculate_activation_energy_from_data(temps, sigmas)
            result['Ea_gb_eV'] = ea_result['Ea_eV']
            result['Ea_gb_R2'] = ea_result['R_squared']
            result['Ea_gb_n_points'] = ea_result['n_points']
        else:
            result['Ea_gb_eV'] = None
            result['Ea_gb_R2'] = None
            result['Ea_gb_n_points'] = 0
        
        Ea_results.append(result)
    
    return pd.DataFrame(Ea_results)


# ============================================================================
# NEW CLASS: CONDUCTIVITY DESCRIPTOR CALCULATOR (ENHANCED)
# ============================================================================
class ConductivityDescriptorCalculator:
    """
    Class for calculating all physicochemical and microstructural descriptors for conductivity analysis.
    
    This enhanced version includes additional descriptors:
    - radius_mismatch: absolute difference from ideal B-site radius
    - electronegativity_difference_B_O: difference between average B and oxygen
    - lattice_distortion_index: approximate lattice distortion from tolerance factor
    - additive_incorporation_likely: binary indicator from literature
    - D_type and D_conc: acceptor dopant type and concentration
    """
    
    def __init__(self, a_element='Ba'):
        """
        Initialize the descriptor calculator.
        
        Parameters
        ----------
        a_element : str
            A-site cation (default: 'Ba')
        """
        self.a_element = a_element
        self.r_O = IONIC_RADII.get(('O', -2, 6), 1.4)
        self.χ_O = ELECTRONEGATIVITY.get('O', 3.44)
        self.r_A = IONIC_RADII.get((a_element, 2, 12), None)
        self.χ_A = ELECTRONEGATIVITY.get(a_element, None)
        self.z_A = IONIC_CHARGES.get(a_element, 2)
        # Theoretical density for porosity calculation
        self.theoretical_density = None
        # Reference radius for mismatch calculation (typically Zr or Ce)
        self.reference_B_radius = 0.72  # Zr4+ radius as reference
    
    def get_ionic_radius(self, element, charge=None, coordination=6):
        """
        Get ionic radius with automatic charge detection.
        
        Parameters
        ----------
        element : str
            Element symbol
        charge : int, optional
            Ionic charge (auto-detected if not provided)
        coordination : int
            Coordination number (default: 6)
            
        Returns
        -------
        float or None
            Ionic radius in Angstroms
        """
        if charge is None:
            charge = IONIC_CHARGES.get(element, 4)
        return IONIC_RADII.get((element, charge, coordination), None)
    
    def get_electronegativity(self, element):
        """
        Get electronegativity of an element.
        
        Parameters
        ----------
        element : str
            Element symbol
            
        Returns
        -------
        float or None
            Pauling electronegativity
        """
        return ELECTRONEGATIVITY.get(element, None)
    
    def get_charge(self, element):
        """
        Get typical ionic charge of an element.
        
        Parameters
        ----------
        element : str
            Element symbol
            
        Returns
        -------
        int or None
            Ionic charge
        """
        return IONIC_CHARGES.get(element, None)
    
    def get_atomic_mass(self, element):
        """
        Get atomic mass of an element.
        
        Parameters
        ----------
        element : str
            Element symbol
            
        Returns
        -------
        float or None
            Atomic mass in g/mol
        """
        return ATOMIC_MASSES.get(element, None)
    
    def calculate_formula(self, b1_element, b2_element, b2_cont, dopant, dop_cont):
        """
        Calculate composition and basic parameters from table columns.
        
        Parameters
        ----------
        b1_element : str
            Main B-site element (e.g., Zr)
        b2_element : str or NaN
            Second B-site element (e.g., Ce)
        b2_cont : float
            Content of second B-element (e.g., 0.3)
        dopant : str
            Doping element (acceptor, e.g., Y)
        dop_cont : float
            Doping content (e.g., 0.2)
        
        Returns
        -------
        dict
            Dictionary with calculated composition parameters
        """
        # Safe conversion of inputs
        b2_cont_safe = 0.0
        if b2_cont is not None and not pd.isna(b2_cont):
            b2_cont_safe = safe_float_converter(b2_cont) or 0.0
        
        dop_cont_safe = 0.0
        if dop_cont is not None and not pd.isna(dop_cont):
            dop_cont_safe = safe_float_converter(dop_cont) or 0.0
        
        result = {
            'formula_type': 'simple',
            'b1_element': b1_element,
            'b2_element': b2_element,
            'b2_cont': b2_cont_safe,
            'dopant': dopant,
            'dop_cont': dop_cont_safe,
            'x_B2': b2_cont_safe,
            'y_dop': dop_cont_safe,
        }
        
        # Determine formula type
        if pd.isna(b2_element) or b2_element == '' or b2_cont_safe == 0:
            # Simple dopant: AB1_{1-y}D_yO_{3-y/2}
            result['formula_type'] = 'simple'
            result['b_main'] = b1_element
            result['x_B2'] = 0
        else:
            # Complex composition: AB1_{1-x-y}B2_xD_yO_{3-y/2}
            result['formula_type'] = 'complex'
            result['b_main'] = b1_element
        
        # Calculate average B-site ionic radius
        r_B1 = self.get_ionic_radius(b1_element, 4, 6)
        r_B2 = self.get_ionic_radius(b2_element, 4, 6) if not pd.isna(b2_element) and b2_element != '' else None
        r_D = self.get_ionic_radius(dopant, 3, 6) if not pd.isna(dopant) and dopant != '' else None
        
        x = result['x_B2']
        y = result['y_dop']
        
        if r_B1 is not None:
            result['r_B1'] = r_B1
        else:
            result['r_B1'] = None
        
        if r_B2 is not None:
            result['r_B2'] = r_B2
        else:
            result['r_B2'] = None
        
        if r_D is not None:
            result['r_D'] = r_D
        else:
            result['r_D'] = None
        
        # Average B-site radius
        if result['formula_type'] == 'simple' and r_B1 is not None and r_D is not None:
            result['r_avg_B'] = (1 - y) * r_B1 + y * r_D
        elif result['formula_type'] == 'complex' and r_B1 is not None and r_B2 is not None and r_D is not None:
            result['r_avg_B'] = (1 - x - y) * r_B1 + x * r_B2 + y * r_D
        else:
            result['r_avg_B'] = None
        
        # NEW: Radius mismatch (absolute difference from reference)
        if result['r_avg_B'] is not None:
            result['radius_mismatch'] = abs(result['r_avg_B'] - self.reference_B_radius)
        else:
            result['radius_mismatch'] = None
        
        # Tolerance factor
        if self.r_A is not None and result['r_avg_B'] is not None and self.r_O is not None:
            result['tolerance_factor'] = (self.r_A + self.r_O) / (np.sqrt(2) * (result['r_avg_B'] + self.r_O))
        else:
            result['tolerance_factor'] = None
        
        # NEW: Lattice distortion index (from tolerance factor deviation)
        if result['tolerance_factor'] is not None:
            # Distortion increases as tolerance factor deviates from 1.0
            result['lattice_distortion_index'] = abs(result['tolerance_factor'] - 1.0)
        else:
            result['lattice_distortion_index'] = None
        
        # Average B-site electronegativity
        χ_B1 = self.get_electronegativity(b1_element)
        χ_B2 = self.get_electronegativity(b2_element) if not pd.isna(b2_element) and b2_element != '' else None
        χ_D = self.get_electronegativity(dopant) if not pd.isna(dopant) and dopant != '' else None
        
        if χ_B1 is not None:
            result['χ_B1'] = χ_B1
        else:
            result['χ_B1'] = None
        
        if χ_B2 is not None:
            result['χ_B2'] = χ_B2
        else:
            result['χ_B2'] = None
        
        if χ_D is not None:
            result['χ_D'] = χ_D
        else:
            result['χ_D'] = None
        
        if result['formula_type'] == 'simple' and χ_B1 is not None and χ_D is not None:
            result['χ_avg_B'] = (1 - y) * χ_B1 + y * χ_D
        elif result['formula_type'] == 'complex' and χ_B1 is not None and χ_B2 is not None and χ_D is not None:
            result['χ_avg_B'] = (1 - x - y) * χ_B1 + x * χ_B2 + y * χ_D
        else:
            result['χ_avg_B'] = None
        
        # Difference in electronegativity (B-site average vs A-site)
        if result['χ_avg_B'] is not None and self.χ_A is not None:
            result['Δχ'] = abs(result['χ_avg_B'] - self.χ_A)
        else:
            result['Δχ'] = None
        
        # NEW: Electronegativity difference between B-site and oxygen
        if result['χ_avg_B'] is not None and self.χ_O is not None:
            result['electronegativity_difference_B_O'] = abs(result['χ_avg_B'] - self.χ_O)
        else:
            result['electronegativity_difference_B_O'] = None
        
        # Oxygen vacancy concentration
        result['oxygen_vacancy_conc'] = y / 2 if y is not None else None
        
        # Molar mass calculation
        M_A = self.get_atomic_mass(self.a_element)
        M_B1 = self.get_atomic_mass(b1_element)
        M_B2 = self.get_atomic_mass(b2_element) if not pd.isna(b2_element) and b2_element != '' else None
        M_D = self.get_atomic_mass(dopant) if not pd.isna(dopant) and dopant != '' else None
        M_O = ATOMIC_MASSES['O']
        
        if M_A is not None and M_B1 is not None and M_D is not None:
            if result['formula_type'] == 'simple':
                result['molar_mass'] = M_A + (1 - y) * M_B1 + y * M_D + (3 - y/2) * M_O
            elif result['formula_type'] == 'complex' and M_B2 is not None:
                result['molar_mass'] = M_A + (1 - x - y) * M_B1 + x * M_B2 + y * M_D + (3 - y/2) * M_O
            else:
                result['molar_mass'] = None
        else:
            result['molar_mass'] = None
        
        # Base structure for theoretical density (Vegard's law approximation)
        base_compound = f"{self.a_element}{b1_element}O3"
        base_props = MATERIAL_PROPERTIES.get(base_compound, None)
        if base_props is not None:
            result['theoretical_density'] = base_props.get('density', None)
        else:
            result['theoretical_density'] = None
        
        return result
    
    def calculate_microstructure_descriptors(self, density_percent, grain_size_um):
        """
        Calculate microstructural descriptors.
        
        Parameters
        ----------
        density_percent : float
            Relative density in percent
        grain_size_um : float
            Grain size in micrometers
        
        Returns
        -------
        dict
            Dictionary with microstructural descriptors
        """
        # Safe conversion
        density_percent_safe = safe_float_converter(density_percent)
        grain_size_um_safe = safe_float_converter(grain_size_um)
        
        descriptors = {}
        
        # Density
        if density_percent_safe is not None and not pd.isna(density_percent_safe):
            descriptors['density_percent'] = density_percent_safe
            descriptors['density_fraction'] = density_percent_safe / 100.0
            descriptors['porosity'] = 1.0 - descriptors['density_fraction']
        else:
            descriptors['density_percent'] = None
            descriptors['density_fraction'] = None
            descriptors['porosity'] = None
        
        # Grain size
        if grain_size_um_safe is not None and not pd.isna(grain_size_um_safe) and grain_size_um_safe > 0:
            descriptors['grain_size_um'] = grain_size_um_safe
            # S/V ratio calculation - grain boundary area per unit volume
            # S/V = 9/4 * (4/3)^(2/3) * 1/req * π^(-1/3) ≈ 1.861 / req
            # where req is the equivalent radius of spherical grain
            req = grain_size_um_safe / 2.0  # radius in μm
            descriptors['S_V_ratio'] = 1.861 / req  # μm⁻¹
            # In m⁻¹ for physical calculations
            descriptors['S_V_ratio_m'] = descriptors['S_V_ratio'] * 1e6
            descriptors['inverse_d'] = 1.0 / grain_size_um_safe
        else:
            descriptors['grain_size_um'] = None
            descriptors['S_V_ratio'] = None
            descriptors['S_V_ratio_m'] = None
            descriptors['inverse_d'] = None
        
        return descriptors
    
    def calculate_sintering_additive_descriptors(self, additive_type, additive_concentration_wt):
        """
        Calculate descriptors for sintering additive.
        
        Parameters
        ----------
        additive_type : str
            Additive type (Pure, Cu, Ni, Zn, Co)
        additive_concentration_wt : float
            Additive concentration in wt%
        
        Returns
        -------
        dict
            Dictionary with additive descriptors
        """
        # Safe conversion
        additive_type_safe = additive_type if not pd.isna(additive_type) else 'Pure'
        additive_conc_safe = safe_float_converter(additive_concentration_wt) or 0.0
        
        descriptors = {
            'additive_type': additive_type_safe,
            'additive_concentration_wt': additive_conc_safe,
            'is_pure': True if (additive_type_safe == 'Pure' or additive_conc_safe == 0) else False
        }
        
        if not descriptors['is_pure']:
            # Ionic radius of additive cation
            descriptors['additive_radius'] = self.get_ionic_radius(additive_type_safe, 2, 6)
            # Electronegativity
            descriptors['additive_electronegativity'] = self.get_electronegativity(additive_type_safe)
            # Charge
            descriptors['additive_charge'] = self.get_charge(additive_type_safe)
            # Atomic mass
            descriptors['additive_atomic_mass'] = self.get_atomic_mass(additive_type_safe)
            # NEW: Incorporation likelihood (from literature)
            descriptors['additive_incorporation_likely'] = ADDITIVE_INCORPORATION.get(additive_type_safe, False)
        else:
            descriptors['additive_radius'] = None
            descriptors['additive_electronegativity'] = None
            descriptors['additive_charge'] = None
            descriptors['additive_atomic_mass'] = None
            descriptors['additive_incorporation_likely'] = True  # Pure has no segregation issue
        
        return descriptors


# ============================================================================
# NEW CLASS: CONDUCTIVITY DATA PROCESSOR (ENHANCED)
# ============================================================================
class ConductivityDataProcessor:
    """
    Class for processing conductivity data of proton-conducting oxides.
    
    This enhanced version includes:
    - Flexible column mapping
    - Outlier detection
    - Progress tracking
    - Conductivity extrapolation
    - Activation energy calculation
    """
    
    def __init__(self):
        """Initialize the data processor with temperatures and column mapper."""
        self.temperatures = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
        self.calculator = ConductivityDescriptorCalculator(a_element='Ba')
        self.column_mapper = FlexibleColumnMapper()
    
    def extract_conductivity_data(self, row, col_type='sigma_total'):
        """
        Extract conductivity data from a row using flexible column mapping.
        
        Parameters
        ----------
        row : pandas.Series
            Row with data
        col_type : str
            Type of conductivity ('sigma_total', 'sigma_bulk', 'sigma_gb')
        
        Returns
        -------
        list of dict
            List with conductivity data at different temperatures
        """
        conductivity_data = []
        
        for T in self.temperatures:
            # Find column using flexible mapper
            col_name = self.column_mapper.find_column(row.to_frame().T, col_type, T)
            
            # Also check for direct column names like 'sigma_total_mS_600' or 'σ total, 600'
            if col_name is None:
                for col in row.index:
                    col_str = str(col).lower()
                    # Check for pattern like "σ total, 600" or "σ total 600"
                    if col_type.replace('sigma_', '') in col_str and str(T) in col_str:
                        col_name = col
                        break
            
            if col_name is not None:
                sigma_value = row[col_name]
                if not pd.isna(sigma_value) and sigma_value != '' and sigma_value is not None:
                    try:
                        sigma_val = safe_float_converter(sigma_value)
                        if sigma_val is not None:
                            conductivity_data.append({
                                'temperature_K': T + 273.15,
                                'temperature_C': T,
                                f'sigma_{col_type.replace("sigma_", "")}': sigma_val,
                                f'sigma_{col_type.replace("sigma_", "")}_mS': sigma_val,
                                f'sigma_{col_type.replace("sigma_", "")}_S_cm': sigma_val / 1000.0
                            })
                    except (ValueError, TypeError):
                        pass
        
        return conductivity_data
    
    def detect_outliers_iqr(self, data, column, multiplier=1.5):
        """
        Detect outliers using IQR method.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Data to analyze
        column : str
            Column name to check for outliers
        multiplier : float
            IQR multiplier (default: 1.5)
            
        Returns
        -------
        pandas.Series
            Boolean mask where True indicates outlier
        """
        if column not in data.columns:
            return pd.Series([False] * len(data), index=data.index)
        
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (data[column] < lower_bound) | (data[column] > upper_bound)
    
    def calculate_gb_contribution(self, sigma_total_data, sigma_bulk_data, sigma_gb_data):
        """
        Calculate grain boundary contribution to total resistance.
        
        Parameters
        ----------
        sigma_total_data : list of dict
            Total conductivity data
        sigma_bulk_data : list of dict
            Bulk conductivity data
        sigma_gb_data : list of dict
            Grain boundary conductivity data
        
        Returns
        -------
        dict
            Relative contribution of grain boundary conductivity
        """
        result = {}
        
        if sigma_total_data and sigma_bulk_data and sigma_gb_data:
            # Create dictionaries by temperature
            total_by_T = {}
            for d in sigma_total_data:
                T = d.get('temperature_C')
                sigma_key = [k for k in d.keys() if 'sigma' in k and 'mS' in k][0] if d else None
                if T is not None and sigma_key:
                    total_by_T[T] = d[sigma_key]
            
            bulk_by_T = {}
            for d in sigma_bulk_data:
                T = d.get('temperature_C')
                sigma_key = [k for k in d.keys() if 'sigma' in k and 'mS' in k][0] if d else None
                if T is not None and sigma_key:
                    bulk_by_T[T] = d[sigma_key]
            
            gb_by_T = {}
            for d in sigma_gb_data:
                T = d.get('temperature_C')
                sigma_key = [k for k in d.keys() if 'sigma' in k and 'mS' in k][0] if d else None
                if T is not None and sigma_key:
                    gb_by_T[T] = d[sigma_key]
            
            for T in total_by_T.keys():
                if T in bulk_by_T and T in gb_by_T:
                    sigma_total = total_by_T[T]
                    sigma_bulk = bulk_by_T[T]
                    sigma_gb = gb_by_T[T]
                    
                    if sigma_total is not None and sigma_total > 0:
                        # Resistance calculation
                        R_total = 1.0 / sigma_total
                        R_bulk = 1.0 / sigma_bulk if sigma_bulk is not None and sigma_bulk > 0 else None
                        R_gb = 1.0 / sigma_gb if sigma_gb is not None and sigma_gb > 0 else None
                        
                        if R_bulk is not None and R_gb is not None:
                            gb_fraction = R_gb / (R_bulk + R_gb) if (R_bulk + R_gb) > 0 else None
                            bulk_fraction = R_bulk / (R_bulk + R_gb) if (R_bulk + R_gb) > 0 else None
                            
                            result[T] = {
                                'gb_resistance_fraction': gb_fraction,
                                'bulk_resistance_fraction': bulk_fraction,
                                'R_total': R_total,
                                'R_bulk': R_bulk,
                                'R_gb': R_gb
                            }
        
        return result


# ============================================================================
# NEW FUNCTIONS FOR ENHANCED ANALYSIS
# ============================================================================

def partial_correlation_analysis(df, target, features, control_variables):
    """
    Calculate partial correlations between target and features, controlling for variables.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data
    target : str
        Target variable name
    features : list
        List of feature variable names
    control_variables : list
        List of control variable names
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with partial correlations and p-values
    """
    results = []
    
    for feature in features:
        if feature not in df.columns or target not in df.columns:
            continue
        
        # Drop NaN values
        all_vars = [target, feature] + control_variables
        clean_df = df[all_vars].dropna()
        
        if len(clean_df) < 5:
            results.append({
                'feature': feature,
                'partial_correlation': np.nan,
                'p_value': np.nan,
                'n_points': len(clean_df)
            })
            continue
        
        # Calculate residuals for target and feature after regressing out controls
        if len(control_variables) > 0:
            # Regress target on controls
            X_controls = clean_df[control_variables].values
            y_target = clean_df[target].values
            y_feature = clean_df[feature].values
            
            # Add constant term
            X_controls = np.column_stack([np.ones(len(X_controls)), X_controls])
            
            try:
                # Linear regression for target
                beta_target = np.linalg.lstsq(X_controls, y_target, rcond=None)[0]
                residual_target = y_target - X_controls @ beta_target
                
                # Linear regression for feature
                beta_feature = np.linalg.lstsq(X_controls, y_feature, rcond=None)[0]
                residual_feature = y_feature - X_controls @ beta_feature
                
                # Correlation of residuals
                corr, p_val = pearsonr(residual_target, residual_feature)
            except:
                corr, p_val = np.nan, np.nan
        else:
            # Simple correlation if no control variables
            corr, p_val = pearsonr(clean_df[target], clean_df[feature])
        
        results.append({
            'feature': feature,
            'partial_correlation': corr,
            'p_value': p_val,
            'n_points': len(clean_df)
        })
    
    return pd.DataFrame(results)


def polynomial_regression_analysis(df, x_col, y_col, degree=2):
    """
    Perform polynomial regression analysis.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data
    x_col : str
        Independent variable column
    y_col : str
        Dependent variable column
    degree : int
        Polynomial degree (default: 2)
    
    Returns
    -------
    dict
        Regression results including model, predictions, and R²
    """
    clean_df = df[[x_col, y_col]].dropna()
    
    if len(clean_df) < degree + 2:
        return {
            'model': None,
            'x_pred': None,
            'y_pred': None,
            'r2': None,
            'coefficients': None
        }
    
    X = clean_df[x_col].values.reshape(-1, 1)
    y = clean_df[y_col].values
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predictions
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    
    # Generate smooth curve for plotting
    x_range = np.linspace(X.min(), X.max(), 100)
    x_range_poly = poly.transform(x_range.reshape(-1, 1))
    y_range_pred = model.predict(x_range_poly)
    
    return {
        'model': model,
        'x_pred': x_range,
        'y_pred': y_range_pred,
        'r2': r2,
        'coefficients': model.coef_,
        'intercept': model.intercept_
    }


def cluster_materials_by_properties(df, feature_columns, eps=0.5, min_samples=3):
    """
    Cluster materials using DBSCAN based on their properties.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data (long format, will be pivoted)
    feature_columns : list
        List of feature columns to use for clustering
    eps : float
        DBSCAN epsilon parameter
    min_samples : int
        DBSCAN minimum samples parameter
    
    Returns
    -------
    pandas.DataFrame
        Data with cluster labels
    """
    # Aggregate by sample_id
    agg_df = df.groupby('sample_id')[feature_columns].mean().dropna()
    
    if len(agg_df) < 3:
        return None, None
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg_df)
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clustering.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    agg_df['cluster'] = cluster_labels
    
    return agg_df, scaler


def get_cluster_profiles(df_long, feature_columns, eps=0.5, min_samples=3):
    """
    Get detailed profiles for each cluster.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    feature_columns : list
        Features to use for clustering
    eps : float
        DBSCAN epsilon
    min_samples : int
        DBSCAN min_samples
    
    Returns
    -------
    pandas.DataFrame
        Cluster profiles with mean, std, and count for each feature
    """
    # Aggregate by sample
    agg_df = df_long.groupby('sample_id')[feature_columns].mean().dropna()
    
    if len(agg_df) < 3:
        return None
    
    # Standardize and cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg_df)
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clustering.fit_predict(X_scaled)
    agg_df['cluster'] = cluster_labels
    
    # Get additive and dopant info for each sample
    additive_info = df_long.groupby('sample_id')['additive_type'].first()
    dopant_info = df_long.groupby('sample_id')['dopant'].first()
    agg_df['additive_type'] = additive_info
    agg_df['dopant'] = dopant_info
    
    # Calculate cluster profiles
    profiles = []
    for cluster in sorted(agg_df['cluster'].unique()):
        cluster_data = agg_df[agg_df['cluster'] == cluster]
        profile = {'cluster': cluster, 'n_samples': len(cluster_data)}
        
        # Add feature statistics
        for col in feature_columns:
            if col in cluster_data.columns:
                profile[f'{col}_mean'] = cluster_data[col].mean()
                profile[f'{col}_std'] = cluster_data[col].std()
        
        # Add mode for categorical
        profile['additive_types'] = cluster_data['additive_type'].value_counts().to_dict()
        profile['dopant_types'] = cluster_data['dopant'].value_counts().to_dict()
        
        # Add conductivity summary (if available)
        if 'sigma_total_mS' in df_long.columns:
            sample_conductivities = []
            for sample_id in cluster_data.index:
                sample_cond = df_long[df_long['sample_id'] == sample_id]['sigma_total_mS'].mean()
                if not pd.isna(sample_cond):
                    sample_conductivities.append(sample_cond)
            if sample_conductivities:
                profile['mean_sigma_total_mS'] = np.mean(sample_conductivities)
                profile['std_sigma_total_mS'] = np.std(sample_conductivities)
        
        profiles.append(profile)
    
    return pd.DataFrame(profiles)


def shap_analysis(df, features, target, model_type='xgboost'):
    """
    Perform SHAP analysis for model interpretability.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data
    features : list
        Feature columns
    target : str
        Target column
    model_type : str
        Model type ('xgboost' or 'random_forest')
    
    Returns
    -------
    dict
        SHAP values, model, and explainer
    """
    # Prepare data
    clean_df = df[features + [target]].dropna()
    
    if len(clean_df) < 10:
        return None
    
    X = clean_df[features].values
    y = clean_df[target].values
    feature_names = features
    
    # Train model
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X, y)
    
    # Create SHAP explainer
    if model_type == 'xgboost':
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X)
    
    return {
        'model': model,
        'explainer': explainer,
        'shap_values': shap_values,
        'X': X,
        'feature_names': feature_names,
        'y': y
    }


# ============================================================================
# NEW FUNCTIONS FOR CONTOUR MAPS
# ============================================================================

def plot_contour_map(df_long, x_col, y_col, z_col, temperature, 
                     filter_dict=None, n_grid=50, cmap='viridis',
                     show_points=True, show_labels=True, n_levels=20):
    """
    Universal contour map with interpolation on irregular grid.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    x_col : str
        X-axis column name (wt%, Ce_ratio, t, Δχ, rBav/rO, T_sin, d, S/V, ρ, D_conc)
    y_col : str
        Y-axis column name (Ce_ratio, wt%, t, Δχ, Ea, T_sin, 1/d, D_conc)
    z_col : str
        Z-axis (color) column name (σ_total, σ_bulk, σ_gb, σ_gb/σ_total, Ea_total, Ea_gb, log(σ))
    temperature : int
        Temperature in Celsius (used if z_col contains conductivity)
    filter_dict : dict
        Dictionary of filters {'additive_type': [...], 'dopant': [...], 'atmosphere': [...], etc.}
    n_grid : int
        Grid resolution for interpolation
    cmap : str
        Colormap name
    show_points : bool
        Show original data points on contour
    show_labels : bool
        Show contour labels
    n_levels : int
        Number of contour levels
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with contour plot
    """
    # Apply filters
    plot_df = df_long.copy()
    
    # Filter by temperature if z_col contains conductivity
    if 'sigma' in z_col or 'σ' in z_col:
        plot_df = plot_df[plot_df['temperature_C'] == temperature]
    
    if filter_dict:
        for col, values in filter_dict.items():
            if col in plot_df.columns and values:
                plot_df = plot_df[plot_df[col].isin(values)]
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    # Prepare data for contour
    plot_df = plot_df.dropna(subset=[x_col, y_col, z_col])
    
    if len(plot_df) < 4:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Insufficient data for contour map\n(need at least 4 points, have {len(plot_df)})',
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Extract coordinates
    x = plot_df[x_col].values
    y = plot_df[y_col].values
    z = plot_df[z_col].values
    
    # Create grid for interpolation
    xi = np.linspace(x.min(), x.max(), n_grid)
    yi = np.linspace(y.min(), y.max(), n_grid)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolation using griddata
    try:
        Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    except:
        try:
            Zi = griddata((x, y), z, (Xi, Yi), method='linear')
        except:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'Interpolation failed (insufficient or poorly distributed data)',
                    ha='center', va='center', transform=ax.transAxes)
            return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot contour
    contour = ax.contourf(Xi, Yi, Zi, levels=n_levels, cmap=cmap, alpha=0.8)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(z_col.replace('_', ' ').title())
    
    # Add contour lines
    if n_levels > 5:
        contour_lines = ax.contour(Xi, Yi, Zi, levels=n_levels//2, colors='black', linewidths=0.5, alpha=0.3)
    
    # Add contour labels
    if show_labels and n_levels > 5:
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Add original data points
    if show_points:
        scatter = ax.scatter(x, y, c=z, cmap=cmap, s=st.session_state.plot_settings.get('marker_size', 80),
                            edgecolors='black', linewidth=0.5, alpha=st.session_state.plot_settings.get('bubble_alpha', 0.7), zorder=5)
    
    # Labels and title
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    
    title = f'Contour Map: {z_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()} and {y_col.replace("_", " ").title()}'
    if 'sigma' in z_col:
        title += f' at {temperature}°C'
    ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    
    return fig


# ============================================================================
# NEW FUNCTIONS FOR BUBBLE DIAGRAMS (PARAMETERIZED)
# ============================================================================

def get_available_bubble_params(df_long):
    """
    Get list of available parameters for bubble diagrams.
    
    Returns
    -------
    dict
        Dictionary with numeric and categorical parameters
    """
    numeric_params = [
        'additive_concentration_wt', 'grain_size_um', 'S_V_ratio', 'density_percent',
        'inverse_d', 'T_sin', 'Ce_ratio', 'D_conc', 'Δχ', 'tolerance_factor',
        'oxygen_vacancy_conc', 'radius_mismatch', 'lattice_distortion_index',
        'sigma_total_mS', 'sigma_bulk_mS', 'sigma_gb_mS', 'sigma_gb_ratio',
        'Ea_total_eV', 'Ea_bulk_eV', 'Ea_gb_eV', 'porosity'
    ]
    
    categorical_params = [
        'additive_type', 'dopant', 'atmosphere', 'atmosphere_type', 
        'method', 'structure', 'additive_incorporation_likely'
    ]
    
    return {
        'numeric': [p for p in numeric_params],
        'categorical': [p for p in categorical_params]
    }


def plot_bubble_diagram(df_long, x_col, y_col, size_col, color_col, temperature,
                        filter_dict=None, show_trend=True, cmap='viridis'):
    """
    Universal bubble diagram for multi-parameter analysis.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    x_col : str
        X-axis column (wt%, Ce_ratio, t, T_sin, d, S/V, ρ, 1/d, Δχ, D_conc)
    y_col : str
        Y-axis column (σ_total, σ_bulk, σ_gb, σ_gb/σ_total, Ea, log(σ))
    size_col : str
        Column for bubble size (d, S/V, ρ, wt%, 1/d, σ_bulk/σ_gb, D_conc)
    color_col : str
        Column for bubble color (ρ, T_sin, Ea, Ce_ratio, Δχ, D_type, additive_type)
    temperature : int
        Temperature in Celsius (used if y_col contains conductivity)
    filter_dict : dict
        Dictionary of filters
    show_trend : bool
        Show trend lines
    cmap : str
        Colormap for color scale
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with bubble diagram
    """
    # Apply filters
    plot_df = df_long.copy()
    
    # Filter by temperature if y_col contains conductivity
    if 'sigma' in y_col or 'σ' in y_col:
        plot_df = plot_df[plot_df['temperature_C'] == temperature]
    
    if filter_dict:
        for col, values in filter_dict.items():
            if col in plot_df.columns and values:
                plot_df = plot_df[plot_df[col].isin(values)]
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    # Prepare data
    plot_df = plot_df.dropna(subset=[x_col, y_col])
    
    if len(plot_df) < 3:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f'Insufficient data for bubble diagram (need at least 3 points, have {len(plot_df)})',
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare bubble sizes
    if size_col in plot_df.columns and plot_df[size_col].notna().any():
        size_min = plot_df[size_col].min()
        size_max = plot_df[size_col].max()
        if size_max > size_min:
            sizes = 50 + (plot_df[size_col] - size_min) / (size_max - size_min) * 400
        else:
            sizes = 100
    else:
        sizes = 100
    
    # Prepare colors
    settings = st.session_state.plot_settings
    marker_size_setting = settings.get('marker_size', 80)
    bubble_alpha = settings.get('bubble_alpha', 0.7)
    
    if color_col in plot_df.columns and plot_df[color_col].notna().any():
        # Numeric color
        if pd.api.types.is_numeric_dtype(plot_df[color_col]):
            scatter = ax.scatter(plot_df[x_col], plot_df[y_col],
                                s=sizes, c=plot_df[color_col],
                                cmap=cmap, alpha=bubble_alpha,
                                edgecolors='black', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_col.replace('_', ' ').title())
        else:
            # Categorical color
            for cat_val in plot_df[color_col].unique():
                subset = plot_df[plot_df[color_col] == cat_val]
                if color_col == 'additive_type':
                    color = SINTERING_ADDITIVE_COLORS.get(cat_val, SINTERING_ADDITIVE_COLORS['default'])
                elif color_col == 'dopant':
                    color = DOPANT_COLORS.get(cat_val, DOPANT_COLORS['default'])
                else:
                    color = '#3B82F6'
                ax.scatter(subset[x_col], subset[y_col],
                          s=sizes[subset.index] if isinstance(sizes, pd.Series) else sizes,
                          c=[color], label=cat_val, alpha=bubble_alpha,
                          edgecolors='black', linewidth=0.5)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Default: group by additive type
        for additive in plot_df['additive_type'].unique():
            subset = plot_df[plot_df['additive_type'] == additive]
            color = SINTERING_ADDITIVE_COLORS.get(additive, SINTERING_ADDITIVE_COLORS['default'])
            ax.scatter(subset[x_col], subset[y_col],
                      s=sizes[subset.index] if isinstance(sizes, pd.Series) else sizes,
                      c=[color], label=additive, alpha=bubble_alpha,
                      edgecolors='black', linewidth=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add trend lines
    if show_trend:
        if color_col in plot_df.columns and not pd.api.types.is_numeric_dtype(plot_df[color_col]):
            # Trend per category
            for cat_val in plot_df[color_col].unique():
                subset = plot_df[plot_df[color_col] == cat_val]
                if len(subset) >= 3:
                    try:
                        z = np.polyfit(subset[x_col], subset[y_col], 1)
                        x_trend = np.linspace(subset[x_col].min(), subset[x_col].max(), 50)
                        ax.plot(x_trend, np.polyval(z, x_trend), '--',
                               linewidth=1.5, alpha=0.5)
                    except:
                        pass
        else:
            # Single trend line for all data
            if len(plot_df) >= 3:
                try:
                    z = np.polyfit(plot_df[x_col], plot_df[y_col], 1)
                    x_trend = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 50)
                    ax.plot(x_trend, np.polyval(z, x_trend), 'k--',
                           linewidth=1.5, alpha=0.5, label='Trend line')
                except:
                    pass
    
    # Add legend for bubble size
    from matplotlib.lines import Line2D
    if size_col in plot_df.columns and plot_df[size_col].notna().any():
        size_quantiles = plot_df[size_col].quantile([0.25, 0.5, 0.75])
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=5, label=f'{size_col}: small (<{size_quantiles[0.25]:.2f})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=10, label=f'{size_col}: medium'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=15, label=f'{size_col}: large (>{size_quantiles[0.75]:.2f})')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # Labels and title
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    
    title = f'Bubble Diagram: {y_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()}'
    if 'sigma' in y_col:
        title += f' at {temperature}°C'
    title += f'\nSize = {size_col.replace("_", " ").title()}, Color = {color_col.replace("_", " ").title()}'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_multi_panel_bubble_analysis(df_long, temperature=600, filter_dict=None):
    """
    Multi-panel bubble analysis for comprehensive understanding.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    temperature : int
        Temperature in Celsius
    filter_dict : dict
        Dictionary of filters
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with multiple subplots
    """
    # Apply filters
    plot_df = df_long.copy()
    plot_df = plot_df[plot_df['temperature_C'] == temperature]
    
    if filter_dict:
        for col, values in filter_dict.items():
            if col in plot_df.columns and values:
                plot_df = plot_df[plot_df[col].isin(values)]
    
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    settings = st.session_state.plot_settings
    bubble_alpha = settings.get('bubble_alpha', 0.7)
    marker_size = settings.get('marker_size', 80)
    
    # Panel 1: Conductivity vs Additive Concentration
    ax1 = axes[0, 0]
    if 'additive_concentration_wt' in plot_df.columns and len(plot_df[plot_df['additive_concentration_wt'] > 0]) > 0:
        panel_df = plot_df[plot_df['additive_concentration_wt'] > 0]
        if 'grain_size_um' in panel_df.columns and panel_df['grain_size_um'].notna().any():
            size_min = panel_df['grain_size_um'].min()
            size_max = panel_df['grain_size_um'].max()
            if size_max > size_min:
                sizes = 50 + (panel_df['grain_size_um'] - size_min) / (size_max - size_min) * 300
            else:
                sizes = 100
        else:
            sizes = 100
        
        if 'density_percent' in panel_df.columns and panel_df['density_percent'].notna().any():
            scatter = ax1.scatter(panel_df['additive_concentration_wt'], panel_df['sigma_total_mS'],
                                 s=sizes, c=panel_df['density_percent'],
                                 cmap='viridis', alpha=bubble_alpha, edgecolors='black')
            plt.colorbar(scatter, ax=ax1, label='Density (%)')
        else:
            for additive in panel_df['additive_type'].unique():
                subset = panel_df[panel_df['additive_type'] == additive]
                color = SINTERING_ADDITIVE_COLORS.get(additive, 'gray')
                ax1.scatter(subset['additive_concentration_wt'], subset['sigma_total_mS'],
                           s=100, c=[color], label=additive, alpha=bubble_alpha, edgecolors='black')
            ax1.legend()
        
        ax1.set_xlabel('Additive Concentration (wt%)')
        ax1.set_ylabel(f'σ at {temperature}°C (mS/cm)')
        ax1.set_title('Conductivity vs Additive Concentration')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Conductivity vs Additive Concentration')
    
    # Panel 2: Conductivity vs Tolerance Factor
    ax2 = axes[0, 1]
    if 'tolerance_factor' in plot_df.columns and len(plot_df.dropna(subset=['tolerance_factor'])) > 0:
        panel_df = plot_df.dropna(subset=['tolerance_factor'])
        if 'density_percent' in panel_df.columns and panel_df['density_percent'].notna().any():
            size_min = panel_df['density_percent'].min()
            size_max = panel_df['density_percent'].max()
            if size_max > size_min:
                sizes = 50 + (panel_df['density_percent'] - size_min) / (size_max - size_min) * 300
            else:
                sizes = 100
        else:
            sizes = 100
        
        for additive in panel_df['additive_type'].unique():
            subset = panel_df[panel_df['additive_type'] == additive]
            color = SINTERING_ADDITIVE_COLORS.get(additive, 'gray')
            ax2.scatter(subset['tolerance_factor'], subset['sigma_total_mS'],
                       s=sizes[subset.index] if isinstance(sizes, pd.Series) else sizes,
                       c=[color], label=additive, alpha=bubble_alpha, edgecolors='black')
        
        ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
        ax2.axvspan(0.96, 1.04, alpha=0.15, color='green')
        ax2.set_xlabel('Tolerance Factor (t)')
        ax2.set_ylabel(f'σ at {temperature}°C (mS/cm)')
        ax2.set_title('Conductivity vs Structural Stability')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Conductivity vs Tolerance Factor')
    
    # Panel 3: Conductivity vs Grain Size
    ax3 = axes[1, 0]
    if 'grain_size_um' in plot_df.columns and len(plot_df.dropna(subset=['grain_size_um'])) > 0:
        panel_df = plot_df.dropna(subset=['grain_size_um'])
        if 'density_percent' in panel_df.columns and panel_df['density_percent'].notna().any():
            size_min = panel_df['density_percent'].min()
            size_max = panel_df['density_percent'].max()
            if size_max > size_min:
                sizes = 50 + (panel_df['density_percent'] - size_min) / (size_max - size_min) * 300
            else:
                sizes = 100
        else:
            sizes = 100
        
        for additive in panel_df['additive_type'].unique():
            subset = panel_df[panel_df['additive_type'] == additive]
            if len(subset) > 0:
                color = SINTERING_ADDITIVE_COLORS.get(additive, 'gray')
                ax3.scatter(subset['grain_size_um'], subset['sigma_total_mS'],
                           s=sizes[subset.index] if isinstance(sizes, pd.Series) else sizes,
                           c=[color], label=additive, alpha=bubble_alpha, edgecolors='black')
        
        ax3.set_xlabel('Grain Size (μm)')
        ax3.set_ylabel(f'σ at {temperature}°C (mS/cm)')
        ax3.set_title('Microstructural Effect (Size = Density)')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Conductivity vs Grain Size')
    
    # Panel 4: Conductivity vs Density
    ax4 = axes[1, 1]
    if 'density_percent' in plot_df.columns and len(plot_df.dropna(subset=['density_percent'])) > 0:
        panel_df = plot_df.dropna(subset=['density_percent'])
        if 'grain_size_um' in panel_df.columns and panel_df['grain_size_um'].notna().any():
            size_min = panel_df['grain_size_um'].min()
            size_max = panel_df['grain_size_um'].max()
            if size_max > size_min:
                sizes = 50 + (panel_df['grain_size_um'] - size_min) / (size_max - size_min) * 300
            else:
                sizes = 100
        else:
            sizes = 100
        
        for additive in panel_df['additive_type'].unique():
            subset = panel_df[panel_df['additive_type'] == additive]
            if len(subset) > 0:
                color = SINTERING_ADDITIVE_COLORS.get(additive, 'gray')
                ax4.scatter(subset['density_percent'], subset['sigma_total_mS'],
                           s=sizes[subset.index] if isinstance(sizes, pd.Series) else sizes,
                           c=[color], label=additive, alpha=bubble_alpha, edgecolors='black')
        
        ax4.set_xlabel('Relative Density (%)')
        ax4.set_ylabel(f'σ at {temperature}°C (mS/cm)')
        ax4.set_title('Densification Effect (Size = Grain Size)')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Conductivity vs Density')
    
    plt.suptitle(f'Multi-Panel Bubble Analysis at {temperature}°C\nComprehensive View of Sintering Additive Effects',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ============================================================================
# NEW FUNCTIONS FOR POISONING ANALYSIS (DENSIFICATION VS POISONING)
# ============================================================================

def plot_poisoning_analysis(df_long, temperature=600):
    """
    Comprehensive poisoning analysis showing densification vs grain boundary poisoning.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    temperature : int
        Temperature in Celsius
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with multiple subplots
    """
    # Prepare data
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Panel 1: Fixed T_sin comparison (Additive helps)
    ax1 = axes[0, 0]
    # Find a common T_sin with both Pure and Additive samples
    t_sin_values = plot_df[plot_df['T_sin'].notna()]['T_sin'].unique()
    common_t_sin = []
    for ts in t_sin_values:
        has_pure = len(plot_df[(plot_df['T_sin'] == ts) & (plot_df['additive_type'] == 'Pure')]) > 0
        has_add = len(plot_df[(plot_df['T_sin'] == ts) & (plot_df['additive_type'] != 'Pure')]) > 0
        if has_pure and has_add:
            common_t_sin.append(ts)
    
    if common_t_sin:
        ts_fixed = common_t_sin[0]
        fixed_T_df = plot_df[plot_df['T_sin'] == ts_fixed]
        
        pure_data = fixed_T_df[fixed_T_df['additive_type'] == 'Pure']
        additive_data = fixed_T_df[fixed_T_df['additive_type'] != 'Pure']
        
        if len(pure_data) > 0 and len(additive_data) > 0:
            pure_mean = pure_data['sigma_total_mS'].mean()
            pure_std = pure_data['sigma_total_mS'].std()
            
            additives = additive_data['additive_type'].unique()
            additive_means = []
            additive_stds = []
            additive_names = []
            
            for add in additives:
                add_data = additive_data[additive_data['additive_type'] == add]
                additive_means.append(add_data['sigma_total_mS'].mean())
                additive_stds.append(add_data['sigma_total_mS'].std())
                additive_names.append(add)
            
            x_pos = np.arange(len(additives) + 1)
            means = [pure_mean] + additive_means
            stds = [pure_std] + additive_stds
            labels = ['Pure'] + additive_names
            colors = ['#10B981'] + [SINTERING_ADDITIVE_COLORS.get(a, '#6B7280') for a in additive_names]
            
            bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, color=colors, edgecolor='black')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
            ax1.set_title(f'Fixed Sintering Temperature ({ts_fixed:.0f}°C): Additives Improve Conductivity')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, means)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.01,
                        f'{val:.3f}', ha='center', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No common sintering temperature with both Pure and Additive samples', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Fixed T_sin Comparison')
    
    # Panel 2: Matched density comparison (Poisoning effect)
    ax2 = axes[0, 1]
    # Find samples with similar density but different T_sin
    density_bins = np.linspace(85, 100, 4)
    poisoning_data = []
    
    for i in range(len(density_bins)-1):
        low_d = density_bins[i]
        high_d = density_bins[i+1]
        
        pure_in_bin = plot_df[(plot_df['additive_type'] == 'Pure') & 
                              (plot_df['density_percent'] >= low_d) & 
                              (plot_df['density_percent'] <= high_d)]
        add_in_bin = plot_df[(plot_df['additive_type'] != 'Pure') & 
                             (plot_df['density_percent'] >= low_d) & 
                             (plot_df['density_percent'] <= high_d)]
        
        if len(pure_in_bin) > 0 and len(add_in_bin) > 0:
            pure_cond = pure_in_bin['sigma_total_mS'].mean()
            add_cond = add_in_bin['sigma_total_mS'].mean()
            ratio = add_cond / pure_cond if pure_cond > 0 else np.nan
            poisoning_data.append({
                'density_range': f'{low_d:.0f}-{high_d:.0f}%',
                'ratio': ratio,
                'n_pure': len(pure_in_bin),
                'n_add': len(add_in_bin)
            })
    
    if poisoning_data:
        poisoning_df = pd.DataFrame(poisoning_data)
        x_pos = np.arange(len(poisoning_df))
        colors = ['#EF4444' if r < 0.9 else '#10B981' if r > 1.1 else '#F59E0B' for r in poisoning_df['ratio']]
        bars = ax2.bar(x_pos, poisoning_df['ratio'], color=colors, edgecolor='black')
        ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(poisoning_df['density_range'], rotation=45, ha='right')
        ax2.set_ylabel('σ_additive / σ_pure')
        ax2.set_title('Matched Density Comparison (Poisoning Effect)')
        ax2.set_ylim(0, max(1.5, poisoning_df['ratio'].max() + 0.2))
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add annotations
        for i, row in poisoning_df.iterrows():
            if row['ratio'] < 0.9:
                ax2.text(i, row['ratio'] - 0.08, 'Poisoning!', ha='center', fontsize=8, color='red')
    else:
        ax2.text(0.5, 0.5, 'No matched density data for comparison', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Matched Density Comparison')
    
    # Panel 3: GB ratio vs Densification boost
    ax3 = axes[1, 0]
    # Calculate for each additive type
    additives = plot_df[plot_df['additive_type'] != 'Pure']['additive_type'].unique()
    
    for additive in additives:
        add_data = plot_df[plot_df['additive_type'] == additive]
        pure_data = plot_df[plot_df['additive_type'] == 'Pure']
        
        if len(add_data) > 0 and len(pure_data) > 0:
            # Find matching compositions
            add_avg_cond = add_data['sigma_total_mS'].mean()
            pure_avg_cond = pure_data['sigma_total_mS'].mean()
            ratio = add_avg_cond / pure_avg_cond if pure_avg_cond > 0 else np.nan
            
            add_avg_dens = add_data['density_percent'].mean()
            pure_avg_dens = pure_data['density_percent'].mean()
            dens_boost = add_avg_dens - pure_avg_dens if not pd.isna(pure_avg_dens) else np.nan
            
            if not np.isnan(ratio) and not np.isnan(dens_boost):
                color = SINTERING_ADDITIVE_COLORS.get(additive, '#6B7280')
                ax3.scatter(dens_boost, ratio, s=200, c=[color], marker='o', 
                           edgecolors='black', linewidth=1.5, label=additive)
                ax3.annotate(additive, (dens_boost, ratio), fontsize=9, ha='center', va='bottom')
    
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Densification Boost (Δρ, %)')
    ax3.set_ylabel('σ_additive / σ_pure')
    ax3.set_title('Trade-off: Densification vs Conductivity')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: GB resistance fraction comparison
    ax4 = axes[1, 1]
    if 'gb_resistance_fraction' in plot_df.columns:
        gb_data = plot_df.dropna(subset=['gb_resistance_fraction', 'additive_type'])
        gb_pure = gb_data[gb_data['additive_type'] == 'Pure']['gb_resistance_fraction'].mean()
        
        additives_gb = []
        add_names_gb = []
        for add in additives:
            add_gb = gb_data[gb_data['additive_type'] == add]['gb_resistance_fraction'].mean()
            if not pd.isna(add_gb):
                additives_gb.append(add_gb)
                add_names_gb.append(add)
        
        if gb_pure and additives_gb:
            x_pos = np.arange(len(additives_gb) + 1)
            means = [gb_pure] + additives_gb
            labels = ['Pure'] + add_names_gb
            colors = ['#10B981'] + [SINTERING_ADDITIVE_COLORS.get(a, '#6B7280') for a in add_names_gb]
            
            bars = ax4.bar(x_pos, means, color=colors, edgecolor='black')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(labels, rotation=45, ha='right')
            ax4.set_ylabel('Grain Boundary Resistance Fraction')
            ax4.set_title('GB Contribution to Total Resistance')
            ax4.grid(True, alpha=0.3, axis='y')
            
            for bar, val in zip(bars, means):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No grain boundary resistance fraction data available',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('GB Contribution Analysis')
    
    plt.suptitle('Poisoning Analysis: Densification vs Grain Boundary Blocking', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_pure_vs_additive_matched_density(df_long, temperature=600):
    """
    Plot comparison of Pure (sintered at high T) vs Additive (sintered at low T) at matched density.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    temperature : int
        Temperature in Celsius
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with comparison plot
    """
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by composition (B1_cation, dopant, Ce_ratio range)
    compositions = plot_df.groupby(['B1_cation', 'dopant'])['sample_id'].first().reset_index()
    
    matched_data = []
    
    for _, comp in compositions.iterrows():
        b_cation = comp['B1_cation']
        dop = comp['dopant']
        
        # Find Pure samples with this composition
        pure_samples = plot_df[(plot_df['additive_type'] == 'Pure') & 
                               (plot_df['B1_cation'] == b_cation) & 
                               (plot_df['dopant'] == dop)]
        
        # Find Additive samples with this composition
        add_samples = plot_df[(plot_df['additive_type'] != 'Pure') & 
                              (plot_df['B1_cation'] == b_cation) & 
                              (plot_df['dopant'] == dop)]
        
        if len(pure_samples) > 0 and len(add_samples) > 0:
            # Try to match by density
            for _, pure in pure_samples.iterrows():
                pure_dens = pure['density_percent']
                if pd.isna(pure_dens):
                    continue
                
                # Find additive with closest density
                best_match = None
                best_diff = float('inf')
                for _, add in add_samples.iterrows():
                    add_dens = add['density_percent']
                    if pd.isna(add_dens):
                        continue
                    diff = abs(add_dens - pure_dens)
                    if diff < best_diff and diff < 5:  # Within 5% density
                        best_diff = diff
                        best_match = add
                
                if best_match is not None:
                    matched_data.append({
                        'composition': f"{b_cation}-{dop}",
                        'pure_cond': pure['sigma_total_mS'],
                        'add_cond': best_match['sigma_total_mS'],
                        'pure_T_sin': pure['T_sin'],
                        'add_T_sin': best_match['T_sin'],
                        'pure_dens': pure_dens,
                        'add_dens': best_match['density_percent'],
                        'additive_type': best_match['additive_type'],
                        'density_match_diff': best_diff
                    })
    
    if matched_data:
        matched_df = pd.DataFrame(matched_data)
        
        x_pos = np.arange(len(matched_df))
        width = 0.35
        
        pure_means = matched_df['pure_cond'].values
        add_means = matched_df['add_cond'].values
        
        bars1 = ax.bar(x_pos - width/2, pure_means, width, label='Pure (high T_sin)', color='#3B82F6', edgecolor='black')
        bars2 = ax.bar(x_pos + width/2, add_means, width, label='Additive (low T_sin)', color='#EF4444', edgecolor='black')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(matched_df['composition'], rotation=45, ha='right')
        ax.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
        ax.set_title('Matched Density Comparison: Pure (high T) vs Additive (low T)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add annotations for poisoning
        for i, row in matched_df.iterrows():
            if row['add_cond'] < row['pure_cond']:
                ax.annotate('Poisoning!', (i + width/2, row['add_cond']), 
                           ha='center', va='bottom', fontsize=8, color='red', rotation=90)
        
        # Add T_sin difference annotation
        for i, row in matched_df.iterrows():
            delta_T = row['pure_T_sin'] - row['add_T_sin'] if not pd.isna(row['pure_T_sin']) and not pd.isna(row['add_T_sin']) else None
            if delta_T and delta_T > 0:
                ax.annotate(f'ΔT={delta_T:.0f}°C', (i, max(row['pure_cond'], row['add_cond']) + 0.02),
                           ha='center', fontsize=7, alpha=0.7)
    else:
        ax.text(0.5, 0.5, 'No matched density pairs found (same composition, similar density)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Matched Density Comparison')
    
    plt.tight_layout()
    return fig


def plot_gb_ratio_vs_densification_boost(df_long, temperature=600):
    """
    Plot GB conductivity ratio vs densification boost.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    temperature : int
        Temperature in Celsius
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with scatter plot
    """
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate for each additive type
    additives = plot_df[plot_df['additive_type'] != 'Pure']['additive_type'].unique()
    
    for additive in additives:
        add_data = plot_df[plot_df['additive_type'] == additive]
        pure_data = plot_df[plot_df['additive_type'] == 'Pure']
        
        if len(add_data) > 0 and len(pure_data) > 0:
            # Calculate GB conductivity ratio
            if 'sigma_gb_mS' in add_data.columns and 'sigma_gb_mS' in pure_data.columns:
                add_gb_mean = add_data['sigma_gb_mS'].mean()
                pure_gb_mean = pure_data['sigma_gb_mS'].mean()
                gb_ratio = add_gb_mean / pure_gb_mean if pure_gb_mean > 0 else np.nan
            else:
                # Use total conductivity as proxy
                add_total_mean = add_data['sigma_total_mS'].mean()
                pure_total_mean = pure_data['sigma_total_mS'].mean()
                gb_ratio = add_total_mean / pure_total_mean if pure_total_mean > 0 else np.nan
            
            # Calculate densification boost
            add_dens_mean = add_data['density_percent'].mean()
            pure_dens_mean = pure_data['density_percent'].mean()
            dens_boost = add_dens_mean - pure_dens_mean if not pd.isna(pure_dens_mean) else np.nan
            
            # Get concentration info
            add_conc = add_data['additive_concentration_wt'].mean()
            
            if not np.isnan(gb_ratio) and not np.isnan(dens_boost):
                color = SINTERING_ADDITIVE_COLORS.get(additive, '#6B7280')
                size = 100 + add_conc * 50 if not pd.isna(add_conc) else 100
                
                ax.scatter(dens_boost, gb_ratio, s=size, c=[color], marker='o',
                          edgecolors='black', linewidth=1.5, alpha=0.8, label=additive)
                ax.annotate(f'{additive}\n({add_conc:.1f} wt%)', (dens_boost, gb_ratio), 
                           fontsize=9, ha='center', va='bottom')
    
    # Add reference lines
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Equal conductivity')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='No densification')
    
    # Add quadrants
    ax.axhspan(0, 1, xmin=0, xmax=1, alpha=0.1, color='red', label='Poisoning region')
    ax.axhspan(1, max(2, gb_ratio if 'gb_ratio' in locals() else 2), xmin=0, xmax=1, alpha=0.1, color='green', label='Beneficial region')
    
    ax.set_xlabel('Densification Boost (Δρ, %)')
    ax.set_ylabel('σ_gb(additive) / σ_gb(pure)')
    ax.set_title('Trade-off: Grain Boundary Conductivity vs Densification')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# NEW FUNCTIONS FOR BRICK-LAYER MODEL VALIDATION
# ============================================================================

def plot_brick_layer_validation(df_long, temperature=600):
    """
    Brick-layer model validation: 1/σ_gb - 1/σ_bulk vs 1/d.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    temperature : int
        Temperature in Celsius
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with brick-layer validation plot
    """
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['sigma_bulk_mS', 'sigma_gb_mS', 'grain_size_um'])
    
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) < 5:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Insufficient data for brick-layer validation (need at least 5 points, have {len(plot_df)})',
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate brick-layer variables
    plot_df['inv_d'] = 1.0 / plot_df['grain_size_um']
    plot_df['inv_sigma_gb_minus_inv_sigma_bulk'] = (1.0 / plot_df['sigma_gb_mS']) - (1.0 / plot_df['sigma_bulk_mS'])
    
    # Filter valid data
    valid_df = plot_df.dropna(subset=['inv_d', 'inv_sigma_gb_minus_inv_sigma_bulk'])
    valid_df = valid_df[valid_df['inv_sigma_gb_minus_inv_sigma_bulk'] > 0]
    
    if len(valid_df) < 5:
        ax.text(0.5, 0.5, 'Invalid data for brick-layer model (negative or zero values)',
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Plot by additive type
    for additive in valid_df['additive_type'].unique():
        subset = valid_df[valid_df['additive_type'] == additive]
        color = SINTERING_ADDITIVE_COLORS.get(additive, '#6B7280')
        marker = SINTERING_ADDITIVE_MARKERS.get(additive, 'o')
        
        ax.scatter(subset['inv_d'], subset['inv_sigma_gb_minus_inv_sigma_bulk'],
                  c=[color], marker=marker, s=100, alpha=0.7,
                  edgecolors='black', linewidth=0.5, label=additive)
        
        # Linear regression per additive
        if len(subset) >= 3:
            try:
                slope, intercept, r_value, p_value, std_err = linregress(subset['inv_d'], 
                                                                          subset['inv_sigma_gb_minus_inv_sigma_bulk'])
                x_line = np.linspace(subset['inv_d'].min(), subset['inv_d'].max(), 50)
                y_line = intercept + slope * x_line
                ax.plot(x_line, y_line, '--', color=color, alpha=0.5,
                       label=f'{additive}: R²={r_value**2:.2f}')
                
                # Annotate slope (specific GB conductivity)
                ax.annotate(f'{additive}: slope={slope:.2f}', 
                           xy=(0.02, 0.98 - 0.08 * list(valid_df['additive_type'].unique()).index(additive)),
                           xycoords='axes fraction', fontsize=8, color=color)
            except:
                pass
    
    ax.set_xlabel('1 / Grain Size (μm⁻¹)')
    ax.set_ylabel('1/σ_gb - 1/σ_bulk (cm/ms)')
    ax.set_title(f'Brick-Layer Model Validation at {temperature}°C\nSlope ∝ Specific GB Conductivity, Intercept ∝ GB Thickness')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# NEW FUNCTIONS FOR PHASE-SPACE 3D DIAGRAMS
# ============================================================================

def plot_phase_space_3d(df_long, x_col, y_col, z_col, color_col, temperature,
                        filter_dict=None, show_scatter=True, show_surface=False):
    """
    Interactive 3D phase-space diagram using Plotly.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    x_col : str
        X-axis column (Ce_ratio, wt%, T_sin, etc.)
    y_col : str
        Y-axis column
    z_col : str
        Z-axis column (usually conductivity)
    color_col : str
        Color column (additive_type, dopant, density, etc.)
    temperature : int
        Temperature in Celsius (if z_col contains conductivity)
    filter_dict : dict
        Dictionary of filters
    show_scatter : bool
        Show scatter points
    show_surface : bool
        Show interpolated surface
    
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D plot
    """
    # Apply filters
    plot_df = df_long.copy()
    
    if 'sigma' in z_col or 'σ' in z_col:
        plot_df = plot_df[plot_df['temperature_C'] == temperature]
    
    if filter_dict:
        for col, values in filter_dict.items():
            if col in plot_df.columns and values:
                plot_df = plot_df[plot_df[col].isin(values)]
    
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    plot_df = plot_df.dropna(subset=[x_col, y_col, z_col])
    
    if len(plot_df) < 5:
        fig = go.Figure()
        fig.add_annotation(text=f'Insufficient data for 3D plot (need at least 5 points, have {len(plot_df)})',
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Prepare color mapping
    if color_col in plot_df.columns:
        if pd.api.types.is_numeric_dtype(plot_df[color_col]):
            color_values = plot_df[color_col]
            colorbar_title = color_col.replace('_', ' ').title()
            colorscale = st.session_state.plot_settings.get('default_cmap', 'viridis')
        else:
            # Categorical color mapping
            unique_cats = plot_df[color_col].unique()
            if color_col == 'additive_type':
                color_map = SINTERING_ADDITIVE_COLORS
            elif color_col == 'dopant':
                color_map = DOPANT_COLORS
            else:
                color_map = {cat: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                            for i, cat in enumerate(unique_cats)}
            plot_df['color_code'] = plot_df[color_col].map(lambda x: color_map.get(x, '#6B7280'))
            color_values = plot_df['color_code']
            colorbar_title = color_col.replace('_', ' ').title()
            colorscale = None
    else:
        color_values = plot_df[z_col]
        colorbar_title = z_col.replace('_', ' ').title()
        colorscale = st.session_state.plot_settings.get('default_cmap', 'viridis')
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    if show_scatter:
        if color_col in plot_df.columns and not pd.api.types.is_numeric_dtype(plot_df[color_col]):
            # Separate traces for each category (for legend)
            for cat in plot_df[color_col].unique():
                subset = plot_df[plot_df[color_col] == cat]
                fig.add_trace(go.Scatter3d(
                    x=subset[x_col], y=subset[y_col], z=subset[z_col],
                    mode='markers',
                    marker=dict(
                        size=st.session_state.plot_settings.get('marker_size', 80) / 10,
                        color=color_map.get(cat, '#6B7280'),
                        opacity=st.session_state.plot_settings.get('bubble_alpha', 0.7),
                        line=dict(color='black', width=0.5)
                    ),
                    name=str(cat),
                    text=[f"{color_col}: {cat}<br>{x_col}: {x:.2f}<br>{y_col}: {y:.2f}<br>{z_col}: {z:.2f}" 
                          for x, y, z in zip(subset[x_col], subset[y_col], subset[z_col])],
                    hoverinfo='text'
                ))
        else:
            # Single trace with color scaling
            fig.add_trace(go.Scatter3d(
                x=plot_df[x_col], y=plot_df[y_col], z=plot_df[z_col],
                mode='markers',
                marker=dict(
                    size=st.session_state.plot_settings.get('marker_size', 80) / 10,
                    color=color_values,
                    colorscale=colorscale,
                    opacity=st.session_state.plot_settings.get('bubble_alpha', 0.7),
                    colorbar=dict(title=colorbar_title),
                    line=dict(color='black', width=0.5)
                ),
                text=[f"{x_col}: {x:.2f}<br>{y_col}: {y:.2f}<br>{z_col}: {z:.2f}<br>{color_col}: {c}" 
                      for x, y, z, c in zip(plot_df[x_col], plot_df[y_col], plot_df[z_col], 
                                            plot_df[color_col] if color_col in plot_df.columns else [None]*len(plot_df))],
                hoverinfo='text'
            ))
    
    # Add surface interpolation (optional)
    if show_surface and len(plot_df) >= 10:
        try:
            from scipy.interpolate import griddata
            xi = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 30)
            yi = np.linspace(plot_df[y_col].min(), plot_df[y_col].max(), 30)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((plot_df[x_col].values, plot_df[y_col].values), 
                         plot_df[z_col].values, (Xi, Yi), method='cubic')
            
            fig.add_trace(go.Surface(
                x=xi, y=yi, z=Zi,
                colorscale='Viridis',
                opacity=0.5,
                showscale=False,
                name='Interpolated surface'
            ))
        except:
            pass
    
    # Layout
    title = f'Phase-Space Diagram: {z_col.replace("_", " ").title()}'
    if 'sigma' in z_col:
        title += f' at {temperature}°C'
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            zaxis_title=z_col.replace('_', ' ').title(),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig


# ============================================================================
# NEW FUNCTIONS FOR UMAP AND PCA BIPLOT
# ============================================================================

def plot_umap_clustering(df_long, feature_columns, temperature=600, n_neighbors=15, min_dist=0.1):
    """
    UMAP clustering for non-linear dimensionality reduction.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    feature_columns : list
        Features to use for UMAP
    temperature : int
        Temperature in Celsius
    n_neighbors : int
        UMAP n_neighbors parameter
    min_dist : float
        UMAP min_dist parameter
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with UMAP plot
    """
    # Prepare data
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    # Aggregate by sample
    available_features = [f for f in feature_columns if f in plot_df.columns]
    if len(available_features) < 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'Need at least 2 features for UMAP', ha='center', va='center')
        return fig
    
    agg_df = plot_df.groupby('sample_id')[available_features].mean().dropna()
    
    if len(agg_df) < 5:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Insufficient samples for UMAP (need at least 5, have {len(agg_df)})',
                ha='center', va='center')
        return fig
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg_df)
    
    # Perform UMAP
    try:
        reducer = umap.UMAP(n_neighbors=min(n_neighbors, len(agg_df)-1), min_dist=min_dist, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'UMAP failed: {str(e)}', ha='center', va='center')
        return fig
    
    # Get metadata
    additive_types = plot_df.groupby('sample_id')['additive_type'].first()
    dopant_types = plot_df.groupby('sample_id')['dopant'].first()
    conductivities = plot_df.groupby('sample_id')['sigma_total_mS'].mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Color by additive type
    for additive in additive_types.unique():
        mask = [additive_types[idx] == additive for idx in agg_df.index]
        if any(mask):
            ax1.scatter(X_umap[mask, 0], X_umap[mask, 1],
                       c=[SINTERING_ADDITIVE_COLORS.get(additive, '#6B7280')],
                       s=100, alpha=0.7, edgecolors='black', label=additive)
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_title('UMAP Clustering: Color by Additive Type')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Color by conductivity
    valid_cond = conductivities.dropna()
    if len(valid_cond) > 0:
        umap_indices = [i for i, idx in enumerate(agg_df.index) if idx in valid_cond.index]
        umap_cond = [valid_cond[idx] for idx in agg_df.index if idx in valid_cond.index]
        scatter = ax2.scatter(X_umap[umap_indices, 0], X_umap[umap_indices, 1],
                             c=umap_cond, cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label(f'σ total at {temperature}°C (mS/cm)')
    else:
        ax2.scatter(X_umap[:, 0], X_umap[:, 1], c='gray', s=100, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.set_title('UMAP Clustering: Color by Conductivity')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('UMAP Non-linear Dimensionality Reduction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_pca_biplot(df_long, feature_columns, temperature=600):
    """
    PCA biplot with loading vectors.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    feature_columns : list
        Features for PCA
    temperature : int
        Temperature in Celsius
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with PCA biplot
    """
    # Prepare data
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    # Aggregate by sample
    available_features = [f for f in feature_columns if f in plot_df.columns]
    if len(available_features) < 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.text(0.5, 0.5, 'Need at least 2 features for PCA', ha='center', va='center')
        return fig
    
    agg_df = plot_df.groupby('sample_id')[available_features].mean().dropna()
    
    if len(agg_df) < 3:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.text(0.5, 0.5, f'Insufficient samples for PCA (need at least 3, have {len(agg_df)})',
                ha='center', va='center')
        return fig
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg_df)
    
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Get loadings
    loadings = pca.components_.T
    explained_var = pca.explained_variance_ratio_
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot points
    additive_types = plot_df.groupby('sample_id')['additive_type'].first()
    for additive in additive_types.unique():
        mask = [additive_types[idx] == additive for idx in agg_df.index]
        if any(mask):
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[SINTERING_ADDITIVE_COLORS.get(additive, '#6B7280')],
                      s=100, alpha=0.7, edgecolors='black', label=additive)
    
    # Plot loading vectors
    for i, feature in enumerate(available_features):
        ax.arrow(0, 0, loadings[i, 0] * 3, loadings[i, 1] * 3,
                head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
        ax.text(loadings[i, 0] * 3.2, loadings[i, 1] * 3.2, feature.replace('_', ' ').title(),
               fontsize=10, ha='center', va='center', color='red')
    
    # Add unit circle
    circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', alpha=0.5, color='gray')
    ax.add_patch(circle)
    
    ax.set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
    ax.set_title('PCA Biplot: Principal Component Analysis with Loading Vectors')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


# ============================================================================
# NEW FUNCTIONS FOR ADDITIVE INCORPORATION ANALYSIS
# ============================================================================

def plot_incorporation_effect(df_long, temperature=600):
    """
    Compare incorporating (Zn, Co) vs segregating (Cu, Ni) additives.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    temperature : int
        Temperature in Celsius
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure with boxplot comparison
    """
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df[plot_df['additive_type'] != 'Pure']
    
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No additive data available for incorporation analysis', ha='center', va='center')
        return fig
    
    # Add incorporation classification
    plot_df['incorporation'] = plot_df['additive_type'].map(
        lambda x: 'Incorporating (Zn, Co)' if ADDITIVE_INCORPORATION.get(x, False) else 'Segregating (Cu, Ni)'
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Total conductivity
    ax1 = axes[0]
    incorporation_groups = plot_df.groupby('incorporation')['sigma_total_mS'].apply(list)
    
    positions = [0, 1]
    bp1 = ax1.boxplot([incorporation_groups.get('Incorporating (Zn, Co)', []),
                       incorporation_groups.get('Segregating (Cu, Ni)', [])],
                      positions=positions, widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor='#3B82F6', alpha=0.7),
                      medianprops=dict(color='black', linewidth=2),
                      whiskerprops=dict(color='black'),
                      capprops=dict(color='black'),
                      flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['Incorporating\n(Zn, Co)', 'Segregating\n(Cu, Ni)'])
    ax1.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
    ax1.set_title('Total Conductivity: Incorporating vs Segregating Additives')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add statistical annotation
    from scipy.stats import mannwhitneyu
    inc_data = incorporation_groups.get('Incorporating (Zn, Co)', [])
    seg_data = incorporation_groups.get('Segregating (Cu, Ni)', [])
    if len(inc_data) > 2 and len(seg_data) > 2:
        stat, p_value = mannwhitneyu(inc_data, seg_data, alternative='two-sided')
        if p_value < 0.05:
            ax1.annotate(f'p = {p_value:.3f}', xy=(0.5, 0.95), xycoords='axes fraction',
                        ha='center', fontsize=10, style='italic')
    
    # Plot 2: GB conductivity (if available)
    ax2 = axes[1]
    if 'sigma_gb_mS' in plot_df.columns:
        gb_inc = plot_df[plot_df['incorporation'] == 'Incorporating (Zn, Co)']['sigma_gb_mS'].dropna()
        gb_seg = plot_df[plot_df['incorporation'] == 'Segregating (Cu, Ni)']['sigma_gb_mS'].dropna()
        
        if len(gb_inc) > 0 and len(gb_seg) > 0:
            bp2 = ax2.boxplot([gb_inc, gb_seg], positions=positions, widths=0.6, patch_artist=True,
                              boxprops=dict(facecolor='#EF4444', alpha=0.7),
                              medianprops=dict(color='black', linewidth=2),
                              whiskerprops=dict(color='black'),
                              capprops=dict(color='black'),
                              flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.5))
            
            ax2.set_xticks(positions)
            ax2.set_xticklabels(['Incorporating\n(Zn, Co)', 'Segregating\n(Cu, Ni)'])
            ax2.set_ylabel(f'σ gb at {temperature}°C (mS/cm)')
            ax2.set_title('Grain Boundary Conductivity')
            ax2.grid(True, alpha=0.3, axis='y')
            
            if len(gb_inc) > 2 and len(gb_seg) > 2:
                stat_gb, p_gb = mannwhitneyu(gb_inc, gb_seg, alternative='two-sided')
                if p_gb < 0.05:
                    ax2.annotate(f'p = {p_gb:.3f}', xy=(0.5, 0.95), xycoords='axes fraction',
                                ha='center', fontsize=10, style='italic')
        else:
            ax2.text(0.5, 0.5, 'Insufficient GB conductivity data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Grain Boundary Conductivity')
    else:
        ax2.text(0.5, 0.5, 'No GB conductivity data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Grain Boundary Conductivity')
    
    plt.suptitle('Additive Incorporation Effect on Conductivity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ============================================================================
# ENHANCED FUNCTIONS FOR EXISTING PLOTS (WITH D_TYPE AND D_CONC)
# ============================================================================

def plot_conductivity_vs_dopant_concentration(df_long, ax, temperature=600, selected_additives=None, selected_dopants=None):
    """
    Plot conductivity vs dopant concentration for different dopant types.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    ax : matplotlib.axes.Axes
        Axes to plot on
    temperature : int
        Temperature in Celsius
    selected_additives : list
        List of additive types to include
    selected_dopants : list
        List of dopant types to include
    """
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    if selected_additives:
        plot_df = plot_df[plot_df['additive_type'].isin(selected_additives)]
    if selected_dopants and 'dopant' in plot_df.columns:
        plot_df = plot_df[plot_df['dopant'].isin(selected_dopants)]
    
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    plot_df = plot_df.dropna(subset=['dop_cont', 'sigma_total_mS'])
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No data at {temperature}°C with selected filters', ha='center', va='center')
        return ax
    
    for dopant in plot_df['dopant'].unique():
        if pd.isna(dopant) or dopant == '':
            continue
        dopant_data = plot_df[plot_df['dopant'] == dopant]
        color = DOPANT_COLORS.get(dopant, DOPANT_COLORS['default'])
        
        # Group by concentration
        grouped = dopant_data.groupby('dop_cont')['sigma_total_mS'].agg(['mean', 'std', 'count']).reset_index()
        grouped = grouped.sort_values('dop_cont')
        
        ax.errorbar(grouped['dop_cont'], grouped['mean'], yerr=grouped['std'],
                   color=color, marker='o', markersize=6, linewidth=1.5,
                   capsize=3, label=dopant, alpha=0.8)
    
    ax.set_xlabel('Dopant Concentration (D_cont)')
    ax.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
    ax.set_title(f'Effect of Dopant Type and Concentration on Conductivity')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    return ax


def plot_conductivity_vs_ce_ratio_by_dopant(df_long, ax, temperature=600, selected_additives=None, selected_dopants=None):
    """
    Plot conductivity vs Ce/Zr ratio, separated by dopant type.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    ax : matplotlib.axes.Axes
        Axes to plot on
    temperature : int
        Temperature in Celsius
    selected_additives : list
        List of additive types to include
    selected_dopants : list
        List of dopant types to include
    """
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    if selected_additives:
        plot_df = plot_df[plot_df['additive_type'].isin(selected_additives)]
    if selected_dopants and 'dopant' in plot_df.columns:
        plot_df = plot_df[plot_df['dopant'].isin(selected_dopants)]
    
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    plot_df = plot_df.dropna(subset=['Ce_ratio', 'sigma_total_mS'])
    
    if 'Ce_ratio' not in plot_df.columns:
        # Calculate from B2_cont if needed
        if 'B2_cont' in plot_df.columns:
            plot_df['Ce_ratio'] = plot_df['B2_cont']
        else:
            ax.text(0.5, 0.5, 'Ce_ratio data not available', ha='center', va='center')
            return ax
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No data at {temperature}°C with selected filters', ha='center', va='center')
        return ax
    
    for dopant in plot_df['dopant'].unique():
        if pd.isna(dopant) or dopant == '':
            continue
        dopant_data = plot_df[plot_df['dopant'] == dopant]
        color = DOPANT_COLORS.get(dopant, DOPANT_COLORS['default'])
        
        ax.scatter(dopant_data['Ce_ratio'], dopant_data['sigma_total_mS'],
                  c=[color], marker='o', s=80, alpha=0.7,
                  edgecolors='black', linewidth=0.5, label=dopant)
        
        # Add trend line
        if len(dopant_data) >= 3:
            try:
                z = np.polyfit(dopant_data['Ce_ratio'], dopant_data['sigma_total_mS'], 1)
                x_trend = np.linspace(dopant_data['Ce_ratio'].min(), dopant_data['Ce_ratio'].max(), 50)
                ax.plot(x_trend, np.polyval(z, x_trend), '--', color=color, alpha=0.5, linewidth=1)
            except:
                pass
    
    ax.set_xlabel('Ce/(Ce+Zr) Ratio')
    ax.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
    ax.set_title(f'Effect of Ce/Zr Ratio on Conductivity by Dopant Type')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    return ax


# ============================================================================
# ENHANCED CORRELATION MATRIX FUNCTION (WITH DOPANT INFO)
# ============================================================================

def plot_enhanced_correlation_matrix(df_long, temperature=600):
    """
    Enhanced correlation matrix with separation of conductivity types and dopant info.
    """
    # Prepare data for specified temperature
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) == 0:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.text(0.5, 0.5, f'No data available at {temperature}°C', 
                ha='center', va='center', fontsize=14)
        ax.set_title(f'Correlation Matrix at {temperature}°C')
        return fig
    
    # Define parameter groups (updated with dopant info)
    conductivity_cols = ['sigma_total_mS', 'sigma_bulk_mS', 'sigma_gb_mS']
    structural_cols = ['tolerance_factor', 'lattice_distortion_index', 'radius_mismatch', 
                       'oxygen_vacancy_conc', 'r_avg_B']
    electronic_cols = ['electronegativity_difference_B_O', 'Δχ']
    microstructural_cols = ['density_percent', 'grain_size_um', 'porosity', 'S_V_ratio']
    composition_cols = ['Ce_ratio', 'dop_cont', 'additive_concentration_wt']
    
    # Calculate Ce_ratio if not present
    if 'Ce_ratio' not in plot_df.columns and 'B2_cont' in plot_df.columns:
        plot_df['Ce_ratio'] = plot_df['B2_cont']
    
    # Collect available columns
    available_conductivity = [c for c in conductivity_cols if c in plot_df.columns and plot_df[c].notna().any()]
    available_structural = [c for c in structural_cols if c in plot_df.columns and plot_df[c].notna().any()]
    available_electronic = [c for c in electronic_cols if c in plot_df.columns and plot_df[c].notna().any()]
    available_micro = [c for c in microstructural_cols if c in plot_df.columns and plot_df[c].notna().any()]
    available_composition = [c for c in composition_cols if c in plot_df.columns and plot_df[c].notna().any()]
    
    all_available = available_conductivity + available_structural + available_electronic + available_micro + available_composition
    
    if len(all_available) < 3:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.text(0.5, 0.5, 'Insufficient data for correlation matrix (need at least 3 parameters with data)', 
                ha='center', va='center', fontsize=12)
        ax.set_title(f'Correlation Matrix at {temperature}°C')
        return fig
    
    # Create correlation matrix
    corr_matrix = plot_df[all_available].corr(method='pearson')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Conductivity vs Structural
    if available_conductivity and available_structural:
        sub_corr = corr_matrix.loc[available_conductivity, available_structural]
        if sub_corr.size > 0 and not sub_corr.isnull().all().all():
            im1 = axes[0, 0].imshow(sub_corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            axes[0, 0].set_xticks(range(len(available_structural)))
            axes[0, 0].set_xticklabels(available_structural, rotation=45, ha='right', fontsize=8)
            axes[0, 0].set_yticks(range(len(available_conductivity)))
            axes[0, 0].set_yticklabels(available_conductivity, fontsize=8)
            axes[0, 0].set_title('Conductivity vs Structural')
            plt.colorbar(im1, ax=axes[0, 0])
            
            for i in range(min(len(available_conductivity), sub_corr.shape[0])):
                for j in range(min(len(available_structural), sub_corr.shape[1])):
                    val = sub_corr.values[i, j] if i < sub_corr.shape[0] and j < sub_corr.shape[1] else np.nan
                    if not np.isnan(val):
                        text_color = 'white' if abs(val) > 0.5 else 'black'
                        axes[0, 0].text(j, i, f'{val:.2f}', ha="center", va="center", color=text_color, fontsize=8)
        else:
            axes[0, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Conductivity vs Structural')
    else:
        axes[0, 0].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Conductivity vs Structural')
    
    # Plot 2: Conductivity vs Electronic
    if available_conductivity and available_electronic:
        sub_corr = corr_matrix.loc[available_conductivity, available_electronic]
        if sub_corr.size > 0 and not sub_corr.isnull().all().all():
            im2 = axes[0, 1].imshow(sub_corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            axes[0, 1].set_xticks(range(len(available_electronic)))
            axes[0, 1].set_xticklabels(available_electronic, rotation=45, ha='right', fontsize=8)
            axes[0, 1].set_yticks(range(len(available_conductivity)))
            axes[0, 1].set_yticklabels(available_conductivity, fontsize=8)
            axes[0, 1].set_title('Conductivity vs Electronic')
            plt.colorbar(im2, ax=axes[0, 1])
            
            for i in range(min(len(available_conductivity), sub_corr.shape[0])):
                for j in range(min(len(available_electronic), sub_corr.shape[1])):
                    val = sub_corr.values[i, j] if i < sub_corr.shape[0] and j < sub_corr.shape[1] else np.nan
                    if not np.isnan(val):
                        text_color = 'white' if abs(val) > 0.5 else 'black'
                        axes[0, 1].text(j, i, f'{val:.2f}', ha="center", va="center", color=text_color, fontsize=8)
        else:
            axes[0, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Conductivity vs Electronic')
    else:
        axes[0, 1].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Conductivity vs Electronic')
    
    # Plot 3: Conductivity vs Microstructural
    if available_conductivity and available_micro:
        sub_corr = corr_matrix.loc[available_conductivity, available_micro]
        if sub_corr.size > 0 and not sub_corr.isnull().all().all():
            im3 = axes[0, 2].imshow(sub_corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            axes[0, 2].set_xticks(range(len(available_micro)))
            axes[0, 2].set_xticklabels(available_micro, rotation=45, ha='right', fontsize=8)
            axes[0, 2].set_yticks(range(len(available_conductivity)))
            axes[0, 2].set_yticklabels(available_conductivity, fontsize=8)
            axes[0, 2].set_title('Conductivity vs Microstructural')
            plt.colorbar(im3, ax=axes[0, 2])
            
            for i in range(min(len(available_conductivity), sub_corr.shape[0])):
                for j in range(min(len(available_micro), sub_corr.shape[1])):
                    val = sub_corr.values[i, j] if i < sub_corr.shape[0] and j < sub_corr.shape[1] else np.nan
                    if not np.isnan(val):
                        text_color = 'white' if abs(val) > 0.5 else 'black'
                        axes[0, 2].text(j, i, f'{val:.2f}', ha="center", va="center", color=text_color, fontsize=8)
        else:
            axes[0, 2].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Conductivity vs Microstructural')
    else:
        axes[0, 2].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Conductivity vs Microstructural')
    
    # Plot 4: Conductivity vs Composition (NEW)
    if available_conductivity and available_composition:
        sub_corr = corr_matrix.loc[available_conductivity, available_composition]
        if sub_corr.size > 0 and not sub_corr.isnull().all().all():
            im4 = axes[1, 0].imshow(sub_corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            axes[1, 0].set_xticks(range(len(available_composition)))
            axes[1, 0].set_xticklabels(available_composition, rotation=45, ha='right', fontsize=8)
            axes[1, 0].set_yticks(range(len(available_conductivity)))
            axes[1, 0].set_yticklabels(available_conductivity, fontsize=8)
            axes[1, 0].set_title('Conductivity vs Composition (NEW)')
            plt.colorbar(im4, ax=axes[1, 0])
            
            for i in range(min(len(available_conductivity), sub_corr.shape[0])):
                for j in range(min(len(available_composition), sub_corr.shape[1])):
                    val = sub_corr.values[i, j] if i < sub_corr.shape[0] and j < sub_corr.shape[1] else np.nan
                    if not np.isnan(val):
                        text_color = 'white' if abs(val) > 0.5 else 'black'
                        axes[1, 0].text(j, i, f'{val:.2f}', ha="center", va="center", color=text_color, fontsize=8)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Conductivity vs Composition')
    else:
        axes[1, 0].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Conductivity vs Composition')
    
    # Plot 5: Full correlation matrix heatmap
    if corr_matrix.size > 0 and not corr_matrix.isnull().all().all():
        im5 = axes[1, 1].imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        axes[1, 1].set_xticks(range(len(all_available)))
        axes[1, 1].set_xticklabels(all_available, rotation=90, ha='right', fontsize=7)
        axes[1, 1].set_yticks(range(len(all_available)))
        axes[1, 1].set_yticklabels(all_available, fontsize=7)
        axes[1, 1].set_title('Full Correlation Matrix')
        plt.colorbar(im5, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data for full correlation matrix', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Full Correlation Matrix')
    
    # Plot 6: Empty or additional info
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, f'Correlation Analysis at {temperature}°C\n'
                   f'Total samples: {len(plot_df)}\n'
                   f'Parameters included: {len(all_available)}\n'
                   f'Conductivity types: {len(available_conductivity)}\n'
                   f'Structural: {len(available_structural)}\n'
                   f'Electronic: {len(available_electronic)}\n'
                   f'Microstructural: {len(available_micro)}\n'
                   f'Compositional: {len(available_composition)}',
                   ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=10)
    
    plt.suptitle(f'Enhanced Correlation Analysis at {temperature}°C', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# ============================================================================
# ENHANCED INSIGHTS GENERATION (WITH DOPANT AND POISONING)
# ============================================================================

def generate_enhanced_conductivity_insights(df_long):
    """
    Enhanced automatic generation of physical insights for conductivity.
    
    This version includes:
    - Per-additive optimal concentration analysis
    - Microstructure correlation analysis
    - Tolerance factor optimal range analysis
    - Grain size effect quantification
    - Density threshold analysis
    - Additive-specific recommendations
    - Temperature-dependent insights
    - Incorporation vs segregation effects
    - Dopant type and concentration effects (NEW)
    - Poisoning effect analysis (NEW)
    """
    insights = []
    
    # Comparison of Pure vs additives at 600°C
    df_600 = df_long[df_long['temperature_C'] == 600].copy()
    
    # Exclude outliers
    if 'is_outlier' in df_600.columns:
        df_600 = df_600[~df_600['is_outlier']]
    
    if len(df_600) > 0:
        pure_mean = df_600[df_600['additive_type'] == 'Pure']['sigma_total_mS'].mean()
        if not pd.isna(pure_mean):
            for additive in df_600['additive_type'].unique():
                if additive != 'Pure':
                    additive_mean = df_600[df_600['additive_type'] == additive]['sigma_total_mS'].mean()
                    if not pd.isna(additive_mean):
                        improvement = (additive_mean - pure_mean) / pure_mean * 100
                        if improvement > 50:
                            insights.append(f"⭐ **{additive} additive** shows {improvement:.0f}% higher conductivity than Pure at 600°C (highly effective).")
                        elif improvement > 20:
                            insights.append(f"✅ **{additive} additive** shows {improvement:.0f}% higher conductivity than Pure at 600°C (moderate improvement).")
                        elif improvement > 5:
                            insights.append(f"📈 **{additive} additive** shows {improvement:.0f}% higher conductivity than Pure at 600°C (slight improvement).")
                        elif improvement < -30:
                            insights.append(f"⚠️ **{additive} additive** shows {abs(improvement):.0f}% lower conductivity than Pure at 600°C (detrimental).")
                        elif improvement < -10:
                            insights.append(f"📉 **{additive} additive** shows {abs(improvement):.0f}% lower conductivity than Pure at 600°C (moderate degradation).")
    
    # Per-additive optimal concentration analysis
    for additive in df_long['additive_type'].unique():
        if additive != 'Pure':
            additive_data = df_long[df_long['additive_type'] == additive].copy()
            additive_data = additive_data.dropna(subset=['additive_concentration_wt', 'sigma_total_mS'])
            
            if 'is_outlier' in additive_data.columns:
                additive_data = additive_data[~additive_data['is_outlier']]
            
            if len(additive_data) > 3:
                grouped = additive_data.groupby('additive_concentration_wt')['sigma_total_mS'].mean().reset_index()
                if len(grouped) > 1:
                    max_row = grouped.loc[grouped['sigma_total_mS'].idxmax()]
                    max_conc = max_row['additive_concentration_wt']
                    max_cond = max_row['sigma_total_mS']
                    insights.append(f"🎯 Optimal concentration for **{additive} additive** is {max_conc:.2f} wt% (σ = {max_cond:.2f} mS/cm at 600°C).")
                    
                    if len(grouped) >= 3:
                        after_opt = grouped[grouped['additive_concentration_wt'] > max_conc]
                        if len(after_opt) >= 2:
                            trend = after_opt['sigma_total_mS'].diff().mean()
                            if trend < -0.1:
                                insights.append(f"⚠️ **{additive}**: Conductivity decreases sharply above {max_conc:.2f} wt%.")
    
    # Dopant type and concentration analysis (NEW)
    df_dopant = df_long.dropna(subset=['dopant', 'dop_cont', 'sigma_total_mS'])
    if 'is_outlier' in df_dopant.columns:
        df_dopant = df_dopant[~df_dopant['is_outlier']]
    
    if len(df_dopant) > 10:
        # Best dopant type
        dopant_means = df_dopant.groupby('dopant')['sigma_total_mS'].mean().sort_values(ascending=False)
        if len(dopant_means) > 1:
            best_dopant = dopant_means.index[0]
            best_value = dopant_means.values[0]
            insights.append(f"🔬 **Best dopant**: {best_dopant} shows the highest average conductivity ({best_value:.2f} mS/cm at 600°C).")
        
        # Dopant concentration correlation
        corr, p_val = spearmanr(df_dopant['dop_cont'], df_dopant['sigma_total_mS'])
        if p_val < 0.05:
            if corr > 0.3:
                insights.append(f"📈 **Positive correlation** between dopant concentration and conductivity (ρ = {corr:.2f}, p < 0.05). Higher doping levels improve conductivity.")
            elif corr < -0.3:
                insights.append(f"📉 **Negative correlation** between dopant concentration and conductivity (ρ = {corr:.2f}, p < 0.05). Optimal doping exists below certain threshold.")
    
    # Grain size effect with quantification
    df_grain = df_long.dropna(subset=['grain_size_um', 'sigma_total_mS'])
    if 'is_outlier' in df_grain.columns:
        df_grain = df_grain[~df_grain['is_outlier']]
    
    if len(df_grain) > 10:
        corr, p_val = spearmanr(df_grain['grain_size_um'], df_grain['sigma_total_mS'])
        if p_val < 0.05:
            if corr > 0.5:
                insights.append(f"🏗️ **Strong positive correlation** between grain size and conductivity (ρ = {corr:.2f}, p < 0.05). Larger grains (>{df_grain['grain_size_um'].quantile(0.75):.1f} μm) improve conductivity by reducing GB resistance.")
            elif corr < -0.5:
                insights.append(f"🔬 **Strong negative correlation** between grain size and conductivity (ρ = {corr:.2f}, p < 0.05). Smaller grains enhance conductivity, possibly due to improved densification.")
            elif corr > 0.2:
                insights.append(f"📊 **Weak positive correlation** between grain size and conductivity (ρ = {corr:.2f}).")
        
        large_grains = df_grain[df_grain['grain_size_um'] > df_grain['grain_size_um'].quantile(0.75)]['sigma_total_mS'].mean()
        small_grains = df_grain[df_grain['grain_size_um'] < df_grain['grain_size_um'].quantile(0.25)]['sigma_total_mS'].mean()
        if not pd.isna(large_grains) and not pd.isna(small_grains) and large_grains > small_grains:
            ratio = large_grains / small_grains if small_grains > 0 else float('inf')
            insights.append(f"📐 Samples with grain size >{df_grain['grain_size_um'].quantile(0.75):.1f} μm show **{ratio:.1f}x higher** conductivity than fine-grained samples.")
    
    # Density effect with threshold analysis
    df_density = df_long.dropna(subset=['density_percent', 'sigma_total_mS'])
    if 'is_outlier' in df_density.columns:
        df_density = df_density[~df_density['is_outlier']]
    
    if len(df_density) > 10:
        corr, p_val = spearmanr(df_density['density_percent'], df_density['sigma_total_mS'])
        if p_val < 0.05 and corr > 0.3:
            insights.append(f"📈 **Positive correlation** between density and conductivity (ρ = {corr:.2f}, p < 0.05). Higher density improves percolation paths.")
        
        high_density = df_density[df_density['density_percent'] >= 95]['sigma_total_mS'].mean()
        low_density = df_density[df_density['density_percent'] < 90]['sigma_total_mS'].mean()
        if not pd.isna(high_density) and not pd.isna(low_density) and high_density > low_density:
            ratio = high_density / low_density if low_density > 0 else float('inf')
            insights.append(f"🔥 **Critical density threshold**: Samples with density >95% have **{ratio:.1f}x higher** conductivity than samples with density <90%.")
    
    # Tolerance factor analysis with optimal range
    df_t = df_long.dropna(subset=['tolerance_factor', 'sigma_total_mS'])
    if 'is_outlier' in df_t.columns:
        df_t = df_t[~df_t['is_outlier']]
    
    if len(df_t) > 10:
        t_opt = df_t[(df_t['tolerance_factor'] >= 0.96) & (df_t['tolerance_factor'] <= 1.04)]
        t_out = df_t[(df_t['tolerance_factor'] < 0.96) | (df_t['tolerance_factor'] > 1.04)]
        if len(t_opt) > 0 and len(t_out) > 0:
            mean_opt = t_opt['sigma_total_mS'].mean()
            mean_out = t_out['sigma_total_mS'].mean()
            if mean_opt > mean_out:
                ratio = mean_opt / mean_out if mean_out > 0 else float('inf')
                insights.append(f"🏛️ **Structural stability matters**: Systems with tolerance factor in [0.96-1.04] have **{ratio:.1f}x higher** average conductivity.")
        
        df_t['distortion'] = abs(df_t['tolerance_factor'] - 1.0)
        corr, p_val = spearmanr(df_t['distortion'], df_t['sigma_total_mS'])
        if p_val < 0.05 and corr < -0.3:
            insights.append(f"📐 **Lattice distortion** negatively impacts conductivity (ρ = {corr:.2f}). More cubic structures (t closer to 1.0) perform better.")
    
    # Oxygen vacancy concentration - non-linear analysis
    df_vac = df_long.dropna(subset=['oxygen_vacancy_conc', 'sigma_total_mS'])
    if 'is_outlier' in df_vac.columns:
        df_vac = df_vac[~df_vac['is_outlier']]
    
    if len(df_vac) > 15:
        optimal = df_vac[(df_vac['oxygen_vacancy_conc'] >= 0.05) & (df_vac['oxygen_vacancy_conc'] <= 0.15)]
        low = df_vac[df_vac['oxygen_vacancy_conc'] < 0.05]
        high = df_vac[df_vac['oxygen_vacancy_conc'] > 0.15]
        
        if len(optimal) > 0:
            opt_mean = optimal['sigma_total_mS'].mean()
            if len(low) > 0:
                low_mean = low['sigma_total_mS'].mean()
                if opt_mean > low_mean:
                    insights.append(f"⚡ **Optimal oxygen vacancy concentration** is in the **0.05-0.15 range** (vs low vacancy samples).")
            if len(high) > 0:
                high_mean = high['sigma_total_mS'].mean()
                if opt_mean > high_mean:
                    insights.append(f"⚠️ **Excessive oxygen vacancies (>0.15)** may reduce conductivity due to defect clustering or trapping effects.")
    
    # Grain boundary contribution analysis
    df_gb = df_long.dropna(subset=['gb_resistance_fraction'])
    if 'is_outlier' in df_gb.columns:
        df_gb = df_gb[~df_gb['is_outlier']]
    
    if len(df_gb) > 5:
        mean_gb = df_gb['gb_resistance_fraction'].mean()
        if mean_gb > 0.5:
            insights.append(f"🚧 **Grain boundaries dominate** the total resistance ({mean_gb:.1%} on average). Improving GB conductivity is critical for overall performance.")
        else:
            insights.append(f"💎 **Bulk conductivity dominates** the total resistance ({1-mean_gb:.1%} on average). Focus on optimizing bulk properties.")
        
        gb_by_temp = df_gb.groupby('temperature_C')['gb_resistance_fraction'].mean()
        if len(gb_by_temp) > 3:
            low_temp_gb = gb_by_temp[gb_by_temp.index < 400].mean() if any(gb_by_temp.index < 400) else None
            high_temp_gb = gb_by_temp[gb_by_temp.index > 700].mean() if any(gb_by_temp.index > 700) else None
            if low_temp_gb is not None and high_temp_gb is not None:
                if low_temp_gb > high_temp_gb:
                    insights.append(f"🌡️ Grain boundary contribution **decreases with temperature** (from {low_temp_gb:.2f} at <400°C to {high_temp_gb:.2f} at >700°C), indicating thermally activated GB transport.")
    
    # Additive incorporation analysis
    if 'additive_incorporation_likely' in df_long.columns:
        incorp_true = df_long[df_long['additive_incorporation_likely'] == True]['sigma_total_mS'].mean()
        incorp_false = df_long[df_long['additive_incorporation_likely'] == False]['sigma_total_mS'].mean()
        if not pd.isna(incorp_true) and not pd.isna(incorp_false):
            if incorp_true > incorp_false:
                insights.append(f"🔬 **Incorporating additives** (Zn, Co) show higher average conductivity than segregating additives (Cu, Ni), suggesting lattice incorporation is beneficial.")
            else:
                insights.append(f"📌 **Segregating additives** (Cu, Ni) show comparable or better performance, possibly due to grain boundary engineering effects.")
    
    # Radius mismatch effect
    df_mismatch = df_long.dropna(subset=['radius_mismatch', 'sigma_total_mS'])
    if 'is_outlier' in df_mismatch.columns:
        df_mismatch = df_mismatch[~df_mismatch['is_outlier']]
    
    if len(df_mismatch) > 10:
        corr, p_val = spearmanr(df_mismatch['radius_mismatch'], df_mismatch['sigma_total_mS'])
        if p_val < 0.05 and corr < -0.3:
            insights.append(f"📏 **Larger B-site radius mismatch** correlates with lower conductivity (ρ = {corr:.2f}). Compositional homogeneity at B-site improves performance.")
    
    # Sintering temperature effect
    df_tsin = df_long.dropna(subset=['T_sin', 'sigma_total_mS'])
    if 'is_outlier' in df_tsin.columns:
        df_tsin = df_tsin[~df_tsin['is_outlier']]
    
    if len(df_tsin) > 10:
        corr, p_val = spearmanr(df_tsin['T_sin'], df_tsin['sigma_total_mS'])
        if p_val < 0.05:
            if corr > 0.3:
                insights.append(f"🔥 **Higher sintering temperature** correlates with better conductivity (ρ = {corr:.2f}). Optimize T_sin between {df_tsin['T_sin'].quantile(0.25):.0f}-{df_tsin['T_sin'].quantile(0.75):.0f}°C.")
    
    # Porosity effect
    df_por = df_long.dropna(subset=['porosity', 'sigma_total_mS'])
    if 'is_outlier' in df_por.columns:
        df_por = df_por[~df_por['is_outlier']]
    
    if len(df_por) > 10:
        corr, p_val = spearmanr(df_por['porosity'], df_por['sigma_total_mS'])
        if p_val < 0.05 and corr < -0.3:
            insights.append(f"🕳️ **Porosity significantly reduces** conductivity (ρ = {corr:.2f}). Target porosity <5% for optimal performance.")
    
    # Poisoning effect insight (NEW)
    # Check if there's evidence of poisoning (additive helps at fixed T but hurts at matched density)
    fixed_T_data = df_long[df_long['temperature_C'] == 600].dropna(subset=['T_sin', 'additive_type', 'sigma_total_mS'])
    if len(fixed_T_data) > 10:
        # Find common T_sin
        t_sin_values = fixed_T_data['T_sin'].unique()
        for ts in t_sin_values:
            pure_at_ts = fixed_T_data[(fixed_T_data['T_sin'] == ts) & (fixed_T_data['additive_type'] == 'Pure')]
            add_at_ts = fixed_T_data[(fixed_T_data['T_sin'] == ts) & (fixed_T_data['additive_type'] != 'Pure')]
            if len(pure_at_ts) > 0 and len(add_at_ts) > 0:
                pure_cond = pure_at_ts['sigma_total_mS'].mean()
                add_cond = add_at_ts['sigma_total_mS'].mean()
                if add_cond > pure_cond:
                    # Now check matched density scenario
                    high_T_pure = fixed_T_data[(fixed_T_data['T_sin'] > ts + 100) & (fixed_T_data['additive_type'] == 'Pure')]
                    if len(high_T_pure) > 0:
                        high_T_pure_cond = high_T_pure['sigma_total_mS'].mean()
                        if high_T_pure_cond > add_cond:
                            insights.append(f"⚠️ **Poisoning effect detected**: At fixed T_sin ({ts:.0f}°C), additives improve conductivity. However, Pure samples sintered at higher temperature achieve even higher conductivity, suggesting additives may segregate to grain boundaries and block proton transport.")
                            break
    
    if len(insights) == 0:
        insights.append("📋 Insufficient data for automated insights. Add more data points (especially with microstructure and compositional parameters) to enable comprehensive pattern detection.")
    
    return insights


# ============================================================================
# CACHED DATA PROCESSING FUNCTIONS
# ============================================================================
@st.cache_data(ttl=3600, show_spinner="Loading and processing data...")
def load_and_process_data(uploaded_file):
    """
    Load and process the Excel file with caching.
    
    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object
        
    Returns
    -------
    tuple
        (long_format_df, wide_format_df) processed data
    """
    # Read the Excel file with simple header
    df = read_excel_simple(uploaded_file)
    
    df_processed = df.copy()
    
    # Preprocess: fill NaN values with 0 for numeric columns where appropriate
    numeric_cols = ['B2_cont', 'D_cont', 'x, wt%']
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].apply(safe_float_converter)
            df_processed[col] = df_processed[col].fillna(0)
    
    processor = ConductivityDataProcessor()
    
    # Create list for storing data in long format
    long_format_data = []
    
    total_rows = len(df_processed)
    
    # Iterate through rows
    for idx, row in df_processed.iterrows():
        # Skip empty rows
        if pd.isna(row.get('A cation')) and pd.isna(row.get('B1 cation')):
            continue
            
        # Basic composition parameters
        a_cation = row.get('A cation', 'Ba')
        if pd.isna(a_cation) or a_cation == '':
            a_cation = 'Ba'
            
        b1_cation = row.get('B1 cation', None)
        if pd.isna(b1_cation) or b1_cation == '':
            b1_cation = None
            
        b2_cation = row.get('B2 cation', None)
        if pd.isna(b2_cation) or b2_cation == '':
            b2_cation = None
            
        b2_cont = row.get('B2_cont', 0)
        if pd.isna(b2_cont):
            b2_cont = 0
            
        dopant = row.get('dopant (D)', None)
        if pd.isna(dopant) or dopant == '':
            dopant = None
            
        dop_cont = row.get('D_cont', 0)
        if pd.isna(dop_cont):
            dop_cont = 0
        
        # Read pre-calculated descriptors if available
        rA = row.get('rA', None)
        rB1 = row.get('rB1', None)
        rB2 = row.get('rB2', None)
        rD = row.get('rD', None)
        rBav = row.get('rBav', None)
        t_factor = row.get('t', None)
        rBav_rO = row.get('rBav/rO', None)
        χA = row.get('χA', None)
        χB1 = row.get('χB1', None)
        χB2 = row.get('χB2', None)
        χD = row.get('χD', None)
        χBav = row.get('χBav', None)
        Δχ = row.get('|χA-χBav|', None)
        χ_ratio = row.get('χBav/χA', None)
        
        # Sintering additive
        additive_type = row.get('Sint addit oxide (MOx), M =', 'Pure')
        if pd.isna(additive_type) or additive_type == '':
            additive_type = 'Pure'
            
        additive_conc = row.get('x, wt%', 0.0)
        if pd.isna(additive_conc):
            additive_conc = 0.0
        
        # Synthesis parameters
        method = row.get('Method', None)
        T_sin = row.get('T sin', None)
        structure = row.get('Structure', None)
        space_group = row.get('Space group', None)
        a_latt = row.get('a, Å', None)
        b_latt = row.get('b, Å', None)
        c_latt = row.get('c, Å', None)
        density_percent = row.get('ρ, %', None)
        grain_size_um = row.get('d, mkm', None)
        S_V = row.get('S/V, mkm-1', None)
        
        # Measurement conditions
        atmosphere = row.get('Atmosphere', None)
        atmosphere_type = row.get('Atmosphere type', None)
        doi = row.get('Reference', None)
        
        # Ea if available
        Ea_eV = row.get('Ea, eV', None)
        
        # Update calculator's A-element if different
        if a_cation != processor.calculator.a_element:
            processor.calculator = ConductivityDescriptorCalculator(a_element=a_cation)
        
        # Calculate composition descriptors (or use pre-calculated)
        if b1_cation is not None and not pd.isna(b1_cation):
            formula_desc = processor.calculator.calculate_formula(
                b1_cation, b2_cation, b2_cont, dopant, dop_cont
            )
        else:
            formula_desc = {}
        
        # Use pre-calculated values if available, otherwise use calculated
        final_r_avg_B = rBav if rBav is not None and not pd.isna(rBav) else formula_desc.get('r_avg_B')
        final_tolerance_factor = t_factor if t_factor is not None and not pd.isna(t_factor) else formula_desc.get('tolerance_factor')
        final_χ_avg_B = χBav if χBav is not None and not pd.isna(χBav) else formula_desc.get('χ_avg_B')
        final_Δχ = Δχ if Δχ is not None and not pd.isna(Δχ) else formula_desc.get('Δχ')
        final_oxygen_vacancy = formula_desc.get('oxygen_vacancy_conc')
        final_radius_mismatch = formula_desc.get('radius_mismatch')
        final_lattice_distortion = formula_desc.get('lattice_distortion_index')
        
        # Calculate microstructural descriptors
        micro_desc = processor.calculator.calculate_microstructure_descriptors(
            density_percent, grain_size_um
        )
        
        # Use pre-calculated S/V if available
        if S_V is not None and not pd.isna(S_V):
            micro_desc['S_V_ratio'] = S_V
        
        # Calculate sintering additive descriptors
        additive_desc = processor.calculator.calculate_sintering_additive_descriptors(
            additive_type, additive_conc
        )
        
        # Calculate Ce_ratio
        Ce_ratio = b2_cont if b2_cation == 'Ce' else (1 - b2_cont if b1_cation == 'Ce' else 0)
        
        # Extract conductivity data - find all temperature columns
        sigma_total_data = []
        sigma_bulk_data = []
        sigma_gb_data = []
        
        # Look for columns with temperatures - ENHANCED PATTERN MATCHING
        for col in df_processed.columns:
            col_str = str(col).lower()
            
            # Check for temperature in column name (numbers like 200, 250, etc.)
            temp_match = re.search(r'(\d{3})', col_str)
            if temp_match:
                temperature = int(temp_match.group(1))
                
                # Check if it's total conductivity
                if ('total' in col_str or 'σ total' in col_str or 'sigma_total' in col_str or 
                    ('sigma' in col_str and 'bulk' not in col_str and 'gb' not in col_str and 'grain' not in col_str) or
                    col_str.startswith(str(temperature))):
                    sigma_value = row[col]
                    if not pd.isna(sigma_value) and sigma_value != '' and sigma_value is not None:
                        try:
                            sigma_val = safe_float_converter(sigma_value)
                            if sigma_val is not None and sigma_val > 0:
                                sigma_total_data.append({
                                    'temperature_K': temperature + 273.15,
                                    'temperature_C': temperature,
                                    'sigma_total_mS': sigma_val,
                                    'sigma_total_S_cm': sigma_val / 1000.0
                                })
                        except (ValueError, TypeError):
                            pass
                
                # Check if it's bulk conductivity
                elif ('bulk' in col_str or 'σ bulk' in col_str or 'sigma_bulk' in col_str):
                    sigma_value = row[col]
                    if not pd.isna(sigma_value) and sigma_value != '' and sigma_value is not None:
                        try:
                            sigma_val = safe_float_converter(sigma_value)
                            if sigma_val is not None and sigma_val > 0:
                                sigma_bulk_data.append({
                                    'temperature_K': temperature + 273.15,
                                    'temperature_C': temperature,
                                    'sigma_bulk_mS': sigma_val,
                                    'sigma_bulk_S_cm': sigma_val / 1000.0
                                })
                        except (ValueError, TypeError):
                            pass
                
                # Check if it's grain boundary conductivity
                elif ('gb' in col_str or 'σ gb' in col_str or 'sigma_gb' in col_str or 'grain boundary' in col_str):
                    sigma_value = row[col]
                    if not pd.isna(sigma_value) and sigma_value != '' and sigma_value is not None:
                        try:
                            sigma_val = safe_float_converter(sigma_value)
                            if sigma_val is not None and sigma_val > 0:
                                sigma_gb_data.append({
                                    'temperature_K': temperature + 273.15,
                                    'temperature_C': temperature,
                                    'sigma_gb_mS': sigma_val,
                                    'sigma_gb_S_cm': sigma_val / 1000.0
                                })
                        except (ValueError, TypeError):
                            pass
        
        # Also check for columns with exact temperature names (200, 250, etc.)
        for temp in processor.temperatures:
            # Total conductivity
            col_name = str(temp)
            if col_name in df_processed.columns:
                sigma_value = row[col_name]
                if not pd.isna(sigma_value) and sigma_value != '' and sigma_value is not None:
                    try:
                        sigma_val = safe_float_converter(sigma_value)
                        if sigma_val is not None and sigma_val > 0:
                            existing = [d for d in sigma_total_data if d['temperature_C'] == temp]
                            if not existing:
                                sigma_total_data.append({
                                    'temperature_K': temp + 273.15,
                                    'temperature_C': temp,
                                    'sigma_total_mS': sigma_val,
                                    'sigma_total_S_cm': sigma_val / 1000.0
                                })
                    except (ValueError, TypeError):
                        pass
        
        # Calculate grain boundary contribution
        gb_contribution = processor.calculate_gb_contribution(
            sigma_total_data, sigma_bulk_data, sigma_gb_data
        )
        
        # Add records to long format
        for sigma_data in sigma_total_data:
            T_C = sigma_data['temperature_C']
            
            # Find matching bulk and gb data for this temperature
            bulk_at_T = next((b for b in sigma_bulk_data if b['temperature_C'] == T_C), None)
            gb_at_T = next((g for g in sigma_gb_data if g['temperature_C'] == T_C), None)
            
            record = {
                'sample_id': idx,
                'A_cation': a_cation,
                'B1_cation': b1_cation,
                'B2_cation': b2_cation,
                'B2_cont': b2_cont,
                'Ce_ratio': Ce_ratio,
                'dopant': dopant,
                'D_type': dopant,
                'dop_cont': dop_cont,
                'D_conc': dop_cont,
                'additive_type': additive_desc['additive_type'],
                'additive_concentration_wt': additive_desc['additive_concentration_wt'],
                'is_pure': additive_desc['is_pure'],
                'additive_radius': additive_desc.get('additive_radius'),
                'additive_electronegativity': additive_desc.get('additive_electronegativity'),
                'additive_incorporation_likely': additive_desc.get('additive_incorporation_likely'),
                'method': method,
                'T_sin': T_sin,
                'structure': structure,
                'space_group': space_group,
                'a_latt': a_latt,
                'b_latt': b_latt,
                'c_latt': c_latt,
                'density_percent': micro_desc['density_percent'],
                'density_fraction': micro_desc['density_fraction'],
                'porosity': micro_desc['porosity'],
                'grain_size_um': grain_size_um,
                'S_V_ratio': micro_desc['S_V_ratio'],
                'inverse_d': micro_desc.get('inverse_d'),
                'atmosphere': atmosphere,
                'atmosphere_type': atmosphere_type,
                'doi': doi,
                'temperature_C': sigma_data['temperature_C'],
                'temperature_K': sigma_data['temperature_K'],
                'sigma_total_mS': sigma_data.get('sigma_total_mS'),
                'sigma_total_S_cm': sigma_data.get('sigma_total_S_cm'),
                'Ea_provided_eV': Ea_eV,
            }
            
            # Add bulk conductivity if available
            if bulk_at_T:
                record['sigma_bulk_mS'] = bulk_at_T.get('sigma_bulk_mS')
                record['sigma_bulk_S_cm'] = bulk_at_T.get('sigma_bulk_S_cm')
            else:
                record['sigma_bulk_mS'] = None
                record['sigma_bulk_S_cm'] = None
            
            # Add grain boundary conductivity if available
            if gb_at_T:
                record['sigma_gb_mS'] = gb_at_T.get('sigma_gb_mS')
                record['sigma_gb_S_cm'] = gb_at_T.get('sigma_gb_S_cm')
            else:
                record['sigma_gb_mS'] = None
                record['sigma_gb_S_cm'] = None
            
            # Calculate sigma_gb_ratio if both available
            if record['sigma_bulk_mS'] is not None and record['sigma_gb_mS'] is not None and record['sigma_gb_mS'] > 0:
                record['sigma_gb_ratio'] = record['sigma_gb_mS'] / record['sigma_bulk_mS']
            else:
                record['sigma_gb_ratio'] = None
            
            # Add grain boundary contribution
            if T_C in gb_contribution:
                record['gb_resistance_fraction'] = gb_contribution[T_C]['gb_resistance_fraction']
                record['bulk_resistance_fraction'] = gb_contribution[T_C]['bulk_resistance_fraction']
            else:
                record['gb_resistance_fraction'] = None
                record['bulk_resistance_fraction'] = None
            
            # Add geometric descriptors
            record['r_avg_B'] = final_r_avg_B
            record['radius_mismatch'] = final_radius_mismatch
            record['tolerance_factor'] = final_tolerance_factor
            record['lattice_distortion_index'] = final_lattice_distortion
            record['χ_avg_B'] = final_χ_avg_B
            record['Δχ'] = final_Δχ
            record['electronegativity_difference_B_O'] = formula_desc.get('electronegativity_difference_B_O')
            record['oxygen_vacancy_conc'] = final_oxygen_vacancy
            record['molar_mass'] = formula_desc.get('molar_mass')
            record['theoretical_density'] = formula_desc.get('theoretical_density')
            
            long_format_data.append(record)
    
    # Create long format DataFrame
    long_df = pd.DataFrame(long_format_data)
    
    # Calculate activation energies
    with st.spinner("Calculating activation energies..."):
        Ea_df = compute_activation_energies_for_df(long_df)
        # Merge Ea data back to long_df
        for col in ['Ea_total_eV', 'Ea_bulk_eV', 'Ea_gb_eV', 'Ea_total_R2', 'Ea_bulk_R2', 'Ea_gb_R2']:
            if col in Ea_df.columns:
                Ea_map = Ea_df.set_index('sample_id')[col].to_dict()
                long_df[col] = long_df['sample_id'].map(Ea_map)
    
    # Detect outliers using IQR with user-defined sensitivity
    outlier_multiplier = get_outlier_iqr_multiplier()
    if 'sigma_total_mS' in long_df.columns and len(long_df) > 0:
        outlier_mask = ConductivityDataProcessor().detect_outliers_iqr(long_df, 'sigma_total_mS', multiplier=outlier_multiplier)
        long_df['is_outlier'] = outlier_mask
    else:
        long_df['is_outlier'] = False
    
    # Create wide format FROM long format (not recalculating)
    wide_format_data = []

    if 'sample_id' not in long_df.columns or len(long_df) == 0:
        if len(long_df) > 0:
            long_df['sample_id'] = long_df.apply(
                lambda row: f"{row.get('A_cation', '')}_{row.get('B1_cation', '')}_{row.get('dopant', '')}_{row.get('additive_type', '')}_{row.get('additive_concentration_wt', 0)}",
                axis=1
            )
    
    if len(long_df) > 0:
        for sample_id in long_df['sample_id'].unique():
            sample_data = long_df[long_df['sample_id'] == sample_id]
            
            # Take first row for non-temperature dependent fields
            first_row = sample_data.iloc[0]
            
            wide_record = {
                'sample_id': sample_id,
                'A_cation': first_row.get('A_cation'),
                'B1_cation': first_row.get('B1_cation'),
                'B2_cation': first_row.get('B2_cation'),
                'B2_cont': first_row.get('B2_cont'),
                'Ce_ratio': first_row.get('Ce_ratio'),
                'dopant': first_row.get('dopant'),
                'D_type': first_row.get('D_type'),
                'dop_cont': first_row.get('dop_cont'),
                'D_conc': first_row.get('D_conc'),
                'additive_type': first_row.get('additive_type'),
                'additive_concentration_wt': first_row.get('additive_concentration_wt'),
                'additive_incorporation_likely': first_row.get('additive_incorporation_likely'),
                'method': first_row.get('method'),
                'T_sin': first_row.get('T_sin'),
                'structure': first_row.get('structure'),
                'space_group': first_row.get('space_group'),
                'a_latt': first_row.get('a_latt'),
                'b_latt': first_row.get('b_latt'),
                'c_latt': first_row.get('c_latt'),
                'density_percent': first_row.get('density_percent'),
                'grain_size_um': first_row.get('grain_size_um'),
                'atmosphere': first_row.get('atmosphere'),
                'atmosphere_type': first_row.get('atmosphere_type'),
                'doi': first_row.get('doi'),
                'r_avg_B': first_row.get('r_avg_B'),
                'radius_mismatch': first_row.get('radius_mismatch'),
                'tolerance_factor': first_row.get('tolerance_factor'),
                'lattice_distortion_index': first_row.get('lattice_distortion_index'),
                'oxygen_vacancy_conc': first_row.get('oxygen_vacancy_conc'),
                'Ea_total_eV': first_row.get('Ea_total_eV'),
                'Ea_bulk_eV': first_row.get('Ea_bulk_eV'),
                'Ea_gb_eV': first_row.get('Ea_gb_eV'),
            }
            
            # Add conductivity at each temperature
            for _, temp_row in sample_data.iterrows():
                T = temp_row['temperature_C']
                sigma_total = temp_row.get('sigma_total_mS')
                sigma_bulk = temp_row.get('sigma_bulk_mS')
                sigma_gb = temp_row.get('sigma_gb_mS')
                
                if sigma_total is not None and not pd.isna(sigma_total):
                    wide_record[f'sigma_total_{T}C'] = sigma_total
                if sigma_bulk is not None and not pd.isna(sigma_bulk):
                    wide_record[f'sigma_bulk_{T}C'] = sigma_bulk
                if sigma_gb is not None and not pd.isna(sigma_gb):
                    wide_record[f'sigma_gb_{T}C'] = sigma_gb
            
            wide_format_data.append(wide_record)
    
    wide_df = pd.DataFrame(wide_format_data)
    
    return long_df, wide_df


# ============================================================================
# CACHED ML FUNCTIONS
# ============================================================================
@st.cache_data(ttl=3600, show_spinner="Computing feature importance...")
def compute_feature_importance(_df_long, selected_features, target, temperature):
    """
    Compute feature importance with caching.
    """
    plot_df = _df_long[_df_long['temperature_C'] == temperature].copy()
    
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    available_features = [f for f in selected_features if f in plot_df.columns]
    
    if len(available_features) < 2:
        return None, None
    
    plot_df = plot_df.dropna(subset=[target] + available_features)
    
    if len(plot_df) < 10:
        return None, None
    
    X = plot_df[available_features].copy()
    
    # Encode categorical variables
    if 'additive_type' in plot_df.columns:
        X = pd.concat([X, pd.get_dummies(plot_df['additive_type'], prefix='additive')], axis=1)
    if 'dopant' in plot_df.columns:
        X = pd.concat([X, pd.get_dummies(plot_df['dopant'], prefix='dopant')], axis=1)
    
    y = plot_df[target]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    r2 = rf.score(X, y)
    
    return importance_df, r2


@st.cache_data(ttl=3600, show_spinner="Comparing ML models...")
def compare_ml_models(_df_long, selected_features, target, temperature):
    """
    Compare ML models with caching.
    """
    plot_df = _df_long[_df_long['temperature_C'] == temperature].copy()
    
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    available_features = [f for f in selected_features if f in plot_df.columns]
    
    if len(available_features) < 2:
        return None, None, None, None
    
    plot_df = plot_df.dropna(subset=[target] + available_features)
    
    if len(plot_df) < 10:
        return None, None, None, None
    
    X = plot_df[available_features].copy()
    
    if 'additive_type' in plot_df.columns:
        X = pd.concat([X, pd.get_dummies(plot_df['additive_type'], prefix='additive')], axis=1)
    if 'dopant' in plot_df.columns:
        X = pd.concat([X, pd.get_dummies(plot_df['dopant'], prefix='dopant')], axis=1)
    
    y = plot_df[target]
    
    # Define models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        
        model.fit(X, y)
        train_r2 = model.score(X, y)
        
        results.append({
            'Model': name,
            'CV R² (mean)': f'{scores.mean():.3f}',
            'CV R² (std)': f'{scores.std():.3f}',
            'Train R²': f'{train_r2:.3f}',
            'CV MAE': f'{mae_scores.mean():.3f}',
            'CV MAE (std)': f'{mae_scores.std():.3f}'
        })
    
    return pd.DataFrame(results), models, X, y


@st.cache_data(ttl=3600, show_spinner="Performing SHAP analysis...")
def compute_shap_analysis(_df, features, target, temperature, model_type='xgboost'):
    """
    Compute SHAP analysis with caching.
    """
    df_filtered = _df[_df['temperature_C'] == temperature].copy()
    
    if 'is_outlier' in df_filtered.columns:
        df_filtered = df_filtered[~df_filtered['is_outlier']]
    
    # Add categorical encoding
    if 'additive_type' in df_filtered.columns:
        df_filtered['additive_encoded'] = df_filtered['additive_type'].map(
            {v: i for i, v in enumerate(df_filtered['additive_type'].unique())}
        )
        if 'additive_encoded' not in features:
            features = features + ['additive_encoded'] if 'additive_encoded' in df_filtered.columns else features
    
    if 'dopant' in df_filtered.columns:
        df_filtered['dopant_encoded'] = df_filtered['dopant'].map(
            {v: i for i, v in enumerate(df_filtered['dopant'].unique())}
        )
        if 'dopant_encoded' not in features:
            features = features + ['dopant_encoded'] if 'dopant_encoded' in df_filtered.columns else features
    
    return shap_analysis(df_filtered, features, target, model_type)


# ============================================================================
# MAIN STREAMLIT APPLICATION
# ============================================================================
def main():
    st.set_page_config(
        page_title="SintAddvsCond - Sintering Additives vs Conductivity",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    apply_custom_css()
    
    # Initialize plot settings
    init_plot_settings()
    
    # Title with gradient effect
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="background: linear-gradient(135deg, #3B82F6, #10B981); 
                   -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent;
                   font-size: 48px;">
            🧪 SintAddvsCond
        </h1>
        <p style="color: #94A3B8; font-size: 18px;">
            Sintering Additives versus Conductivity Analysis Platform
        </p>
        <p style="color: #64748B; font-size: 14px;">
            Proton-Conducting Perovskites | Bulk & Grain Boundary Analysis | Machine Learning Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        
        # Debug mode checkbox
        debug_mode = st.checkbox("🔧 Debug Mode", value=False, help="Show debug information about column detection")
        
        # Global plot settings
        with st.expander("🎨 Global Plot Settings"):
            default_cmap = st.selectbox(
                "Default colormap",
                options=['viridis', 'plasma', 'inferno', 'coolwarm', 'RdYlBu_r', 'jet', 'magma', 'cividis'],
                index=0,
                key='global_cmap'
            )
            st.session_state.plot_settings['default_cmap'] = default_cmap
            
            show_trendlines = st.checkbox("Show trend lines", value=True, key='global_trend')
            st.session_state.plot_settings['show_trendlines'] = show_trendlines
            
            marker_size = st.slider("Marker size", min_value=20, max_value=200, value=80, key='global_marker')
            st.session_state.plot_settings['marker_size'] = marker_size
            
            font_scale = st.slider("Font scale", min_value=0.8, max_value=1.5, value=1.0, step=0.05, key='global_font')
            st.session_state.plot_settings['font_scale'] = font_scale
            plt.rcParams.update({'font.size': 10 * font_scale})
            
            outlier_sensitivity = st.select_slider(
                "Outlier detection sensitivity",
                options=['low (IQR=2.0)', 'medium (IQR=1.5)', 'high (IQR=1.0)'],
                value='medium (IQR=1.5)',
                key='global_outlier'
            )
            st.session_state.plot_settings['outlier_sensitivity'] = outlier_sensitivity
            
            contour_grid_resolution = st.slider("Contour grid resolution", min_value=20, max_value=100, value=50, key='global_grid')
            st.session_state.plot_settings['contour_grid_resolution'] = contour_grid_resolution
            
            bubble_alpha = st.slider("Bubble transparency", min_value=0.3, max_value=1.0, value=0.7, step=0.05, key='global_alpha')
            st.session_state.plot_settings['bubble_alpha'] = bubble_alpha
            
            show_contour_labels = st.checkbox("Show contour labels", value=True, key='global_labels')
            st.session_state.plot_settings['show_contour_labels'] = show_contour_labels
            
            contour_levels = st.slider("Number of contour levels", min_value=5, max_value=50, value=20, key='global_levels')
            st.session_state.plot_settings['contour_levels'] = contour_levels
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload Excel file", 
            type=['xlsx', 'xls'],
            help="Upload your data file with perovskite compositions and conductivity measurements"
        )
        
        if uploaded_file is None:
            st.info("👈 Please upload a data file to begin")
            st.markdown("""
            ### Expected format:
            - **A cation**: A-site element (Ba)
            - **B1 cation**: Main B-site element (Zr, Ce, Sn)
            - **B2 cation**: Second B-site element (Ce)
            - **B2_cont**: Content of B2 cation
            - **dopant (D)**: Acceptor dopant (Y, Gd, Sm)
            - **D_cont**: Dopant concentration
            - **Sint addit oxide (MOx), M =**: Sintering additive (Pure, Cu, Ni, Zn, Co)
            - **x, wt%**: Additive concentration
            - **Method**: Synthesis method
            - **T sin**: Sintering temperature (°C)
            - **ρ, %**: Relative density
            - **d, mkm**: Grain size (μm)
            - **σ total, mS**: Total conductivity at 200-900°C
            - **σ bulk, mS**: Bulk conductivity (optional)
            - **σ gb, mS**: Grain boundary conductivity (optional)
            - **Atmosphere**: Measurement atmosphere
            - **Atmosphere type**: Ox/Redox/Inert
            - **Reference**: DOI reference
            """)
            return
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Settings")
        
        # Temperature selection for analysis
        temperature_analysis = st.slider(
            "Reference Temperature for Analysis (°C)",
            min_value=200,
            max_value=900,
            value=600,
            step=50,
            help="Most important temperature range for proton ceramics is 500-700°C",
            key='temp_analysis'
        )
        
        st.markdown("---")
        st.markdown("### 🔍 Filters")
        
        # Initialize variables before try
        df_long = pd.DataFrame()
        df_wide = pd.DataFrame()
        filtered_long = pd.DataFrame()
        filtered_wide = pd.DataFrame()
        selected_additives = []
        selected_b_sites = []
        selected_humidity = []
        selected_atmosphere = []
        selected_dopants = []
        
        try:
            # Load and process data with caching
            with st.spinner("🔄 Loading and processing data..."):
                df_long, df_wide = load_and_process_data(uploaded_file)
            
            st.success(f"✅ Data loaded: {len(df_long)} measurements, {df_long['sample_id'].nunique() if 'sample_id' in df_long.columns else 0} unique samples")
            
            # Create filters in sidebar
            if 'additive_type' in df_long.columns:
                selected_additives = st.multiselect(
                    "Sintering Additives",
                    options=sorted(df_long['additive_type'].unique()),
                    default=sorted(df_long['additive_type'].unique()),
                    key='filter_additives'
                )
            
            if 'B1_cation' in df_long.columns:
                selected_b_sites = st.multiselect(
                    "B-site cations",
                    options=sorted(df_long['B1_cation'].dropna().unique()),
                    default=sorted(df_long['B1_cation'].dropna().unique()),
                    key='filter_bsites'
                )
            
            if 'dopant' in df_long.columns:
                selected_dopants = st.multiselect(
                    "Dopant types",
                    options=sorted(df_long['dopant'].dropna().unique()),
                    default=sorted(df_long['dopant'].dropna().unique()) if len(df_long['dopant'].dropna().unique()) <= 5 else sorted(df_long['dopant'].dropna().unique())[:5],
                    key='filter_dopants'
                )
            
            if 'atmosphere_type' in df_long.columns:
                selected_atmosphere = st.multiselect(
                    "Atmosphere type",
                    options=sorted(df_long['atmosphere_type'].dropna().unique()),
                    default=sorted(df_long['atmosphere_type'].dropna().unique()),
                    key='filter_atmosphere'
                )
            
            # Apply filters
            filtered_long = df_long.copy()
            
            if selected_additives:
                filtered_long = filtered_long[filtered_long['additive_type'].isin(selected_additives)]
            if selected_b_sites:
                filtered_long = filtered_long[filtered_long['B1_cation'].isin(selected_b_sites)]
            if selected_dopants:
                filtered_long = filtered_long[filtered_long['dopant'].isin(selected_dopants)]
            if selected_atmosphere:
                filtered_long = filtered_long[filtered_long['atmosphere_type'].isin(selected_atmosphere)]
            
            filtered_wide = df_wide.copy()
            if selected_additives and len(filtered_wide) > 0 and 'additive_type' in filtered_wide.columns:
                filtered_wide = filtered_wide[filtered_wide['additive_type'].isin(selected_additives)]
            if selected_b_sites and len(filtered_wide) > 0 and 'B1_cation' in filtered_wide.columns:
                filtered_wide = filtered_wide[filtered_wide['B1_cation'].isin(selected_b_sites)]
            if selected_dopants and len(filtered_wide) > 0 and 'dopant' in filtered_wide.columns:
                filtered_wide = filtered_wide[filtered_wide['dopant'].isin(selected_dopants)]
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.exception(e)
            return
    
    # ============================================================================
    # MAIN DISPLAY AREA
    # ============================================================================
    
    if uploaded_file is not None and len(filtered_long) > 0:
        # Display data information
        st.subheader("📈 Data Overview")
        
        # Calculate outlier count
        outlier_count = filtered_long['is_outlier'].sum() if 'is_outlier' in filtered_long.columns else 0
        
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        
        with col1:
            st.metric("Total measurements", len(filtered_long))
        with col2:
            n_samples = filtered_long['sample_id'].nunique() if 'sample_id' in filtered_long.columns else 0
            st.metric("Unique samples", n_samples)
        with col3:
            n_additives = filtered_long['additive_type'].nunique() if 'additive_type' in filtered_long.columns else 0
            st.metric("Additive types", n_additives)
        with col4:
            n_b_sites = filtered_long['B1_cation'].nunique() if 'B1_cation' in filtered_long.columns else 0
            st.metric("B-site cations", n_b_sites)
        with col5:
            n_dopants = filtered_long['dopant'].nunique() if 'dopant' in filtered_long.columns else 0
            st.metric("Dopant types", n_dopants)
        with col6:
            temp_min = filtered_long['temperature_C'].min() if 'temperature_C' in filtered_long.columns else 0
            temp_max = filtered_long['temperature_C'].max() if 'temperature_C' in filtered_long.columns else 0
            st.metric("Temp range", f"{temp_min}-{temp_max}°C")
        with col7:
            st.metric("Outliers detected", outlier_count)
        
        if outlier_count > 0:
            st.warning(f"⚠️ {outlier_count} outlier measurements detected and will be excluded from analysis.")
        
        st.markdown("---")
        
        # Update tabs with new analysis modules
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
            "📊 Conductivity Overview",
            "🔬 Additive Analysis",
            "⚡ Bulk vs GB Analysis",
            "📈 Microstructure Effects",
            "🧪 Compositional Effects",
            "🎯 Contour Maps",
            "🔵 Bubble Diagrams",
            "🧠 Poisoning Analysis",
            "🤖 ML & Feature Importance",
            "📐 Advanced Analysis",
            "💡 Insights & Data"
        ])
        
        # ====================================================================
        # TAB 1: CONDUCTIVITY OVERVIEW (Preserved from original)
        # ====================================================================
        with tab1:
            st.subheader("Conductivity vs Temperature")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.spinner("Generating conductivity plot..."):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    # Using the existing plot_conductivity_vs_temperature function (preserved)
                    from functools import partial
                    # Re-define or use existing - for now, create simple version
                    plot_df = filtered_long.copy()
                    if 'is_outlier' in plot_df.columns:
                        plot_df = plot_df[~plot_df['is_outlier']]
                    
                    for additive in plot_df['additive_type'].unique():
                        add_data = plot_df[plot_df['additive_type'] == additive]
                        grouped = add_data.groupby('temperature_C')['sigma_total_mS'].agg(['mean', 'std']).reset_index()
                        color = SINTERING_ADDITIVE_COLORS.get(additive, '#6B7280')
                        marker = SINTERING_ADDITIVE_MARKERS.get(additive, 'o')
                        ax.errorbar(grouped['temperature_C'], grouped['mean'], yerr=grouped['std'],
                                   color=color, marker=marker, markersize=6, linewidth=1.5,
                                   capsize=3, label=additive, alpha=0.8)
                    ax.set_xlabel('Temperature (°C)')
                    ax.set_ylabel('σ total (mS/cm)')
                    ax.set_title('Conductivity vs Temperature for Different Sintering Additives')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                with st.spinner("Generating Arrhenius plot..."):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_df = filtered_long.copy()
                    if 'is_outlier' in plot_df.columns:
                        plot_df = plot_df[~plot_df['is_outlier']]
                    plot_df['ln_sigmaT'] = np.log(plot_df['sigma_total_mS'] * (plot_df['temperature_C'] + 273.15))
                    plot_df['invT_1000'] = 1000.0 / (plot_df['temperature_C'] + 273.15)
                    plot_df = plot_df.dropna(subset=['ln_sigmaT', 'invT_1000'])
                    
                    for additive in plot_df['additive_type'].unique():
                        add_data = plot_df[plot_df['additive_type'] == additive]
                        grouped = add_data.groupby('invT_1000')['ln_sigmaT'].agg(['mean', 'std']).reset_index()
                        color = SINTERING_ADDITIVE_COLORS.get(additive, '#6B7280')
                        marker = SINTERING_ADDITIVE_MARKERS.get(additive, 'o')
                        ax.errorbar(grouped['invT_1000'], grouped['mean'], yerr=grouped['std'],
                                   color=color, marker=marker, markersize=5, linewidth=1.5,
                                   capsize=3, label=additive, alpha=0.8)
                    ax.set_xlabel('1000/T (K⁻¹)')
                    ax.set_ylabel('ln(σT)')
                    ax.set_title('Arrhenius Plot: ln(σT) vs 1000/T')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
            
            st.subheader(f"Conductivity at {temperature_analysis}°C")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.spinner("Generating comparison bar chart..."):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_df = filtered_long[filtered_long['temperature_C'] == temperature_analysis].copy()
                    if 'is_outlier' in plot_df.columns:
                        plot_df = plot_df[~plot_df['is_outlier']]
                    grouped = plot_df.groupby('additive_type')['sigma_total_mS'].agg(['mean', 'std', 'count']).reset_index()
                    grouped = grouped.sort_values('mean', ascending=False)
                    bars = ax.bar(range(len(grouped)), grouped['mean'], yerr=grouped['std'], capsize=5,
                                color=[SINTERING_ADDITIVE_COLORS.get(atype, '#6B7280') for atype in grouped['additive_type']],
                                edgecolor='black', linewidth=0.5)
                    ax.set_xticks(range(len(grouped)))
                    ax.set_xticklabels(grouped['additive_type'], rotation=45, ha='right')
                    ax.set_ylabel(f'σ total at {temperature_analysis}°C (mS/cm)')
                    ax.set_title(f'Conductivity Comparison at {temperature_analysis}°C')
                    ax.grid(True, alpha=0.3, axis='y')
                    for i, (_, row) in enumerate(grouped.iterrows()):
                        ax.text(i, row['mean'] + row['std'] + 0.01, f'{row["mean"]:.3f}', ha='center', fontsize=8)
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                with st.spinner("Generating improvement chart..."):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_df = filtered_long[filtered_long['temperature_C'] == temperature_analysis].copy()
                    if 'is_outlier' in plot_df.columns:
                        plot_df = plot_df[~plot_df['is_outlier']]
                    pure_mean = plot_df[plot_df['additive_type'] == 'Pure']['sigma_total_mS'].mean()
                    if not pd.isna(pure_mean) and pure_mean > 0:
                        improvement_data = []
                        for additive in plot_df['additive_type'].unique():
                            if additive == 'Pure':
                                continue
                            add_mean = plot_df[plot_df['additive_type'] == additive]['sigma_total_mS'].mean()
                            if not pd.isna(add_mean):
                                improvement = (add_mean - pure_mean) / pure_mean * 100
                                improvement_data.append({'additive': additive, 'improvement': improvement})
                        improvement_df = pd.DataFrame(improvement_data).sort_values('improvement', ascending=False)
                        colors = ['#10B981' if imp > 0 else '#EF4444' for imp in improvement_df['improvement']]
                        ax.barh(range(len(improvement_df)), improvement_df['improvement'], color=colors, edgecolor='black')
                        ax.set_yticks(range(len(improvement_df)))
                        ax.set_yticklabels(improvement_df['additive'])
                        ax.set_xlabel(f'Improvement relative to Pure at {temperature_analysis}°C (%)')
                        ax.set_title(f'Conductivity Improvement: Additives vs Pure at {temperature_analysis}°C')
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
                        for i, (_, row) in enumerate(improvement_df.iterrows()):
                            ax.text(row['improvement'] + 1, i, f'{row["improvement"]:.1f}%', va='center', fontsize=8)
                        ax.grid(True, alpha=0.3, axis='x')
                    else:
                        ax.text(0.5, 0.5, 'No Pure reference data available', ha='center', va='center')
                    st.pyplot(fig)
                    plt.close(fig)
            
            # Enhanced correlation matrix
            st.subheader("Enhanced Correlation Matrix")
            with st.spinner("Generating correlation matrix..."):
                fig = plot_enhanced_correlation_matrix(filtered_long, temperature_analysis)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
        
        # ====================================================================
        # TAB 2: ADDITIVE ANALYSIS (Enhanced with concentration effects)
        # ====================================================================
        with tab2:
            st.subheader("Effect of Additive Concentration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.spinner("Generating concentration plot..."):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_df = filtered_long[filtered_long['temperature_C'] == temperature_analysis].copy()
                    plot_df = plot_df[plot_df['additive_concentration_wt'] > 0]
                    if 'is_outlier' in plot_df.columns:
                        plot_df = plot_df[~plot_df['is_outlier']]
                    
                    for additive in plot_df['additive_type'].unique():
                        if additive == 'Pure':
                            continue
                        add_data = plot_df[plot_df['additive_type'] == additive]
                        grouped = add_data.groupby('additive_concentration_wt')['sigma_total_mS'].agg(['mean', 'std', 'count']).reset_index()
                        color = SINTERING_ADDITIVE_COLORS.get(additive, '#6B7280')
                        marker = SINTERING_ADDITIVE_MARKERS.get(additive, 'o')
                        ax.errorbar(grouped['additive_concentration_wt'], grouped['mean'], yerr=grouped['std'],
                                   color=color, marker=marker, markersize=8, linewidth=1.5,
                                   capsize=5, label=additive, alpha=0.8)
                    ax.set_xlabel('Additive Concentration (wt%)')
                    ax.set_ylabel(f'σ total at {temperature_analysis}°C (mS/cm)')
                    ax.set_title(f'Conductivity vs Additive Concentration at {temperature_analysis}°C')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                st.info("Activation energy analysis integrated into Insights tab.")
            
            st.subheader("Influence of Sintering Temperature (Enhanced)")
            
            with st.spinner("Generating sintering temperature bubble plot..."):
                fig = plot_bubble_diagram(
                    filtered_long, 
                    x_col='T_sin', 
                    y_col='sigma_total_mS', 
                    size_col='additive_concentration_wt', 
                    color_col='additive_type',
                    temperature=temperature_analysis,
                    show_trend=st.session_state.plot_settings.get('show_trendlines', True),
                    cmap=st.session_state.plot_settings.get('default_cmap', 'viridis')
                )
                st.pyplot(fig)
                plt.close(fig)
        
        # ====================================================================
        # TAB 3: BULK VS GB ANALYSIS
        # ====================================================================
        with tab3:
            st.subheader("Bulk vs Grain Boundary Conductivity")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.spinner("Generating bulk vs GB plot..."):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_df = filtered_long[filtered_long['temperature_C'] == temperature_analysis].copy()
                    plot_df = plot_df.dropna(subset=['sigma_bulk_mS', 'sigma_gb_mS'])
                    if 'is_outlier' in plot_df.columns:
                        plot_df = plot_df[~plot_df['is_outlier']]
                    
                    if len(plot_df) > 0:
                        grouped_bulk = plot_df.groupby('additive_type')['sigma_bulk_mS'].mean()
                        grouped_gb = plot_df.groupby('additive_type')['sigma_gb_mS'].mean()
                        additives = grouped_bulk.index.tolist()
                        x = np.arange(len(additives))
                        width = 0.35
                        ax.bar(x - width/2, grouped_bulk.values, width, label='σ bulk', color='#3B82F6', edgecolor='black')
                        ax.bar(x + width/2, grouped_gb.values, width, label='σ gb', color='#EF4444', edgecolor='black')
                        ax.set_xticks(x)
                        ax.set_xticklabels(additives, rotation=45, ha='right')
                        ax.set_ylabel(f'Conductivity at {temperature_analysis}°C (mS/cm)')
                        ax.set_title(f'Bulk vs Grain Boundary Conductivity at {temperature_analysis}°C')
                        ax.legend()
                        ax.grid(True, alpha=0.3, axis='y')
                    else:
                        ax.text(0.5, 0.5, f'No bulk/gb data at {temperature_analysis}°C', ha='center', va='center')
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                with st.spinner("Generating GB fraction plot..."):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_df = filtered_long[filtered_long['temperature_C'] == temperature_analysis].copy()
                    plot_df = plot_df.dropna(subset=['gb_resistance_fraction'])
                    if 'is_outlier' in plot_df.columns:
                        plot_df = plot_df[~plot_df['is_outlier']]
                    
                    if len(plot_df) > 0:
                        grouped = plot_df.groupby('additive_type')['gb_resistance_fraction'].agg(['mean', 'std']).reset_index()
                        grouped = grouped.sort_values('mean', ascending=False)
                        bars = ax.bar(range(len(grouped)), grouped['mean'], yerr=grouped['std'], capsize=5,
                                    color=[SINTERING_ADDITIVE_COLORS.get(atype, '#6B7280') for atype in grouped['additive_type']],
                                    edgecolor='black', linewidth=0.5)
                        ax.set_xticks(range(len(grouped)))
                        ax.set_xticklabels(grouped['additive_type'], rotation=45, ha='right')
                        ax.set_ylabel(f'Grain Boundary Resistance Fraction at {temperature_analysis}°C')
                        ax.set_title(f'GB Contribution to Total Resistance at {temperature_analysis}°C')
                        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% GB contribution')
                        ax.grid(True, alpha=0.3, axis='y')
                    else:
                        ax.text(0.5, 0.5, f'No gb fraction data at {temperature_analysis}°C', ha='center', va='center')
                    st.pyplot(fig)
                    plt.close(fig)
            
            # Verification of mixing rule
            st.subheader("Verification of the Mixing Rule")
            
            df_check = filtered_long.dropna(subset=['sigma_total_mS', 'sigma_bulk_mS', 'sigma_gb_mS'])
            df_check = df_check[df_check['temperature_C'] == temperature_analysis]
            
            if 'is_outlier' in df_check.columns:
                df_check = df_check[~df_check['is_outlier']]
            
            if len(df_check) > 0:
                df_check['sigma_total_calc'] = 1.0 / (1.0/df_check['sigma_bulk_mS'] + 1.0/df_check['sigma_gb_mS'])
                df_check['error_pct'] = abs(df_check['sigma_total_calc'] - df_check['sigma_total_mS']) / df_check['sigma_total_mS'] * 100
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(df_check['sigma_total_mS'], df_check['sigma_total_calc'], 
                          c='#3B82F6', s=80, alpha=0.7, edgecolors='black')
                min_val = min(df_check['sigma_total_mS'].min(), df_check['sigma_total_calc'].min())
                max_val = max(df_check['sigma_total_mS'].max(), df_check['sigma_total_calc'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y = x')
                ax.set_xlabel('Measured σ total (mS/cm)')
                ax.set_ylabel('Calculated σ total (mS/cm)')
                ax.set_title(f'Verification: 1/σ_total = 1/σ_bulk + 1/σ_gb at {temperature_analysis}°C')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
                st.metric("Mean prediction error", f"{df_check['error_pct'].mean():.1f}%")
            else:
                st.info(f"No data with both bulk and GB conductivity at {temperature_analysis}°C")
        
        # ====================================================================
        # TAB 4: MICROSTRUCTURE EFFECTS
        # ====================================================================
        with tab4:
            st.subheader("Microstructure Effects on Conductivity")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.spinner("Generating grain size plot..."):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_df = filtered_long[filtered_long['temperature_C'] == temperature_analysis].copy()
                    plot_df = plot_df.dropna(subset=['grain_size_um', 'sigma_total_mS'])
                    if 'is_outlier' in plot_df.columns:
                        plot_df = plot_df[~plot_df['is_outlier']]
                    
                    for additive in plot_df['additive_type'].unique():
                        add_data = plot_df[plot_df['additive_type'] == additive]
                        color = SINTERING_ADDITIVE_COLORS.get(additive, '#6B7280')
                        marker = SINTERING_ADDITIVE_MARKERS.get(additive, 'o')
                        ax.scatter(add_data['grain_size_um'], add_data['sigma_total_mS'],
                                  color=color, marker=marker, s=80, alpha=0.7, edgecolors='black', linewidth=0.5, label=additive)
                    ax.set_xlabel('Grain Size (μm)')
                    ax.set_ylabel(f'σ total at {temperature_analysis}°C (mS/cm)')
                    ax.set_title(f'Conductivity vs Grain Size at {temperature_analysis}°C')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                with st.spinner("Generating density plot..."):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_df = filtered_long[filtered_long['temperature_C'] == temperature_analysis].copy()
                    plot_df = plot_df.dropna(subset=['density_percent', 'sigma_total_mS'])
                    if 'is_outlier' in plot_df.columns:
                        plot_df = plot_df[~plot_df['is_outlier']]
                    
                    for additive in plot_df['additive_type'].unique():
                        add_data = plot_df[plot_df['additive_type'] == additive]
                        color = SINTERING_ADDITIVE_COLORS.get(additive, '#6B7280')
                        marker = SINTERING_ADDITIVE_MARKERS.get(additive, 'o')
                        ax.scatter(add_data['density_percent'], add_data['sigma_total_mS'],
                                  color=color, marker=marker, s=80, alpha=0.7, edgecolors='black', linewidth=0.5, label=additive)
                    ax.set_xlabel('Relative Density (%)')
                    ax.set_ylabel(f'σ total at {temperature_analysis}°C (mS/cm)')
                    ax.set_title(f'Conductivity vs Density at {temperature_analysis}°C')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
            
            # Brick-layer model validation (NEW)
            st.subheader("Brick-Layer Model Validation")
            with st.spinner("Generating brick-layer validation plot..."):
                fig = plot_brick_layer_validation(filtered_long, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
        
        # ====================================================================
        # TAB 5: COMPOSITIONAL EFFECTS (Enhanced with dopant info)
        # ====================================================================
        with tab5:
            st.subheader("Compositional Effects on Conductivity")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.spinner("Generating tolerance factor plot..."):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_df = filtered_long[filtered_long['temperature_C'] == temperature_analysis].copy()
                    plot_df = plot_df.dropna(subset=['tolerance_factor', 'sigma_total_mS'])
                    if 'is_outlier' in plot_df.columns:
                        plot_df = plot_df[~plot_df['is_outlier']]
                    
                    for additive in plot_df['additive_type'].unique():
                        add_data = plot_df[plot_df['additive_type'] == additive]
                        color = SINTERING_ADDITIVE_COLORS.get(additive, '#6B7280')
                        marker = SINTERING_ADDITIVE_MARKERS.get(additive, 'o')
                        ax.scatter(add_data['tolerance_factor'], add_data['sigma_total_mS'],
                                  color=color, marker=marker, s=80, alpha=0.7, edgecolors='black', linewidth=0.5, label=additive)
                    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Ideal cubic (t=1)')
                    ax.axvspan(0.96, 1.04, alpha=0.2, color='green', label='Optimal range')
                    ax.set_xlabel('Tolerance Factor (t)')
                    ax.set_ylabel(f'σ total at {temperature_analysis}°C (mS/cm)')
                    ax.set_title(f'Effect of Tolerance Factor on Conductivity at {temperature_analysis}°C')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                with st.spinner("Generating oxygen vacancy plot..."):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_df = filtered_long[filtered_long['temperature_C'] == temperature_analysis].copy()
                    plot_df = plot_df.dropna(subset=['oxygen_vacancy_conc', 'sigma_total_mS'])
                    if 'is_outlier' in plot_df.columns:
                        plot_df = plot_df[~plot_df['is_outlier']]
                    
                    for additive in plot_df['additive_type'].unique():
                        add_data = plot_df[plot_df['additive_type'] == additive]
                        color = SINTERING_ADDITIVE_COLORS.get(additive, '#6B7280')
                        marker = SINTERING_ADDITIVE_MARKERS.get(additive, 'o')
                        ax.scatter(add_data['oxygen_vacancy_conc'], add_data['sigma_total_mS'],
                                  color=color, marker=marker, s=80, alpha=0.7, edgecolors='black', linewidth=0.5, label=additive)
                    ax.set_xlabel('Oxygen Vacancy Concentration [V_O]')
                    ax.set_ylabel(f'σ total at {temperature_analysis}°C (mS/cm)')
                    ax.set_title(f'Effect of Oxygen Vacancy Concentration on Conductivity at {temperature_analysis}°C')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
            
            # Dopant type and concentration effects (NEW)
            st.subheader("Dopant Effects")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.spinner("Generating dopant concentration plot..."):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_df = filtered_long[filtered_long['temperature_C'] == temperature_analysis].copy()
                    plot_df = plot_df.dropna(subset=['dop_cont', 'sigma_total_mS', 'dopant'])
                    if 'is_outlier' in plot_df.columns:
                        plot_df = plot_df[~plot_df['is_outlier']]
                    
                    for dopant in plot_df['dopant'].unique():
                        if pd.isna(dopant) or dopant == '':
                            continue
                        dop_data = plot_df[plot_df['dopant'] == dopant]
                        color = DOPANT_COLORS.get(dopant, DOPANT_COLORS['default'])
                        grouped = dop_data.groupby('dop_cont')['sigma_total_mS'].agg(['mean', 'std']).reset_index()
                        ax.errorbar(grouped['dop_cont'], grouped['mean'], yerr=grouped['std'],
                                   color=color, marker='o', markersize=6, linewidth=1.5,
                                   capsize=3, label=dopant, alpha=0.8)
                    ax.set_xlabel('Dopant Concentration (D_cont)')
                    ax.set_ylabel(f'σ total at {temperature_analysis}°C (mS/cm)')
                    ax.set_title('Effect of Dopant Type and Concentration on Conductivity')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
            
            with col2:
                with st.spinner("Generating Ce/Zr ratio by dopant plot..."):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_df = filtered_long[filtered_long['temperature_C'] == temperature_analysis].copy()
                    plot_df = plot_df.dropna(subset=['Ce_ratio', 'sigma_total_mS', 'dopant'])
                    if 'is_outlier' in plot_df.columns:
                        plot_df = plot_df[~plot_df['is_outlier']]
                    
                    for dopant in plot_df['dopant'].unique():
                        if pd.isna(dopant) or dopant == '':
                            continue
                        dop_data = plot_df[plot_df['dopant'] == dopant]
                        color = DOPANT_COLORS.get(dopant, DOPANT_COLORS['default'])
                        ax.scatter(dop_data['Ce_ratio'], dop_data['sigma_total_mS'],
                                  c=[color], marker='o', s=80, alpha=0.7, edgecolors='black', label=dopant)
                        if len(dop_data) >= 3:
                            z = np.polyfit(dop_data['Ce_ratio'], dop_data['sigma_total_mS'], 1)
                            x_trend = np.linspace(dop_data['Ce_ratio'].min(), dop_data['Ce_ratio'].max(), 50)
                            ax.plot(x_trend, np.polyval(z, x_trend), '--', color=color, alpha=0.5)
                    ax.set_xlabel('Ce/(Ce+Zr) Ratio')
                    ax.set_ylabel(f'σ total at {temperature_analysis}°C (mS/cm)')
                    ax.set_title('Effect of Ce/Zr Ratio on Conductivity by Dopant Type')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)
        
        # ====================================================================
        # TAB 6: CONTOUR MAPS (NEW - Parameterized)
        # ====================================================================
        with tab6:
            st.subheader("🎯 Contour Maps")
            st.markdown("2D heatmaps to identify optimal composition regions ('golden zones')")
            
            # Parameter selection for contour map
            col1, col2, col3 = st.columns(3)
            
            # Define available options for contour maps
            contour_x_options = ['additive_concentration_wt', 'Ce_ratio', 'tolerance_factor', 'Δχ', 
                                'T_sin', 'grain_size_um', 'S_V_ratio', 'density_percent', 'dop_cont']
            contour_y_options = ['Ce_ratio', 'additive_concentration_wt', 'tolerance_factor', 'Δχ',
                                'Ea_total_eV', 'T_sin', 'inverse_d', 'dop_cont']
            contour_z_options = ['sigma_total_mS', 'sigma_bulk_mS', 'sigma_gb_mS', 'sigma_gb_ratio',
                                'Ea_total_eV', 'Ea_bulk_eV', 'Ea_gb_eV']
            
            with col1:
                contour_x = st.selectbox("X-axis", contour_x_options, index=0, key='contour_x')
            with col2:
                contour_y = st.selectbox("Y-axis", contour_y_options, index=0, key='contour_y')
            with col3:
                contour_z = st.selectbox("Color (Z-axis)", contour_z_options, index=0, key='contour_z')
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                contour_additive_filter = st.multiselect(
                    "Filter by additive type",
                    options=filtered_long['additive_type'].unique() if 'additive_type' in filtered_long.columns else [],
                    default=filtered_long['additive_type'].unique() if 'additive_type' in filtered_long.columns else [],
                    key='contour_additive'
                )
            with col2:
                contour_dopant_filter = st.multiselect(
                    "Filter by dopant type",
                    options=filtered_long['dopant'].unique() if 'dopant' in filtered_long.columns else [],
                    default=filtered_long['dopant'].unique() if 'dopant' in filtered_long.columns else [],
                    key='contour_dopant'
                )
            
            # Advanced settings
            with st.expander("Advanced Contour Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    show_points = st.checkbox("Show data points", value=True, key='contour_points')
                    show_labels = st.checkbox("Show contour labels", 
                                             value=st.session_state.plot_settings.get('show_contour_labels', True),
                                             key='contour_labels')
                with col2:
                    n_levels = st.slider("Number of contour levels", min_value=5, max_value=50,
                                        value=st.session_state.plot_settings.get('contour_levels', 20),
                                        key='contour_levels_slider')
                    n_grid = st.slider("Grid resolution", min_value=20, max_value=100,
                                      value=st.session_state.plot_settings.get('contour_grid_resolution', 50),
                                      key='contour_grid')
            
            # Create filter dict
            filter_dict = {}
            if contour_additive_filter:
                filter_dict['additive_type'] = contour_additive_filter
            if contour_dopant_filter:
                filter_dict['dopant'] = contour_dopant_filter
            
            # Generate contour map
            with st.spinner("Generating contour map..."):
                fig = plot_contour_map(
                    filtered_long,
                    x_col=contour_x,
                    y_col=contour_y,
                    z_col=contour_z,
                    temperature=temperature_analysis,
                    filter_dict=filter_dict if filter_dict else None,
                    n_grid=n_grid,
                    cmap=st.session_state.plot_settings.get('default_cmap', 'viridis'),
                    show_points=show_points,
                    show_labels=show_labels,
                    n_levels=n_levels
                )
                st.pyplot(fig)
                plt.close(fig)
            
            # Example suggestions
            st.markdown("---")
            st.markdown("### Suggested Contour Maps for Key Insights")
            st.markdown("""
            | # | X | Y | Z | Insight |
            |---|---|---|---|----------|
            | 1 | additive_concentration_wt | Ce_ratio | sigma_gb_mS | Optimal Ce/ratio + additive concentration for GB conductivity |
            | 2 | tolerance_factor | Δχ | sigma_gb_ratio | Where GB contribution is minimized |
            | 3 | dop_cont | Ce_ratio | Ea_total_eV | Activation energy landscape vs composition |
            | 4 | T_sin | additive_concentration_wt | densification_boost | Optimal sintering conditions |
            | 5 | grain_size_um | S_V_ratio | sigma_gb_mS | Microstructure-property relationship |
            """)
        
        # ====================================================================
        # TAB 7: BUBBLE DIAGRAMS (NEW - Parameterized with smart selection)
        # ====================================================================
        with tab7:
            st.subheader("🔵 Bubble Diagrams")
            st.markdown("Multi-parameter visualization for comprehensive material analysis")
            
            # Get available parameters
            available_params = get_available_bubble_params(filtered_long)
            all_numeric_params = [p for p in available_params['numeric'] if p in filtered_long.columns]
            all_categorical_params = [p for p in available_params['categorical'] if p in filtered_long.columns]
            
            # Smart parameter selection with mutual exclusivity
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # X-axis (numeric only)
                bubble_x = st.selectbox(
                    "X-axis",
                    options=all_numeric_params,
                    index=all_numeric_params.index('additive_concentration_wt') if 'additive_concentration_wt' in all_numeric_params else 0,
                    key='bubble_x'
                )
            
            with col2:
                # Y-axis (numeric only - conductivity or Ea)
                y_options = [p for p in all_numeric_params if 'sigma' in p or 'Ea' in p or 'conductivity' in p]
                if not y_options:
                    y_options = all_numeric_params
                bubble_y = st.selectbox(
                    "Y-axis",
                    options=y_options,
                    index=0,
                    key='bubble_y'
                )
            
            # Remaining options for size and color (exclude selected X and Y)
            remaining_numeric = [p for p in all_numeric_params if p not in [bubble_x, bubble_y]]
            remaining_categorical = all_categorical_params
            
            with col3:
                bubble_size = st.selectbox(
                    "Bubble size",
                    options=remaining_numeric + ['None'],
                    index=0,
                    key='bubble_size'
                )
                if bubble_size == 'None':
                    bubble_size = None
            
            with col4:
                bubble_color = st.selectbox(
                    "Bubble color",
                    options=remaining_numeric + remaining_categorical + ['None'],
                    index=0,
                    key='bubble_color'
                )
                if bubble_color == 'None':
                    bubble_color = None
            
            # Filter options
            st.markdown("### Filters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bubble_additive_filter = st.multiselect(
                    "Additive type",
                    options=filtered_long['additive_type'].unique() if 'additive_type' in filtered_long.columns else [],
                    default=filtered_long['additive_type'].unique() if 'additive_type' in filtered_long.columns else [],
                    key='bubble_additive'
                )
            with col2:
                bubble_dopant_filter = st.multiselect(
                    "Dopant type",
                    options=filtered_long['dopant'].unique() if 'dopant' in filtered_long.columns else [],
                    default=filtered_long['dopant'].unique() if 'dopant' in filtered_long.columns else [],
                    key='bubble_dopant'
                )
            with col3:
                bubble_atmosphere_filter = st.multiselect(
                    "Atmosphere type",
                    options=filtered_long['atmosphere_type'].unique() if 'atmosphere_type' in filtered_long.columns else [],
                    default=filtered_long['atmosphere_type'].unique() if 'atmosphere_type' in filtered_long.columns else [],
                    key='bubble_atmosphere'
                )
            
            # Trend line option
            show_trend = st.checkbox("Show trend lines", 
                                    value=st.session_state.plot_settings.get('show_trendlines', True),
                                    key='bubble_trend')
            
            # Create filter dict
            filter_dict = {}
            if bubble_additive_filter:
                filter_dict['additive_type'] = bubble_additive_filter
            if bubble_dopant_filter:
                filter_dict['dopant'] = bubble_dopant_filter
            if bubble_atmosphere_filter:
                filter_dict['atmosphere_type'] = bubble_atmosphere_filter
            
            # Generate bubble diagram
            with st.spinner("Generating bubble diagram..."):
                if bubble_size is not None and bubble_color is not None:
                    fig = plot_bubble_diagram(
                        filtered_long,
                        x_col=bubble_x,
                        y_col=bubble_y,
                        size_col=bubble_size,
                        color_col=bubble_color,
                        temperature=temperature_analysis,
                        filter_dict=filter_dict if filter_dict else None,
                        show_trend=show_trend,
                        cmap=st.session_state.plot_settings.get('default_cmap', 'viridis')
                    )
                elif bubble_size is not None:
                    # Use additive_type as color
                    fig = plot_bubble_diagram(
                        filtered_long,
                        x_col=bubble_x,
                        y_col=bubble_y,
                        size_col=bubble_size,
                        color_col='additive_type',
                        temperature=temperature_analysis,
                        filter_dict=filter_dict if filter_dict else None,
                        show_trend=show_trend,
                        cmap=st.session_state.plot_settings.get('default_cmap', 'viridis')
                    )
                else:
                    # Use additive_type as size and color
                    fig = plot_bubble_diagram(
                        filtered_long,
                        x_col=bubble_x,
                        y_col=bubble_y,
                        size_col='additive_concentration_wt',
                        color_col='additive_type',
                        temperature=temperature_analysis,
                        filter_dict=filter_dict if filter_dict else None,
                        show_trend=show_trend,
                        cmap=st.session_state.plot_settings.get('default_cmap', 'viridis')
                    )
                st.pyplot(fig)
                plt.close(fig)
            
            # Multi-panel bubble analysis
            st.subheader("Multi-Panel Bubble Analysis")
            with st.spinner("Generating multi-panel bubble analysis..."):
                fig = plot_multi_panel_bubble_analysis(filtered_long, temperature_analysis, filter_dict if filter_dict else None)
                st.pyplot(fig)
                plt.close(fig)
        
        # ====================================================================
        # TAB 8: POISONING ANALYSIS (NEW - Key physical insight)
        # ====================================================================
        with tab8:
            st.subheader("🧠 Poisoning Analysis: Densification vs Grain Boundary Blocking")
            st.markdown("""
            This analysis reveals the key trade-off in sintering additive design:
            - **At fixed sintering temperature**: Additives improve densification and conductivity
            - **At matched density**: Pure samples sintered at higher temperature often outperform additive-containing samples
            - **Conclusion**: Additives may segregate to grain boundaries and block proton transport
            """)
            
            with st.spinner("Generating poisoning analysis plots..."):
                fig = plot_poisoning_analysis(filtered_long, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
            
            st.subheader("Matched Density Comparison")
            with st.spinner("Generating matched density comparison..."):
                fig = plot_pure_vs_additive_matched_density(filtered_long, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
            
            st.subheader("GB Ratio vs Densification Boost")
            with st.spinner("Generating trade-off analysis..."):
                fig = plot_gb_ratio_vs_densification_boost(filtered_long, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
            
            st.subheader("Additive Incorporation Analysis")
            with st.spinner("Generating incorporation effect plot..."):
                fig = plot_incorporation_effect(filtered_long, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
        
        # ====================================================================
        # TAB 9: ML & FEATURE IMPORTANCE
        # ====================================================================
        with tab9:
            st.subheader("Machine Learning Analysis")
            
            # Available features for ML (including dopant and composition)
            available_ml_features = [f for f in ['density_percent', 'grain_size_um', 'tolerance_factor', 
                        'oxygen_vacancy_conc', 'additive_concentration_wt', 'radius_mismatch',
                        'lattice_distortion_index', 'Ce_ratio', 'dop_cont', 'Δχ', 'S_V_ratio']
                                     if f in filtered_long.columns]
            
            if len(available_ml_features) > 0:
                ml_features = st.multiselect(
                    "Select features for ML analysis",
                    options=available_ml_features,
                    default=available_ml_features[:min(5, len(available_ml_features))],
                    key='ml_features'
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Feature Importance (Random Forest)**")
                    with st.spinner("Computing feature importance..."):
                        importance_df, r2 = compute_feature_importance(
                            filtered_long, ml_features, 'sigma_total_mS', temperature_analysis
                        )
                    
                    if importance_df is not None:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        importance_df = importance_df.head(10)
                        ax.barh(range(len(importance_df)), importance_df['importance'], 
                               color='#3B82F6', edgecolor='black')
                        ax.set_yticks(range(len(importance_df)))
                        ax.set_yticklabels(importance_df['feature'], fontsize=9)
                        ax.set_xlabel('Feature Importance')
                        ax.set_title(f'Random Forest R² = {r2:.3f}')
                        ax.invert_yaxis()
                        st.pyplot(fig)
                        plt.close(fig)
                        st.dataframe(importance_df, use_container_width=True)
                    else:
                        st.warning("Insufficient data for feature importance analysis")
                
                with col2:
                    st.markdown("**Model Comparison**")
                    with st.spinner("Comparing ML models..."):
                        models_df, models, X, y = compare_ml_models(
                            filtered_long, ml_features, 'sigma_total_mS', temperature_analysis
                        )
                    
                    if models_df is not None:
                        st.dataframe(models_df, use_container_width=True)
                    else:
                        st.warning("Insufficient data for model comparison")
                
                # SHAP Analysis
                st.subheader("🔬 SHAP Analysis (Model Interpretability)")
                
                if len(ml_features) >= 2 and models_df is not None:
                    with st.spinner("Performing SHAP analysis..."):
                        shap_result = compute_shap_analysis(
                            filtered_long, ml_features, 'sigma_total_mS', temperature_analysis, model_type='xgboost'
                        )
                    
                    if shap_result is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**SHAP Feature Importance**")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            mean_abs_shap = np.mean(np.abs(shap_result['shap_values']), axis=0)
                            sorted_idx = np.argsort(mean_abs_shap)[::-1]
                            ax.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx], color='#3B82F6', edgecolor='black')
                            ax.set_yticks(range(len(sorted_idx)))
                            ax.set_yticklabels([shap_result['feature_names'][i] for i in sorted_idx], fontsize=9)
                            ax.set_xlabel('Mean |SHAP value|')
                            ax.set_title('Feature Importance (SHAP)')
                            ax.invert_yaxis()
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        with col2:
                            st.markdown("**SHAP Dependence Plot**")
                            feature_for_dependence = st.selectbox(
                                "Select feature for SHAP dependence plot",
                                options=ml_features,
                                key='shap_feature'
                            )
                            
                            if feature_for_dependence in shap_result['feature_names']:
                                fig, ax = plt.subplots(figsize=(8, 6))
                                idx = shap_result['feature_names'].index(feature_for_dependence)
                                shap_values = shap_result['shap_values']
                                X_data = shap_result['X']
                                ax.scatter(X_data[:, idx], shap_values[:, idx], 
                                          c='#3B82F6', alpha=0.6, edgecolors='black')
                                ax.set_xlabel(feature_for_dependence)
                                ax.set_ylabel('SHAP value')
                                ax.set_title(f'SHAP Dependence: {feature_for_dependence}')
                                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close(fig)
                    else:
                        st.info("Insufficient data for SHAP analysis (need at least 10 samples)")
                else:
                    st.info("Select at least 2 features and ensure sufficient data for SHAP analysis")
            else:
                st.warning("No ML features available in the data")
        
        # ====================================================================
        # TAB 10: ADVANCED ANALYSIS (PCA, UMAP, clustering, phase-space)
        # ====================================================================
        with tab10:
            st.subheader("Advanced Statistical Analysis")
            
            # Feature selection for advanced analysis
            adv_features = st.multiselect(
                "Select features for dimensionality reduction",
                options=[f for f in available_ml_features if f in filtered_long.columns],
                default=[f for f in available_ml_features if f in filtered_long.columns][:4] if len(available_ml_features) > 0 else [],
                key='adv_features'
            )
            
            if len(adv_features) >= 2:
                # UMAP Clustering
                st.markdown("### UMAP Non-linear Dimensionality Reduction")
                with st.spinner("Generating UMAP plot..."):
                    fig = plot_umap_clustering(filtered_long, adv_features, temperature_analysis, n_neighbors=15, min_dist=0.1)
                    st.pyplot(fig)
                    plt.close(fig)
                
                # PCA Biplot
                st.markdown("### PCA Biplot with Loading Vectors")
                with st.spinner("Generating PCA biplot..."):
                    fig = plot_pca_biplot(filtered_long, adv_features, temperature_analysis)
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Phase-space 3D diagram
                st.markdown("### Interactive 3D Phase-Space Diagram")
                st.markdown("Explore multi-dimensional relationships interactively")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    ps_x = st.selectbox("X-axis", adv_features, index=0, key='ps_x')
                with col2:
                    ps_y = st.selectbox("Y-axis", adv_features, index=1 if len(adv_features) > 1 else 0, key='ps_y')
                with col3:
                    ps_z = st.selectbox("Z-axis", ['sigma_total_mS', 'sigma_bulk_mS', 'sigma_gb_mS', 'Ea_total_eV'], 
                                       index=0, key='ps_z')
                
                with st.spinner("Generating 3D plot..."):
                    fig = plot_phase_space_3d(
                        filtered_long,
                        x_col=ps_x,
                        y_col=ps_y,
                        z_col=ps_z,
                        color_col='additive_type',
                        temperature=temperature_analysis,
                        show_scatter=True,
                        show_surface=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster profiles
                st.markdown("### Cluster Profiles")
                with st.spinner("Generating cluster profiles..."):
                    profiles_df = get_cluster_profiles(filtered_long, adv_features, eps=0.5, min_samples=3)
                    if profiles_df is not None:
                        st.dataframe(profiles_df, use_container_width=True)
                    else:
                        st.info("Insufficient data for clustering analysis")
                
                # Partial correlations
                st.markdown("### Partial Correlation Analysis")
                st.markdown("*Controlling for density and grain size effects*")
                
                control_options = [f for f in ['density_percent', 'grain_size_um', 'porosity', 'S_V_ratio'] if f in filtered_long.columns]
                if control_options:
                    control_vars = st.multiselect("Control variables", control_options, default=control_options[:2], key='control_vars')
                    if control_vars:
                        features_for_partial = [f for f in adv_features if f not in control_vars]
                        if features_for_partial:
                            with st.spinner("Computing partial correlations..."):
                                fig, ax = plt.subplots(figsize=(10, 6))
                                plot_partial_correlations(filtered_long, features_for_partial, 'sigma_total_mS', 
                                                         control_vars, ax, temperature_analysis)
                                st.pyplot(fig)
                                plt.close(fig)
            else:
                st.warning("Select at least 2 features for advanced analysis")
        
        # ====================================================================
        # TAB 11: INSIGHTS & DATA
        # ====================================================================
        with tab11:
            st.subheader("💡 Automated Physical Insights")
            
            with st.spinner("Generating insights from data..."):
                insights = generate_enhanced_conductivity_insights(filtered_long)
                for insight in insights:
                    st.info(insight)
            
            st.markdown("---")
            st.subheader("📋 Processed Data")
            
            # Show data in long format
            display_cols = ['sample_id', 'B1_cation', 'dopant', 'D_conc', 'additive_type', 
                           'additive_concentration_wt', 'additive_incorporation_likely',
                           'temperature_C', 'sigma_total_mS', 'sigma_bulk_mS', 'sigma_gb_mS',
                           'density_percent', 'grain_size_um', 'tolerance_factor', 
                           'radius_mismatch', 'oxygen_vacancy_conc', 'Ea_total_eV', 'is_outlier']
            available_cols = [col for col in display_cols if col in filtered_long.columns]
            
            st.dataframe(filtered_long[available_cols].head(100), use_container_width=True)
            
            # Export data
            csv = filtered_long.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download processed data as CSV",
                csv,
                "conductivity_data_processed.csv",
                "text/csv"
            )
    
    else:
        st.info("👈 Please upload an Excel file to begin analysis")
        
        st.markdown("### Expected data format example:")
        example_data = pd.DataFrame({
            'A cation': ['Ba', 'Ba', 'Ba'],
            'B1 cation': ['Zr', 'Zr', 'Ce'],
            'B2 cation': ['Ce', 'Ce', ''],
            'B2_cont': [0.3, 0.3, ''],
            'dopant (D)': ['Y', 'Y', 'Gd'],
            'D_cont': [0.2, 0.2, 0.1],
            'Sint addit oxide (MOx), M =': ['Pure', 'Cu', 'Ni'],
            'x, wt%': [0, 1.36, 1],
            'Method': ['solid-state', 'solid-state', 'solid-state'],
            'T sin': [1550, 1550, 1450],
            'ρ, %': ['', '', 95.7],
            'd, mkm': ['', '', 15.3],
            'σ total, 600': [3.41, 7.40, 12.76],
            'Atmosphere': ['Ox', 'Ox', 'Ox'],
            'Atmosphere type': ['wet', 'wet', 'dry'],
            'Reference': ['10.1016/j.ceramint.2022.01.039', 
                         '10.1016/j.ceramint.2022.01.039',
                         '10.1016/j.ijhydene.2022.07.237']
        })
        
        st.dataframe(example_data, use_container_width=True)
        
        st.markdown("""
        ### Key Features of SintAddvsCond (Enhanced Version):
        
        #### 🎯 **New Analysis Modules**
        - **Contour Maps**: Identify optimal composition regions (Ce/ratio vs additive concentration)
        - **Bubble Diagrams**: Multi-parameter visualization with smart parameter selection
        - **Poisoning Analysis**: Reveals the trade-off between densification and GB blocking
        - **Brick-Layer Model**: Quantitative GB conductivity analysis
        - **Phase-Space 3D**: Interactive 3D exploration of composition-processing-property space
        
        #### 🔬 **Enhanced Analysis**
        - **Dopant Effects**: Full integration of D_type and D_conc across all analyses
        - **Activation Energy**: Automatic Ea calculation from Arrhenius plots
        - **UMAP & PCA Biplot**: Advanced dimensionality reduction with loadings
        - **Additive Incorporation**: Comparison of incorporating (Zn, Co) vs segregating (Cu, Ni) additives
        
        #### 🎨 **User Controls**
        - Global plot settings (colormap, marker size, font scale)
        - Outlier sensitivity adjustment
        - Trend line toggle
        - Interactive filtering by additive, dopant, atmosphere
        
        #### 📊 **All Original Features Preserved**
        - Temperature-dependent conductivity
        - Bulk vs GB analysis
        - Microstructure effects
        - Machine Learning with SHAP
        - Automated insights generation
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>SintAddvsCond - Sintering Additives versus Conductivity Analysis Platform</p>
        <p>Enhanced Version | Powered by Streamlit, Scikit-learn, UMAP, and SHAP</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
