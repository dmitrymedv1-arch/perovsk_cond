import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import time
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================
AVOGADRO_NUMBER = 6.02214076e23  # mol⁻¹
OXYGEN_RADIUS = 1.4  # Å
PREFACTOR_VOLUME = 16 * np.pi / 3  # 16π/3 for sphere volume calculation
GAS_CONSTANT = 8.314  # J/(mol·K)

# Scientific plot style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'xtick.color': 'black',
    'ytick.color': 'black',
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
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.facecolor': 'white',
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
    'Nd': 144.24,
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
    'Pure': '#4DAF4A',
    'Cu': '#E41A1C',
    'Ni': '#377EB8',
    'Zn': '#984EA3',
    'Co': '#FF7F00',
    'default': '#999999'
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
                r'σ\s*total\s*,\s*mS'
            ],
            'sigma_bulk': [
                r'σ\s*bulk',
                r'sigma\s*_?\s*bulk',
                r'bulk\s*conductivity',
                r'σ\s*\(bulk\)'
            ],
            'sigma_gb': [
                r'σ\s*gb',
                r'sigma\s*_?\s*gb',
                r'grain\s*boundary',
                r'σ\s*\(gb\)'
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
            f'@{temperature}'
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
        else:
            descriptors['grain_size_um'] = None
            descriptors['S_V_ratio'] = None
            descriptors['S_V_ratio_m'] = None
        
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
    
    def calculate_arrhenius_params(self, sigma_data):
        """
        Calculate Arrhenius parameters from conductivity data.
        
        Parameters
        ----------
        sigma_data : list of dict
            Conductivity data at different temperatures
        
        Returns
        -------
        dict
            Arrhenius parameters: Ea, A, R²
        """
        if len(sigma_data) < 3:
            return {'Ea': None, 'A': None, 'R2': None, 'has_data': False}
        
        # Transform for Arrhenius plot: ln(σT) vs 1000/T
        temps = []
        ln_sigmaT = []
        
        for data in sigma_data:
            T_K = data['temperature_K']
            # Get the sigma value (key may vary)
            sigma_key = [k for k in data.keys() if 'sigma' in k and 'mS' in k][0] if data else None
            if sigma_key:
                sigma = data[sigma_key]
                if sigma is not None and sigma > 0 and T_K > 0:
                    ln_val = np.log(sigma * T_K)
                    if np.isfinite(ln_val):
                        ln_sigmaT.append(ln_val)
                        temps.append(1000.0 / T_K)
        
        if len(temps) < 3:
            return {'Ea': None, 'A': None, 'R2': None, 'has_data': False}
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(temps, ln_sigmaT)
        
        # Ea = slope * R (where R = 8.314 J/(mol·K) = 0.008314 kJ/(mol·K))
        # For convenience, Ea in eV: 1 eV = 96.485 kJ/mol
        Ea_kJ = slope * GAS_CONSTANT / 1000.0  # kJ/mol
        Ea_eV = Ea_kJ / 96.485  # eV
        
        return {
            'Ea': Ea_eV,
            'Ea_kJ': Ea_kJ,
            'A': np.exp(intercept),
            'R2': r_value ** 2,
            'has_data': True,
            'n_points': len(temps)
        }
    
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
# ENHANCED DATA PROCESSING FUNCTION
# ============================================================================
@st.cache_data
def process_conductivity_data(df):
    """
    Main function for processing conductivity data.
    Enhanced version with proper handling of multi-row headers.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source data (already loaded with proper column names)
    
    Returns
    -------
    tuple
        (long_format_df, wide_format_df) processed data
    """
    df_processed = df.copy()
    
    # Preprocess: fill NaN values with 0 for numeric columns where appropriate
    numeric_cols = ['B2_cont', 'dop_cont', 'x, wt%']
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
            
        dopant = row.get('dopant', None)
        if pd.isna(dopant) or dopant == '':
            dopant = None
            
        dop_cont = row.get('dop_cont', 0)
        if pd.isna(dop_cont):
            dop_cont = 0
        
        # Sintering additive
        additive_type = row.get('Сд', 'Pure')
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
        
        # Measurement conditions
        atmosphere = row.get('Атмосфера', None)
        humidity = row.get('Влажность', None)
        doi = row.get('ссылка', None)
        
        # Ea from table (if available)
        Ea_table = row.get('Ea, эВ', None)
        
        # Update calculator's A-element if different
        if a_cation != processor.calculator.a_element:
            processor.calculator = ConductivityDescriptorCalculator(a_element=a_cation)
        
        # Calculate composition descriptors
        if b1_cation is not None and not pd.isna(b1_cation):
            formula_desc = processor.calculator.calculate_formula(
                b1_cation, b2_cation, b2_cont, dopant, dop_cont
            )
        else:
            formula_desc = {}
        
        # Calculate microstructural descriptors
        micro_desc = processor.calculator.calculate_microstructure_descriptors(
            density_percent, grain_size_um
        )
        
        # Calculate sintering additive descriptors
        additive_desc = processor.calculator.calculate_sintering_additive_descriptors(
            additive_type, additive_conc
        )
        
        # Extract conductivity data - find all temperature columns
        sigma_total_data = []
        sigma_bulk_data = []
        sigma_gb_data = []
        
        # Look for columns with temperatures
        for col in df_processed.columns:
            col_str = str(col).lower()
            
            # Check for temperature in column name (numbers like 200, 250, etc.)
            temp_match = re.search(r'(\d{3})', col_str)
            if temp_match:
                temperature = int(temp_match.group(1))
                
                # Check if it's total conductivity
                if 'total' in col_str or 'σ total' in col_str or ('sigma' in col_str and 'bulk' not in col_str and 'gb' not in col_str):
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
                elif 'bulk' in col_str or 'σ bulk' in col_str:
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
                elif 'gb' in col_str or 'σ gb' in col_str:
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
                            # Check if this is already added
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
        
        # Calculate Arrhenius parameters
        arrhenius = processor.calculate_arrhenius_params(sigma_total_data)
        
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
                'dopant': dopant,
                'dop_cont': dop_cont,
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
                'atmosphere': atmosphere,
                'humidity': humidity,
                'doi': doi,
                'Ea_table': Ea_table,
                'Ea_calculated': arrhenius['Ea'],
                'Ea_calculated_kJ': arrhenius['Ea_kJ'],
                'arrhenius_R2': arrhenius['R2'],
                'arrhenius_n_points': arrhenius.get('n_points', 0),
                'temperature_C': sigma_data['temperature_C'],
                'temperature_K': sigma_data['temperature_K'],
                'sigma_total_mS': sigma_data.get('sigma_total_mS'),
                'sigma_total_S_cm': sigma_data.get('sigma_total_S_cm'),
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
            
            # Add grain boundary contribution
            if T_C in gb_contribution:
                record['gb_resistance_fraction'] = gb_contribution[T_C]['gb_resistance_fraction']
                record['bulk_resistance_fraction'] = gb_contribution[T_C]['bulk_resistance_fraction']
            else:
                record['gb_resistance_fraction'] = None
                record['bulk_resistance_fraction'] = None
            
            # Add geometric descriptors
            record['r_avg_B'] = formula_desc.get('r_avg_B')
            record['radius_mismatch'] = formula_desc.get('radius_mismatch')
            record['tolerance_factor'] = formula_desc.get('tolerance_factor')
            record['lattice_distortion_index'] = formula_desc.get('lattice_distortion_index')
            record['χ_avg_B'] = formula_desc.get('χ_avg_B')
            record['Δχ'] = formula_desc.get('Δχ')
            record['electronegativity_difference_B_O'] = formula_desc.get('electronegativity_difference_B_O')
            record['oxygen_vacancy_conc'] = formula_desc.get('oxygen_vacancy_conc')
            record['molar_mass'] = formula_desc.get('molar_mass')
            record['theoretical_density'] = formula_desc.get('theoretical_density')
            
            long_format_data.append(record)
    
    # Create long format DataFrame
    long_df = pd.DataFrame(long_format_data)
    
    # Detect outliers in conductivity
    if 'sigma_total_mS' in long_df.columns and len(long_df) > 0:
        outlier_mask = ConductivityDataProcessor().detect_outliers_iqr(long_df, 'sigma_total_mS')
        long_df['is_outlier'] = outlier_mask
    else:
        long_df['is_outlier'] = False
    
    # Create wide format FROM long format (not recalculating)
    wide_format_data = []

    if 'sample_id' not in long_df.columns or len(long_df) == 0:
        # Create composite ID from key parameters
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
                'dopant': first_row.get('dopant'),
                'dop_cont': first_row.get('dop_cont'),
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
                'humidity': first_row.get('humidity'),
                'doi': first_row.get('doi'),
                'Ea_table': first_row.get('Ea_table'),
                'Ea_calculated': first_row.get('Ea_calculated'),
                'Ea_calculated_kJ': first_row.get('Ea_calculated_kJ'),
                'arrhenius_R2': first_row.get('arrhenius_R2'),
                'r_avg_B': first_row.get('r_avg_B'),
                'radius_mismatch': first_row.get('radius_mismatch'),
                'tolerance_factor': first_row.get('tolerance_factor'),
                'lattice_distortion_index': first_row.get('lattice_distortion_index'),
                'oxygen_vacancy_conc': first_row.get('oxygen_vacancy_conc'),
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
# PLOTTING FUNCTIONS (all original plots preserved, new ones added)
# ============================================================================

def plot_conductivity_vs_temperature(df_long, ax, selected_additives=None, selected_b_sites=None, temperature_min=400, temperature_max=700):
    """Plot 1: Conductivity as function of temperature for different additives"""
    
    plot_df = df_long.copy()
    
    # Filtering
    if selected_additives:
        plot_df = plot_df[plot_df['additive_type'].isin(selected_additives)]
    if selected_b_sites and 'B1_cation' in plot_df.columns:
        plot_df = plot_df[plot_df['B1_cation'].isin(selected_b_sites)]
    
    # Filter by temperature
    plot_df = plot_df[(plot_df['temperature_C'] >= temperature_min) & (plot_df['temperature_C'] <= temperature_max)]
    
    # Exclude outliers if flag exists
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, 'No data for selected filters', ha='center', va='center')
        return ax
    
    # Grouping for averaging
    grouped = plot_df.groupby(['additive_type', 'temperature_C'])['sigma_total_mS'].agg(['mean', 'std', 'count']).reset_index()
    
    for additive in grouped['additive_type'].unique():
        additive_data = grouped[grouped['additive_type'] == additive]
        color = SINTERING_ADDITIVE_COLORS.get(additive, SINTERING_ADDITIVE_COLORS['default'])
        marker = SINTERING_ADDITIVE_MARKERS.get(additive, SINTERING_ADDITIVE_MARKERS['default'])
        
        ax.errorbar(
            additive_data['temperature_C'],
            additive_data['mean'],
            yerr=additive_data['std'],
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.5,
            capsize=3,
            label=additive,
            alpha=0.8
        )
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('σ total (mS/cm)')
    ax.set_title('Conductivity vs Temperature for Different Sintering Additives')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    return ax


def plot_arrhenius(df_long, ax, selected_additives=None, selected_b_sites=None):
    """Plot 2: Arrhenius plot ln(σT) vs 1000/T"""
    
    plot_df = df_long.copy()
    
    if selected_additives:
        plot_df = plot_df[plot_df['additive_type'].isin(selected_additives)]
    if selected_b_sites and 'B1_cation' in plot_df.columns:
        plot_df = plot_df[plot_df['B1_cation'].isin(selected_b_sites)]
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    # Calculate ln(σT)
    plot_df['ln_sigmaT'] = np.log(plot_df['sigma_total_mS'] * (plot_df['temperature_C'] + 273.15))
    plot_df['invT_1000'] = 1000.0 / (plot_df['temperature_C'] + 273.15)
    
    plot_df = plot_df.dropna(subset=['ln_sigmaT', 'invT_1000'])
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, 'No data for selected filters', ha='center', va='center')
        return ax
    
    for additive in plot_df['additive_type'].unique():
        additive_data = plot_df[plot_df['additive_type'] == additive]
        color = SINTERING_ADDITIVE_COLORS.get(additive, SINTERING_ADDITIVE_COLORS['default'])
        marker = SINTERING_ADDITIVE_MARKERS.get(additive, SINTERING_ADDITIVE_MARKERS['default'])
        
        # Averaging by temperature
        grouped = additive_data.groupby('invT_1000')['ln_sigmaT'].agg(['mean', 'std']).reset_index()
        
        ax.errorbar(
            grouped['invT_1000'],
            grouped['mean'],
            yerr=grouped['std'],
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.5,
            capsize=3,
            label=additive,
            alpha=0.8
        )
    
    ax.set_xlabel('1000/T (K⁻¹)')
    ax.set_ylabel('ln(σT)')
    ax.set_title('Arrhenius Plot: ln(σT) vs 1000/T')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    return ax


def plot_additive_comparison_bar(df_long, ax, temperature=600):
    """Plot 3: Comparison of different additives at fixed temperature"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No data at {temperature}°C', ha='center', va='center')
        return ax
    
    # Grouping by additive type
    grouped = plot_df.groupby('additive_type')['sigma_total_mS'].agg(['mean', 'std', 'count']).reset_index()
    grouped = grouped.sort_values('mean', ascending=False)
    
    bars = ax.bar(
        range(len(grouped)),
        grouped['mean'],
        yerr=grouped['std'],
        capsize=5,
        color=[SINTERING_ADDITIVE_COLORS.get(atype, SINTERING_ADDITIVE_COLORS['default']) for atype in grouped['additive_type']],
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped['additive_type'], rotation=45, ha='right')
    ax.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
    ax.set_title(f'Conductivity Comparison at {temperature}°C')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values above bars
    for i, (_, row) in enumerate(grouped.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.01, f'{row["mean"]:.3f}', ha='center', fontsize=8)
    
    return ax


def plot_conductivity_vs_additive_concentration(df_long, ax, temperature=600, b_site_filter=None):
    """Plot 4: Conductivity vs additive concentration"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df[plot_df['additive_concentration_wt'] > 0]
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if b_site_filter and 'B1_cation' in plot_df.columns:
        plot_df = plot_df[plot_df['B1_cation'].isin(b_site_filter)]
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No data at {temperature}°C', ha='center', va='center')
        return ax
    
    for additive in plot_df['additive_type'].unique():
        if additive == 'Pure':
            continue
        additive_data = plot_df[plot_df['additive_type'] == additive]
        color = SINTERING_ADDITIVE_COLORS.get(additive, SINTERING_ADDITIVE_COLORS['default'])
        marker = SINTERING_ADDITIVE_MARKERS.get(additive, SINTERING_ADDITIVE_MARKERS['default'])
        
        # Grouping by concentration
        grouped = additive_data.groupby('additive_concentration_wt')['sigma_total_mS'].agg(['mean', 'std', 'count']).reset_index()
        
        ax.errorbar(
            grouped['additive_concentration_wt'],
            grouped['mean'],
            yerr=grouped['std'],
            color=color,
            marker=marker,
            markersize=8,
            linewidth=1.5,
            capsize=5,
            label=additive,
            alpha=0.8
        )
    
    ax.set_xlabel('Additive Concentration (wt%)')
    ax.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
    ax.set_title(f'Conductivity vs Additive Concentration at {temperature}°C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_pure_vs_additive_comparison(df_long, ax, temperature=600):
    """Plot 5: Pure vs additive comparison (relative improvement)"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No data at {temperature}°C', ha='center', va='center')
        return ax
    
    # Get baseline for Pure
    pure_data = plot_df[plot_df['additive_type'] == 'Pure']['sigma_total_mS'].mean()
    
    if pd.isna(pure_data) or pure_data == 0:
        ax.text(0.5, 0.5, 'No Pure reference data available', ha='center', va='center')
        return ax
    
    # Calculate improvement for each additive
    improvement_data = []
    for additive in plot_df['additive_type'].unique():
        if additive == 'Pure':
            continue
        additive_mean = plot_df[plot_df['additive_type'] == additive]['sigma_total_mS'].mean()
        if not pd.isna(additive_mean):
            improvement = (additive_mean - pure_data) / pure_data * 100
            improvement_data.append({
                'additive': additive,
                'improvement': improvement,
                'mean_conductivity': additive_mean
            })
    
    improvement_df = pd.DataFrame(improvement_data).sort_values('improvement', ascending=False)
    
    if len(improvement_df) == 0:
        ax.text(0.5, 0.5, 'No additive data available', ha='center', va='center')
        return ax
    
    colors = ['#4DAF4A' if imp > 0 else '#E41A1C' for imp in improvement_df['improvement']]
    bars = ax.barh(range(len(improvement_df)), improvement_df['improvement'], color=colors, edgecolor='black')
    
    ax.set_yticks(range(len(improvement_df)))
    ax.set_yticklabels(improvement_df['additive'])
    ax.set_xlabel(f'Improvement relative to Pure at {temperature}°C (%)')
    ax.set_title(f'Conductivity Improvement: Additives vs Pure at {temperature}°C')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Add values
    for i, (_, row) in enumerate(improvement_df.iterrows()):
        ax.text(row['improvement'] + 1, i, f'{row["improvement"]:.1f}%', va='center', fontsize=8)
    
    ax.grid(True, alpha=0.3, axis='x')
    return ax


def plot_bulk_vs_gb_contribution(df_long, ax, temperature=600):
    """Plot 6: Comparison of bulk and grain boundary conductivity"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['sigma_bulk_mS', 'sigma_gb_mS'])
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No bulk/gb data at {temperature}°C', ha='center', va='center')
        return ax
    
    # Grouping by additives
    grouped_bulk = plot_df.groupby('additive_type')['sigma_bulk_mS'].mean()
    grouped_gb = plot_df.groupby('additive_type')['sigma_gb_mS'].mean()
    
    additives = grouped_bulk.index.tolist()
    x = np.arange(len(additives))
    width = 0.35
    
    bulk_bars = ax.bar(x - width/2, grouped_bulk.values, width, label='σ bulk', color='#377EB8', edgecolor='black')
    gb_bars = ax.bar(x + width/2, grouped_gb.values, width, label='σ gb', color='#E41A1C', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(additives, rotation=45, ha='right')
    ax.set_ylabel(f'Conductivity at {temperature}°C (mS/cm)')
    ax.set_title(f'Bulk vs Grain Boundary Conductivity at {temperature}°C')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_gb_resistance_fraction(df_long, ax, temperature=600):
    """Plot 7: Grain boundary resistance fraction"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['gb_resistance_fraction'])
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No gb fraction data at {temperature}°C', ha='center', va='center')
        return ax
    
    # Grouping by additives
    grouped = plot_df.groupby('additive_type')['gb_resistance_fraction'].agg(['mean', 'std']).reset_index()
    grouped = grouped.sort_values('mean', ascending=False)
    
    bars = ax.bar(
        range(len(grouped)),
        grouped['mean'],
        yerr=grouped['std'],
        capsize=5,
        color=[SINTERING_ADDITIVE_COLORS.get(atype, SINTERING_ADDITIVE_COLORS['default']) for atype in grouped['additive_type']],
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped['additive_type'], rotation=45, ha='right')
    ax.set_ylabel(f'Grain Boundary Resistance Fraction at {temperature}°C')
    ax.set_title(f'GB Contribution to Total Resistance at {temperature}°C')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% GB contribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_conductivity_vs_grain_size(df_long, ax, temperature=600):
    """Plot 8: Conductivity vs grain size"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['grain_size_um', 'sigma_total_mS'])
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No grain size data at {temperature}°C', ha='center', va='center')
        return ax
    
    for additive in plot_df['additive_type'].unique():
        additive_data = plot_df[plot_df['additive_type'] == additive]
        color = SINTERING_ADDITIVE_COLORS.get(additive, SINTERING_ADDITIVE_COLORS['default'])
        marker = SINTERING_ADDITIVE_MARKERS.get(additive, SINTERING_ADDITIVE_MARKERS['default'])
        
        ax.scatter(
            additive_data['grain_size_um'],
            additive_data['sigma_total_mS'],
            color=color,
            marker=marker,
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=additive
        )
    
    ax.set_xlabel('Grain Size (μm)')
    ax.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
    ax.set_title(f'Conductivity vs Grain Size at {temperature}°C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_conductivity_vs_density(df_long, ax, temperature=600):
    """Plot 9: Conductivity vs density"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['density_percent', 'sigma_total_mS'])
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No density data at {temperature}°C', ha='center', va='center')
        return ax
    
    for additive in plot_df['additive_type'].unique():
        additive_data = plot_df[plot_df['additive_type'] == additive]
        color = SINTERING_ADDITIVE_COLORS.get(additive, SINTERING_ADDITIVE_COLORS['default'])
        marker = SINTERING_ADDITIVE_MARKERS.get(additive, SINTERING_ADDITIVE_MARKERS['default'])
        
        ax.scatter(
            additive_data['density_percent'],
            additive_data['sigma_total_mS'],
            color=color,
            marker=marker,
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=additive
        )
    
    ax.set_xlabel('Relative Density (%)')
    ax.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
    ax.set_title(f'Conductivity vs Density at {temperature}°C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_conductivity_heatmap(df_wide, ax, temperature=600):
    """Plot 10: Conductivity heatmap (B-cation vs additive)"""
    
    if 'B1_cation' not in df_wide.columns or 'additive_type' not in df_wide.columns:
        ax.text(0.5, 0.5, 'Required columns missing', ha='center', va='center')
        return ax
    
    # Create matrix
    pivot_data = []
    for _, row in df_wide.iterrows():
        b_cation = row.get('B1_cation')
        additive = row.get('additive_type')
        sigma_col = f'sigma_total_{temperature}C'
        if sigma_col in row and not pd.isna(row[sigma_col]) and b_cation and additive:
            pivot_data.append({
                'B_cation': b_cation,
                'Additive': additive,
                'sigma': row[sigma_col]
            })
    
    if len(pivot_data) == 0:
        ax.text(0.5, 0.5, f'No data at {temperature}°C', ha='center', va='center')
        return ax
    
    pivot_df = pd.DataFrame(pivot_data)
    heatmap_data = pivot_df.pivot(index='B_cation', columns='Additive', values='sigma')
    
    if len(heatmap_data) == 0:
        ax.text(0.5, 0.5, 'Cannot create heatmap', ha='center', va='center')
        return ax
    
    im = ax.imshow(heatmap_data.values, cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    ax.set_xlabel('Sintering Additive')
    ax.set_ylabel('B-cation')
    ax.set_title(f'Conductivity Heatmap at {temperature}°C (mS/cm)')
    
    # Add values in cells
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.values[i, j]
            if not pd.isna(value):
                text_color = 'white' if value > heatmap_data.values.max() / 2 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='σ (mS/cm)')
    return ax


def plot_activation_energy_comparison(df_long, ax):
    """Plot 11: Activation energy comparison for different additives"""
    
    # Get unique samples with calculated Ea
    samples = df_long[['sample_id', 'additive_type', 'Ea_calculated']].drop_duplicates().dropna(subset=['Ea_calculated'])
    
    # Exclude outliers if flag exists (but Ea is per sample, not per measurement)
    if len(samples) == 0:
        ax.text(0.5, 0.5, 'No Ea data available', ha='center', va='center')
        return ax
    
    # Grouping by additives
    grouped = samples.groupby('additive_type')['Ea_calculated'].agg(['mean', 'std', 'count']).reset_index()
    grouped = grouped.sort_values('mean', ascending=False)
    
    bars = ax.bar(
        range(len(grouped)),
        grouped['mean'],
        yerr=grouped['std'],
        capsize=5,
        color=[SINTERING_ADDITIVE_COLORS.get(atype, SINTERING_ADDITIVE_COLORS['default']) for atype in grouped['additive_type']],
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped['additive_type'], rotation=45, ha='right')
    ax.set_ylabel('Activation Energy (eV)')
    ax.set_title('Activation Energy Comparison by Additive Type')
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_t_sin_influence(df_long, ax, temperature=600):
    """Plot 12: Influence of sintering temperature on conductivity"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['T_sin', 'sigma_total_mS'])
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No sintering temperature data at {temperature}°C', ha='center', va='center')
        return ax
    
    for additive in plot_df['additive_type'].unique():
        additive_data = plot_df[plot_df['additive_type'] == additive]
        color = SINTERING_ADDITIVE_COLORS.get(additive, SINTERING_ADDITIVE_COLORS['default'])
        marker = SINTERING_ADDITIVE_MARKERS.get(additive, SINTERING_ADDITIVE_MARKERS['default'])
        
        ax.scatter(
            additive_data['T_sin'],
            additive_data['sigma_total_mS'],
            color=color,
            marker=marker,
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=additive
        )
    
    ax.set_xlabel('Sintering Temperature (°C)')
    ax.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
    ax.set_title(f'Effect of Sintering Temperature on Conductivity at {temperature}°C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_porosity_influence(df_long, ax, temperature=600):
    """Plot 13: Influence of porosity on conductivity"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['porosity', 'sigma_total_mS'])
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No porosity data at {temperature}°C', ha='center', va='center')
        return ax
    
    for additive in plot_df['additive_type'].unique():
        additive_data = plot_df[plot_df['additive_type'] == additive]
        color = SINTERING_ADDITIVE_COLORS.get(additive, SINTERING_ADDITIVE_COLORS['default'])
        marker = SINTERING_ADDITIVE_MARKERS.get(additive, SINTERING_ADDITIVE_MARKERS['default'])
        
        ax.scatter(
            additive_data['porosity'] * 100,  # in percent
            additive_data['sigma_total_mS'],
            color=color,
            marker=marker,
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=additive
        )
    
    ax.set_xlabel('Porosity (%)')
    ax.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
    ax.set_title(f'Effect of Porosity on Conductivity at {temperature}°C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_tolerance_factor_influence(df_long, ax, temperature=600):
    """Plot 14: Influence of tolerance factor on conductivity"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['tolerance_factor', 'sigma_total_mS'])
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No tolerance factor data at {temperature}°C', ha='center', va='center')
        return ax
    
    for additive in plot_df['additive_type'].unique():
        additive_data = plot_df[plot_df['additive_type'] == additive]
        color = SINTERING_ADDITIVE_COLORS.get(additive, SINTERING_ADDITIVE_COLORS['default'])
        marker = SINTERING_ADDITIVE_MARKERS.get(additive, SINTERING_ADDITIVE_MARKERS['default'])
        
        ax.scatter(
            additive_data['tolerance_factor'],
            additive_data['sigma_total_mS'],
            color=color,
            marker=marker,
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=additive
        )
    
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Ideal cubic (t=1)')
    ax.axvspan(0.96, 1.04, alpha=0.2, color='green', label='Optimal range')
    ax.set_xlabel('Tolerance Factor (t)')
    ax.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
    ax.set_title(f'Effect of Tolerance Factor on Conductivity at {temperature}°C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_oxygen_vacancy_influence(df_long, ax, temperature=600):
    """Plot 15: Influence of oxygen vacancy concentration on conductivity"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['oxygen_vacancy_conc', 'sigma_total_mS'])
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No oxygen vacancy data at {temperature}°C', ha='center', va='center')
        return ax
    
    for additive in plot_df['additive_type'].unique():
        additive_data = plot_df[plot_df['additive_type'] == additive]
        color = SINTERING_ADDITIVE_COLORS.get(additive, SINTERING_ADDITIVE_COLORS['default'])
        marker = SINTERING_ADDITIVE_MARKERS.get(additive, SINTERING_ADDITIVE_MARKERS['default'])
        
        ax.scatter(
            additive_data['oxygen_vacancy_conc'],
            additive_data['sigma_total_mS'],
            color=color,
            marker=marker,
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            label=additive
        )
    
    ax.set_xlabel('Oxygen Vacancy Concentration [V_O]')
    ax.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
    ax.set_title(f'Effect of Oxygen Vacancy Concentration on Conductivity at {temperature}°C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_correlation_matrix_conductivity(df_long, features, temperature=600):
    """Plot 16: Correlation matrix for conductivity parameters"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    available_features = [f for f in features if f in plot_df.columns]
    available_features.append('sigma_total_mS')
    
    plot_df = plot_df[available_features].dropna()
    
    if len(plot_df) < 5:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'Insufficient data for correlation matrix', ha='center', va='center')
        return fig
    
    corr_matrix = plot_df.corr(method='pearson')
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.set_title(f'Correlation Matrix for Conductivity at {temperature}°C')
    
    return fig


# ============================================================================
# NEW PLOTTING FUNCTIONS FOR ENHANCED ANALYSIS
# ============================================================================

def plot_shap_summary(shap_result, ax):
    """
    Plot SHAP summary plot.
    
    Parameters
    ----------
    shap_result : dict
        Result from shap_analysis function
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    if shap_result is None:
        ax.text(0.5, 0.5, 'Insufficient data for SHAP analysis', ha='center', va='center')
        return ax
    
    shap_values = shap_result['shap_values']
    X = shap_result['X']
    feature_names = shap_result['feature_names']
    
    # Create SHAP summary plot
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    
    # Get current figure and transfer to ax
    fig = plt.gcf()
    # This is a bit tricky - for simplicity, we'll use a different approach
    # Clear ax and use shap's plotting
    ax.clear()
    
    # Alternative: manually create a bar plot of mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    
    ax.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Feature Importance (SHAP)')
    ax.invert_yaxis()
    
    return ax


def plot_polynomial_fit(df_long, x_col, y_col, ax, degree=2, temperature=600):
    """
    Plot polynomial fit for non-linear relationships.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    x_col : str
        Independent variable column
    y_col : str
        Dependent variable column (usually 'sigma_total_mS')
    ax : matplotlib.axes.Axes
        Axes to plot on
    degree : int
        Polynomial degree
    temperature : int
        Temperature in Celsius
    """
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=[x_col, y_col])
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if len(plot_df) < 5:
        ax.text(0.5, 0.5, f'Insufficient data for {x_col}', ha='center', va='center')
        return ax
    
    # Plot original data points by additive type
    for additive in plot_df['additive_type'].unique():
        additive_data = plot_df[plot_df['additive_type'] == additive]
        color = SINTERING_ADDITIVE_COLORS.get(additive, SINTERING_ADDITIVE_COLORS['default'])
        marker = SINTERING_ADDITIVE_MARKERS.get(additive, SINTERING_ADDITIVE_MARKERS['default'])
        
        ax.scatter(
            additive_data[x_col],
            additive_data[y_col],
            color=color,
            marker=marker,
            s=60,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            label=additive
        )
    
    # Perform polynomial regression on all data
    poly_result = polynomial_regression_analysis(plot_df, x_col, y_col, degree)
    
    if poly_result['model'] is not None:
        ax.plot(poly_result['x_pred'], poly_result['y_pred'], 
               'k-', linewidth=2, alpha=0.7, 
               label=f'Polynomial fit (R² = {poly_result["r2"]:.3f})')
    
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
    ax.set_title(f'{x_col.replace("_", " ").title()} vs Conductivity')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_partial_correlations(df_long, features, target, control_vars, ax, temperature=600):
    """
    Plot partial correlations bar chart.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    features : list
        Feature columns
    target : str
        Target column
    control_vars : list
        Control variables
    ax : matplotlib.axes.Axes
        Axes to plot on
    temperature : int
        Temperature in Celsius
    """
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    # Calculate partial correlations
    partial_df = partial_correlation_analysis(plot_df, target, features, control_vars)
    
    if len(partial_df) == 0:
        ax.text(0.5, 0.5, 'Insufficient data for partial correlation', ha='center', va='center')
        return ax
    
    partial_df = partial_df.sort_values('partial_correlation', ascending=False)
    
    colors = ['#4DAF4A' if corr > 0 else '#E41A1C' for corr in partial_df['partial_correlation']]
    
    bars = ax.barh(range(len(partial_df)), partial_df['partial_correlation'], color=colors, edgecolor='black')
    
    ax.set_yticks(range(len(partial_df)))
    ax.set_yticklabels(partial_df['feature'])
    ax.set_xlabel(f'Partial Correlation with {target}')
    ax.set_title(f'Partial Correlations (controlling for {", ".join(control_vars[:2])})')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add p-value annotations
    for i, (_, row) in enumerate(partial_df.iterrows()):
        if row['p_value'] < 0.05:
            ax.text(row['partial_correlation'] + 0.02, i, '*', fontsize=12, va='center')
    
    ax.invert_yaxis()
    return ax


def plot_clustering_results(df_long, feature_columns, ax, eps=0.5, min_samples=3):
    """
    Plot clustering results using PCA for visualization.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    feature_columns : list
        Features to use for clustering
    ax : matplotlib.axes.Axes
        Axes to plot on
    eps : float
        DBSCAN epsilon
    min_samples : int
        DBSCAN min_samples
    """
    # Aggregate by sample
    agg_df = df_long.groupby('sample_id')[feature_columns].mean().dropna()
    
    if len(agg_df) < 3:
        ax.text(0.5, 0.5, 'Insufficient data for clustering', ha='center', va='center')
        return ax
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg_df)
    
    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = clustering.fit_predict(X_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Get additive types for coloring
    additive_map = df_long.groupby('sample_id')['additive_type'].first()
    additive_colors = [SINTERING_ADDITIVE_COLORS.get(additive_map.get(idx, 'Pure'), SINTERING_ADDITIVE_COLORS['default']) 
                      for idx in agg_df.index]
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    
    # Add labels for each point
    for i, idx in enumerate(agg_df.index):
        additive = additive_map.get(idx, '')
        ax.annotate(f'{additive}', (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.7)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
    
    # Add colorbar for clusters
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster')
    
    return ax


def plot_correlation_by_temperature(df_long, feature, target='sigma_total_mS', ax=None):
    """
    Plot correlation between feature and target as function of temperature.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    feature : str
        Feature column name
    target : str
        Target column name (default: 'sigma_total_mS')
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    temperatures = sorted(df_long['temperature_C'].unique())
    
    correlations = []
    p_values = []
    
    for T in temperatures:
        temp_df = df_long[df_long['temperature_C'] == T].dropna(subset=[feature, target])
        
        # Exclude outliers
        if 'is_outlier' in temp_df.columns:
            temp_df = temp_df[~temp_df['is_outlier']]
        
        if len(temp_df) > 3:
            corr, p_val = spearmanr(temp_df[feature], temp_df[target])
            correlations.append(corr)
            p_values.append(p_val)
        else:
            correlations.append(np.nan)
            p_values.append(np.nan)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot correlation
    ax.plot(temperatures, correlations, 'o-', color='#377EB8', linewidth=2, markersize=6)
    
    # Add significance indicators
    for i, (T, p_val) in enumerate(zip(temperatures, p_values)):
        if p_val < 0.05:
            ax.scatter(T, correlations[i], s=150, facecolors='none', edgecolors='red', linewidth=2, zorder=5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.fill_between(temperatures, -0.2, 0.2, alpha=0.1, color='gray')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel(f'Spearman Correlation: {feature} vs {target}')
    ax.set_title(f'Temperature-Dependent Correlation: {feature}')
    ax.grid(True, alpha=0.3)
    
    # Add annotation for significant points
    if any(p < 0.05 for p in p_values if not np.isnan(p)):
        ax.annotate('* p < 0.05', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=8)
    
    return ax


# ============================================================================
# INSIGHTS GENERATION (ENHANCED)
# ============================================================================

def generate_conductivity_insights(df_long):
    """
    Automatic generation of physical insights for conductivity.
    
    Enhanced version with additional insights:
    - Tolerance factor optimal range analysis
    - Grain size effect quantification
    - Density threshold analysis
    - Additive-specific recommendations
    - Temperature-dependent insights
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
                            insights.append(f"**{additive} additive** shows {improvement:.0f}% higher conductivity than Pure at 600°C.")
                        elif improvement < -50:
                            insights.append(f"**{additive} additive** shows {abs(improvement):.0f}% lower conductivity than Pure at 600°C.")
                        elif improvement > 10:
                            insights.append(f"**{additive} additive** shows moderate improvement ({improvement:.0f}%) at 600°C.")
    
    # Optimal additive concentration
    for additive in df_long['additive_type'].unique():
        if additive != 'Pure':
            additive_data = df_long[df_long['additive_type'] == additive].copy()
            additive_data = additive_data.dropna(subset=['additive_concentration_wt', 'sigma_total_mS'])
            
            # Exclude outliers
            if 'is_outlier' in additive_data.columns:
                additive_data = additive_data[~additive_data['is_outlier']]
            
            if len(additive_data) > 3:
                # Group by concentration
                grouped = additive_data.groupby('additive_concentration_wt')['sigma_total_mS'].mean().reset_index()
                if len(grouped) > 1:
                    max_conc = grouped.loc[grouped['sigma_total_mS'].idxmax(), 'additive_concentration_wt']
                    insights.append(f"Optimal concentration for **{additive} additive** appears around {max_conc:.2f} wt%.")
    
    # Grain size effect
    df_grain = df_long.dropna(subset=['grain_size_um', 'sigma_total_mS'])
    
    # Exclude outliers
    if 'is_outlier' in df_grain.columns:
        df_grain = df_grain[~df_grain['is_outlier']]
    
    if len(df_grain) > 10:
        corr, p_val = stats.spearmanr(df_grain['grain_size_um'], df_grain['sigma_total_mS'])
        if p_val < 0.05:
            if corr > 0.5:
                insights.append(f"**Strong positive correlation** between grain size and conductivity (ρ = {corr:.2f}, p < 0.05). Larger grains improve conductivity.")
            elif corr < -0.5:
                insights.append(f"**Strong negative correlation** between grain size and conductivity (ρ = {corr:.2f}, p < 0.05). Smaller grains improve conductivity.")
            elif corr > 0.2:
                insights.append(f"**Weak positive correlation** between grain size and conductivity (ρ = {corr:.2f}).")
    
    # Density effect
    df_density = df_long.dropna(subset=['density_percent', 'sigma_total_mS'])
    
    # Exclude outliers
    if 'is_outlier' in df_density.columns:
        df_density = df_density[~df_density['is_outlier']]
    
    if len(df_density) > 10:
        corr, p_val = stats.spearmanr(df_density['density_percent'], df_density['sigma_total_mS'])
        if p_val < 0.05 and corr > 0.3:
            insights.append(f"**Positive correlation** between density and conductivity (ρ = {corr:.2f}, p < 0.05). Higher density improves conductivity.")
        
        # Density threshold analysis
        high_density = df_density[df_density['density_percent'] >= 95]['sigma_total_mS'].mean()
        low_density = df_density[df_density['density_percent'] < 95]['sigma_total_mS'].mean()
        if not pd.isna(high_density) and not pd.isna(low_density) and high_density > low_density:
            ratio = high_density / low_density if low_density > 0 else float('inf')
            insights.append(f"Samples with density >95% have **{ratio:.1f}x higher** conductivity than less dense samples.")
    
    # Tolerance factor analysis
    df_t = df_long.dropna(subset=['tolerance_factor', 'sigma_total_mS'])
    
    # Exclude outliers
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
                insights.append(f"Systems with tolerance factor in [0.96-1.04] have **{ratio:.1f}x higher** average conductivity.")
        
        # Distortion effect
        df_t['distortion'] = abs(df_t['tolerance_factor'] - 1.0)
        corr, p_val = stats.spearmanr(df_t['distortion'], df_t['sigma_total_mS'])
        if p_val < 0.05 and corr < -0.3:
            insights.append(f"**Negative correlation** between lattice distortion and conductivity (ρ = {corr:.2f}). More cubic structures perform better.")
    
    # Oxygen vacancy concentration (non-linear analysis)
    df_vac = df_long.dropna(subset=['oxygen_vacancy_conc', 'sigma_total_mS'])
    
    # Exclude outliers
    if 'is_outlier' in df_vac.columns:
        df_vac = df_vac[~df_vac['is_outlier']]
    
    if len(df_vac) > 15:
        # Check for optimal range (0.05-0.15)
        optimal = df_vac[(df_vac['oxygen_vacancy_conc'] >= 0.05) & (df_vac['oxygen_vacancy_conc'] <= 0.15)]
        low = df_vac[df_vac['oxygen_vacancy_conc'] < 0.05]
        high = df_vac[df_vac['oxygen_vacancy_conc'] > 0.15]
        
        if len(optimal) > 0 and len(low) > 0:
            opt_mean = optimal['sigma_total_mS'].mean()
            low_mean = low['sigma_total_mS'].mean()
            if opt_mean > low_mean:
                insights.append(f"Optimal oxygen vacancy concentration appears in **0.05-0.15 range** (vs low vacancy samples).")
        
        if len(optimal) > 0 and len(high) > 0:
            opt_mean = optimal['sigma_total_mS'].mean()
            high_mean = high['sigma_total_mS'].mean()
            if opt_mean > high_mean:
                insights.append(f"**Excessive oxygen vacancies (>0.15)** may reduce conductivity due to trapping effects.")
    
    # Grain boundary contribution
    df_gb = df_long.dropna(subset=['gb_resistance_fraction'])
    
    # Exclude outliers
    if 'is_outlier' in df_gb.columns:
        df_gb = df_gb[~df_gb['is_outlier']]
    
    if len(df_gb) > 5:
        mean_gb = df_gb['gb_resistance_fraction'].mean()
        if mean_gb > 0.5:
            insights.append(f"**Grain boundaries dominate** the total resistance ({mean_gb:.1%} on average). Improving grain boundary conductivity is critical.")
        else:
            insights.append(f"**Bulk conductivity dominates** the total resistance ({1-mean_gb:.1%} on average).")
        
        # Temperature-dependent GB analysis
        gb_by_temp = df_gb.groupby('temperature_C')['gb_resistance_fraction'].mean()
        if len(gb_by_temp) > 3:
            temp_range = gb_by_temp.index.max() - gb_by_temp.index.min()
            if temp_range > 300:
                low_temp = gb_by_temp[gb_by_temp.index < 400].mean() if any(gb_by_temp.index < 400) else None
                high_temp = gb_by_temp[gb_by_temp.index > 700].mean() if any(gb_by_temp.index > 700) else None
                if low_temp is not None and high_temp is not None:
                    if low_temp > high_temp:
                        insights.append(f"Grain boundary contribution **decreases with temperature** (from {low_temp:.2f} at <400°C to {high_temp:.2f} at >700°C).")
    
    # Additive incorporation analysis
    if 'additive_incorporation_likely' in df_long.columns:
        incorp_true = df_long[df_long['additive_incorporation_likely'] == True]['sigma_total_mS'].mean()
        incorp_false = df_long[df_long['additive_incorporation_likely'] == False]['sigma_total_mS'].mean()
        if not pd.isna(incorp_true) and not pd.isna(incorp_false):
            if incorp_true > incorp_false:
                insights.append(f"Additives that **incorporate into the lattice** (Zn, Co) show higher average conductivity than segregating additives (Cu, Ni).")
    
    # Radius mismatch effect
    df_mismatch = df_long.dropna(subset=['radius_mismatch', 'sigma_total_mS'])
    
    # Exclude outliers
    if 'is_outlier' in df_mismatch.columns:
        df_mismatch = df_mismatch[~df_mismatch['is_outlier']]
    
    if len(df_mismatch) > 10:
        corr, p_val = stats.spearmanr(df_mismatch['radius_mismatch'], df_mismatch['sigma_total_mS'])
        if p_val < 0.05 and corr < -0.3:
            insights.append(f"**Larger B-site radius mismatch** correlates with lower conductivity (ρ = {corr:.2f}). Compositional homogeneity is important.")
    
    if len(insights) == 0:
        insights.append("Insufficient data for automated insights. Add more data points to enable pattern detection.")
    
    return insights


# ============================================================================
# ML FUNCTIONS FOR CONDUCTIVITY ANALYSIS (ENHANCED with SHAP)
# ============================================================================

@st.cache_data
def feature_importance_conductivity(df_long, selected_features=None, target='sigma_total_mS', temperature=600):
    """
    Random Forest feature importance analysis for conductivity.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    selected_features : list
        Features to include
    target : str
        Target column
    temperature : int
        Temperature in Celsius
    
    Returns
    -------
    tuple
        (importance_df, r2) or (None, None) if insufficient data
    """
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if selected_features is None:
        selected_features = ['density_percent', 'grain_size_um', 'tolerance_factor', 
                            'oxygen_vacancy_conc', 'additive_concentration_wt', 'radius_mismatch']
    
    available_features = [f for f in selected_features if f in plot_df.columns]
    
    if len(available_features) < 2:
        return None, None
    
    plot_df = plot_df.dropna(subset=[target] + available_features)
    
    if len(plot_df) < 10:
        return None, None
    
    X = plot_df[available_features].copy()
    
    # Add one-hot encoding for additive type
    if 'additive_type' in plot_df.columns:
        X = pd.concat([X, pd.get_dummies(plot_df['additive_type'], prefix='additive')], axis=1)
    
    y = plot_df[target]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    r2 = rf.score(X, y)
    
    return importance_df, r2


@st.cache_data
def compare_ml_models_conductivity(df_long, selected_features=None, target='sigma_total_mS', temperature=600):
    """
    Compare multiple ML models for conductivity prediction.
    
    Parameters
    ----------
    df_long : pandas.DataFrame
        Long format data
    selected_features : list
        Features to include
    target : str
        Target column
    temperature : int
        Temperature in Celsius
    
    Returns
    -------
    tuple
        (results_df, models, X, y) or (None, None, None, None) if insufficient data
    """
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    # Exclude outliers
    if 'is_outlier' in plot_df.columns:
        plot_df = plot_df[~plot_df['is_outlier']]
    
    if selected_features is None:
        selected_features = ['density_percent', 'grain_size_um', 'tolerance_factor', 
                            'oxygen_vacancy_conc', 'additive_concentration_wt', 'radius_mismatch']
    
    available_features = [f for f in selected_features if f in plot_df.columns]
    
    if len(available_features) < 2:
        return None, None, None, None
    
    plot_df = plot_df.dropna(subset=[target] + available_features)
    
    if len(plot_df) < 10:
        return None, None, None, None
    
    X = plot_df[available_features].copy()
    
    if 'additive_type' in plot_df.columns:
        X = pd.concat([X, pd.get_dummies(plot_df['additive_type'], prefix='additive')], axis=1)
    
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


# ============================================================================
# MAIN STREAMLIT APPLICATION
# ============================================================================
def main():
    st.set_page_config(
        page_title="Proton-Conducting Perovskite Analyzer",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Proton-Conducting Perovskite: Sintering Additive Analysis")
    st.markdown("Analyzing the effect of sintering additives (Cu, Ni, Zn, Co) on total, bulk, and grain boundary conductivity")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Excel file", 
            type=['xlsx', 'xls'],
            help="Upload your data file with perovskite compositions and conductivity measurements"
        )
        
        if uploaded_file is None:
            st.info("Please upload a data file to begin")
            st.markdown("""
            ### Expected format:
            - **A cation**: A-site element (Ba)
            - **B1 cation**: Main B-site element (Zr, Ce, Sn)
            - **B2 cation**: Second B-site element (Ce)
            - **B2_cont**: Content of B2 cation
            - **dopant**: Acceptor dopant (Y, Gd, Sm)
            - **dop_cont**: Dopant concentration
            - **Сд**: Sintering additive (Pure, Cu, Ni, Zn, Co)
            - **x, wt%**: Additive concentration
            - **Method**: Synthesis method
            - **T sin**: Sintering temperature (°C)
            - **ρ, %**: Relative density
            - **d, mkm**: Grain size (μm)
            - **σ total, mS**: Total conductivity at 200-900°C
            - **σ bulk, mS**: Bulk conductivity (optional)
            - **σ gb, mS**: Grain boundary conductivity (optional)
            - **Атмосфера**: Measurement atmosphere
            - **Влажность**: Humidity condition
            - **ссылка**: DOI reference
            """)
            return
        
        st.markdown("---")
        st.header("📊 Analysis Settings")
        
        # Temperature selection for analysis
        temperature_analysis = st.slider(
            "Reference Temperature for Analysis (°C)",
            min_value=200,
            max_value=900,
            value=600,
            step=50,
            help="Most important temperature range for proton ceramics is 500-700°C"
        )
        
        st.markdown("---")
        st.header("🔍 Filters")
        
        # ИНИЦИАЛИЗИРУЕМ ПЕРЕМЕННЫЕ ДО TRY (ВАЖНО!)
        df_long = pd.DataFrame()
        df_wide = pd.DataFrame()
        filtered_long = pd.DataFrame()
        filtered_wide = pd.DataFrame()
        selected_additives = []
        selected_b_sites = []
        selected_humidity = []
        selected_atmosphere = []
        
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            st.success(f"✅ File loaded: {len(df)} rows")
            
            # Process data without progress_bar in cache
            with st.spinner("🔄 Processing conductivity data... Please wait."):
                df_long, df_wide = process_conductivity_data(df)
            
            st.success(f"✅ Processed: {len(df_long)} measurements")
            
            # Create filters in sidebar
            if 'additive_type' in df_long.columns:
                selected_additives = st.multiselect(
                    "Sintering Additives",
                    options=sorted(df_long['additive_type'].unique()),
                    default=sorted(df_long['additive_type'].unique())
                )
            
            if 'B1_cation' in df_long.columns:
                selected_b_sites = st.multiselect(
                    "B-site cations",
                    options=sorted(df_long['B1_cation'].dropna().unique()),
                    default=sorted(df_long['B1_cation'].dropna().unique())
                )
            
            if 'humidity' in df_long.columns:
                selected_humidity = st.multiselect(
                    "Humidity conditions",
                    options=sorted(df_long['humidity'].dropna().unique()),
                    default=sorted(df_long['humidity'].dropna().unique())
                )
            
            if 'atmosphere' in df_long.columns:
                selected_atmosphere = st.multiselect(
                    "Atmosphere",
                    options=sorted(df_long['atmosphere'].dropna().unique()),
                    default=sorted(df_long['atmosphere'].dropna().unique())
                )
            
            # Apply filters
            filtered_long = df_long.copy()
            
            if selected_additives:
                filtered_long = filtered_long[filtered_long['additive_type'].isin(selected_additives)]
            if selected_b_sites:
                filtered_long = filtered_long[filtered_long['B1_cation'].isin(selected_b_sites)]
            if selected_humidity:
                filtered_long = filtered_long[filtered_long['humidity'].isin(selected_humidity)]
            if selected_atmosphere:
                filtered_long = filtered_long[filtered_long['atmosphere'].isin(selected_atmosphere)]
            
            filtered_wide = df_wide.copy()
            if selected_additives and len(filtered_wide) > 0 and 'additive_type' in filtered_wide.columns:
                filtered_wide = filtered_wide[filtered_wide['additive_type'].isin(selected_additives)]
            if selected_b_sites and len(filtered_wide) > 0 and 'B1_cation' in filtered_wide.columns:
                filtered_wide = filtered_wide[filtered_wide['B1_cation'].isin(selected_b_sites)]
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.exception(e)  # Показывает полную ошибку для отладки
            return
    
    # ============================================================================
    # MAIN DISPLAY AREA
    # ============================================================================
    
    if uploaded_file is not None and len(filtered_long) > 0:
        # Display data information
        st.subheader("📈 Data Overview")
        
        # Calculate outlier count
        outlier_count = filtered_long['is_outlier'].sum() if 'is_outlier' in filtered_long.columns else 0
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
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
            temp_min = filtered_long['temperature_C'].min() if 'temperature_C' in filtered_long.columns else 0
            temp_max = filtered_long['temperature_C'].max() if 'temperature_C' in filtered_long.columns else 0
            st.metric("Temp range", f"{temp_min}-{temp_max}°C")
        with col6:
            st.metric("Outliers detected", outlier_count)
        
        if outlier_count > 0:
            st.warning(f"⚠️ {outlier_count} outlier measurements detected and will be excluded from analysis.")
        
        st.markdown("---")
        
        # Tabs (оставляем как в оригинале, без изменений)
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Conductivity Overview",
            "🔬 Additive Analysis",
            "⚡ Bulk vs GB Analysis",
            "📈 Microstructure Effects",
            "🧪 Compositional Effects",
            "🤖 ML & Feature Importance",
            "📐 Advanced Analysis",
            "💡 Insights & Data"
        ])
        
        # ====================================================================
        # TAB 1: CONDUCTIVITY OVERVIEW (оригинал, без изменений)
        # ====================================================================
        with tab1:
            st.subheader("Conductivity vs Temperature")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_conductivity_vs_temperature(filtered_long, ax, selected_additives, selected_b_sites, 200, 900)
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_arrhenius(filtered_long, ax, selected_additives, selected_b_sites)
                st.pyplot(fig)
                plt.close(fig)
            
            st.subheader(f"Conductivity at {temperature_analysis}°C")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_additive_comparison_bar(filtered_long, ax, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_pure_vs_additive_comparison(filtered_long, ax, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
            
            # Heatmap
            if len(filtered_wide) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_conductivity_heatmap(filtered_wide, ax, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
        
        # ====================================================================
        # TAB 2: ADDITIVE ANALYSIS (оригинал)
        # ====================================================================
        with tab2:
            st.subheader("Effect of Additive Concentration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_conductivity_vs_additive_concentration(filtered_long, ax, temperature_analysis, selected_b_sites)
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_activation_energy_comparison(filtered_long, ax)
                st.pyplot(fig)
                plt.close(fig)
            
            st.subheader("Influence of Sintering Temperature")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_t_sin_influence(filtered_long, ax, temperature_analysis)
            st.pyplot(fig)
            plt.close(fig)
        
        # ====================================================================
        # TAB 3: BULK VS GB ANALYSIS (оригинал)
        # ====================================================================
        with tab3:
            st.subheader("Bulk vs Grain Boundary Conductivity")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_bulk_vs_gb_contribution(filtered_long, ax, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_gb_resistance_fraction(filtered_long, ax, temperature_analysis)
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
                          c='blue', s=80, alpha=0.7, edgecolors='black')
                
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
        # TAB 4: MICROSTRUCTURE EFFECTS (оригинал)
        # ====================================================================
        with tab4:
            st.subheader("Microstructure Effects on Conductivity")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_conductivity_vs_grain_size(filtered_long, ax, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_conductivity_vs_density(filtered_long, ax, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_porosity_influence(filtered_long, ax, temperature_analysis)
            st.pyplot(fig)
            plt.close(fig)
        
        # ====================================================================
        # TAB 5: COMPOSITIONAL EFFECTS (оригинал)
        # ====================================================================
        with tab5:
            st.subheader("Compositional Effects on Conductivity")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_tolerance_factor_influence(filtered_long, ax, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_oxygen_vacancy_influence(filtered_long, ax, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
            
            # Radius mismatch effect
            st.subheader("B-site Radius Mismatch Effect")
            if 'radius_mismatch' in filtered_long.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_polynomial_fit(filtered_long, 'radius_mismatch', 'sigma_total_mS', ax, degree=2, temperature=temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Radius mismatch data not available")
            
            # Correlation matrix
            st.subheader("Correlation Matrix")
            features = ['density_percent', 'grain_size_um', 'tolerance_factor', 
                       'oxygen_vacancy_conc', 'additive_concentration_wt', 'radius_mismatch',
                       'lattice_distortion_index', 'electronegativity_difference_B_O']
            # Фильтруем только существующие колонки
            available_features = [f for f in features if f in filtered_long.columns]
            if len(available_features) > 1:
                fig = plot_correlation_matrix_conductivity(filtered_long, available_features, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Insufficient features for correlation matrix")
        
        # ====================================================================
        # TAB 6: ML & FEATURE IMPORTANCE (оригинал, но с проверками)
        # ====================================================================
        with tab6:
            st.subheader("Machine Learning Analysis")
            
            # Доступные фичи для ML
            available_ml_features = [f for f in ['density_percent', 'grain_size_um', 'tolerance_factor', 
                        'oxygen_vacancy_conc', 'additive_concentration_wt', 'radius_mismatch',
                        'lattice_distortion_index'] if f in filtered_long.columns]
            
            if len(available_ml_features) > 0:
                ml_features = st.multiselect(
                    "Select features for ML analysis",
                    options=available_ml_features,
                    default=available_ml_features[:min(4, len(available_ml_features))]
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Feature Importance (Random Forest)**")
                    importance_df, r2 = feature_importance_conductivity(
                        filtered_long, ml_features, 'sigma_total_mS', temperature_analysis
                    )
                    
                    if importance_df is not None:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        importance_df = importance_df.head(10)
                        ax.barh(range(len(importance_df)), importance_df['importance'], 
                               color='steelblue', edgecolor='black')
                        ax.set_yticks(range(len(importance_df)))
                        ax.set_yticklabels(importance_df['feature'])
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
                    models_df, models, X, y = compare_ml_models_conductivity(
                        filtered_long, ml_features, 'sigma_total_mS', temperature_analysis
                    )
                    
                    if models_df is not None:
                        st.dataframe(models_df, use_container_width=True)
                    else:
                        st.warning("Insufficient data for model comparison")
                
                # SHAP Analysis
                st.subheader("🔬 SHAP Analysis (Model Interpretability)")
                
                if len(ml_features) >= 2 and models_df is not None:
                    shap_result = shap_analysis(
                        filtered_long[filtered_long['temperature_C'] == temperature_analysis],
                        ml_features, 'sigma_total_mS', model_type='xgboost'
                    )
                    
                    if shap_result is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**SHAP Feature Importance**")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            plot_shap_summary(shap_result, ax)
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        with col2:
                            st.markdown("**SHAP Dependence Plot**")
                            feature_for_dependence = st.selectbox(
                                "Select feature for SHAP dependence plot",
                                options=ml_features
                            )
                            
                            if feature_for_dependence in shap_result['feature_names']:
                                fig, ax = plt.subplots(figsize=(8, 6))
                                idx = shap_result['feature_names'].index(feature_for_dependence)
                                shap_values = shap_result['shap_values']
                                X_data = shap_result['X']
                                
                                ax.scatter(X_data[:, idx], shap_values[:, idx], 
                                          c='steelblue', alpha=0.6, edgecolors='black')
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
        # TAB 7: ADVANCED ANALYSIS (оригинал, но с проверками)
        # ====================================================================
        with tab7:
            st.subheader("Advanced Statistical Analysis")
            
            # Partial Correlation Analysis
            st.markdown("### Partial Correlation Analysis")
            st.markdown("*Controlling for density and grain size effects*")
            
            control_options = [f for f in ['density_percent', 'grain_size_um', 'porosity'] if f in filtered_long.columns]
            
            if len(control_options) > 0:
                control_vars = st.multiselect(
                    "Select control variables",
                    options=control_options,
                    default=control_options[:min(2, len(control_options))]
                )
                
                if len(control_vars) > 0:
                    features_for_partial = [f for f in ['tolerance_factor', 'oxygen_vacancy_conc', 
                                            'additive_concentration_wt', 'radius_mismatch'] if f in filtered_long.columns]
                    
                    if len(features_for_partial) > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plot_partial_correlations(filtered_long, features_for_partial, 'sigma_total_mS', 
                                                 control_vars, ax, temperature_analysis)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("No features available for partial correlation analysis")
                else:
                    st.info("Select at least one control variable")
            else:
                st.info("No control variables available")
            
            # Temperature-dependent correlation
            st.markdown("### Temperature-Dependent Correlations")
            
            temp_corr_options = [f for f in ['tolerance_factor', 'oxygen_vacancy_conc', 'radius_mismatch', 
                        'grain_size_um', 'density_percent'] if f in filtered_long.columns]
            
            if len(temp_corr_options) > 0:
                temp_corr_feature = st.selectbox(
                    "Select feature to analyze temperature-dependent correlation",
                    options=temp_corr_options,
                    help="Shows how correlation between this feature and conductivity changes with temperature"
                )
                
                if temp_corr_feature:
                    fig = plot_correlation_by_temperature(filtered_long, temp_corr_feature, 'sigma_total_mS')
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("No features available for temperature-dependent correlation")
            
            # Polynomial regression
            st.markdown("### Non-linear Relationship Analysis")
            
            poly_options = [f for f in ['oxygen_vacancy_conc', 'additive_concentration_wt', 'grain_size_um', 'radius_mismatch'] if f in filtered_long.columns]
            
            if len(poly_options) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    poly_feature = st.selectbox(
                        "Select X variable",
                        options=poly_options,
                        key='poly_x'
                    )
                
                with col2:
                    poly_degree = st.slider("Polynomial degree", min_value=2, max_value=4, value=2)
                
                if poly_feature:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_polynomial_fit(filtered_long, poly_feature, 'sigma_total_mS', ax, degree=poly_degree, temperature=temperature_analysis)
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("No features available for polynomial regression")
            
            # Clustering analysis
            st.markdown("### Material Clustering Analysis")
            
            cluster_options = [f for f in ['tolerance_factor', 'oxygen_vacancy_conc', 'density_percent', 
                        'grain_size_um', 'additive_concentration_wt'] if f in filtered_long.columns]
            
            if len(cluster_options) >= 2:
                cluster_features = st.multiselect(
                    "Select features for clustering",
                    options=cluster_options,
                    default=cluster_options[:min(2, len(cluster_options))]
                )
                
                if len(cluster_features) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        eps = st.slider("DBSCAN eps (neighborhood radius)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
                    
                    with col2:
                        min_samples = st.slider("DBSCAN min_samples", min_value=2, max_value=10, value=3)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    plot_clustering_results(filtered_long, cluster_features, ax, eps=eps, min_samples=min_samples)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Select at least 2 features for clustering analysis")
            else:
                st.info("Need at least 2 features for clustering analysis")
        
        # ====================================================================
        # TAB 8: INSIGHTS & DATA (оригинал)
        # ====================================================================
        with tab8:
            st.subheader("💡 Automated Physical Insights")
            
            insights = generate_conductivity_insights(filtered_long)
            for insight in insights:
                st.info(insight)
            
            st.markdown("---")
            st.subheader("📋 Processed Data")
            
            # Show data in long format
            display_cols = ['sample_id', 'B1_cation', 'dopant', 'additive_type', 
                           'additive_concentration_wt', 'additive_incorporation_likely',
                           'temperature_C', 'sigma_total_mS', 'sigma_bulk_mS', 'sigma_gb_mS',
                           'density_percent', 'grain_size_um', 'tolerance_factor', 
                           'radius_mismatch', 'oxygen_vacancy_conc', 'Ea_calculated', 'is_outlier']
            available_cols = [col for col in display_cols if col in filtered_long.columns]
            
            st.dataframe(filtered_long[available_cols].head(100), use_container_width=True)
            
            # Export data
            csv = filtered_long.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download processed data as CSV",
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
            'dopant': ['Y', 'Y', 'Gd'],
            'dop_cont': [0.2, 0.2, 0.1],
            'Сд': ['Pure', 'Cu', 'Ni'],
            'x, wt%': [0, 1.36, 1],
            'Method': ['solid-state', 'solid-state', 'solid-state'],
            'T sin': [1550, 1550, 1450],
            'ρ, %': ['', '', 95.7],
            'd, mkm': ['', '', 15.3],
            'σ total, mS 600': [3.41, 7.40, 12.76],
            'Атмосфера': ['Ox', 'Ox', 'Ox'],
            'Влажность': ['wet', 'wet', 'dry'],
            'ссылка': ['10.1016/j.ceramint.2022.01.039', 
                      '10.1016/j.ceramint.2022.01.039',
                      '10.1016/j.ijhydene.2022.07.237']
        })
        
        st.dataframe(example_data, use_container_width=True)
        
        st.markdown("""
        ### Key Features of This Analyzer:
        
        #### 🔬 **Conductivity Analysis**
        - Temperature-dependent conductivity (200-900°C)
        - Arrhenius plots and activation energy calculation
        - Comparison of different sintering additives
        
        #### ⚡ **Bulk vs Grain Boundary**
        - Separation of bulk and GB contributions
        - Verification of mixing rule: 1/σ_total = 1/σ_bulk + 1/σ_gb
        - GB resistance fraction analysis
        
        #### 📈 **Microstructure Effects**
        - Grain size effect on conductivity
        - Density and porosity analysis
        - Sintering temperature influence
        
        #### 🧪 **Compositional Effects**
        - Tolerance factor analysis with optimal range [0.96-1.04]
        - Oxygen vacancy concentration effects
        - Additive concentration optimization
        - B-site radius mismatch analysis
        """)

if __name__ == "__main__":
    main()


# ============================================================================
# UNIT TESTS (for development and debugging)
# ============================================================================

# Unit tests for key functions
import sys

def test_safe_float_converter():
    """Test safe_float_converter function"""
    assert safe_float_converter(123) == 123.0
    assert safe_float_converter("123") == 123.0
    assert safe_float_converter("123.45") == 123.45
    assert safe_float_converter("123,45") == 123.45
    assert safe_float_converter("123%") == 123.0
    assert safe_float_converter("") is None
    assert safe_float_converter(None) is None
    assert safe_float_converter("abc") is None
    print("✓ test_safe_float_converter passed")

def test_flexible_column_mapper():
    """Test FlexibleColumnMapper"""
    mapper = FlexibleColumnMapper()
    
    # Test pattern matching
    assert any(re.search(pattern, "σ total, mS", re.IGNORECASE) for pattern in mapper.patterns['sigma_total'])
    assert any(re.search(pattern, "sigma_total_mS", re.IGNORECASE) for pattern in mapper.patterns['sigma_total'])
    assert any(re.search(pattern, "σ bulk", re.IGNORECASE) for pattern in mapper.patterns['sigma_bulk'])
    assert any(re.search(pattern, "σ gb", re.IGNORECASE) for pattern in mapper.patterns['sigma_gb'])
    
    print("✓ test_flexible_column_mapper passed")

def test_polynomial_regression():
    """Test polynomial_regression_analysis"""
    # Create synthetic data
    np.random.seed(42)
    x = np.linspace(0, 1, 20)
    y = x**2 + 0.1 * np.random.randn(20)
    df = pd.DataFrame({'x': x, 'y': y})
    
    result = polynomial_regression_analysis(df, 'x', 'y', degree=2)
    
    assert result['model'] is not None
    assert result['r2'] is not None
    assert result['r2'] > 0.8  # Should fit well
    assert len(result['x_pred']) == 100
    assert len(result['y_pred']) == 100
    
    print("✓ test_polynomial_regression passed")

def test_partial_correlation():
    """Test partial_correlation_analysis"""
    np.random.seed(42)
    n = 50
    data = {
        'target': np.random.randn(n),
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'control': np.random.randn(n)
    }
    df = pd.DataFrame(data)
    
    result = partial_correlation_analysis(df, 'target', ['feature1', 'feature2'], ['control'])
    
    assert len(result) == 2
    assert 'partial_correlation' in result.columns
    assert 'p_value' in result.columns
    
    print("✓ test_partial_correlation passed")

# Run tests
print("\n=== Running Unit Tests ===\n")
try:
    test_safe_float_converter()
    test_flexible_column_mapper()
    test_polynomial_regression()
    test_partial_correlation()
    print("\n✅ All tests passed!")
except Exception as e:
    print(f"\n❌ Test failed: {e}")
