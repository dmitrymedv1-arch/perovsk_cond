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
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
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
warnings.filterwarnings('ignore')

# Константы
AVOGADRO_NUMBER = 6.02214076e23  # моль⁻¹
OXYGEN_RADIUS = 1.4  # Å
PREFACTOR_VOLUME = 16 * np.pi / 3  # 16π/3 для расчета объема сфер
GAS_CONSTANT = 8.314  # Дж/(моль·К)

# Научный стиль графиков
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
# БАЗА ДАННЫХ ИОННЫХ РАДИУСОВ (Шеннон)
# ============================================================================
IONIC_RADII = {
    # Формат: (ион, заряд, КЧ): (кристаллический радиус, ионный радиус)
    # Для A-сайта используем КЧ=12, для B-сайта КЧ=6, для O - фиксированное значение
    ('Ba', 2, 12): 1.61,
    ('Sr', 2, 12): 1.44,
    ('O', -2, 6): 1.4,
    
    # B-катионы (КЧ=6)
    ('Ce', 4, 6): 0.87,
    ('Zr', 4, 6): 0.72,
    ('Sn', 4, 6): 0.69,
    ('Ti', 4, 6): 0.605,
    ('Hf', 4, 6): 0.71,
    
    # D-допанты (акцепторы, обычно 3+, КЧ=6)
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
    
    # Спекающие добавки (переходные металлы, КЧ=6 для простоты)
    ('Cu', 2, 6): 0.73,
    ('Ni', 2, 6): 0.69,
    ('Zn', 2, 6): 0.74,
    ('Co', 2, 6): 0.65,
}

# ============================================================================
# БАЗА ДАННЫХ ЭЛЕКТРООТРИЦАТЕЛЬНОСТИ (Полинг)
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
# БАЗА ДАННЫХ ЗАРЯДОВ ИОНОВ
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
# БАЗА ДАННЫХ СВОЙСТВ БАЗОВЫХ СТРУКТУР
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
# АТОМНЫЕ МАССЫ (г/моль)
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

# Цветовая карта для B-катионов
B_COLORS = {
    'Ce': '#E41A1C',
    'Zr': '#377EB8',
    'Sn': '#4DAF4A',
    'Ti': '#984EA3',
    'Hf': '#FF7F00',
    'default': '#999999'
}

# Цветовая карта для спекающих добавок
SINTERING_ADDITIVE_COLORS = {
    'Pure': '#4DAF4A',
    'Cu': '#E41A1C',
    'Ni': '#377EB8',
    'Zn': '#984EA3',
    'Co': '#FF7F00',
    'default': '#999999'
}

# Маркеры для спекающих добавок
SINTERING_ADDITIVE_MARKERS = {
    'Pure': 'o',
    'Cu': 's',
    'Ni': '^',
    'Zn': 'D',
    'Co': 'v',
    'default': 'o'
}

# ============================================================================
# НОВЫЙ КЛАСС: РАСЧЕТЧИК ДЕСКРИПТОРОВ ДЛЯ АНАЛИЗА ПРОВОДИМОСТИ
# ============================================================================
class ConductivityDescriptorCalculator:
    """Класс для расчета всех физико-химических и микроструктурных дескрипторов для анализа проводимости"""
    
    def __init__(self, a_element='Ba'):
        self.a_element = a_element
        self.r_O = IONIC_RADII.get(('O', -2, 6), 1.4)
        self.χ_O = ELECTRONEGATIVITY.get('O', 3.44)
        self.r_A = IONIC_RADII.get((a_element, 2, 12), None)
        self.χ_A = ELECTRONEGATIVITY.get(a_element, None)
        self.z_A = IONIC_CHARGES.get(a_element, 2)
        # Теоретическая плотность для расчета пористости
        self.theoretical_density = None
    
    def get_ionic_radius(self, element, charge=None, coordination=6):
        """Получение ионного радиуса с автоматическим определением заряда"""
        if charge is None:
            charge = IONIC_CHARGES.get(element, 4)
        return IONIC_RADII.get((element, charge, coordination), None)
    
    def get_electronegativity(self, element):
        return ELECTRONEGATIVITY.get(element, None)
    
    def get_charge(self, element):
        return IONIC_CHARGES.get(element, None)
    
    def get_atomic_mass(self, element):
        return ATOMIC_MASSES.get(element, None)
    
    def calculate_formula(self, b1_element, b2_element, b2_cont, dopant, dop_cont):
        """
        Расчет состава и базовых параметров из столбцов таблицы
        
        Parameters
        ----------
        b1_element : str
            Основной элемент B-сайта (например, Zr)
        b2_element : str or NaN
            Второй элемент B-сайта (например, Ce)
        b2_cont : float
            Содержание второго B-элемента (например, 0.3)
        dopant : str
            Легирующий элемент (акцептор, например, Y)
        dop_cont : float
            Содержание легирующего элемента (например, 0.2)
        
        Returns
        -------
        dict
            Словарь с рассчитанными параметрами состава
        """
        result = {
            'formula_type': 'simple',
            'b1_element': b1_element,
            'b2_element': b2_element,
            'b2_cont': b2_cont if not pd.isna(b2_cont) else 0,
            'dopant': dopant,
            'dop_cont': dop_cont if not pd.isna(dop_cont) else 0,
            'x_B2': b2_cont if not pd.isna(b2_cont) else 0,
            'y_dop': dop_cont if not pd.isna(dop_cont) else 0,
        }
        
        # Определение формулы
        if pd.isna(b2_element) or b2_element == '' or b2_cont == 0:
            # Простой допант: AB1_{1-y}D_yO_{3-y/2}
            result['formula_type'] = 'simple'
            result['b_main'] = b1_element
            result['x_B2'] = 0
        else:
            # Сложный состав: AB1_{1-x-y}B2_xD_yO_{3-y/2}
            result['formula_type'] = 'complex'
            result['b_main'] = b1_element
        
        # Расчет среднего ионного радиуса B-сайта
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
        
        # Средний радиус B-сайта
        if result['formula_type'] == 'simple' and r_B1 is not None and r_D is not None:
            result['r_avg_B'] = (1 - y) * r_B1 + y * r_D
        elif result['formula_type'] == 'complex' and r_B1 is not None and r_B2 is not None and r_D is not None:
            result['r_avg_B'] = (1 - x - y) * r_B1 + x * r_B2 + y * r_D
        else:
            result['r_avg_B'] = None
        
        # Tolerance factor
        if self.r_A is not None and result['r_avg_B'] is not None and self.r_O is not None:
            result['tolerance_factor'] = (self.r_A + self.r_O) / (np.sqrt(2) * (result['r_avg_B'] + self.r_O))
        else:
            result['tolerance_factor'] = None
        
        # Средняя электроотрицательность B-сайта
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
        
        # Разница электроотрицательностей
        if result['χ_avg_B'] is not None and self.χ_A is not None:
            result['Δχ'] = abs(result['χ_avg_B'] - self.χ_A)
        else:
            result['Δχ'] = None
        
        # Концентрация кислородных вакансий
        result['oxygen_vacancy_conc'] = y / 2 if y is not None else None
        
        # Молярная масса
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
        
        # Базовая структура для теоретической плотности
        base_compound = f"{self.a_element}{b1_element}O3"
        base_props = MATERIAL_PROPERTIES.get(base_compound, None)
        if base_props is not None:
            result['theoretical_density'] = base_props.get('density', None)
        else:
            result['theoretical_density'] = None
        
        return result
    
    def calculate_microstructure_descriptors(self, density_percent, grain_size_um):
        """
        Расчет микроструктурных дескрипторов
        
        Parameters
        ----------
        density_percent : float
            Относительная плотность в процентах
        grain_size_um : float
            Размер зерен в микрометрах
        
        Returns
        -------
        dict
            Словарь с микроструктурными дескрипторами
        """
        descriptors = {}
        
        # Плотность
        if density_percent is not None and not pd.isna(density_percent):
            descriptors['density_percent'] = density_percent
            descriptors['density_fraction'] = density_percent / 100.0
            descriptors['porosity'] = 1.0 - descriptors['density_fraction']
        else:
            descriptors['density_percent'] = None
            descriptors['density_fraction'] = None
            descriptors['porosity'] = None
        
        # Размер зерен
        if grain_size_um is not None and not pd.isna(grain_size_um) and grain_size_um > 0:
            descriptors['grain_size_um'] = grain_size_um
            # Расчет S/V - площади контакта зерен на единицу объема
            # S/V = 9/4 * (4/3)^(2/3) * 1/req * π^(-1/3) ≈ 1.861 / req
            # где req - эквивалентный радиус сферического зерна
            req = grain_size_um / 2.0  # радиус в мкм
            descriptors['S_V_ratio'] = 1.861 / req  # мкм⁻¹
            # В м⁻¹ для физических расчетов
            descriptors['S_V_ratio_m'] = descriptors['S_V_ratio'] * 1e6
        else:
            descriptors['grain_size_um'] = None
            descriptors['S_V_ratio'] = None
            descriptors['S_V_ratio_m'] = None
        
        return descriptors
    
    def calculate_sintering_additive_descriptors(self, additive_type, additive_concentration_wt):
        """
        Расчет дескрипторов для спекающей добавки
        
        Parameters
        ----------
        additive_type : str
            Тип добавки (Pure, Cu, Ni, Zn, Co)
        additive_concentration_wt : float
            Концентрация добавки в масс.%
        
        Returns
        -------
        dict
            Словарь с дескрипторами добавки
        """
        descriptors = {
            'additive_type': additive_type if not pd.isna(additive_type) else 'Pure',
            'additive_concentration_wt': additive_concentration_wt if not pd.isna(additive_concentration_wt) else 0.0,
            'is_pure': True if (pd.isna(additive_type) or additive_type == 'Pure' or additive_concentration_wt == 0) else False
        }
        
        if not descriptors['is_pure']:
            # Ионный радиус катиона добавки
            descriptors['additive_radius'] = self.get_ionic_radius(additive_type, 2, 6)
            # Электроотрицательность
            descriptors['additive_electronegativity'] = self.get_electronegativity(additive_type)
            # Заряд
            descriptors['additive_charge'] = self.get_charge(additive_type)
            # Атомная масса
            descriptors['additive_atomic_mass'] = self.get_atomic_mass(additive_type)
        else:
            descriptors['additive_radius'] = None
            descriptors['additive_electronegativity'] = None
            descriptors['additive_charge'] = None
            descriptors['additive_atomic_mass'] = None
        
        return descriptors

# ============================================================================
# НОВЫЙ КЛАСС: ОБРАБОТЧИК ДАННЫХ ПРОВОДИМОСТИ
# ============================================================================
class ConductivityDataProcessor:
    """Класс для обработки данных проводимости протонпроводящих оксидов"""
    
    def __init__(self):
        self.temperatures = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
        self.calculator = ConductivityDescriptorCalculator(a_element='Ba')
    
    def extract_conductivity_data(self, row, sigma_cols_prefix='σ total, mS'):
        """
        Извлечение данных проводимости из строки
        
        Parameters
        ----------
        row : pandas.Series
            Строка с данными
        sigma_cols_prefix : str
            Префикс колонок с проводимостью
        
        Returns
        -------
        list of dict
            Список с данными проводимости при разных температурах
        """
        conductivity_data = []
        
        for i, T in enumerate(self.temperatures):
            # Поиск колонки с проводимостью
            sigma_col = None
            for col in row.index:
                if sigma_cols_prefix in str(col) and str(T) in str(col):
                    sigma_col = col
                    break
            
            if sigma_col is not None:
                sigma_value = row[sigma_col]
                if not pd.isna(sigma_value) and sigma_value != '' and sigma_value is not None:
                    try:
                        sigma_val = float(sigma_value)
                        conductivity_data.append({
                            'temperature_K': T + 273.15,
                            'temperature_C': T,
                            'sigma_total': sigma_val,
                            'sigma_total_mS': sigma_val,
                            'sigma_total_S_cm': sigma_val / 1000.0 if sigma_val is not None else None
                        })
                    except (ValueError, TypeError):
                        pass
        
        return conductivity_data
    
    def extract_bulk_conductivity_data(self, row):
        """Извлечение объемной проводимости из строки"""
        bulk_data = []
        
        for i, T in enumerate(self.temperatures):
            bulk_col = None
            for col in row.index:
                if 'σ bulk, mS' in str(col) and str(T) in str(col):
                    bulk_col = col
                    break
            
            if bulk_col is not None:
                bulk_value = row[bulk_col]
                if not pd.isna(bulk_value) and bulk_value != '' and bulk_value is not None:
                    try:
                        bulk_val = float(bulk_value)
                        bulk_data.append({
                            'temperature_K': T + 273.15,
                            'temperature_C': T,
                            'sigma_bulk': bulk_val,
                            'sigma_bulk_mS': bulk_val,
                            'sigma_bulk_S_cm': bulk_val / 1000.0 if bulk_val is not None else None
                        })
                    except (ValueError, TypeError):
                        pass
        
        return bulk_data
    
    def extract_gb_conductivity_data(self, row):
        """Извлечение зернограничной проводимости из строки"""
        gb_data = []
        
        for i, T in enumerate(self.temperatures):
            gb_col = None
            for col in row.index:
                if 'σ gb, mS' in str(col) and str(T) in str(col):
                    gb_col = col
                    break
            
            if gb_col is not None:
                gb_value = row[gb_col]
                if not pd.isna(gb_value) and gb_value != '' and gb_value is not None:
                    try:
                        gb_val = float(gb_value)
                        gb_data.append({
                            'temperature_K': T + 273.15,
                            'temperature_C': T,
                            'sigma_gb': gb_val,
                            'sigma_gb_mS': gb_val,
                            'sigma_gb_S_cm': gb_val / 1000.0 if gb_val is not None else None
                        })
                    except (ValueError, TypeError):
                        pass
        
        return gb_data
    
    def calculate_arrhenius_params(self, sigma_data):
        """
        Расчет параметров Аррениуса из данных проводимости
        
        Parameters
        ----------
        sigma_data : list of dict
            Данные проводимости при разных температурах
        
        Returns
        -------
        dict
            Параметры Аррениуса: Ea, A, R²
        """
        if len(sigma_data) < 3:
            return {'Ea': None, 'A': None, 'R2': None, 'has_data': False}
        
        # Преобразование для графика Аррениуса: ln(σT) vs 1000/T
        temps = []
        ln_sigmaT = []
        
        for data in sigma_data:
            T_K = data['temperature_K']
            sigma = data['sigma_total']
            if sigma is not None and sigma > 0 and T_K > 0:
                ln_val = np.log(sigma * T_K)
                if np.isfinite(ln_val):
                    ln_sigmaT.append(ln_val)
                    temps.append(1000.0 / T_K)
        
        if len(temps) < 3:
            return {'Ea': None, 'A': None, 'R2': None, 'has_data': False}
        
        # Линейная регрессия
        slope, intercept, r_value, p_value, std_err = stats.linregress(temps, ln_sigmaT)
        
        # Ea = slope * R (где R = 8.314 кДж/(моль·К) = 0.008314 кДж/(моль·К))
        # Для удобства Ea в эВ: 1 эВ = 96.485 кДж/моль
        Ea_kJ = slope * GAS_CONSTANT / 1000.0  # кДж/моль
        Ea_eV = Ea_kJ / 96.485  # эВ
        
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
        Расчет вклада зернограничной проводимости
        
        Returns
        -------
        dict
            Относительный вклад зернограничной проводимости
        """
        result = {}
        
        if sigma_total_data and sigma_bulk_data and sigma_gb_data:
            # Создаем словари по температурам
            total_by_T = {d['temperature_C']: d['sigma_total'] for d in sigma_total_data}
            bulk_by_T = {d['temperature_C']: d['sigma_bulk'] for d in sigma_bulk_data}
            gb_by_T = {d['temperature_C']: d['sigma_gb'] for d in sigma_gb_data}
            
            for T in total_by_T.keys():
                if T in bulk_by_T and T in gb_by_T:
                    sigma_total = total_by_T[T]
                    sigma_bulk = bulk_by_T[T]
                    sigma_gb = gb_by_T[T]
                    
                    if sigma_total is not None and sigma_total > 0:
                        # Расчет сопротивлений
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
# ФУНКЦИИ ДЛЯ ПРОЦЕССИНГА ДАННЫХ ПРОВОДИМОСТИ
# ============================================================================
@st.cache_data
def process_conductivity_data(df):
    """
    Основная функция обработки данных проводимости
    
    Parameters
    ----------
    df : pandas.DataFrame
        Исходные данные
    
    Returns
    -------
    pandas.DataFrame
        Обработанные данные в длинном формате
    """
    df_processed = df.copy()
    processor = ConductivityDataProcessor()
    
    # Создаем список для хранения данных в длинном формате
    long_format_data = []
    
    # Перебираем строки
    for idx, row in df_processed.iterrows():
        # Базовые параметры состава
        a_cation = row.get('A cation', 'Ba')
        b1_cation = row.get('B1 cation', None)
        b2_cation = row.get('B2 cation', None)
        b2_cont = row.get('B2_cont', None)
        dopant = row.get('dopant', None)
        dop_cont = row.get('dop_cont', None)
        
        # Спекающая добавка
        additive_type = row.get('Сд', 'Pure')
        additive_conc = row.get('x, wt%', 0.0)
        
        # Параметры синтеза
        method = row.get('Method', None)
        T_sin = row.get('T sin', None)
        structure = row.get('Structure', None)
        space_group = row.get('Space group', None)
        a_latt = row.get('a, Å', None)
        b_latt = row.get('b, Å', None)
        c_latt = row.get('c, Å', None)
        density_percent = row.get('ρ, %', None)
        grain_size_um = row.get('d, mkm', None)
        
        # Условия измерений
        atmosphere = row.get('Атмосфера', None)
        humidity = row.get('Влажность', None)
        doi = row.get('ссылка', None)
        
        # Ea из таблицы (если есть)
        Ea_table = row.get('Ea, эВ', None)
        
        # Расчет дескрипторов состава
        if b1_cation is not None and not pd.isna(b1_cation):
            formula_desc = processor.calculator.calculate_formula(
                b1_cation, b2_cation, b2_cont, dopant, dop_cont
            )
        else:
            formula_desc = {}
        
        # Расчет микроструктурных дескрипторов
        micro_desc = processor.calculator.calculate_microstructure_descriptors(
            density_percent, grain_size_um
        )
        
        # Расчет дескрипторов спекающей добавки
        additive_desc = processor.calculator.calculate_sintering_additive_descriptors(
            additive_type, additive_conc
        )
        
        # Извлечение данных проводимости
        sigma_total_data = processor.extract_conductivity_data(row, 'σ total, mS')
        sigma_bulk_data = processor.extract_bulk_conductivity_data(row)
        sigma_gb_data = processor.extract_gb_conductivity_data(row)
        
        # Расчет параметров Аррениуса
        arrhenius = processor.calculate_arrhenius_params(sigma_total_data)
        
        # Расчет вклада зернограничной проводимости
        gb_contribution = processor.calculate_gb_contribution(
            sigma_total_data, sigma_bulk_data, sigma_gb_data
        )
        
        # Добавляем записи в длинный формат
        for sigma_data in sigma_total_data:
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
                'method': method,
                'T_sin': T_sin,
                'structure': structure,
                'space_group': space_group,
                'a_latt': a_latt,
                'b_latt': b_latt,
                'c_latt': c_latt,
                'density_percent': density_percent,
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
                'sigma_total_mS': sigma_data['sigma_total'],
                'sigma_total_S_cm': sigma_data['sigma_total_S_cm'],
            }
            
            # Добавляем объемную проводимость, если есть
            bulk_at_T = next((b for b in sigma_bulk_data if b['temperature_C'] == sigma_data['temperature_C']), None)
            if bulk_at_T:
                record['sigma_bulk_mS'] = bulk_at_T['sigma_bulk']
                record['sigma_bulk_S_cm'] = bulk_at_T['sigma_bulk_S_cm']
            else:
                record['sigma_bulk_mS'] = None
                record['sigma_bulk_S_cm'] = None
            
            # Добавляем зернограничную проводимость, если есть
            gb_at_T = next((g for g in sigma_gb_data if g['temperature_C'] == sigma_data['temperature_C']), None)
            if gb_at_T:
                record['sigma_gb_mS'] = gb_at_T['sigma_gb']
                record['sigma_gb_S_cm'] = gb_at_T['sigma_gb_S_cm']
            else:
                record['sigma_gb_mS'] = None
                record['sigma_gb_S_cm'] = None
            
            # Добавляем вклад зернограничной проводимости
            if sigma_data['temperature_C'] in gb_contribution:
                record['gb_resistance_fraction'] = gb_contribution[sigma_data['temperature_C']]['gb_resistance_fraction']
                record['bulk_resistance_fraction'] = gb_contribution[sigma_data['temperature_C']]['bulk_resistance_fraction']
            else:
                record['gb_resistance_fraction'] = None
                record['bulk_resistance_fraction'] = None
            
            # Добавляем геометрические дескрипторы
            record['r_avg_B'] = formula_desc.get('r_avg_B')
            record['tolerance_factor'] = formula_desc.get('tolerance_factor')
            record['χ_avg_B'] = formula_desc.get('χ_avg_B')
            record['Δχ'] = formula_desc.get('Δχ')
            record['oxygen_vacancy_conc'] = formula_desc.get('oxygen_vacancy_conc')
            record['molar_mass'] = formula_desc.get('molar_mass')
            record['theoretical_density'] = formula_desc.get('theoretical_density')
            
            long_format_data.append(record)
    
    # Создаем DataFrame в длинном формате
    long_df = pd.DataFrame(long_format_data)
    
    # Создаем также широкий формат для некоторых анализов
    wide_format_data = []
    for idx, row in df_processed.iterrows():
        wide_record = {
            'sample_id': idx,
            'A_cation': row.get('A cation', 'Ba'),
            'B1_cation': row.get('B1 cation', None),
            'B2_cation': row.get('B2 cation', None),
            'B2_cont': row.get('B2_cont', None),
            'dopant': row.get('dopant', None),
            'dop_cont': row.get('dop_cont', None),
            'additive_type': row.get('Сд', 'Pure'),
            'additive_concentration_wt': row.get('x, wt%', 0.0),
            'method': row.get('Method', None),
            'T_sin': row.get('T sin', None),
            'structure': row.get('Structure', None),
            'space_group': row.get('Space group', None),
            'a_latt': row.get('a, Å', None),
            'b_latt': row.get('b, Å', None),
            'c_latt': row.get('c, Å', None),
            'density_percent': row.get('ρ, %', None),
            'grain_size_um': row.get('d, mkm', None),
            'atmosphere': row.get('Атмосфера', None),
            'humidity': row.get('Влажность', None),
            'doi': row.get('ссылка', None),
            'Ea_table': row.get('Ea, эВ', None),
        }
        
        # Добавляем данные проводимости при разных температурах
        processor_local = ConductivityDataProcessor()
        sigma_total_data = processor_local.extract_conductivity_data(row, 'σ total, mS')
        sigma_bulk_data = processor_local.extract_bulk_conductivity_data(row)
        sigma_gb_data = processor_local.extract_gb_conductivity_data(row)
        
        for sigma_data in sigma_total_data:
            T = sigma_data['temperature_C']
            wide_record[f'sigma_total_{T}C'] = sigma_data['sigma_total']
        
        for bulk_data in sigma_bulk_data:
            T = bulk_data['temperature_C']
            wide_record[f'sigma_bulk_{T}C'] = bulk_data['sigma_bulk']
        
        for gb_data in sigma_gb_data:
            T = gb_data['temperature_C']
            wide_record[f'sigma_gb_{T}C'] = gb_data['sigma_gb']
        
        # Расчет Arrhenius
        arrhenius = processor_local.calculate_arrhenius_params(sigma_total_data)
        wide_record['Ea_calculated_eV'] = arrhenius['Ea']
        wide_record['Ea_calculated_kJ'] = arrhenius['Ea_kJ']
        wide_record['arrhenius_R2'] = arrhenius['R2']
        
        wide_format_data.append(wide_record)
    
    wide_df = pd.DataFrame(wide_format_data)
    
    return long_df, wide_df

# ============================================================================
# ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ АНАЛИЗА ПРОВОДИМОСТИ
# ============================================================================
def plot_conductivity_vs_temperature(df_long, ax, selected_additives=None, selected_b_sites=None, temperature_min=400, temperature_max=700):
    """График 1: Проводимость как функция температуры для разных добавок"""
    
    plot_df = df_long.copy()
    
    # Фильтрация
    if selected_additives:
        plot_df = plot_df[plot_df['additive_type'].isin(selected_additives)]
    if selected_b_sites and 'B1_cation' in plot_df.columns:
        plot_df = plot_df[plot_df['B1_cation'].isin(selected_b_sites)]
    
    # Фильтрация по температуре
    plot_df = plot_df[(plot_df['temperature_C'] >= temperature_min) & (plot_df['temperature_C'] <= temperature_max)]
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, 'No data for selected filters', ha='center', va='center')
        return ax
    
    # Группировка для усреднения
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
    """График 2: График Аррениуса ln(σT) vs 1000/T"""
    
    plot_df = df_long.copy()
    
    if selected_additives:
        plot_df = plot_df[plot_df['additive_type'].isin(selected_additives)]
    if selected_b_sites and 'B1_cation' in plot_df.columns:
        plot_df = plot_df[plot_df['B1_cation'].isin(selected_b_sites)]
    
    # Расчет ln(σT)
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
        
        # Усреднение по температуре
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
    """График 3: Сравнение проводимости разных добавок при фиксированной температуре"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No data at {temperature}°C', ha='center', va='center')
        return ax
    
    # Группировка по типу добавки
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
    
    # Добавляем значения над столбцами
    for i, (_, row) in enumerate(grouped.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.01, f'{row["mean"]:.3f}', ha='center', fontsize=8)
    
    return ax


def plot_conductivity_vs_additive_concentration(df_long, ax, temperature=600, b_site_filter=None):
    """График 4: Зависимость проводимости от концентрации добавки"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df[plot_df['additive_concentration_wt'] > 0]
    
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
        
        # Группировка по концентрации
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
    """График 5: Сравнение Pure vs добавки (относительное улучшение)"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No data at {temperature}°C', ha='center', va='center')
        return ax
    
    # Получаем baseline для Pure
    pure_data = plot_df[plot_df['additive_type'] == 'Pure']['sigma_total_mS'].mean()
    
    if pd.isna(pure_data) or pure_data == 0:
        ax.text(0.5, 0.5, 'No Pure reference data available', ha='center', va='center')
        return ax
    
    # Расчет улучшения для каждой добавки
    improvement_data = []
    for additive in plot_df['additive_type'].unique():
        if additive == 'Pure':
            continue
        additive_mean = plot_df[plot_df['additive_type'] == additive]['sigma_total_mS'].mean()
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
    
    # Добавляем значения
    for i, (_, row) in enumerate(improvement_df.iterrows()):
        ax.text(row['improvement'] + 1, i, f'{row["improvement"]:.1f}%', va='center', fontsize=8)
    
    ax.grid(True, alpha=0.3, axis='x')
    return ax


def plot_bulk_vs_gb_contribution(df_long, ax, temperature=600):
    """График 6: Сравнение объемной и зернограничной проводимости"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['sigma_bulk_mS', 'sigma_gb_mS'])
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No bulk/gb data at {temperature}°C', ha='center', va='center')
        return ax
    
    # Группировка по добавкам
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
    """График 7: Доля зернограничного сопротивления"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['gb_resistance_fraction'])
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No gb fraction data at {temperature}°C', ha='center', va='center')
        return ax
    
    # Группировка по добавкам
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
    """График 8: Зависимость проводимости от размера зерен"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['grain_size_um', 'sigma_total_mS'])
    
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
    """График 9: Зависимость проводимости от плотности"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['density_percent', 'sigma_total_mS'])
    
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
    """График 10: Тепловая карта проводимости (B-катион vs добавка)"""
    
    if 'B1_cation' not in df_wide.columns or 'additive_type' not in df_wide.columns:
        ax.text(0.5, 0.5, 'Required columns missing', ha='center', va='center')
        return ax
    
    # Создаем матрицу
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
    
    # Добавляем значения в ячейки
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.values[i, j]
            if not pd.isna(value):
                text_color = 'white' if value > heatmap_data.values.max() / 2 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='σ (mS/cm)')
    return ax


def plot_activation_energy_comparison(df_long, ax):
    """График 11: Сравнение энергий активации для разных добавок"""
    
    # Получаем уникальные образцы с рассчитанной Ea
    samples = df_long[['sample_id', 'additive_type', 'Ea_calculated']].drop_duplicates().dropna(subset=['Ea_calculated'])
    
    if len(samples) == 0:
        ax.text(0.5, 0.5, 'No Ea data available', ha='center', va='center')
        return ax
    
    # Группировка по добавкам
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
    """График 12: Влияние температуры спекания на проводимость"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['T_sin', 'sigma_total_mS'])
    
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
    """График 13: Влияние пористости на проводимость"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['porosity', 'sigma_total_mS'])
    
    if len(plot_df) == 0:
        ax.text(0.5, 0.5, f'No porosity data at {temperature}°C', ha='center', va='center')
        return ax
    
    for additive in plot_df['additive_type'].unique():
        additive_data = plot_df[plot_df['additive_type'] == additive]
        color = SINTERING_ADDITIVE_COLORS.get(additive, SINTERING_ADDITIVE_COLORS['default'])
        marker = SINTERING_ADDITIVE_MARKERS.get(additive, SINTERING_ADDITIVE_MARKERS['default'])
        
        ax.scatter(
            additive_data['porosity'] * 100,  # в процентах
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
    """График 14: Влияние tolerance factor на проводимость"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['tolerance_factor', 'sigma_total_mS'])
    
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
    ax.set_xlabel('Tolerance Factor (t)')
    ax.set_ylabel(f'σ total at {temperature}°C (mS/cm)')
    ax.set_title(f'Effect of Tolerance Factor on Conductivity at {temperature}°C')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_oxygen_vacancy_influence(df_long, ax, temperature=600):
    """График 15: Влияние концентрации кислородных вакансий на проводимость"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    plot_df = plot_df.dropna(subset=['oxygen_vacancy_conc', 'sigma_total_mS'])
    
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
    """График 16: Корреляционная матрица для параметров проводимости"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
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


def generate_conductivity_insights(df_long):
    """Автоматическая генерация инсайтов по проводимости"""
    insights = []
    
    # Сравнение Pure vs добавки при 600°C
    df_600 = df_long[df_long['temperature_C'] == 600].copy()
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
    
    # Оптимальная концентрация добавки
    for additive in df_long['additive_type'].unique():
        if additive != 'Pure':
            additive_data = df_long[df_long['additive_type'] == additive].copy()
            additive_data = additive_data.dropna(subset=['additive_concentration_wt', 'sigma_total_mS'])
            if len(additive_data) > 3:
                # Группировка по концентрации
                grouped = additive_data.groupby('additive_concentration_wt')['sigma_total_mS'].mean().reset_index()
                if len(grouped) > 1:
                    max_conc = grouped.loc[grouped['sigma_total_mS'].idxmax(), 'additive_concentration_wt']
                    insights.append(f"Optimal concentration for **{additive} additive** appears around {max_conc:.2f} wt%.")
    
    # Влияние размера зерен
    df_grain = df_long.dropna(subset=['grain_size_um', 'sigma_total_mS'])
    if len(df_grain) > 10:
        corr, p_val = stats.spearmanr(df_grain['grain_size_um'], df_grain['sigma_total_mS'])
        if p_val < 0.05:
            if corr > 0.5:
                insights.append(f"**Strong positive correlation** between grain size and conductivity (ρ = {corr:.2f}, p < 0.05). Larger grains improve conductivity.")
            elif corr < -0.5:
                insights.append(f"**Strong negative correlation** between grain size and conductivity (ρ = {corr:.2f}, p < 0.05). Smaller grains improve conductivity.")
    
    # Влияние плотности
    df_density = df_long.dropna(subset=['density_percent', 'sigma_total_mS'])
    if len(df_density) > 10:
        corr, p_val = stats.spearmanr(df_density['density_percent'], df_density['sigma_total_mS'])
        if p_val < 0.05 and corr > 0.3:
            insights.append(f"**Positive correlation** between density and conductivity (ρ = {corr:.2f}, p < 0.05). Higher density improves conductivity.")
    
    # Влияние tolerance factor
    df_t = df_long.dropna(subset=['tolerance_factor', 'sigma_total_mS'])
    if len(df_t) > 10:
        t_opt = df_t[(df_t['tolerance_factor'] >= 0.96) & (df_t['tolerance_factor'] <= 1.04)]
        t_out = df_t[(df_t['tolerance_factor'] < 0.96) | (df_t['tolerance_factor'] > 1.04)]
        if len(t_opt) > 0 and len(t_out) > 0:
            mean_opt = t_opt['sigma_total_mS'].mean()
            mean_out = t_out['sigma_total_mS'].mean()
            if mean_opt > mean_out:
                ratio = mean_opt / mean_out if mean_out > 0 else float('inf')
                insights.append(f"Systems with tolerance factor in [0.96-1.04] have **{ratio:.1f}x higher** average conductivity.")
    
    # Вклад зернограничной проводимости
    df_gb = df_long.dropna(subset=['gb_resistance_fraction'])
    if len(df_gb) > 5:
        mean_gb = df_gb['gb_resistance_fraction'].mean()
        if mean_gb > 0.5:
            insights.append(f"**Grain boundaries dominate** the total resistance ({mean_gb:.1%} on average). Improving grain boundary conductivity is critical.")
        else:
            insights.append(f"**Bulk conductivity dominates** the total resistance ({1-mean_gb:.1%} on average).")
    
    if len(insights) == 0:
        insights.append("Insufficient data for automated insights. Add more data points to enable pattern detection.")
    
    return insights


# ============================================================================
# ФУНКЦИИ ДЛЯ ML АНАЛИЗА ПРОВОДИМОСТИ
# ============================================================================
@st.cache_data
def feature_importance_conductivity(df_long, selected_features=None, target='sigma_total_mS', temperature=600):
    """Random Forest анализ важности признаков для проводимости"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    if selected_features is None:
        selected_features = ['density_percent', 'grain_size_um', 'tolerance_factor', 
                            'oxygen_vacancy_conc', 'additive_concentration_wt']
    
    available_features = [f for f in selected_features if f in plot_df.columns]
    
    if len(available_features) < 2:
        return None, None
    
    plot_df = plot_df.dropna(subset=[target] + available_features)
    
    if len(plot_df) < 10:
        return None, None
    
    X = plot_df[available_features].copy()
    
    # Добавляем one-hot кодирование для типа добавки
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
    """Сравнение нескольких моделей ML для проводимости"""
    
    plot_df = df_long[df_long['temperature_C'] == temperature].copy()
    
    if selected_features is None:
        selected_features = ['density_percent', 'grain_size_um', 'tolerance_factor', 
                            'oxygen_vacancy_conc', 'additive_concentration_wt']
    
    available_features = [f for f in selected_features if f in plot_df.columns]
    
    if len(available_features) < 2:
        return None, None, None
    
    plot_df = plot_df.dropna(subset=[target] + available_features)
    
    if len(plot_df) < 10:
        return None, None, None
    
    X = plot_df[available_features].copy()
    
    if 'additive_type' in plot_df.columns:
        X = pd.concat([X, pd.get_dummies(plot_df['additive_type'], prefix='additive')], axis=1)
    
    y = plot_df[target]
    
    # Определяем модели
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
# ОСНОВНОЕ STREAMLIT-ПРИЛОЖЕНИЕ
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
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Загрузка файла
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
        
        # Выбор температуры для анализа
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
        
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            with st.spinner("Processing conductivity data..."):
                df_long, df_wide = process_conductivity_data(df)
                
                if 'additive_type' in df_long.columns:
                    selected_additives = st.multiselect(
                        "Sintering Additives",
                        options=sorted(df_long['additive_type'].unique()),
                        default=sorted(df_long['additive_type'].unique())
                    )
                else:
                    selected_additives = []
                
                if 'B1_cation' in df_long.columns:
                    selected_b_sites = st.multiselect(
                        "B-site cations",
                        options=sorted(df_long['B1_cation'].dropna().unique()),
                        default=sorted(df_long['B1_cation'].dropna().unique())
                    )
                else:
                    selected_b_sites = []
                
                if 'humidity' in df_long.columns:
                    selected_humidity = st.multiselect(
                        "Humidity conditions",
                        options=sorted(df_long['humidity'].dropna().unique()),
                        default=sorted(df_long['humidity'].dropna().unique())
                    )
                else:
                    selected_humidity = []
                
                if 'atmosphere' in df_long.columns:
                    selected_atmosphere = st.multiselect(
                        "Atmosphere",
                        options=sorted(df_long['atmosphere'].dropna().unique()),
                        default=sorted(df_long['atmosphere'].dropna().unique())
                    )
                else:
                    selected_atmosphere = []
                
                # Применяем фильтры
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
                if selected_additives:
                    filtered_wide = filtered_wide[filtered_wide['additive_type'].isin(selected_additives)]
                if selected_b_sites:
                    filtered_wide = filtered_wide[filtered_wide['B1_cation'].isin(selected_b_sites)]
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    
    if uploaded_file is not None and len(filtered_long) > 0:
        # Отображение информации о данных
        st.subheader("📈 Data Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total measurements", len(filtered_long))
        with col2:
            st.metric("Unique samples", filtered_long['sample_id'].nunique())
        with col3:
            st.metric("Additive types", filtered_long['additive_type'].nunique())
        with col4:
            st.metric("B-site cations", filtered_long['B1_cation'].nunique())
        with col5:
            st.metric("Temp range", f"{filtered_long['temperature_C'].min()}-{filtered_long['temperature_C'].max()}°C")
        
        st.markdown("---")
        
        # Вкладки
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Conductivity Overview",
            "🔬 Additive Analysis",
            "⚡ Bulk vs GB Analysis",
            "📈 Microstructure Effects",
            "🧪 Compositional Effects",
            "🤖 ML & Feature Importance",
            "💡 Insights & Data"
        ])
        
        # ============================================================================
        # ВКЛАДКА 1: CONDUCTIVITY OVERVIEW
        # ============================================================================
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
            
            # Тепловая карта
            if len(filtered_wide) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_conductivity_heatmap(filtered_wide, ax, temperature_analysis)
                st.pyplot(fig)
                plt.close(fig)
        
        # ============================================================================
        # ВКЛАДКА 2: ADDITIVE ANALYSIS
        # ============================================================================
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
        
        # ============================================================================
        # ВКЛАДКА 3: BULK VS GB ANALYSIS
        # ============================================================================
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
            
            # Проверка формулы 1/σ_total = 1/σ_bulk + 1/σ_gb
            st.subheader("Verification of the Mixing Rule")
            
            df_check = filtered_long.dropna(subset=['sigma_total_mS', 'sigma_bulk_mS', 'sigma_gb_mS'])
            df_check = df_check[df_check['temperature_C'] == temperature_analysis]
            
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
        
        # ============================================================================
        # ВКЛАДКА 4: MICROSTRUCTURE EFFECTS
        # ============================================================================
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
        
        # ============================================================================
        # ВКЛАДКА 5: COMPOSITIONAL EFFECTS
        # ============================================================================
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
            
            # Корреляционная матрица
            st.subheader("Correlation Matrix")
            features = ['density_percent', 'grain_size_um', 'tolerance_factor', 
                       'oxygen_vacancy_conc', 'additive_concentration_wt']
            fig = plot_correlation_matrix_conductivity(filtered_long, features, temperature_analysis)
            st.pyplot(fig)
            plt.close(fig)
        
        # ============================================================================
        # ВКЛАДКА 6: ML & FEATURE IMPORTANCE
        # ============================================================================
        with tab6:
            st.subheader("Machine Learning Analysis")
            
            ml_features = st.multiselect(
                "Select features for ML analysis",
                options=['density_percent', 'grain_size_um', 'tolerance_factor', 
                        'oxygen_vacancy_conc', 'additive_concentration_wt'],
                default=['density_percent', 'grain_size_um', 'tolerance_factor', 'additive_concentration_wt']
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
        
        # ============================================================================
        # ВКЛАДКА 7: INSIGHTS & DATA
        # ============================================================================
        with tab7:
            st.subheader("💡 Automated Physical Insights")
            
            insights = generate_conductivity_insights(filtered_long)
            for insight in insights:
                st.info(insight)
            
            st.markdown("---")
            st.subheader("📋 Processed Data")
            
            # Показываем данные в длинном формате
            display_cols = ['sample_id', 'B1_cation', 'dopant', 'additive_type', 
                           'additive_concentration_wt', 'temperature_C', 'sigma_total_mS',
                           'sigma_bulk_mS', 'sigma_gb_mS', 'density_percent', 'grain_size_um',
                           'tolerance_factor', 'Ea_calculated']
            available_cols = [col for col in display_cols if col in filtered_long.columns]
            
            st.dataframe(filtered_long[available_cols].head(100), use_container_width=True)
            
            # Экспорт данных
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
        - Tolerance factor analysis
        - Oxygen vacancy concentration effects
        - Additive concentration optimization
        
        #### 🤖 **Machine Learning**
        - Feature importance (Random Forest)
        - Model comparison (RF, GB, XGBoost)
        - Cross-validation evaluation
        """)

if __name__ == "__main__":
    main()
