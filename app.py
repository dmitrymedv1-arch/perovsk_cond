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
matplotlib.use('Agg')  # Для предотвращения ошибок с backend

warnings.filterwarnings('ignore')

# ============================================
# КОНФИГУРАЦИЯ И ИНИЦИАЛИЗАЦИЯ
# ============================================

# Настройка стиля для научных публикаций
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

# Доступные цветовые палитры (10 вариантов)
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

# Словари для расчета дескрипторов
IONIC_RADII = {
    'Ba': 1.61,   # XII координационное число
    'Zr': 0.72, 'Ce': 0.87, 'Sn': 0.69,  # IV координационное число
    'Y': 0.90, 'Gd': 0.94, 'Yb': 0.87, 'Sm': 0.96,  # III координационное число
    'O': 1.40
}

ELECTRONEGATIVITY = {
    'Ba': 0.89, 'Zr': 1.33, 'Ce': 1.12, 'Sn': 1.96,
    'Y': 1.22, 'Gd': 1.20, 'Yb': 1.10, 'Sm': 1.17, 'O': 3.44
}

MOLAR_MASS = {
    'Ba': 137.33, 'Zr': 91.22, 'Ce': 140.12, 'Sn': 118.71,
    'Y': 88.91, 'Gd': 157.25, 'Yb': 173.05, 'Sm': 150.36, 'O': 16.00
}

# ============================================
# ФУНКЦИИ РАСЧЕТА ДЕСКРИПТОРОВ
# ============================================

def compute_descriptors(df):
    """
    Вычисление структурных, электроотрицательных и массовых дескрипторов
    для всех строк датафрейма
    """
    desc_df = pd.DataFrame(index=df.index)
    
    for idx, row in df.iterrows():
        # Извлечение мольных долей
        x_B2 = row['B2_cont'] if pd.notna(row['B2_cont']) else 0.0
        x_dop = row['dop_cont'] if pd.notna(row['dop_cont']) else 0.0
        
        # Проверка наличия B2 катиона
        has_B2 = pd.notna(row['B2 cation']) and row['B2 cation'] != ''
        
        # Расчет x_B1
        if has_B2:
            x_B1 = 1.0 - x_B2 - x_dop
        else:
            x_B1 = 1.0 - x_dop
            x_B2 = 0.0
        
        # Расчет среднего ионного радиуса B-подрешетки
        r_B = 0.0
        if x_B1 > 0:
            r_B += x_B1 * IONIC_RADII[row['B1 cation']]
        if x_B2 > 0 and has_B2:
            r_B += x_B2 * IONIC_RADII[row['B2 cation']]
        if x_dop > 0:
            r_B += x_dop * IONIC_RADII[row['dopant']]
        
        # Tолеранс-фактор Гольдшмидта (для кубического перовскита)
        r_A = IONIC_RADII['Ba']
        r_O = IONIC_RADII['O']
        t_factor = (r_A + r_O) / (np.sqrt(2) * (r_B + r_O))
        
        # Средняя электроотрицательность B-подрешетки
        chi_B = 0.0
        if x_B1 > 0:
            chi_B += x_B1 * ELECTRONEGATIVITY[row['B1 cation']]
        if x_B2 > 0 and has_B2:
            chi_B += x_B2 * ELECTRONEGATIVITY[row['B2 cation']]
        if x_dop > 0:
            chi_B += x_dop * ELECTRONEGATIVITY[row['dopant']]
        
        chi_ratio = chi_B / ELECTRONEGATIVITY['Ba']
        
        # Молярная масса (приближенная)
        molar_mass = 0.0
        molar_mass += x_B1 * MOLAR_MASS[row['B1 cation']]
        if x_B2 > 0 and has_B2:
            molar_mass += x_B2 * MOLAR_MASS[row['B2 cation']]
        if x_dop > 0:
            molar_mass += x_dop * MOLAR_MASS[row['dopant']]
        molar_mass += 3 * MOLAR_MASS['O']  # приближенно O3
        
        # Микроструктурные метрики
        rho = row['ρ, %'] if pd.notna(row['ρ, %']) else np.nan
        d = row['d, mkm'] if pd.notna(row['d, mkm']) else np.nan
        
        porosity = 100 - rho if pd.notna(rho) else np.nan
        gb_area = 3.722 / d if pd.notna(d) and d > 0 else np.nan
        
        # Сохранение
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
# ФУНКЦИЯ РАСЧЕТА Ea
# ============================================

def calculate_ea(row):
    """
    Расчет энергии активации по уравнению Аррениуса
    Используются все доступные температуры проводимости
    """
    # Если Ea уже есть, возвращаем его
    if pd.notna(row['Ea (eV)']) and row['Ea (eV)'] != '':
        return row['Ea (eV)']
    
    # Собираем все температуры и проводимости
    temp_cols = ['σ total, 200', 'σ total, 250', 'σ total, 300', 'σ total, 350',
                 'σ total, 400', 'σ total, 450', 'σ total, 500', 'σ total, 550',
                 'σ total, 600', 'σ total, 650', 'σ total, 700', 'σ total, 750',
                 'σ total, 800', 'σ total, 850', 'σ total, 900']
    
    temps = []
    sigmas = []
    
    for col in temp_cols:
        if col in row.index and pd.notna(row[col]) and row[col] > 0:
            # Извлекаем температуру из названия колонки
            T = int(col.split(', ')[1])
            temps.append(T + 273.15)  # перевод в Кельвины
            sigmas.append(row[col] * 1e-3)  # перевод из mS/cm в S/cm
    
    # Если меньше 2 точек, возвращаем NaN
    if len(temps) < 2:
        return np.nan
    
    # Уравнение Аррениуса: ln(σ*T) = ln(A) - Ea/(R*T)
    # Преобразуем: y = ln(σ*T), x = 1000/T
    y = np.log(np.array(sigmas) * np.array(temps))
    x = 1000 / np.array(temps)
    
    # Линейная регрессия
    try:
        slope, intercept = np.polyfit(x, y, 1)
        # Ea = -slope * R, где R = 8.314 Дж/(моль·К)
        # Для перевода в эВ: 1 эВ = 96485 Дж/моль
        ea = -slope * 8.314 / 96485  # в эВ
        return ea
    except:
        return np.nan

# ============================================
# ФУНКЦИИ ДЛЯ ГРАФИКОВ
# ============================================

def create_scatter_heatmap(df, x_col, y_col, z_col, x_log, y_log, z_log, 
                           palette, title, xlabel, ylabel, zlabel):
    """
    Создание scatter plot с цветовой шкалой
    """
    # Подготовка данных
    plot_data = df.dropna(subset=[x_col, y_col, z_col])
    
    if len(plot_data) == 0:
        st.warning("Нет данных для построения графика")
        return None
    
    x = plot_data[x_col].values
    y = plot_data[y_col].values
    z = plot_data[z_col].values
    
    # Логарифмирование
    if x_log:
        mask = x > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) == 0:
            st.warning("Все значения x ≤ 0, невозможно построить логарифмический график")
            return None
        x = np.log10(x)
        xlabel = f'log10({xlabel})'
    
    if y_log:
        mask = y > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(y) == 0:
            st.warning("Все значения y ≤ 0, невозможно построить логарифмический график")
            return None
        y = np.log10(y)
        ylabel = f'log10({ylabel})'
    
    if z_log:
        mask = z > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(z) == 0:
            st.warning("Все значения z ≤ 0, невозможно построить логарифмический график")
            return None
        z = np.log10(z)
        zlabel = f'log10({zlabel})'
    
    # Создание графика
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(x, y, c=z, cmap=palette, s=50, 
                         edgecolors='black', linewidth=0.5, alpha=0.8)
    
    # Цветовая шкала
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(zlabel, fontsize=11, fontweight='bold')
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def create_contour_heatmap(df, x_col, y_col, z_col, x_log, y_log, z_log,
                           palette, title, xlabel, ylabel, zlabel, grid_resolution=50):
    """
    Создание контурного графика с интерполяцией
    """
    # Подготовка данных
    plot_data = df.dropna(subset=[x_col, y_col, z_col])
    
    if len(plot_data) < 4:
        st.warning("Недостаточно данных для контурного графика (нужно минимум 4 точки)")
        return None
    
    x = plot_data[x_col].values
    y = plot_data[y_col].values
    z = plot_data[z_col].values
    
    # Логарифмирование
    if x_log:
        mask = x > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) < 4:
            st.warning("Недостаточно данных после логарифмирования")
            return None
        x = np.log10(x)
        xlabel = f'log10({xlabel})'
    
    if y_log:
        mask = y > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) < 4:
            st.warning("Недостаточно данных после логарифмирования")
            return None
        y = np.log10(y)
        ylabel = f'log10({ylabel})'
    
    if z_log:
        mask = z > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) < 4:
            st.warning("Недостаточно данных после логарифмирования")
            return None
        z = np.log10(z)
        zlabel = f'log10({zlabel})'
    
    # Создание регулярной сетки для интерполяции
    xi = np.linspace(x.min(), x.max(), grid_resolution)
    yi = np.linspace(y.min(), y.max(), grid_resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # Интерполяция
    try:
        zi = griddata((x, y), z, (xi, yi), method='cubic')
    except:
        try:
            zi = griddata((x, y), z, (xi, yi), method='linear')
        except:
            st.warning("Не удалось выполнить интерполяцию данных")
            return None
    
    # Создание графика
    fig, ax = plt.subplots(figsize=(8, 6))
    
    contour = ax.contourf(xi, yi, zi, levels=20, cmap=palette)
    contour_lines = ax.contour(xi, yi, zi, levels=20, colors='black', 
                               linewidths=0.5, alpha=0.3)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.2f')
    
    # Точки данных
    ax.scatter(x, y, color='red', s=30, edgecolors='white', 
               linewidth=1, alpha=0.7, label='Data points')
    
    # Цветовая шкала
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(zlabel, fontsize=11, fontweight='bold')
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

def create_3d_surface(df, x_col, y_col, z_col, x_log, y_log, z_log,
                      palette, title, xlabel, ylabel, zlabel):
    """
    Создание 3D поверхностного графика
    """
    # Подготовка данных
    plot_data = df.dropna(subset=[x_col, y_col, z_col])
    
    if len(plot_data) < 4:
        st.warning("Недостаточно данных для 3D графика (нужно минимум 4 точки)")
        return None
    
    x = plot_data[x_col].values
    y = plot_data[y_col].values
    z = plot_data[z_col].values
    
    # Логарифмирование
    if x_log:
        mask = x > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) < 4:
            st.warning("Недостаточно данных после логарифмирования")
            return None
        x = np.log10(x)
        xlabel = f'log10({xlabel})'
    
    if y_log:
        mask = y > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) < 4:
            st.warning("Недостаточно данных после логарифмирования")
            return None
        y = np.log10(y)
        ylabel = f'log10({ylabel})'
    
    if z_log:
        mask = z > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        if len(x) < 4:
            st.warning("Недостаточно данных после логарифмирования")
            return None
        z = np.log10(z)
        zlabel = f'log10({zlabel})'
    
    # Интерполяция для поверхности
    xi = np.linspace(x.min(), x.max(), 30)
    yi = np.linspace(y.min(), y.max(), 30)
    xi, yi = np.meshgrid(xi, yi)
    
    try:
        zi = griddata((x, y), z, (xi, yi), method='cubic')
    except:
        try:
            zi = griddata((x, y), z, (xi, yi), method='linear')
        except:
            st.warning("Не удалось выполнить интерполяцию для 3D графика")
            return None
    
    # Создание 3D графика
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(xi, yi, zi, cmap=palette, alpha=0.8, 
                           linewidth=0, antialiased=True)
    
    # Точки данных
    ax.scatter(x, y, z, color='red', s=30, alpha=0.7, label='Data points')
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_zlabel(zlabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Цветовая шкаба
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(zlabel, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_bubble_chart(df, x_col, y_col, color_col, size_col, 
                        x_log, y_log, color_log, size_log,
                        palette, title, xlabel, ylabel):
    """
    Создание пузырьковой диаграммы
    """
    # Подготовка данных
    plot_data = df.dropna(subset=[x_col, y_col, color_col, size_col])
    
    if len(plot_data) == 0:
        st.warning("Нет данных для построения графика")
        return None
    
    x = plot_data[x_col].values
    y = plot_data[y_col].values
    colors = plot_data[color_col].values
    sizes = plot_data[size_col].values
    
    # Логарифмирование
    if x_log:
        mask = x > 0
        x = x[mask]
        y = y[mask]
        colors = colors[mask]
        sizes = sizes[mask]
        if len(x) == 0:
            st.warning("Все значения x ≤ 0, невозможно построить логарифмический график")
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
            st.warning("Все значения y ≤ 0, невозможно построить логарифмический график")
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
            st.warning("Все значения color ≤ 0, невозможно построить логарифмический график")
            return None
        colors = np.log10(colors)
    
    if size_log:
        mask = sizes > 0
        x = x[mask]
        y = y[mask]
        colors = colors[mask]
        sizes = sizes[mask]
        if len(sizes) == 0:
            st.warning("Все значения size ≤ 0, невозможно построить логарифмический график")
            return None
        sizes = np.log10(sizes)
    
    # Масштабирование размера для визуализации
    if len(sizes) > 0:
        size_min, size_max = sizes.min(), sizes.max()
        if size_max > size_min:
            sizes_scaled = 20 + 180 * (sizes - size_min) / (size_max - size_min)
        else:
            sizes_scaled = np.ones_like(sizes) * 50
    else:
        sizes_scaled = np.ones_like(sizes) * 50
    
    # Создание графика
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(x, y, c=colors, s=sizes_scaled, 
                         cmap=palette, alpha=0.7, edgecolors='black', 
                         linewidth=0.5)
    
    # Цветовая шкала
    cbar = plt.colorbar(scatter, ax=ax)
    if color_log:
        cbar.set_label(f'log10({plot_data[color_col].name})', 
                      fontsize=11, fontweight='bold')
    else:
        cbar.set_label(plot_data[color_col].name, 
                      fontsize=11, fontweight='bold')
    
    # Добавляем информацию о размере
    if size_log:
        size_label = f'log10({plot_data[size_col].name})'
    else:
        size_label = plot_data[size_col].name
    
    # Добавляем легенду для размеров
    from matplotlib.patches import Circle
    import matplotlib.lines as mlines
    
    # Создаем точки для легенды размера
    size_legend_values = [np.percentile(sizes, 25), np.percentile(sizes, 50), 
                          np.percentile(sizes, 75)] if len(sizes) > 0 else [1, 2, 3]
    size_legend_sizes = [20 + 180 * (v - sizes.min()) / (sizes.max() - sizes.min()) 
                         if sizes.max() > sizes.min() else 50 for v in size_legend_values]
    
    legend_elements = []
    for val, size in zip(size_legend_values, size_legend_sizes):
        legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w',
                              label=f'{val:.2f}',
                              markersize=np.sqrt(size/2),
                              markerfacecolor='gray', 
                              markeredgecolor='black'))
    
    # Добавляем легенду
    ax.legend(handles=legend_elements, title=f'Size: {size_label}',
              loc='upper right', frameon=True, framealpha=0.9)
    
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

# ============================================
# ФУНКЦИЯ ДЛЯ СКАЧИВАНИЯ ГРАФИКА
# ============================================

def download_plot(fig, filename="plot.png"):
    """
    Преобразование графика в PNG для скачивания
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
    buf.seek(0)
    return buf

# ============================================
# ОСНОВНАЯ ФУНКЦИЯ ПРИЛОЖЕНИЯ
# ============================================

def main():
    st.set_page_config(
        page_title="Ceramic Conductivity Data Explorer",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Интерактивный анализ проводимости керамических материалов")
    st.markdown("---")
    
    # ============================================
    # Блок A: Загрузка данных (текстовое поле)
    # ============================================
    st.header("📊 Загрузка данных")
    
    st.markdown("""
    **Вставьте данные в формате TSV (табуляция) или CSV (запятая):**
    
    *Пример:*
    """)
    
    data_input = st.text_area(
        "Вставьте данные сюда:",
        height=200,
        placeholder="A cation\tB1 cation\tB2 cation\tB2_cont\tdopant\tdop_cont\tMethod\tT sin\tStructure\tSpace group\ta, Å\tb, Å\tc, Å\tSintering additive\tx, wt%\tρ, %\td, mkm\tσ total, 200\tσ total, 250\tσ total, 300\tσ total, 350\tσ total, 400\tσ total, 450\tσ total, 500\tσ total, 550\tσ total, 600\tσ total, 650\tσ total, 700\tσ total, 750\tσ total, 800\tσ total, 850\tσ total, 900\tEa (eV)\tAtmospheres\tHumidity\tReferences"
    )
    
    if not data_input:
        st.info("⏳ Пожалуйста, вставьте данные для начала работы")
        st.stop()
    
    # Парсинг данных
    try:
        # Определяем разделитель (табуляция или запятая)
        lines = data_input.strip().split('\n')
        if '\t' in lines[0]:
            sep = '\t'
        elif ',' in lines[0]:
            sep = ','
        else:
            st.error("Не удалось определить разделитель. Используйте табуляцию или запятую.")
            st.stop()
        
        # Чтение данных
        df = pd.read_csv(io.StringIO(data_input), sep=sep)
        
        # Очистка колонок от лишних пробелов
        df.columns = df.columns.str.strip()
        
        st.success(f"✅ Данные загружены: {len(df)} строк, {len(df.columns)} колонок")
        st.dataframe(df.head(10))
        
    except Exception as e:
        st.error(f"Ошибка при парсинге данных: {str(e)}")
        st.stop()
    
    # ============================================
    # РАСЧЕТ ДЕСКРИПТОРОВ И Ea
    # ============================================
    st.markdown("---")
    st.header("🔄 Автоматический расчет дескрипторов")
    
    with st.spinner("Вычисление структурных и электроотрицательных дескрипторов..."):
        # Расчет дескрипторов
        desc_df = compute_descriptors(df)
        
        # Добавление к основному датафрейму
        for col in desc_df.columns:
            df[col] = desc_df[col]
        
        st.success("✅ Дескрипторы рассчитаны")
    
    with st.spinner("Расчет энергии активации (Ea)..."):
        # Расчет Ea
        df['Ea_calculated'] = df.apply(calculate_ea, axis=1)
        
        # Используем рассчитанное Ea, если оно есть, иначе берем из колонки
        df['Ea_final'] = df['Ea (eV)'].fillna(df['Ea_calculated'])
        
        st.success("✅ Ea рассчитана")
    
    # Показываем результаты расчета
    st.dataframe(df[['References', 'tolerance_factor', 'chi_B_avg', 
                    'chi_ratio', 'molar_mass', 'porosity', 
                    'grain_boundary_area', 'Ea_final']].head(10))
    
    # ============================================
    # Блок C: Фильтры (сайдбар)
    # ============================================
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Фильтры данных")
    
    # Фильтр по атмосфере
    if 'Atmospheres' in df.columns:
        atmos_options = sorted([x for x in df['Atmospheres'].unique() 
                               if pd.notna(x) and x != ''])
        selected_atmos = st.sidebar.multiselect(
            "Атмосфера",
            options=['All'] + atmos_options,
            default=['All']
        )
        if 'All' not in selected_atmos:
            df_filtered = df[df['Atmospheres'].isin(selected_atmos)]
        else:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
    
    # Фильтр по влажности
    if 'Humidity' in df.columns:
        humidity_options = sorted([x for x in df_filtered['Humidity'].unique() 
                                  if pd.notna(x) and x != ''])
        if humidity_options:
            selected_humidity = st.sidebar.multiselect(
                "Влажность",
                options=['All'] + humidity_options,
                default=['All']
            )
            if 'All' not in selected_humidity:
                df_filtered = df_filtered[df_filtered['Humidity'].isin(selected_humidity)]
    
    # Фильтр по структуре
    if 'Structure' in df.columns:
        structure_options = sorted([x for x in df_filtered['Structure'].unique() 
                                   if pd.notna(x) and x != ''])
        if structure_options:
            selected_structure = st.sidebar.multiselect(
                "Структура",
                options=['All'] + structure_options,
                default=['All']
            )
            if 'All' not in selected_structure:
                # Приводим к нижнему регистру для сравнения
                df_filtered['Structure_lower'] = df_filtered['Structure'].str.lower()
                selected_structure_lower = [s.lower() for s in selected_structure]
                df_filtered = df_filtered[df_filtered['Structure_lower'].isin(selected_structure_lower)]
    
    # Фильтр по спекающей добавке
    if 'Sintering additive' in df.columns:
        additive_options = sorted([x for x in df_filtered['Sintering additive'].unique() 
                                  if pd.notna(x) and x != ''])
        if additive_options:
            selected_additive = st.sidebar.multiselect(
                "Спекающая добавка",
                options=['All'] + additive_options,
                default=['All']
            )
            if 'All' not in selected_additive:
                df_filtered = df_filtered[df_filtered['Sintering additive'].isin(selected_additive)]
    
    # Слайдеры для числовых фильтров
    st.sidebar.markdown("---")
    st.sidebar.subheader("Диапазоны значений")
    
    # T sin
    if 'T sin' in df_filtered.columns:
        t_min = float(df_filtered['T sin'].min()) if not df_filtered['T sin'].isna().all() else 0
        t_max = float(df_filtered['T sin'].max()) if not df_filtered['T sin'].isna().all() else 1000
        if t_max > t_min:
            t_range = st.sidebar.slider(
                "Температура синтеза, °C",
                min_value=t_min,
                max_value=t_max,
                value=(t_min, t_max)
            )
            df_filtered = df_filtered[(df_filtered['T sin'] >= t_range[0]) & 
                                     (df_filtered['T sin'] <= t_range[1])]
    
    # dop_cont
    if 'dop_cont' in df_filtered.columns:
        d_min = float(df_filtered['dop_cont'].min()) if not df_filtered['dop_cont'].isna().all() else 0
        d_max = float(df_filtered['dop_cont'].max()) if not df_filtered['dop_cont'].isna().all() else 1
        if d_max > d_min:
            d_range = st.sidebar.slider(
                "Содержание допанта",
                min_value=d_min,
                max_value=d_max,
                value=(d_min, d_max)
            )
            df_filtered = df_filtered[(df_filtered['dop_cont'] >= d_range[0]) & 
                                     (df_filtered['dop_cont'] <= d_range[1])]
    
    # x, wt% (концентрация добавки)
    if 'x, wt%' in df_filtered.columns:
        x_min = float(df_filtered['x, wt%'].min()) if not df_filtered['x, wt%'].isna().all() else 0
        x_max = float(df_filtered['x, wt%'].max()) if not df_filtered['x, wt%'].isna().all() else 10
        if x_max > x_min:
            x_range = st.sidebar.slider(
                "Концентрация спекающей добавки, wt%",
                min_value=x_min,
                max_value=x_max,
                value=(x_min, x_max)
            )
            df_filtered = df_filtered[(df_filtered['x, wt%'] >= x_range[0]) & 
                                     (df_filtered['x, wt%'] <= x_range[1])]
    
    st.sidebar.markdown(f"**Данных после фильтрации: {len(df_filtered)}**")
    
    # ============================================
    # ОСНОВНЫЕ ВКЛАДКИ
    # ============================================
    tab1, tab2, tab3 = st.tabs(["🌡️ Тепловые карты", "🫧 Пузырьковые диаграммы", "📋 Данные"])
    
    # ============================================
    # ВКЛАДКА 1: ТЕПЛОВЫЕ КАРТЫ
    # ============================================
    with tab1:
        st.header("🌡️ Тепловые карты")
        
        # Определение доступных колонок
        numeric_cols = ['dop_cont', 'x, wt%', 'ρ, %', 'd, mkm', 
                       'tolerance_factor', 'chi_B_avg', 'chi_ratio', 
                       'molar_mass', 'porosity', 'grain_boundary_area']
        
        # Добавляем колонки проводимости
        temp_cols = ['σ total, 200', 'σ total, 250', 'σ total, 300', 'σ total, 350',
                    'σ total, 400', 'σ total, 450', 'σ total, 500', 'σ total, 550',
                    'σ total, 600', 'σ total, 650', 'σ total, 700', 'σ total, 750',
                    'σ total, 800', 'σ total, 850', 'σ total, 900']
        
        available_temp_cols = [col for col in temp_cols if col in df_filtered.columns]
        
        # Авто-выбор температуры с максимальным количеством данных
        temp_counts = {}
        for col in available_temp_cols:
            # Считаем непустые значения для выбранных осей
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
                "Ось X (дескриптор)",
                options=numeric_cols,
                index=0
            )
        
        with col2:
            y_axis = st.selectbox(
                "Ось Y (дескриптор)",
                options=numeric_cols,
                index=1 if len(numeric_cols) > 1 else 0
            )
        
        with col3:
            z_axis = st.selectbox(
                "Цветовая шкала (Z)",
                options=temp_options,
                index=0
            )
        
        # Выбор температуры (если выбрана проводимость)
        if z_axis in available_temp_cols:
            selected_temp = st.selectbox(
                "Выберите температуру для проводимости",
                options=available_temp_cols,
                index=available_temp_cols.index(default_temp) if default_temp in available_temp_cols else 0
            )
            z_col = selected_temp
            z_label = selected_temp.replace('σ total, ', 'σ at ') + ' °C (mS/cm)'
        else:
            z_col = 'Ea_final'
            z_label = 'Ea (eV)'
        
        # Тип графика
        plot_type = st.radio(
            "Тип тепловой карты",
            options=['Scatter с цветовой шкалой', 'Контурный график', '3D поверхность'],
            horizontal=True
        )
        
        # Палитра
        palette_name = st.selectbox(
            "Цветовая палитра",
            options=list(COLOR_PALETTES.keys()),
            index=0
        )
        palette = COLOR_PALETTES[palette_name]
        
        # Логарифмические шкалы
        col1, col2, col3 = st.columns(3)
        with col1:
            log_x = st.checkbox("log10(X)", value=False)
        with col2:
            log_y = st.checkbox("log10(Y)", value=False)
        with col3:
            log_z = st.checkbox("log10(Z)", value=False)
        
        # Кнопка построения
        if st.button("Построить тепловую карту", key="heatmap_btn"):
            if len(df_filtered) == 0:
                st.warning("Нет данных после фильтрации")
            else:
                title = f"{z_label} vs {x_axis} и {y_axis}"
                xlabel = x_axis
                ylabel = y_axis
                
                if plot_type == 'Scatter с цветовой шкалой':
                    fig = create_scatter_heatmap(
                        df_filtered, x_axis, y_axis, z_col,
                        log_x, log_y, log_z, palette,
                        title, xlabel, ylabel, z_label
                    )
                elif plot_type == 'Контурный график':
                    fig = create_contour_heatmap(
                        df_filtered, x_axis, y_axis, z_col,
                        log_x, log_y, log_z, palette,
                        title, xlabel, ylabel, z_label
                    )
                else:  # 3D поверхность
                    fig = create_3d_surface(
                        df_filtered, x_axis, y_axis, z_col,
                        log_x, log_y, log_z, palette,
                        title, xlabel, ylabel, z_label
                    )
                
                if fig is not None:
                    st.pyplot(fig)
                    
                    # Кнопка скачивания
                    buf = download_plot(fig, "heatmap.png")
                    st.download_button(
                        label="📥 Скачать график (PNG, 600 dpi)",
                        data=buf,
                        file_name="heatmap.png",
                        mime="image/png"
                    )
                    
                    plt.close(fig)
    
    # ============================================
    # ВКЛАДКА 2: ПУЗЫРЬКОВЫЕ ДИАГРАММЫ
    # ============================================
    with tab2:
        st.header("🫧 Пузырьковые диаграммы")
        
        # Определение доступных колонок
        numeric_cols_bubble = ['dop_cont', 'x, wt%', 'ρ, %', 'd, mkm', 
                              'tolerance_factor', 'chi_B_avg', 'chi_ratio', 
                              'molar_mass', 'porosity', 'grain_boundary_area']
        
        # Для Y - проводимость или Ea
        y_options = available_temp_cols + ['Ea_final']
        
        # Авто-выбор температуры
        if available_temp_cols:
            default_y = max(available_temp_cols, 
                           key=lambda c: df_filtered[c].count() if c in df_filtered.columns else 0)
            y_index = y_options.index(default_y) if default_y in y_options else 0
        else:
            default_y = 'Ea_final'
            y_index = y_options.index('Ea_final') if 'Ea_final' in y_options else 0
        
        # Оси
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            y_axis_bubble = st.selectbox(
                "Ось Y (проводимость/Ea)",
                options=y_options,
                index=y_index
            )
        
        with col2:
            x_axis_bubble = st.selectbox(
                "Ось X",
                options=numeric_cols_bubble,
                index=0
            )
        
        with col3:
            color_axis = st.selectbox(
                "Цвет пузырьков",
                options=['None'] + numeric_cols_bubble + ['Sintering additive', 'Atmospheres', 'Humidity', 'Structure'],
                index=0
            )
        
        with col4:
            size_axis = st.selectbox(
                "Размер пузырьков",
                options=['None'] + numeric_cols_bubble,
                index=0
            )
        
        # Выбор температуры (если выбрана проводимость)
        if y_axis_bubble in available_temp_cols:
            y_label = y_axis_bubble.replace('σ total, ', 'σ at ') + ' °C (mS/cm)'
        else:
            y_label = 'Ea (eV)'
        
        # Палитра
        palette_name_bubble = st.selectbox(
            "Цветовая палитра для пузырьков",
            options=list(COLOR_PALETTES.keys()),
            index=0
        )
        palette_bubble = COLOR_PALETTES[palette_name_bubble]
        
        # Логарифмические шкалы
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            log_x_bubble = st.checkbox("log10(X)", value=False, key="log_x_bubble")
        with col2:
            log_y_bubble = st.checkbox("log10(Y)", value=False, key="log_y_bubble")
        with col3:
            log_color_bubble = st.checkbox("log10(Color)", value=False, key="log_color_bubble")
        with col4:
            log_size_bubble = st.checkbox("log10(Size)", value=False, key="log_size_bubble")
        
        # Кнопка построения
        if st.button("Построить пузырьковую диаграмму", key="bubble_btn"):
            if len(df_filtered) == 0:
                st.warning("Нет данных после фильтрации")
            else:
                # Проверка наличия данных
                if color_axis == 'None':
                    # Используем константный цвет
                    df_temp = df_filtered.copy()
                    df_temp['_color_temp'] = 1
                    color_col_bubble = '_color_temp'
                else:
                    color_col_bubble = color_axis
                
                if size_axis == 'None':
                    df_temp = df_filtered.copy()
                    df_temp['_size_temp'] = 1
                    size_col_bubble = '_size_temp'
                else:
                    size_col_bubble = size_axis
                
                title = f"{y_label} vs {x_axis_bubble}"
                xlabel = x_axis_bubble
                
                fig = create_bubble_chart(
                    df_filtered, x_axis_bubble, y_axis_bubble, 
                    color_col_bubble, size_col_bubble,
                    log_x_bubble, log_y_bubble, log_color_bubble, log_size_bubble,
                    palette_bubble, title, xlabel, y_label
                )
                
                if fig is not None:
                    st.pyplot(fig)
                    
                    # Кнопка скачивания
                    buf = download_plot(fig, "bubble_chart.png")
                    st.download_button(
                        label="📥 Скачать график (PNG, 600 dpi)",
                        data=buf,
                        file_name="bubble_chart.png",
                        mime="image/png"
                    )
                    
                    plt.close(fig)
    
    # ============================================
    # ВКЛАДКА 3: ДАННЫЕ
    # ============================================
    with tab3:
        st.header("📋 Данные с рассчитанными дескрипторами")
        
        st.dataframe(df_filtered)
        
        # Кнопка скачивания данных
        csv_data = df_filtered.to_csv(index=False, sep='\t')
        st.download_button(
            label="📥 Скачать данные (TSV)",
            data=csv_data,
            file_name="filtered_data.tsv",
            mime="text/tab-separated-values"
        )
        
        # Статистика
        st.subheader("📊 Статистика по данным")
        numeric_display = df_filtered.select_dtypes(include=[np.number])
        if not numeric_display.empty:
            st.dataframe(numeric_display.describe())
    
    # ============================================
    # ЗАПУСК ПРИЛОЖЕНИЯ
    # ============================================
    
    if __name__ == "__main__":
    main()
