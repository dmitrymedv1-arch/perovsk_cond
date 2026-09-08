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
