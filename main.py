import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
import geopandas
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import Point
import watchdog

with st.echo(code_location='below'):
    st.title("Wildfire Visualisation")

    st.markdown("Привет! В этом проекте я визуализирую данные о природных пожарах в США с 1992 по 2015. Это "
                "полезное упражнение, которое призвано подсветить проблему пожаров, которые опасны для всех участников "
                "экосистем: растений, животных, людей.")

    st.markdown("Сначала расскажу план:")
    st.markdown("1. Загрузка данных")
    st.markdown("2. Подготовка данных")
    st.markdown("3. Построение карт без данных")
    st.markdown("4. Построение интерактивных карт с точками")
    st.markdown("5. Построение интерактивных карт с пузырьками (bubble maps)")
    st.markdown("5. Подгрузка данных о населенности, температуре и влажности воздуха")
    st.markdown("6. Построение heat maps с данными о населенности, температуре и влажности")
    st.markdown("7. Построение bubble maps с учётом площади, населенности, температуры и влажности")
    st.markdown("8. Выводы о том, в каких штатах ситуация с пожарами лучше/хуже других")

    st.markdown("# 1. Загрузка данных")

    st.markdown("Сначала я подгружаю данные, которые хранятся как база данных SQLite. Чтобы скормить в Pandas, "
                "я использую библиотеку ``sqlite3``. Правда, у меня не получилось загрузить такую "
                "большую базу данных (800+ Mb) в Github, поэтому я локально сохраняю её в .csv и сжимаю. "
                "Попутно я выбрасываю ненужные мне колонки, что облегчает файл в 6 раз.")

    df = pd.read_csv("data compressed.csv", compression="gzip"
                     )
    st.write(df.columns)

    st.markdown("Как выглядят данные? Много непонятных колонок с данными разных типов:")

    st.write(df.head(5))

    st.markdown("В таком виде данные не очень информативны. Хорошая новость в том, что данные — пространственные. "
                "Это значит, что их можно отобразить на карте. Звучит круто, да?")

    st.markdown("# 2. Подготовка данных")

    st.markdown("Датасет *был* довольно большой (но в датафрейме лежит меньшая его версия, где нет многих колонок) "
                "и содержит много технических данных, которые мне не нужны. Такие данные уже дропнуты."
                "Вот, что я оставляю:")
    st.markdown("1. Дата возникновения пожара")
    st.markdown("2. Широта и долгота точки с пожаром")
    st.markdown("3. Код и класс пожара")
    st.markdown("4. Причина пожара")
    st.markdown("5. Штат и округ (county)")
    st.markdown("6. Форма (shape)")

    st.markdown("Теперь я преобразую пространственные данные, пользуясь ``geopandas``. Это удобная библиотека,"
                "расширяющая возможности ``pandas`` в части работы с геоданными.")

    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df["LONGITUDE"], df["LATITUDE"]))

    st.write(gdf.head(5))

    st.markdown("Видишь? Теперь данные хранятся в удобном для переваривания многими пакетами формате, "
                "который называется "
                "[Well-Known Text](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry) (WKT)")

    st.markdown("# 3. Построение карт без данных")

    st.markdown("Начинается самое интересное: картинки.")

    st.markdown("Начинаю с простого: учусь строить карты без данных в каждой библиотеке.")

    plain_map_lib = st.selectbox(
        label="Выбери библиотеку из выпадающего списка, а я построю карту с её помощью.",
        options=['plotly', 'geoplot', 'folium', 'bokeh', 'seaborn', 'altair', 'basemap'],
    )

    st.markdown(f"Покажу карту при помощи ``{plain_map_lib}``")

    if plain_map_lib == "plotly":
        pl_plain_map = go.Figure(go.Scattergeo())
        pl_plain_map.update_geos(scope="usa")
        st.write(pl_plain_map)

    st.markdown("# 4. Построение простых bubble maps")

    scatter_map_lib = st.selectbox(
        label="Выбери библиотеку из выпадающего списка, а я с её помощью построю карту с точками пожаров.",
        options=['plotly', 'geoplot', 'folium', 'bokeh', 'seaborn', 'altair', 'basemap'],
    )

    st.markdown(f"Покажу, как строю карту с точками пожаровпри помощи ``{scatter_map_lib}``")

    scatter_dates = st.slider(
        label="Выбери промежуток времени (оба конца включаются):",
        min_value=2014, max_value=2015,
        value=(2014, 2015)
    )

    scatter_relative_size = st.checkbox(
        label="Хочешь, чтобы график показывал относительный размер пожаров?", value=True
    )

    if scatter_map_lib == "plotly":
        pl_scatter_map = px.scatter_geo(
            gdf.loc[
                (gdf["FIRE_YEAR"] >= scatter_dates[0]) & (gdf["FIRE_YEAR"] <= scatter_dates[1])
                ],
            lon=df["LONGITUDE"], lat=df["LATITUDE"], size=df["FIRE_SIZE"], scope="usa"
        )
        pl_scatter_map.update_layout(title_text=f"Карта пожаров в США с {scatter_dates[0]} по {scatter_dates[1]} год")
        st.write(pl_scatter_map)