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
from sklearn import linear_model


with st.echo(code_location='below'):
    st.title("Wildfire Visualisation")

    st.markdown("Привет! В этом проекте я визуализирую данные о природных пожарах в США с 1992 по 2015. Это "
                "полезное упражнение, которое призвано подсветить проблему пожаров, которые опасны для всех участников "
                "экосистем: растений, животных, людей.")

    st.markdown("## 1. Загрузка данных")

    st.markdown("Сначала я подгружаю данные, которые хранятся как база данных SQLite. Чтобы скормить в Pandas, "
                "я использую библиотеку ``sqlite3``. Правда, у меня не получилось загрузить такую "
                "большую базу данных (800+ Mb) в Github, поэтому я локально сохраняю её в .csv и сжимаю. "
                "Попутно я выбрасываю ненужные мне колонки, что облегчает файл в 6 раз.")

    df = pd.read_csv("data compressed since 2010.csv", compression="gzip")

    st.markdown("Как выглядят данные? Много непонятных колонок с данными разных типов:")

    st.write(df.head(5))

    st.markdown("В таком виде данные не очень информативны. Хорошая новость в том, что данные — пространственные. "
                "Это значит, что их можно отобразить на карте. Звучит многообещающе :smile:")

    st.markdown("# 2. Подготовка и описание данных")

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

    st.write(gdf.drop(["geometry"], inplace=False, axis=1).head(5))

    st.markdown("Видишь? Теперь данные хранятся в удобном для переваривания многими пакетами формате, "
                "который называется "
                "[Well-Known Text](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry) (WKT)")

    st.markdown("## 3. Построение карт без данных")

    st.markdown("Начинается самое интересное: картинки.")

    st.markdown("Начинаю с простого: учусь строить карты без данных в каждой библиотеке.")

    plain_map_lib = st.selectbox(
        label="Выбери библиотеку из выпадающего списка, а я построю карту с её помощью.",
        options=['plotly', 'geoplot', 'folium', 'bokeh', 'seaborn', 'altair', 'basemap'], key=1
    )

    st.markdown(f"Покажу карту при помощи ``{plain_map_lib}``")

    if plain_map_lib == "plotly":
        pl_plain_map = go.Figure(go.Scattergeo())
        pl_plain_map.update_geos(scope="usa")
        pl_plain_map.update_layout(title="Просто карта США")
        st.write(pl_plain_map)


    st.markdown("## 4. Построение простых bubble maps")

    bubble_map_data = df["STATE"].value_counts().reset_index(name="COUNT")

    st.markdown(f"Покажу, как строю bubble map при помощи ``plotly``")

    pl_bubble_map = px.scatter_geo(bubble_map_data, locations="index", locationmode="USA-states", scope="usa",
                                  size="COUNT", title="Bubble map пожаров на территории США с 2010 по 2015",
                                  labels={"index": "State", "COUNT": "# Fires"})
    st.write(pl_bubble_map)


    st.markdown("Выглядит так, как будто хуже всего ситуация в Техасе и Калифорнии (что неудивительно, "
                "учитывая климат в этих штатах и их огромные размеры), а также — к удивлению — в штате Нью-Йорк.")

    st.markdown("## 5. Построение динамических bubble maps")

    st.markdown("Интересно, а что там в динамике? Может, в каких-то штатах все становится хуже и хуже?")

    dynamic_bubble_map_data = df.drop("geometry", axis=1, inplace=False)[["STATE", "FIRE_YEAR"]].groupby(
        ["STATE", "FIRE_YEAR"]).size().reset_index().rename(columns={0: "COUNT"})

    dynamic_bubble_map_lib = st.selectbox(
        label="Выбери библиотеку, которая отрисует картинку в динамике",
        options=['plotly', 'geoplot', 'folium', 'bokeh', 'seaborn', 'altair', 'basemap'], key=3
    )

    if dynamic_bubble_map_lib == "plotly":
        pl_dynamic_bubble_map = px.scatter_geo(dynamic_bubble_map_data, locations="STATE", locationmode="USA-states",
                                               scope="usa", size="COUNT", animation_frame="FIRE_YEAR",
                                               title="Динамическая bubble map пожаров на территории США",
                                               labels={"STATE": "State", "COUNT": "# Fires"})
        st.write(pl_dynamic_bubble_map)

    st.markdown("## 6. Построение heatmaps, отображающих динамику")

    st.markdown("Хочется какой-то более понятной визуализации. Давай сделаем какую-нибудь метрику и отобразим "
                "её на статической карте")

    st.markdown("Придумывать что-то сложное не хочется. Давай просто сделаем регрессию логарифма количества "
                "пожаров на год, и коэффициент наклона будет нашей метрикой. Всё просто: чем он больше, тем хуже "
                "динамика.")

    coefs = pd.DataFrame(data={"STATE": dynamic_bubble_map_data["STATE"].unique(),
                               "COEF": np.array([0 for i in range(dynamic_bubble_map_data["STATE"].nunique())])})
    for state in dynamic_bubble_map_data["STATE"].unique():
        reg = linear_model.LinearRegression().fit(
            X=dynamic_bubble_map_data.loc[dynamic_bubble_map_data["STATE"] == state, ["FIRE_YEAR"]],
            y=dynamic_bubble_map_data.loc[dynamic_bubble_map_data["STATE"] == state, ["COUNT"]]
        )
        coefs.loc[coefs["STATE"] == state, ["COEF"]] = reg.coef_[0]

    regression_heat_map_lib = st.selectbox(
        label="Выбери библиотеку, которая отрисует картинку в динамике",
        options=['plotly', 'geoplot', 'folium', 'bokeh', 'seaborn', 'altair', 'basemap'], key=4
    )

    if regression_heat_map_lib == "plotly":
        pl_regression_heat_map = px.choropleth(coefs, locations="STATE", locationmode="USA-states",
                                               scope="usa", color="COEF", color_continuous_scale='oranges',
                                               labels={"STATE": "State", "COEF": "Avg. growth of # Fires"},
                                               title="Тренд количества пожаров по штатам США")
        st.write(pl_regression_heat_map)

    st.markdown("Похоже, в Канзасе, Аризоне и Массачусетсе ситуация становится всё хуже и хуже, а в Техасе и "
                "Джорджии — напротив, динамика довольно оптимистична.")

    st.markdown("## 7. Построение динамических heatmaps по месяцам")

    st.markdown("Интересно посмотреть, как пожары привязаны к временам года. Давай построим визуализацию!")

    def is_leap(year):
        return (int(year) % 4 == 0) & (int(year) % 100 != 0)

    def get_month(year, doy_str):
        doy = int(np.floor(float(doy_str)))
        if is_leap(year):
            last_days = {31: "01", 60: "02", 91: "03", 121: "04", 152: "05", 182: "06", 213: "07", 244: "08",
                         274: "09", 305: "10", 335: "11", 366: "12"}
        else:
            last_days = {31: "01", 59: "02", 90: "03", 120: "04", 151: "05", 181: "06", 212: "07", 243: "08",
                         273: "09", 304: "10", 334: "11", 365: "12"}
        last_days_list = np.array(list(last_days.keys()))
        try:
            month = last_days[np.amin(last_days_list[last_days_list >= doy])]
        except ValueError:
            month = "NAN"
        return month

    monthly_data = df.drop("geometry", axis=1, inplace=False)[["STATE", "FIRE_YEAR", "DISCOVERY_DOY"]]
    monthly_data["MONTH"] = [get_month(row[2], row[3]) for row in monthly_data.itertuples()]
    monthly_data["YYYYMMM"] = monthly_data.apply(lambda x: str(x["FIRE_YEAR"]) + '-' + str(x["MONTH"]), axis=1)
    monthly_data.drop(["MONTH", "DISCOVERY_DOY", "FIRE_YEAR"], inplace=True, axis=1)
    monthly_data = monthly_data.groupby(["STATE", "YYYYMMM"]).size().reset_index().rename(columns={0: "COUNT"})

    st.markdown("Представлю данные в виде динамической heatmap.")

    dynamic_heat_map_lib = st.selectbox(
        label="Выбери библиотеку, которая отрисует картинку в динамике",
        options=['plotly', 'geoplot', 'folium', 'bokeh', 'seaborn', 'altair', 'basemap'], key=5
    )
    if dynamic_heat_map_lib == "plotly":
        pl_dynamic_heat_map = px.choropleth(monthly_data, locations="STATE", locationmode="USA-states", scope="usa",
                                            color="COUNT", animation_frame="YYYYMMM", color_continuous_scale="oranges",
                                            range_color=[0, monthly_data["COUNT"].max()],
                                            category_orders={"YYYYMMM": pd.Series(monthly_data["YYYYMMM"].unique()).sort_values().tolist()},
                                            labels={"STATE": "State", "COUNT": "# Fires"},
                                            title="Визуализация количества пожаров в США по месяцам"
                                            )
        st.write(pl_dynamic_heat_map)

    st.markdown("Видно, что есть сезонность, особенно заметная на примере Калифорнии и Техаса: летом там существенно "
                "больше пожаров. Тем не менее, это не всегда правда: иногда много пожаров и в зимние месяцы.")

    st.markdown("## 8. Построение самых крупных пожаров")

    st.markdown("Интересно посмотреть на самые крупные пожары.")

    st.markdown("Сначала посмотрим, где произошли 20 самых крупных пожаров")

    st.write(df.columns)

    threshold_size = df["FIRE_SIZE"].sort_values(inplace=False, ascending=False)[20]
    greatest_fires = df[df["FIRE_SIZE"] >= threshold_size].reset_index()

    greatest_fires_map = px.scatter_geo(greatest_fires, lat="LATITUDE", lon="LONGITUDE",
                                        labels={"STATE": "State", "FIRE_SIZE": "Size",
                                                "STATE_CODE_DESCR": "Description"},
                                        title="20 крупнейших пожаров в США за 2010-2015 годы")
    st.write(greatest_fires_map)

    st.markdown("## 9. Выводы")

