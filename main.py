from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import seaborn as sns
from scipy.stats import pearsonr, kendalltau

from utils.geo import osm_geocode
from utils.weather import get_meteostat_data


def generate_sample_data():
    cities = (
        'Москва',
        'Санкт-Петербург',
        'Нижний Новгород',
        'Казань',
        'Уфа',
        'Новосибирск',
        'Владивосток',
    )
    all_data = []

    for city in cities:
        latitude, longitude = osm_geocode(city)
        city_weather_df = get_meteostat_data(latitude, longitude, 2023)
        city_weather_df["city"] = city
        all_data.append(city_weather_df)

    if not all_data:
        st.error("Не удалось получить данные. Проверьте API ключ и подключение к интернету.")
        return pd.DataFrame()

    df = pd.concat(all_data)
    return df


def preprocess_data(data, freq, season):
    df = data.pivot(index='datetime', columns='city', values='temperature')
    df = df.resample(freq).mean()

    seasons = {
        'зима': [12, 1, 2],
        'весна': [3, 4, 5],
        'лето': [6, 7, 8],
        'осень': [9, 10, 11]
    }

    if season == 'зима':
        mask = df.index.month.isin([12, 1, 2])
    else:
        mask = df.index.month.isin(seasons[season])

    return df[mask]


def calculate_correlations(data, method='pearson'):
    cities = data.columns
    corr_matrix = pd.DataFrame(index=cities, columns=cities)

    for (i, j) in combinations(cities, 2):
        if method == 'pearson':
            corr, _ = pearsonr(data[i], data[j])
        elif method == 'kendall':
            corr, _ = kendalltau(data[i], data[j])
        corr_matrix.loc[i, j] = corr
        corr_matrix.loc[j, i] = corr

    np.fill_diagonal(corr_matrix.values, 1.0)
    return corr_matrix.astype(float)


def analyze_graph(corr_matrix, threshold):
    G = nx.Graph()
    cities = corr_matrix.columns

    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                G.add_edge(cities[i], cities[j], weight=corr_matrix.iloc[i, j])

    cliques = list(nx.find_cliques(G))
    independent_sets = list(nx.maximal_independent_set(G))

    return G, cliques, independent_sets


def plot_results(G, cliques, independent_sets):
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12)

    plt.title("Сетевая модель температурных корреляций")
    st.pyplot(plt)

    st.subheader("Клики:")
    st.write(pd.DataFrame(cliques))

    st.subheader("Независимые множества:")
    st.write(independent_sets)


def main():
    st.title("Анализ климатических сетей России")

    # Генерация данных
    data = generate_sample_data()

    # Выбор параметров
    freq = st.sidebar.selectbox("Частота наблюдений:", ['H', 'D', 'M'], index=2)
    season = st.sidebar.selectbox("Сезон:", ['зима', 'весна', 'лето', 'осень'])
    threshold = st.sidebar.slider("Порог корреляции:", 0.0, 1.0, 0.5, 0.05)

    # Предобработка
    processed_data = preprocess_data(data, freq, season)

    # Анализ
    pearson_corr = calculate_correlations(processed_data, 'pearson')
    kendall_corr = calculate_correlations(processed_data, 'kendall')

    # Визуализация матриц
    st.subheader("Матрица корреляций Пирсона")
    plt.figure(figsize=(10, 8))
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm')
    st.pyplot(plt)

    st.subheader("Матрица корреляций Кендалла")
    plt.figure(figsize=(10, 8))
    sns.heatmap(kendall_corr, annot=True, cmap='coolwarm')
    st.pyplot(plt)

    # Анализ графов
    G_pearson, cliq_p, ind_p = analyze_graph(pearson_corr, threshold)
    G_kendall, cliq_k, ind_k = analyze_graph(kendall_corr, threshold)

    st.header("Результаты для Пирсона")
    plot_results(G_pearson, cliq_p, ind_p)

    st.header("Результаты для Кендалла")
    plot_results(G_kendall, cliq_k, ind_k)


if __name__ == "__main__":
    main()