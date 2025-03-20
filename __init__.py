import requests
import pandas as pd


def get_weather_data(city: str, api_key: str) -> pd.DataFrame:
    """
    Получает данные о погоде для указанного города и возвращает DataFrame.

    Параметры:
        city (str): Название города на английском языке (например: 'Moscow')
        api_key (str): Ваш API-ключ от OpenWeatherMap

    Возвращает:
        pd.DataFrame: DataFrame с колонками 'datetime' и 'temperature'
    """
    # Формируем URL запроса
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"

    # Отправляем запрос
    response = requests.get(url)

    # Проверяем статус ответа
    if response.status_code != 200:
        raise Exception(f"Ошибка получения данных. Код: {response.status_code}")

    # Парсим JSON
    data = response.json()

    # Извлекаем необходимые данные
    weather_list = data['list']
    result = []

    for item in weather_list:
        result.append({
            'datetime': item['dt_txt'],
            'temperature': item['main']['temp']
        })

    # Создаем DataFrame
    return pd.DataFrame(result)


# Пример использования
api_key = "072a23353a06497a099adedd4f5b0f00"  # Замените на ваш ключ
df = get_weather_data("Nizhny Novgorod", api_key)
print(df.head())