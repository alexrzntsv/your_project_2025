from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=(retry_if_exception_type(GeocoderUnavailable) | retry_if_exception_type(GeocoderTimedOut)),
)
def osm_geocode(city: str, country: str = "RU") -> tuple:
    """
    Получает координаты города с обработкой ошибок и повторами

    Аргументы:
        city (str): Название города на русском языке
        country (str): Код страны (по умолчанию 'RU')

    Возвращает:
        tuple: (широта, долгота) или None при ошибке
    """
    geolocator = Nominatim(user_agent="my_weather_app/1.0 (contact@example.com)")

    try:
        location = geolocator.geocode(
            query=f"{city}, {country}",
            exactly_one=True,
            timeout=15
        )

        if location:
            return (location.latitude, location.longitude)
        return None

    except (GeocoderUnavailable, GeocoderTimedOut) as e:
        print(f"Временная ошибка геокодера: {str(e)}")
        raise  # Для повторных попыток через tenacity

    except Exception as e:
        print(f"Критическая ошибка при геокодировании {city}: {str(e)}")
        return None
