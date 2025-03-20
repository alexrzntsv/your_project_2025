from geopy.geocoders import Nominatim


def osm_geocode(city: str, country: str = "RU") -> tuple:
    """
    Возвращает (широта, долгота) через OpenStreetMap
    Требует установки geopy: pip install geopy
    """
    geolocator = Nominatim(user_agent="city_locator")
    location = geolocator.geocode(f"{city}, {country}")

    if location:
        return (location.latitude, location.longitude)
    return None