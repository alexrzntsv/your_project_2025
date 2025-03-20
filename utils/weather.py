from datetime import datetime

import pandas as pd
from meteostat import Hourly, Point


def get_meteostat_data(
        latitude: float,
        longitude: float,
        year: int = 2023,
) -> pd.DataFrame:
    location = Point(latitude, longitude)

    data = Hourly(
        location,
        start=datetime(year, 1, 1),
        end=datetime(year, 12, 31),
    )
    data = data.fetch()

    return data[['temp']].reset_index().rename(columns={
        'time': 'datetime',
        'temp': 'temperature'
    })