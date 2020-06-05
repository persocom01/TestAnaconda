# Demonstrates the use of resampling, which is primarily used for time series
# data to up or down scale datetime intervals.
import datetime
import json
import pandas as pd

station_id = 'S116'


def get_temperature_features(station_id):
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    rainfall_api = 'https://api.data.gov.sg/v1/environment/air-temperature'
    request_string = rainfall_api + '?date=' + yesterday.strftime('%Y-%m-%d')
    temperature_features = {}

    # Get data from api.
    # r = requests.get(request_string)
    # if r.status_code == 200:
    #     data = r.json()

    # For testing on file.
    file = './datasets/air_temp_20171212.json'
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def get_reading(readings, station_id):
        for r in readings:
            if r['station_id'] == station_id:
                return r['value']
        return 0

    timestamps = [x['timestamp'] for x in data['items']]
    temperatures = [get_reading(x['readings'], station_id) for x in data['items']]
    df = pd.DataFrame()
    df['timestamp'] = pd.to_datetime(timestamps)
    df['temperature'] = temperatures
    # Some timestamps are missing, so the data has to be resampled. .mean()
    # converts the DatetimeIndexResampler object back into a DataFrame.
    df = df.resample('1T', on='timestamp').mean()
    # The missing timestamps become filled with nan values, which have to be filled
    # in. Demonstrated here is average fill.
    df = (df.ffill()+df.bfill())/2
    # In case the first and last values are nan.
    df = df.bfill().ffill()

    temperature_features = {
        'mean_temperature_c': round(df['temperature'].mean(), 1),
        'maximum_temperature_c': round(df['temperature'].max(), 1),
        'minimum_temperature_c': round(df['temperature'].min(), 1)
    }
    return temperature_features


# The correct test values are 26.6 30.4 24.8
print(get_temperature_features(station_id))
