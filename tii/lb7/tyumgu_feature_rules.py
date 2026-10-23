# Правила преобразования признаков для занятий 7–8
import numpy as np
import pandas as pd
NUM_BASE = ['distance_km', 'queue_size',  'promised_minutes']
CAT = ['zone', 'weather']
NUM = NUM_BASE + ['hour_sin', 'hour_cos', 'planned_speed_kmh']
INPUT = NUM_BASE + CAT + ['created_at']

def make_features(X):
    f = X.loc[:, INPUT].copy()
    for name in NUM_BASE:
        f[name] = pd.to_numeric(f[name], errors='coerce')
    for name in NUM_BASE:
        f.loc[f[name] < 0, name] = np.nan
    f.loc[f['promised_minutes'] == 0, 'promised_minutes'] = np.nan
    for name in CAT:
        s = f[name].astype('string').str.strip()
        s = s.mask(s.str.lower().isin(['', 'nan', 'none', 'нет данных', 'неизвестно', 'null']))
        f[name] = s.astype(object).where(s.notna(), np.nan)
    time = pd.to_datetime(f['created_at'], errors='coerce')
    hour = time.dt.hour.astype(float)
    f['hour_sin'] = np.sin(2*np.pi*hour/24)
    f['hour_cos'] = np.cos(2*np.pi*hour/24)
    f['planned_speed_kmh'] = 60*f['distance_km']/f['promised_minutes']
    f['planned_speed_kmh'] = f['planned_speed_kmh'].replace([np.inf, -np.inf], np.nan)
    return f.drop(columns='created_at')
