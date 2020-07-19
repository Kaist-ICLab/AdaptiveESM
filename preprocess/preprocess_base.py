import os
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from os.path import expanduser
from sklearn import preprocessing
from utils import get_user_esms, load_aggregated, chauvenet


def extraction_fn(segment):
    """Apply window of size 100ms w/o overlap to 1-min segment, and fill in missing values"""
    windowed = segment.resample('100ms', label='right', closed='right').mean()
    windowed.interpolate(method='linear', inplace=True)
    windowed.fillna(method='bfill', inplace=True)
    windowed.index = windowed.index.values.astype(np.int64) / 1e6
    return windowed


def process_segment(i, esm, userdata, savepath_user):
    ts_end = esm.response_ts * 1e3 - (i * 60 * 1e3)
    ts_start = ts_end - 60 * 1e3 + 1
    segment = userdata.loc[lambda x: (x.index >= ts_start) & (x.index < ts_end)]

    # if any data was entered between t_start & t_end, and none among selected features are entirely empty
    if len(segment) > 0 and segment.notnull().sum().all():
        print(f'\tProcessing ESM#{esm.Index}-{i+1}...')

        # reindex segment with datetime index of frequency = 1ms
        segment.index = pd.to_datetime(segment.index, unit='ms')
        dt_end = datetime.fromtimestamp(ts_end / 1e3)
        dt_start = datetime.fromtimestamp(ts_start / 1e3)
        segment = segment.reindex(pd.date_range(start=dt_start, end=dt_end, freq='1ms'))

        segment = extraction_fn(segment)
        saveto = os.path.join(savepath_user, f'{str(esm.Index)}-{i+1}.npy')
        np.save(saveto, segment.values)
        print(f'\t\tSaved feature ndarray to {saveto}.')
    else:
        print(f'\tEmpty segment for ESM#{esm.Index}-{i+1}.')
    return


def preprocess(name, augment, paths, cols, features):
    tic = time.time()
    print('Starting preprocessing...', end='\n'+'='*80+'\n')
    loadpath, savepath, esm_path = expanduser(paths['load']), expanduser(paths['save']), expanduser(paths['esm'])

    if augment:
        name = f'{name}-aug'
    savepath = os.path.join(savepath, name)
    esm_per_user = get_user_esms(esm_path)

    # for each user
    for uid, esms in esm_per_user.items():
        start = time.time()
        savepath_user = os.path.join(savepath, str(uid))
        print(f'Processing data for user {uid}...')

        # make directories to save data from current user
        try:
            os.makedirs(savepath_user)
        except OSError as err:
            print(err)
            continue

        # load user data
        try:
            userdata = load_aggregated(loadpath, uid)[cols]
        except TypeError as err:
            print(err, f'No data for user {uid}.', end='\n'+'-'*80+'\n')
            continue

        # set HeartRate-BPM to nan where HeartRate-Quality = 'ACQUIRING'
        userdata['HeartRate-BPM'] = userdata['HeartRate-BPM'].mask(userdata['HeartRate-Quality'] == 'ACQUIRING')
        userdata.drop(columns=['HeartRate-Quality'], inplace=True)
        # remove outliers
        userdata = userdata.mask(chauvenet(userdata))
        # apply min-max scaling
        scaler = preprocessing.MinMaxScaler()
        userdata = pd.DataFrame(scaler.fit_transform(userdata.values), index=userdata.index, columns=features)

        # for all esms for the current user
        for esm in esms.itertuples():
            if augment and esm.duration > 0:
                for i in range(5):
                    process_segment(i, esm, userdata, savepath_user)
            else:
                process_segment(0, esm, userdata, savepath_user)

        end = time.time()
        print(f'Processed data for user {uid} in {end-start:2f}s.', end='\n'+'-'*80+'\n')

    toc = time.time()
    print(f'Finished preprocessing in {toc-tic}s.', end='\n'+'='*80+'\n')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess DailyLife dataset and save processed features in numpy file format.')
    parser.add_argument('--name', '-n', type=str, required=True)
    parser.add_argument('--augment', default=False, action='store_true')
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    args = parser.parse_args()

    PATHS = {
        'load': '~/data/dailyLife2/aggregated',
        'save': '~/data/dailyLife2/datasets',
        'esm': '~/data/dailyLife2/metadata/esm_data.csv',
    }
    COLS = [
        'Accelerometer-Y', 'Accelerometer-X', 'Accelerometer-Z',
        'Gsr-Resistance', 'HeartRate-Quality', 'HeartRate-BPM',
        'SkinTemperature-Temperature', 'RRInterval-Interval'
    ]
    FEATURES = ['y', 'x', 'z', 'gsr', 'bpm', 'temp', 'rri']

    preprocess(name=args.name, augment=args.augment, paths=PATHS, cols=COLS, features=FEATURES)
