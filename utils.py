import os
import pandas as pd
from scipy import special


def get_user_esms(filepath):
    '''
    Get esms for each user given filepath, which is a path to esm_data.csv.

    Returns
    -------
    dict of the form {'uid': dataframe} where a dataframe contains esms for a user specified by the uid.
    '''
    esm_data = pd.read_csv(filepath_or_buffer=filepath,
                           names=['uid', 'response_dt', 'response_ts',
                                  'valence', 'arousal', 'attention',
                                  'stress', 'duration', 'disturbance', 'emotion_change'],
                           skiprows=[0])

    # fill NaN values in duration column
    esm_data.duration.fillna(value=0, inplace=True)
    return {uid: esm_data.loc[esm_data['uid'] == uid] for uid in esm_data['uid'].unique()}


def load_aggregated(loadpath, uid):
    '''Loads aggregated data as pandas dataframe from loadpath given uid.'''
    for filename in os.listdir(loadpath):
        if str(uid) == filename.split('_')[0]:
            filepath = os.path.join(loadpath, filename)
            return pd.read_pickle(filepath, compression='gzip')
            
    return None


def chauvenet(data):
    '''
    Implements Chauvenet's criterion for outlier detection.
    
    * See: https://en.wikipedia.org/wiki/Chauvenet%27s_criterion
    Parameters
    ----------
    data: a Pandas DataFrame or named Series.
    
    Returns
    -------
    A masked object of same shape and type as data, with Trues in where input
    values were outliers, and Falses in where values were not outliers.
    
    '''

    mean, std, N = data.mean(), data.std(), len(data)  # mean, standard deviation, and length of input data
    criterion = 1.0 / (2 * N)                          # Chauvenet's criterion
    d = abs(data - mean) / std                         # distance of a value to mean in stdev.'s
    prob = special.erfc(d)                             # area of normal dist.
    
    return prob < criterion                            # if prob is below criterion, a value is an outlier