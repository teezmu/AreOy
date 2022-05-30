import pandas as pd
import datetime as dt
import numpy as np


def data_for_LSTM_model(df, scale=False):

    df['Date'] = pd.to_datetime(df['Date'])
    # New column seconds
    df['Seconds'] = df['Date'].map(pd.Timestamp.timestamp)
    df = df[['Date', 'Kosteus', 'Lampotila', 'temperature', 'humidity', 'Seconds']]
    df = df.set_index(df['Date'], drop=True)

    # New columns 
    day = 60*60*24
    year = 365.2425*day
    df['Day sin'] = np.sin(df['Seconds'] * (2* np.pi / day))
    df['Day cos'] = np.cos(df['Seconds'] * (2* np.pi / day))
    df['Year sin'] = np.sin(df['Seconds'] * (2* np.pi / year))
    df['Year cos'] = np.cos(df['Seconds'] * (2* np.pi / year))
    df = df.drop(columns=['Seconds', 'Date'], axis=1)
    # Using scaling function
    if scale:
        df = scale_data(df)
        
    return df



def dataframe_to_csv(df,name):
    try:
        df.to_csv(name + '.csv', header=True)
        print('Tallennettu!')
    except:
        print('Virhe')
        return None
    return df.head()



def datetime(df, utf=0):
    
    """ 
    df = must have colum called date
    utf = amout of hours added to date

    """
    # Convert date column to datetime type
    df['Date'] = pd.to_datetime(df['Date'])
    if utf != 0:
        df['Date'] = df['Date'] + pd.DateOffset(hours=utf)
    # New columns
    # Month
    #df['Month'] = df['Date'].dt.month
    # neukkariDay
    #df['Day'] = df['Date'].dt.day
    # neukkariWeekday
    #df['Weekday'] = df['Date'].dt.day_name()
    # neukkariHour
    #df['Hour'] = df['Date'].dt.hour
    # neukkariMinute
    #df['Minute'] = df['Date'].dt.minute

    return df



def s1temp (df):
    def s1column(row):
        if row['Ulkolämpötila'] <= 0:
            # Raja-arvot
            raja_arvoalas = float(20.5)
            raja_arvoylös = float(22.5)
            
            if (row['SisäLämpötila'] < raja_arvoylös and row['SisäLämpötila'] > raja_arvoalas):
                return int(1)
            else:
                return int(0)
            
        if row['Ulkolämpötila'] > 0:
            if (row['SisäLämpötila'] < (22.5 + 0.166 * float(row['Ulkolämpötila'])) and 
                row['SisäLämpötila'] > (20.5 + 0.0075 * float(row['Ulkolämpötila']))):
                return int(1)
            else:
                return int(0)
    
    df['S1temp'] = df.apply(lambda row: s1column(row), axis=1)
    return df




def df_to_X_y3(df, window_size=5):
    """
    Returns X, y
    Window_size = default=True
    """
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        label = [df_as_np[i+window_size][0], df_as_np[i+window_size][1]]
        y.append(label)
        
    return np.array(X), np.array(y)



def train_test(X, y, split_size=0.8):
    r = int(len(X) * split_size)
    
    X_train, y_train = (X[:r], y[:r])
    X_test, y_test = (X[r:], y[r:])
    #X_val, y_val = (X[2700:], y[2700:])
    
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test



def test_LSTM_model(df, model, return_df=False):
    """
    df = Testattava dataframe
    model = koulutettu malli
    return_df = palauttaa dataframen jos asetettu True, default = False
    """
    
    df = data_for_LSTM_model(df, scale=False)
    X, y = df_to_X_y3(df)
    # Predictions
    predictions = model.predict(X)
    hum_preds, temp_preds = predictions[:,0], predictions[:,1]
    hum_actuals, temp_actuals = y[:,0], y[:,1]

    # Dataframe
    preds_df = pd.DataFrame({'Temprerature Predictions': temp_preds,
                              'Humidity Predictions': hum_preds,
                              'Temprerature Actuals': temp_actuals,
                              'Humidity Actuals': hum_actuals,
                              'Temperature Error': (temp_preds-temp_actuals),
                              'Humidity Error': (hum_preds-hum_actuals)})
    print('\nERRORS')
    print('Highest error < 0:\n', preds_df[['Temperature Error', 'Humidity Error']].min())
    print('\nHighest error > 0:\n', preds_df[['Temperature Error', 'Humidity Error']].max())
    print('\nAvg error\n', preds_df[['Temperature Error', 'Humidity Error']].mean())    
    
    if return_df:
        return preds_df
    
    
    # Scale values
def scale_data(df, return_scalers = False):
    
    from sklearn.preprocessing import StandardScaler
    
    scalerdf = StandardScaler()
    scalertemp = StandardScaler()
    scalerhum = StandardScaler()
    
    
    df['TemperatureMuseo'] = scalertemp.fit_transform(df[['TemperatureMuseo']])
    df['HumidityMuseo'] = scalerhum.fit_transform(df[['HumidityMuseo']])
    df[['Temperature', 'Humidity']] = scalerdf.fit_transform(df[['Temperature','Humidity']])
    
    if return_scalers:
        from joblib import dump
        
        dump(scalerdf, 'scalerdf.joblib')
        dump(scalertemp, 'scalertemp.joblib')
        dump(scalerhum, 'scalerhum.joblib')
        
    return df
    