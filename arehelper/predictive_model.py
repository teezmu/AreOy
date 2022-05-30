import pandas as pd
import numpy as np
import arehelper.dataprocessing
import arehelper.datavisualization
import arehelper.predictive_model
import pyodbc


def get_files():
    import pandas as pd
    ## DATA FROM SQL
    import pyodbc
    conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=sqlserver7w7bbd5rlwweu.database.windows.net;'
                      'Database=AreIoTSQLSensorData;'
                      'Trusted_Connection=no;'
                     'Uid=Teemu_ARE;'
                     'Pwd=Toukokuu2022!;')

    cursor = conn.cursor()
    sqlmuseo = """SELECT * FROM museotesti
                      WHERE Apartment = 1 AND date > '2022-01-01' ORDER BY date DESC"""
    dfmuseo = pd.read_sql(sqlmuseo, conn)
    
    sqlsää = """SELECT time, temperature, humidity FROM weatherrecords WHERE Time > '2022-01-01'
                    AND PLACE = 'Jyväskylä' ORDER BY time DESC"""
    dfsää = pd.read_sql(sqlsää,conn)
    
    return dfmuseo, dfsää, conn


def create_sensordf(rule, prints=False):
    """
    Input dataframessa on oltava columnit 'Sensorid' ja 'Date'
    Functio palauttaa dataframen jokaisesta sensorista
    
    """
    dfmuseo, dfsää, conn = get_files()
    dfmuseo = dfmuseo.rename(columns={'date': 'Date'})
    dfsää = dfsää.rename(columns={'time': 'Date'})

    sensorF1 = dfmuseo['Sensorid'].unique()
    df = {name: (dfmuseo.loc[dfmuseo['Sensorid'] == name].reset_index(drop=True)) for name in sensorF1}
    
    #####################################################
    if prints:
        print(f'Found {len(sensorF1)} sensors')
        print(f'Dataframe names:')
    #####################################################
    

    # Loop list of sensorid´s
    for s in sensorF1:
        # Date-column to datetime format
        df[s]['Date'] = pd.to_datetime(df[s]['Date'])
        dfsää['Date'] = pd.to_datetime(dfsää['Date'])

        # Resampling data based on rule(input)
        df[s] = df[s].resample(rule=rule, on='Date').mean()
        
        # Säädf and sensordf into one dataframe
        df[s] = pd.merge(dfsää,df[s], on='Date')
        # Final dataframe names
        if prints:
            print(f'df["{s}"]')
    
    return df



def process_data_to_ml(df, conn, sää):
    #museo, sää, conn = get_files()
    try:
        sää = sää.drop(columns=['Unnamed: 0'])
    except:
        pass
        
    sää = sää.rename(columns={'time':'Date', 'temperature': 'Temperature', 'humidity': 'Humidity'})
    dfsää = pd.merge(df,sää, on='Date')
    dfsää = arehelper.dataprocessing.data_for_LSTM_model(dfsää)

    #dfsää = dfsää.reset_index(drop=True)
    means = dfsää.mean(numeric_only=True)
    devitations = dfsää.std(numeric_only=True)
    
    columns = list(dfsää.columns)
    df = arehelper.predictive_model.scale_data(dfsää, means, devitations)
    
    return df, means, devitations



def scale_data(df, mean, deviation):
    ennustus_means = mean[['Kosteus' ,'Lampotila', 'temperature', 'humidity', 'Day sin', 'Day cos', 'Year sin', 'Year cos']]
    ennustus_std = deviation[['Kosteus' ,'Lampotila', 'temperature', 'humidity', 'Day sin', 'Day cos', 'Year sin', 'Year cos']]
    for c in range(len(df.columns)):
        df.iloc[:,c] = (df.iloc[:,c].values - ennustus_means[c]) / ennustus_std[c]

    return df

def scale_data2(df, means, std):
    
    df['Temperature'] = (df['Temperature'].values - means['temperature']) / std['temperature']
    df['Humidity'] = (df['Humidity'].values - means['humidity']) / std['humidity']
    df['Day sin'] = (df['Day sin'].values - means['Day sin']) / std['Day sin']
    df['Day cos'] = (df['Day cos'].values - means['Day cos']) / std['Day cos']
    df['Year sin'] = (df['Year sin'].values - means['Year sin']) / std['Year sin']
    df['Year cos'] = (df['Year cos'].values - means['Year cos']) / std['Year cos']
    return df

def prepare_data(df):
    # Ignore copywarning
    import warnings
    import pandas as pd
    from pandas.core.common import SettingWithCopyWarning
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    
    ennustus2h = df[['temp2h','hum2h']]
    ennustus2h = ennustus2h.rename(columns={'temp2h': 'Temperature', 'hum2h': 'Humidity'})
    ennustus3h = df[['temp3h','hum3h']]
    ennustus3h = ennustus3h.rename(columns={'temp3h': 'Temperature', 'hum3h': 'Humidity'})
    ennustus6h = df[['temp6h','hum6h']]
    
    ennustus2h['Date'] = df['timestamp'] + pd.DateOffset(hours=2)
    ennustus3h['Date'] = df['timestamp'] + pd.DateOffset(hours=3)
    ennustus6h['Date'] = df['timestamp'] + pd.DateOffset(hours=6)
    
    ennustus2h['Seconds'] = ennustus2h['Date'].map(pd.Timestamp.timestamp)
    ennustus3h['Seconds'] = ennustus3h['Date'].map(pd.Timestamp.timestamp)
    ennustus6h['Seconds'] = ennustus6h['Date'].map(pd.Timestamp.timestamp)
    
    day = 60*60*24
    year = 365.2425*day
    # Ennustus2h
    ennustus2h['Day sin'] = np.sin(ennustus2h['Seconds'] * (2* np.pi / day))
    ennustus2h['Day cos'] = np.cos(ennustus2h['Seconds'] * (2* np.pi / day))
    ennustus2h['Year sin'] = np.sin(ennustus2h['Seconds'] * (2* np.pi / year))
    ennustus2h['Year cos'] = np.cos(ennustus2h['Seconds'] * (2* np.pi / year))
    ennustus2h = ennustus2h.drop(columns=['Seconds','Date'], axis=1)
    ennustus2h = ennustus2h.rename(columns={'temp2h': 'Temperature', 'hum2h': 'Humidity'})
    
    # Ennustus3h
    ennustus3h['Day sin'] = np.sin(ennustus3h['Seconds'] * (2* np.pi / day))
    ennustus3h['Day cos'] = np.cos(ennustus3h['Seconds'] * (2* np.pi / day))
    ennustus3h['Year sin'] = np.sin(ennustus3h['Seconds'] * (2* np.pi / year))
    ennustus3h['Year cos'] = np.cos(ennustus3h['Seconds'] * (2* np.pi / year))
    ennustus3h = ennustus3h.drop(columns=['Seconds','Date'], axis=1)
    ennustus3h = ennustus3h.rename(columns={'temp3h': 'Temperature', 'hum3h': 'Humidity'})

    # Ennustus6h
    ennustus6h['Day sin'] = np.sin(ennustus6h['Seconds'] * (2* np.pi / day))
    ennustus6h['Day cos'] = np.cos(ennustus6h['Seconds'] * (2* np.pi / day))
    ennustus6h['Year sin'] = np.sin(ennustus6h['Seconds'] * (2* np.pi / year))
    ennustus6h['Year cos'] = np.cos(ennustus6h['Seconds'] * (2* np.pi / year))
    ennustus6h = ennustus6h.drop(columns=['Seconds','Date'], axis=1)
    ennustus6h = ennustus6h.rename(columns={'temp6h': 'Temperature', 'hum6h': 'Humidity'})

    return ennustus2h, ennustus3h, ennustus6h



def train_ML(return_models=False, prints=False, save_model=False, train=True):
    import joblib  
    museo, sää, conn = arehelper.predictive_model.get_files()
    if train:
        # Timer
        import time
        start = time.time()
        
        sensor = arehelper.predictive_model.create_sensordf('1min', prints=False)
        c = 1
        for s in sensor.keys():
            print(f'Training models: {c}/{len(list(sensor))}\n')
            print(f'Sensor: {s}')
            sensordf = sensor[s].dropna()
            df, mean, std = arehelper.predictive_model.process_data_to_ml(sensordf, conn, sää)
            
            trainset = df.sample(frac=0.8)
            testset = df.drop(trainset.index)
            X_train, y_train = trainset.drop(columns=['Kosteus', 'Lampotila']), trainset['Kosteus']
            X_test, y_test = testset.drop(columns=['Kosteus', 'Lampotila']), testset['Kosteus']
            #X_train = X_train.to_numpy()
            #X_test = X_test.to_numpy()
            # RandomForestRegressor
            from sklearn.ensemble import RandomForestRegressor
            rand_regr = RandomForestRegressor(n_estimators = 1000, random_state = 42)
            rand_regr.fit(X_train, y_train)
            # Prediction 
            rand_predictions_scaled = rand_regr.predict(X_test)
            # Inverse scaling
            rand_predictions = (rand_predictions_scaled * std['Kosteus']) + mean['Kosteus']
            #rand_actuals = (y_test * std) + mean
            actuals = (y_test * std.Kosteus) + mean.Kosteus
            c+=1
            if save_model:
                name = '/['+s+']random_forest.joblib'
                joblib.dump(rand_regr, './Models'+ name)
    
    
            test_df = pd.DataFrame({'Rand Predictions': rand_predictions,
                                   'Rand Actuals': actuals,
                                   'Rand Error': (rand_predictions-actuals),
                                   })
            if prints:

                print('\nRandom Forest Regression:')
                print('Max error < 0:', test_df['Rand Error'].min())
                print('Max error > 0:', test_df['Rand Error'].max())
                print('Avg error:', test_df['Rand Error'].mean())
            
                #Plot results
                import matplotlib.pyplot as plt        
                plt.figure()
                plt.plot(test_df['Rand Predictions'][100:200], label='Rand Predictions')
                plt.plot(test_df['Rand Actuals'][100:200], label='Rand Actuals')
                plt.grid(True)
                plt.ylim([25,45])
                plt.legend()
                plt.show()
            
            print(f'Model {s} saved!')
        print(f'\nTotal training time: {round((time.time())-start,0)} seconds\n')
        
    
def predict_museo_hum(figure=False):
    import time
    import joblib
    start = time.time()
    # Luodaan datasetit jokaiselle sensorille
    df = arehelper.predictive_model.create_sensordf('1min', prints=False)
    # Lisätään listaan kaikkien sensoreiden datasettien nimet
    sensordf_list = []
    for s in range(0, len(list(df))):
        sensordf_list.append(list(df)[s])    
    # SQL Connection
    museo, sää, conn = arehelper.predictive_model.get_files()
     
        
    # Sääennustukset  #  
    ###################
    sqlennustus = """SELECT * FROM weatherforecasts WHERE place = 'Jyväskylä, Keskusta'"""
    dfennustus = pd.read_sql(sqlennustus, conn)
    dfennustus['timestamp'] = pd.to_datetime(dfennustus['timestamp'])
    
    enn2h = dfennustus['timestamp'] + pd.DateOffset(hours=2)
    enn3h = dfennustus['timestamp'] + pd.DateOffset(hours=3)
    enn6h = dfennustus['timestamp'] + pd.DateOffset(hours=6)
    print(f'Ennustukset ajoille: {enn2h.values} \n {enn3h.values} \n {enn6h.values}\n')
    
    # Mallien ennustus loop #
    #########################
    # loopataan sensorilista läpi ja käytetään oikeaa modelia oikean sensorin kanssa
    # Lista ennustuksille
    rand_predictions_dict = {}
    for sens in sensordf_list:
        print('\nVuorossa sensori: ',sens, '\n')
        name = "Models/["+sens+"]random_forest.joblib"
        model = joblib.load(name)
        sensordf = df[sens].dropna()
        # Viimeisin sensorilta saatu piste
        print('\nViimeisin datapiste:')
        print(sensordf['Date'].max())
        
        # Scaalaukseen tarvittavat arvot 
        sensordf, mean, std = arehelper.predictive_model.process_data_to_ml(sensordf, conn, sää)
        # Ennustusten skaalaus
        ennustus2h, ennustus3h, ennustus6h = arehelper.predictive_model.prepare_data(dfennustus)
        ennustus2h = arehelper.predictive_model.scale_data2(ennustus2h, mean, std)
        ennustus3h = arehelper.predictive_model.scale_data2(ennustus3h, mean, std)
        ennustus6h = arehelper.predictive_model.scale_data2(ennustus6h, mean, std)
        
        # Predict Humidity #
        ###################
        # RandomRegressor
        rand_pred2h_scaled = model.predict(ennustus2h)
        rand_pred3h_scaled = model.predict(ennustus3h)
        rand_pred6h_scaled = model.predict(ennustus6h)
        # Scaalataan arvot takaisin 'normaaleiksi'
        rand_pred2h = (rand_pred2h_scaled * std.Kosteus) + mean.Kosteus
        rand_pred3h = (rand_pred3h_scaled * std.Kosteus) + mean.Kosteus
        rand_pred6h = (rand_pred6h_scaled * std.Kosteus) + mean.Kosteus
    
        # Ennustuksen tulokset
        rand_predictions_df = pd.DataFrame({'Humidity in 2h': rand_pred2h,
                                           'Humidity in 3h': rand_pred3h,
                                           'Humidity in 6h': rand_pred6h})
        
        #print(f'\nRand predictions: {rand_predictions_df.to_numpy()}')
        
        # Lisätään listaan ennustukset
        rand_predictions_dict.update({sens:rand_predictions_df})
        
        # Jos figure=True tulee kuvaaja esiin
        if figure:
            import plotly.graph_objects as go
            import plotly.express as px
            
            fig = go.Figure()
            fig.add_traces(go.Scatter(x=rand_predictions_df.T.index, y=rand_predictions_df.T[0]))
        
            fig.update_traces(marker={'size': 20})
            
            fig.update_layout(
                title="Predictions",
                yaxis_title="Humidity(%)",
                font=dict(
                    family="Times New Roman",
                    size=18,
                    color="RebeccaPurple"))
            fig.show()
            
    print(f'Total time: {round((time.time())-start,0)} seconds')
    return pd.concat(rand_predictions_dict, axis=0)
       


