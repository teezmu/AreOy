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
    print('connecting...')
    conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=sqlserver7w7bbd5rlwweu.database.windows.net;'
                      'Database=AreIoTSQLSensorData;'
                      'Trusted_Connection=no;'
                     'Uid=Teemu_ARE;'
                     'Pwd=Toukokuu2022!;')

    cursor = conn.cursor()
    print('Connected!')
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
        print(f'Found {len(sensorF1)} sensors:\n{sensorF1}\n')
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
    df = scale_data(dfsää, means, devitations)
    
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



def train_ML(df, conn, sää, return_parameters=True, prints=False, save_model=False, train=True):
    import joblib
    df, mean, std = process_data_to_ml(df, conn, sää)
    df = df.dropna()
    
    if train:
        trainset = df.sample(frac=0.8)
        testset = df.drop(trainset.index)
        X_train, y_train = trainset.drop(columns=['Kosteus', 'Lampotila']), trainset['Kosteus']
        X_test, y_test = testset.drop(columns=['Kosteus', 'Lampotila']), testset['Kosteus']
        # RandomForestRegressor
        from sklearn.ensemble import RandomForestRegressor
        rand_regr = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        rand_regr.fit(X_train, y_train)
        # Prediction 
        rand_predictions_scaled = rand_regr.predict(X_test)
        # Inverse scaling
        rand_predictions = (rand_predictions_scaled * std['Kosteus']) + mean['Kosteus']
        #rand_actuals = (y_test * std) + mean
    
        # DecisionTreeRegressor
        from sklearn.tree import DecisionTreeRegressor
        dec_regr = DecisionTreeRegressor(random_state=0)
        dec_regr.fit(X_train, y_train)
         # Prediction 
        dec_predictions_scaled = dec_regr.predict(X_test)
        # Inverse scaling
        dec_predictions = (dec_predictions_scaled * std['Kosteus']) + mean['Kosteus']
        
        actuals = (y_test * std.Kosteus) + mean.Kosteus
    
    if prints:
        test_df = pd.DataFrame({'Rand Predictions': rand_predictions,
                               'Rand Actuals': actuals,
                               'Rand Error': (rand_predictions-actuals),
                                'Dec Predictions': dec_predictions,
                               'Dec Actuals': actuals,
                               'Dec Error': (dec_predictions-actuals),
                               'Ridge Predictions': ridge_predictions,
                               'Ridge Actuals': actuals,
                               'Ridge Error': (ridge_predictions-actuals),
                               'Lasso Predictions': lasso_predictions,
                               'Lasso Actuals': actuals,
                               'Lasso Error': (lasso_predictions-actuals)})
        
        print('Random Forest Regression:\n')
        print('Max error < 0:', test_df['Rand Error'].min())
        print('Max error > 0:', test_df['Rand Error'].max())
        print('Avg error:', test_df['Rand Error'].mean())
        
        print('\nDecision Tree Regression:\n')
        print('Max error < 0:', test_df['Dec Error'].min())
        print('Max error > 0:', test_df['Dec Error'].max())
        print('Avg error:', test_df['Dec Error'].mean())
        
        print('\Ridge Regression:\n')
        print('Max error < 0:', test_df['Ridge Error'].min())
        print('Max error > 0:', test_df['Ridge Error'].max())
        print('Avg error:', test_df['Ridge Error'].mean())
        
        print('\nLasso Regression:\n')
        print('Max error < 0:', test_df['Lasso Error'].min())
        print('Max error > 0:', test_df['Lasso Error'].max())
        print('Avg error:', test_df['Lasso Error'].mean())
        
        #Plot results
        import matplotlib.pyplot as plt        
        plt.figure()
        plt.plot(test_df['Rand Predictions'][:100], label='Rand Predictions')
        plt.plot(test_df['Rand Actuals'][:100], label='Rand Actuals')
        plt.grid(True)
        #plt.ylim([1,22])
        plt.legend()
        
        plt.figure()
        plt.plot(test_df['Dec Predictions'][:100], label='Dec Predictions')
        plt.plot(test_df['Dec Actuals'][:100], label='Dec Actuals')
        plt.grid(True)
        plt.legend()
        
        plt.figure()
        plt.plot(test_df['Ridge Predictions'][:100], label='Ridge Predictions')
        plt.plot(test_df['Ridge Actuals'][:100], label='Ridge Actuals')
        plt.grid(True)
        plt.legend()
        
        plt.figure()
        plt.plot(test_df['Lasso Predictions'][:100], label='Lasso Predictions')
        plt.plot(test_df['Lasso Actuals'][:100], label='Lasso Actuals')
        plt.grid(True)
        plt.legend()
        
        return test_df
        
    if save_model:
        # save
        joblib.dump(rand_regr, "./random_forest.joblib")
        joblib.dump(dec_regr, "./decision_tree.joblib")
        
    if return_parameters:
        rand_regr = joblib.load("./random_forest.joblib")
        dec_regr = joblib.load("./decision_tree.joblib")

        return dec_regr, rand_regr, mean, std
    
    
def predict_museo_hum(df_list, figure=False):
    print('Viimeisin datapiste:')
    print(df_list[0]['Date'].max())
    # SQL Connection
    museo, sää, conn = get_files()
     
    # Sääennustukset
    sqlennustus = """SELECT * FROM weatherforecasts WHERE place = 'Jyväskylä, Keskusta'"""
    dfennustus = pd.read_sql(sqlennustus, conn)
    dfennustus['timestamp'] = pd.to_datetime(dfennustus['timestamp'])
    
    enn2h = dfennustus['timestamp'] + pd.DateOffset(hours=2)
    enn3h = dfennustus['timestamp'] + pd.DateOffset(hours=3)
    enn6h = dfennustus['timestamp'] + pd.DateOffset(hours=6)
    print(f'Ennustukset ajoille: {enn2h} \n {enn3h} \n {enn6h}')
    
    dec_predictions_list = []
    rand_predictions_list = []
    for i in range(0, len(df_list)):
        # Train and get models
        dec_reg, rand_reg, mean, std = arehelper.predictive_model.train_ML(df_list[i], conn, sää, return_parameters=True, prints=False, train=False, save_model=False)
    
        ennustus2h, ennustus3h, ennustus6h = arehelper.predictive_model.prepare_data(dfennustus)
        ennustus2h = arehelper.predictive_model.scale_data2(ennustus2h, mean, std)
        ennustus3h = arehelper.predictive_model.scale_data2(ennustus3h, mean, std)
        ennustus6h = arehelper.predictive_model.scale_data2(ennustus6h, mean, std)
    
    
    
        ## Predict Humidity ##
        # DecisionTree
        dec_pred2h_scaled = dec_reg.predict(ennustus2h)
        dec_pred3h_scaled = dec_reg.predict(ennustus3h)
        dec_pred6h_scaled = dec_reg.predict(ennustus6h)
        
        dec_pred2h = (dec_pred2h_scaled * std.Kosteus) + mean.Kosteus
        dec_pred3h = (dec_pred3h_scaled * std.Kosteus) + mean.Kosteus
        dec_pred6h = (dec_pred6h_scaled * std.Kosteus) + mean.Kosteus
        
        # RandomRegressor
        rand_pred2h_scaled = rand_reg.predict(ennustus2h)
        rand_pred3h_scaled = rand_reg.predict(ennustus3h)
        rand_pred6h_scaled = rand_reg.predict(ennustus6h)
        
        rand_pred2h = (rand_pred2h_scaled * std.Kosteus) + mean.Kosteus
        rand_pred3h = (rand_pred3h_scaled * std.Kosteus) + mean.Kosteus
        rand_pred6h = (rand_pred6h_scaled * std.Kosteus) + mean.Kosteus
    
        # Dataframes
        dec_predictions_df = pd.DataFrame({'Humidity in 2h': dec_pred2h,
                                           'Humidity in 3h': dec_pred3h,
                                           'Humidity in 6h': dec_pred6h})
        
        rand_predictions_df = pd.DataFrame({'Humidity in 2h': rand_pred2h,
                                           'Humidity in 3h': rand_pred3h,
                                           'Humidity in 6h': rand_pred6h})
        
        print(f'\nDec Predictions:{dec_predictions_df.to_numpy()}\nRand predictions: {rand_predictions_df.to_numpy()}')
        
        dec_predictions_list.append(dec_predictions_df)
        rand_predictions_list.append(rand_predictions_df)
        
        if figure:
            import plotly.graph_objects as go
            import plotly.express as px
            
            fig = go.Figure()
            fig.add_traces(go.Scatter(x=dec_predictions_df.T.index, y=dec_predictions_df.T[0]))
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

    
    return dec_predictions_list, rand_predictions_list
       



