import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_training_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0,2])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)


def plot_predictions(model, X, y, start=0, end=100):
    predictions = model.predict(X)
    hum_preds, temp_preds = predictions[:,0], predictions[:,1]
    hum_actuals, temp_actuals = y[:,0], y[:,1]
    
    # Make Dataframe
    df = pd.DataFrame({'Temprerature Predictions': temp_preds,
                      'Humidity Predictions': hum_preds,
                      'Temprerature Actuals':temp_actuals,
                      'Humidity Actuals': hum_actuals})
    
    #Plot results
    plt.plot(df['Temprerature Predictions'][start:end], label='Temp Predictions')
    plt.plot(df['Temprerature Actuals'][start:end], label='Temp Actuals')
    plt.plot(df['Humidity Predictions'][start:end], label='Hum Predictions')
    plt.plot(df['Humidity Actuals'][start:end], label='Hum Actuals')
    plt.legend()
    
    return df[start:end]
