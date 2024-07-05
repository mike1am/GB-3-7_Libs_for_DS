import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics as metrics


allInt = lambda s: np.all(np.mod(s, 1) == 0)


colors = lambda: iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])


def reduce_mem_usage(df, toCat=True, toFloat=False, excluding=None):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    if excluding == None: excluding = []
    
    for col in df.columns:
        if col in excluding:
            continue
        
        col_type = df[col].dtype
        
        if col_type not in (object, "category"):
            c_min = df[col].min()
            c_max = df[col].max()
            if not toFloat and (str(col_type)[:3] == 'int' or allInt(df[col])):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        
        if  df[col].nunique() <= 25 and toCat:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


reduceMem = reduce_mem_usage


def showVectors(M):
    if M.shape[1] != 3:
        return
    
    plt.figure(figsize=(5, 5))
    plt.subplot(projection="3d")

    for row in M:
        plt.plot([0, row[0]], [0, row[1]], [0, row[2]])

    plt.show()


def evaluatePreds(true_values, pred_values, save=False):
    """Оценка качества модели и график preds vs true"""
    
    print("R2:\t" + str(round(metrics.r2_score(true_values, pred_values), 3)) + "\n" +
          "RMSE:\t" + str(round(np.sqrt(metrics.mean_squared_error(true_values, pred_values)), 3)) + "\n" +
          "MSE:\t" + str(round(metrics.mean_squared_error(true_values, pred_values), 3))
         )
    
    plt.figure(figsize=(7,7))
    
    sns.scatterplot(x=pred_values, y=true_values, size=1, legend=False)
    diagMin = min(pred_values.min(), true_values.min())
    diagMax = max(pred_values.max(), true_values.max())
    plt.plot(  # диагональ, где true_values = pred_values
        [diagMin, diagMax],
        [diagMin, diagMax],
        linestyle='--',
        color='black'
    )
    
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('True vs Predicted values')
    
    if save == True:
        plt.savefig('report.png')
    plt.show()


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("lesson02-03\housing.csv")
    print(df.info(), end="\n\n")

    print(reduce_mem_usage(df).info())
