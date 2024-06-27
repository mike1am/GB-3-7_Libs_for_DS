import numpy as np
import matplotlib.pyplot as plt


allInt = lambda s: np.all(np.mod(s, 1) == 0)


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type not in (object, "category"):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int' or allInt(df[col]):
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
        
        if  df[col].nunique() <= 25:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def showVectors(M):
    if M.shape[1] != 3:
        return
    
    plt.figure(figsize=(5, 5))
    plt.subplot(projection="3d")

    for row in M:
        plt.plot([0, row[0]], [0, row[1]], [0, row[2]])

    plt.show()


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("lesson02-03\housing.csv")
    print(df.info(), end="\n\n")

    print(reduce_mem_usage(df).info())
