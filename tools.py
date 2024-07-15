import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.spatial.distance

import sklearn.metrics as metrics
import sklearn.cluster as skl_cluster
import sklearn.manifold as skl_manifold
import sklearn.decomposition as skl_decomp


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


def evaluatePreds(true_values, pred_values, figsize=None):
    """Оценка качества модели и график preds vs true"""
    
    print("R2:\t" + str(round(metrics.r2_score(true_values, pred_values), 3)) + "\n" +
          "RMSE:\t" + str(round(np.sqrt(metrics.mean_squared_error(true_values, pred_values)), 3)) + "\n" +
          "MSE:\t" + str(round(metrics.mean_squared_error(true_values, pred_values), 3))
         )
    
    figsize = figsize or (7, 7)
    plt.figure(figsize=figsize)
    
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
    
    plt.show()


def annotCorr(x, y, hue=None, ax=None, **kwargs):
    ax = ax or plt.gca()
    ax.annotate(
        r"$\eta$" + f" = {np.corrcoef(x, y)[0, 1]:.2f}",
        xy=(.1, .9),
        xycoords=ax.transAxes
    )


def displayIn2D(df, labels=None, figsize=None):
    df_wLabels = pd.concat((
        df.reset_index(drop=True),
        pd.Series(labels).reset_index(drop=True)
    ), axis=1)
    if df.shape[1] != 2:
        raise ValueError("Number of features in the DataFrame differs from 2")

    figsize = figsize or (10, 7)
    plt.figure(figsize=figsize)

    params = dict(
        kind="scatter",
        x=df.columns[0],
        y=df.columns[1],
        alpha=0.5
    )
    if labels is not None:
        params.update(
            c=df_wLabels.iloc[:, -1],
            cmap=plt.get_cmap('jet')
        )

    df_wLabels.plot(**params)

    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title('2D mapping of objects')    
    plt.show()


def reduceDim(data, dims=2, method='pca', **kwargs):
    match method:
        case 'pca':
            reducer = skl_decomp.PCA(**kwargs)
        case 'tsne':
            reducer = skl_manifold.TSNE(**kwargs)
        case _:
            raise ValueError("Unexpected reducer method")

    return pd.DataFrame(
        reducer.fit_transform(data),
        columns=["feat_" + str(i) for i in range(dims)],
        index=data.index if isinstance(data, pd.DataFrame) else None
    )


def scPlot2D(data, labels=None, figsize=None, **kwargs):
    df = reduceDim(
        data.drop(labels, axis=1) if isinstance(labels, str) else data,
        dims=2,
        **kwargs
    )

    figsize = figsize or (8 + (2 if labels is not None else 0), 7)
    plt.figure(figsize=figsize)

    params = dict(
        kind="scatter",
        x=df.columns[0],
        y=df.columns[1],
        alpha=0.5
    )
    if labels is not None:
        df = df.join(
            labels.rename("labels")
            if isinstance(labels, pd.Series)
            else 
                data[labels].rename("labels")
                if isinstance(labels, str)
                else pd.Series(labels, index=df.index, name="labels")
        )
        params.update(
            c=df.labels,
            cmap=plt.get_cmap('jet')
        )

    df.plot(**params)

    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title('Scatter plot of reduced to 2D data')    
    plt.show()


def elbowMethod(df, random_state=33, kRange=None):
    """Визуализация для метода 'локтя'"""
    
    distortions = []
    kRange = range(kRange) if kRange is not None else range(2,30)
    cdist = scipy.spatial.distance.cdist
    for k in kRange:
        kmeanModel = skl_cluster.KMeans(n_clusters=k, random_state=random_state).fit(df)
        distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])

    plt.figure(figsize=(10, 8))
    plt.plot(kRange, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("lesson02-03\housing.csv")
    print(df.info(), end="\n\n")

    # print(reduce_mem_usage(df).info())

    scPlot2D(df.fillna(0).select_dtypes([np.integer, np.floating]))
