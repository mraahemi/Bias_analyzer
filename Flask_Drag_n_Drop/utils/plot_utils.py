import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_histogram(df, column, bins=100, rotate=True, round_xticks=True, ax=None):
    hist = np.histogram(df[column], bins)

    df_hist = pd.DataFrame(hist, index=['counts', column]).T
    df_hist = df_hist[df_hist['counts'] != 0]

    axis = sns.barplot(ax=ax,
        data=df_hist,
        x=column, y="counts", orient='v'
    )
    if round_xticks:
        axis.set_xticklabels([int(x) for x in df_hist[column].values])
    if rotate:
        for item in axis.get_xticklabels():
            item.set_rotation(45)
    return axis