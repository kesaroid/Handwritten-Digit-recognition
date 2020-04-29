import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from openTSNE import TSNE
from openTSNE.callbacks import ErrorLogger
from openTSNE import utils

df = pd.read_csv("train.csv")
df = df[:100]
label = df.label
df.drop("label", axis=1, inplace=True)
standardized_data = StandardScaler().fit_transform(df)
print(standardized_data.shape)

tsne = TSNE(
            perplexity=30,
            metric="euclidean",
            callbacks=ErrorLogger(),
            n_jobs=8,
            random_state=42,
        )

embedding_train = tsne.fit(standardized_data)
utils.plot(embedding_train, label, colors=utils.MACOSKO_COLORS)