import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import sklearn.datasets as skds

iris = skds.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
