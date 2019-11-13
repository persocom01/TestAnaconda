# Demonstrates plotting multiple graphs on the same figure. This is not the
# same as plotting multiple datasets on the same graph.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as skld

iris = skld.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
