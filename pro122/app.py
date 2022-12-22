import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

data= pd.read_csv("labels.csv")
labels=data["labels"]
X,y = labels(return_X_y=True)

print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)
samples_perclass = 5
figure = plt.figure(figsize=(nclasses*2, (1+samples_perclass*2)))

iclass=0

for cls in classes:
  
  id= np.flatnonzero(y==cls)
  id= np.random.choice(id, samples_perclass, replace=False)
  i=0
  
  for j in id:
      plt_j=i*nclasses+iclass+1
      p=plt.subplot(samples_perclass, nclasses, plt_j)
      p= sb.heatmap(np.reshape(X[j], (28,28)), cmap=plt.cm.Blues, xticklabels=False, yticklabels = False, cbar=False)
      i=i+1
  iclass= iclass+1