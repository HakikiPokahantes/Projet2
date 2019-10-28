# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:04:24 2019

@author: hakik
"""

#EXO2
#Q1

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits 


digits=load_digits()

print(digits)

print(digits.data.shape)

print("Nombre des observations:"+str(digits.data.shape[0]))
print("Nombre des categories:"+str(len(digits.target_names)))

#Q2

from sklearn.decomposition import PCA

pca=PCA(n_components=2)
Xpca=pca.fit_transform(digits.data)


plt.scatter(Xpca[:,0],Xpca[:,1],c=digits.target,cmap="jet")
plt.colorbar()



#EXO4


