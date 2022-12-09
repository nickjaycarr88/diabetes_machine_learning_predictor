import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
diabetes = pd.read_csv('diabetes.csv')
print(diabetes)