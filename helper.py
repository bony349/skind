# This File contains all library for concept dry -> don't repeat your self
import os

import pickle
import numpy as np
from sklearn import svm
import skimage.io as img
from sklearn.svm import SVC
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def load_model(model_name):
    model = pickle.load(open(model_name, 'rb'))
    return model


def save_model(model, model_name='model'):  # giving defualt name for the model
    model_name = model_name
    pickle.dump(model, open(model_name, 'wb'))

