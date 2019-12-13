import pandas as pd
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

class Dataset(Dataset):

	def __init__(self):
		self.csv = pd.read_csv('./train_updated.csv')
		self.csv['Survived'] = self.csv['Survived'].astype(float)

	def __len__(self):
		return len(self.csv)

	def __getitem__(self, ind):
		X = self.csv.iloc[ind][1:].values
		Y = self.csv.iloc[ind][0]
		# print(type(Y))
		X = torch.from_numpy(X).float()
		# Y = torch.from_numpy(Y).float()

		return X, Y

