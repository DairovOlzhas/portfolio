import cv2
import numpy as np
import pandas as pd
import glob
import torch

from model import NeuralNetwork

net = NeuralNetwork()
net.load_state_dict(torch.load('./model.pth'))

test = pd.read_csv('test_updated.csv')

X_test = test.drop("PassengerId", axis=1).copy()
X_test = X_test.values

def predict(dataset):
	dataset = torch.from_numpy(dataset).float()

	pred = net(dataset)
	pred = pred.data.numpy()
	print(pred)
	if pred > 0.5:
		return 1;
	return 0;

predictions = [ predict(x) for x in X_test]
print(len(predictions))
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)


# for dataset in X_test:
# 	dataset = torch.from_numpy(dataset).float()

# 	pred = net(dataset)
# 	pred = pred.data.numpy()
	
# output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
# output.to_csv('my_submission.csv', index=False)


	# cv2.imshow("img", cv2.resize(img, (500, 500)))
	# cv2.waitKey(0)