from __future__ import print_function

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import matplotlib.pyplot as plt
from keras.models import model_from_json
from nested_json_handler import getScores
def feedNeuralNet(jsonPayload):
	print(jsonPayload)
	print(jsonPayload.decode("utf-8"))
	scoreList = getScores(jsonPayload.decode("utf-8"))
	# load json and create model
	json_file = open('model_nucthon.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model_nucthon.h5")
	print("Loaded model from disk")
	 

	input = np.array([[0,0,0,0,0]])
	input[0] = scoreList
	out_score= loaded_model.predict(input)
	print(np.argmax(out_score))
	out_score=np.asscalar(np.argmax(out_score))
	return out_score

