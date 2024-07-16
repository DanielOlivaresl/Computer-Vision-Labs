# predict.py
import numpy as np
from keras.models import load_model
import joblib

scaler = joblib.load('scaler.save')
# Assuming the descriptor is loaded similarly as before
descriptor = np.loadtxt('descriptor.txt').reshape(1, -1)
descriptor_scaled = scaler.transform(descriptor)
model = load_model('sceneClassifier.h5')

prediction = model.predict(descriptor_scaled)
np.savetxt('result.txt', (prediction))
