import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("data.csv",header = None,sep = ",")
temp = df.as_matrix()
ldata = np.zeros((100,6))
ldata[:,0:5] = temp
weights = [1.2,1.4,0.95,1.2,1.4]
for i in range(100):
	rf = np.random.rand(1)*0.01 
	ldata[i,5] = np.sum(temp[i] + rf)
scaler = MinMaxScaler(feature_range=(0, 9))
ldata = scaler.fit_transform(ldata)

print(ldata[:,5].astype(int))

df = pd.DataFrame(ldata.astype(int))
df.to_csv("ldata.csv",header = False,index = False,sep=",")			  


	
