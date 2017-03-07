import pickle as p
import numpy as np
file = open("dmy_data.p","wb")
data = {
    "data_dim":5,
    "embeds":np.array([200*[1],200*[-1], 200*[1],200*[-1],200*[1],200*[-1]]),
    "timesteps":3,
    "x":[np.array([[0,0,1],[0,1,0],[1,0,0]]),np.array([[0,0,1],[0,1,0],[1,0,0]])],
    "y":np.array([[1,0,0],[0,1,0],[0,0,1]])}
p.dump(data,file)
file.close()
