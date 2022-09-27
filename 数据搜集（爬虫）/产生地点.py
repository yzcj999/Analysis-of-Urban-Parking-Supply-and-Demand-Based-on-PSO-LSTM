import numpy as np
import matplotlib.pyplot as plt

height  = 121.702968 - 121.150475
weight = 31.378883 - 30.878361

x = np.linspace(121.150475, 121.702968, 400) + np.random.normal(loc= 0.0, scale=1.0, size=400)*0.01
y = np.linspace(30.878361, 31.378883, 400) + np.random.normal(loc= 0.0, scale=1.0, size=400)*0.01

with open("选取的地点.txt", "w") as t:
    for each in zip(x, y):
        #print(type(each))
        t.write(str(each[0])+","+str(each[1]) + "\n")