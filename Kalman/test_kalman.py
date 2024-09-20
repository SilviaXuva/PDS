import kalman
import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np

Q = np.matrix([[1,0],[0,0.01]])
R = np.matrix([1])
Ts = 0.01
kf = kalman.kalmanFilter(Ts,Q,R)

speed = 2 #m/s                
t = np.arange(0,5,Ts)
N = len(t)

#The conveyor belt has constant speed
#the speed variation will be simulated as white noise
measurement = speed*np.ones(N)
noise = np.random.normal(0,0.1,N)
measurement = measurement + noise

output_pos = np.zeros(N)
output_vel = np.zeros(N)

for i in range(N):
    kf.update(measurement[i])
    output_pos[i] = kf.x[0]
    output_vel[i] = kf.x[1]

plt.close("all")
plt.figure()
plt.plot(t,measurement)
plt.plot(t,output_vel)
plt.legend('measurement','filter output')
plt.title("Measurement X Filter Output")

plt.figure()
plt.plot(t,output_pos)
plt.title("Position")

plt.show()