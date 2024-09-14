from scipy import linalg
import numpy as np

class kalmanFilter:
    def __init__(self, _Ts, _Q, _R):
        self.Q = _Q
        self.R = _R
        self.Ts = _Ts

        #Constant state space model
        self.A = np.matrix([[1,self.Ts],[0,1]])
        self.B = np.matrix([[0],[1]])
        self.C = np.matrix([0,1])

        #Initial conditions
        self.S = np.matrix([0])
        self.P = np.matrix([[1,0],[0,1]])
        self.Pp = np.matrix([[1,0],[0,1]])

        self.x = np.matrix([[0],[-0.01]])
        self.xp = np.matrix([[0],[0]])

    def update(self, measurement, pCamera):
        self.x[0] = [pCamera]

        self.xp = self.A*self.x;  
        self.Pp = self.A*self.P*(self.A.T) + self.Q    
    
        self.S = np.linalg.inv(self.C*self.Pp*(self.C.T) + self.R)
        self.K = self.Pp*(self.C.T)*self.S

        #Kalman gain will be used in the error
        self.x = self.xp + self.K*(measurement - self.C*self.xp)
        # self.x = K*measurement + self.xp(1 - K*self.C)
        self.P = self.Pp - self.K*self.C*self.Pp
        return (self.x)