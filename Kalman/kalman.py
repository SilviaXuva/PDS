from scipy import linalg
import numpy as np
from Helpers.log import Log

class kalmanFilter:
    def __init__(self, _Ts, _Q, _R):
        self.Q = _Q # Covariância do processo (ruído da matriz de estados)
        self.R = _R # Covariância em relação à observação
        self.Ts = _Ts

        #Constant state space model
        self.A = np.matrix([[1,self.Ts],[0,1]]) # 
        self.B = np.matrix([[0],[1]]) # Não estamos entrando com a velocidade
        self.C = np.matrix([0,1]) # Matriz de observação

        #Initial conditions
        self.S = np.matrix([0])
        self.P = np.matrix([[1,0],[0,1]])
        self.Pp = np.matrix([[1,0],[0,1]])

        self.x = np.matrix([[0],[-0.01]])
        self.xp = np.matrix([[0],[0]])

    def update(self, measurement, pCamera):
        
        if pCamera != 0:
            self.x[0] = [pCamera]

        self.xp = self.A*self.x; 
        self.Pp = self.A*self.P*(self.A.T) + self.Q
    
        self.S = self.C*self.Pp*(self.C.T) + self.R
        self.K = self.Pp*(self.C.T)*np.linalg.inv(self.S)

        #Kalman gain will be used in the error
        self.x = self.xp + self.K*(measurement - self.C*self.xp)
        self.P = self.Pp - self.K*self.C*self.Pp

        return (self.x)

# F => A
# H => C
# z => measurement
# a priori => _p
# a posteriori => _