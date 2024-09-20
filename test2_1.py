# Pegando as informações do objeto vindas da câmera (esteira normal)
from Helpers.paths import Paths
import os
Paths.execution = fr'{Paths.output}\{os.path.splitext(os.path.basename(__file__))[0]}\{os.path.splitext(os.path.basename(Paths.execution))[0]}'
os.makedirs(Paths.execution, exist_ok=True)
from Helpers.log import Log

from Helpers.input import Motion, Conveyor as Conv
from Helpers.plot import SaveData, Plot
from Simulators import CoppeliaSim
from Simulators.CoppeliaSim import Camera, Conveyor, Cuboids
from VisionProcessing.aruco import ArucoVision

import numpy as np
import random
from Kalman.kalman import kalmanFilter

# Inicializando kalman
Q = np.matrix([[1,0],[0,1]])
R = np.matrix([1])
kalman_filter = kalmanFilter(Motion.ts, Q, R)

# Inicializando Coppelia
coppelia = CoppeliaSim(scene='test2_1.ttt')
coppelia.Camera = Camera()
coppelia.ArucoVision = ArucoVision(coppelia.Camera)
coppelia.Conveyor = Conveyor()
coppelia.Cuboids = Cuboids()

# Começando a simulação
coppelia.Start(None)
coppelia.Step()

# Inicializando variáveis
count_vel = 0
count_camera = 0
vel = Conv.vel
x_pos_integration = 0

# Inicializando vetores
velocity = []
real = []
kalman = []
integration = []
camera = []

# Loop para identificar se existe objeto na cena
while coppelia.Cuboids.CheckToHandle():
    if len(coppelia.ArucoVision.detected) == 0:
        coppelia.Step() # Câmera ainda não detectou objeto
    else:
        # Câmera detectou objeto
        Log('====================================================')
        Log(f'Vel: {vel}');velocity.append(vel)
        
        marker_pose = coppelia.ArucoVision.detected[0].objectWorldT # Matriz pose do objeto a partir da câmera
        real_pose = coppelia.Cuboids.GetRealPose('3', 'red') # Matriz pose do objeto a partir do Coppelia
        
        # Simulando uma oclusão
        if count_camera > 100000000000000000:
            Log('Parou câmera')
            x_pos = 0
        else:
            x_pos = marker_pose.t[0]
        count_camera += 1
        
        # Posição x do objeto estimada pelo Kalman
        x_pos_kalman = kalman_filter.update(-vel,x_pos)[0,0]
        Log('Kalman:', x_pos_kalman);kalman.append(x_pos_kalman)
        
        # Posição x do objeto estimada pela integração
        if x_pos_integration == 0:
            x_pos_integration = marker_pose.t[0]-Motion.ts*vel
        else:
            x_pos_integration -= Motion.ts*vel
        Log('Integration:', x_pos_integration);integration.append(x_pos_integration)

        # # Ruído na velocidade
        # if count_vel == 2:
        #     vel = random.randrange(1,10)*0.01
        #     coppelia.Conveyor.Move(vel)
        #     count_vel = 0
        # else:
        #     count_vel += 1

        coppelia.Step()

        # Posição x do objeto a partir da câmera
        x_pos_camera = x_pos
        Log('Camera:', x_pos_camera);camera.append(x_pos_camera)
        # Posição x do objeto a partir do Coppelia
        x_pos_real = coppelia.Cuboids.GetRealPose('3', 'red').t[0]
        Log('Real:', x_pos_real);real.append(x_pos_real)

coppelia.Stop()

SaveData('test2_1', velocity, real, kalman, integration, camera)
Plot('test2_1', camera=True)