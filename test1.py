# Pegando as informações do objeto vindas do coppelia
from Helpers.paths import Paths
import os
Paths.execution = fr'{Paths.output}\{os.path.splitext(os.path.basename(__file__))[0]}\{os.path.splitext(os.path.basename(Paths.execution))[0]}'
os.makedirs(Paths.execution, exist_ok=True)
from Helpers.log import Log

from Helpers.input import Motion
from Helpers.plot import SaveData, Plot
# Em tools desativar Scene Hierarchy, ativar Video Recorder, mudar a location da gravação para o root e clicar em "Launch at next simulation start" 
# e ao fim da gravação clicar em OK
from Helpers.record import SaveRecording 
# ===================================================================================================================================================
from Simulators import CoppeliaSim
from Simulators.CoppeliaSim import Conveyor, Cuboids

import numpy as np
import random
from Kalman.kalman import kalmanFilter

# Inicializando kalman
Q = np.matrix([[1,0],[0,0.01]])
R = np.matrix([1])
kalman_filter = kalmanFilter(Motion.ts, Q, R)

# Inicializando Coppelia
coppelia = CoppeliaSim(scene='test1.ttt')
coppelia.Conveyor = Conveyor()
coppelia.Cuboids = Cuboids()

# Começando a simulação
coppelia.Start(None)
coppelia.Step()

# Inicializando variáveis
count_vel = 0
vel = 0.01
x_pos_integration = 0

# Inicializando vetores
velocity = []
real = []
kalman = []
integration = []

for i in range(1, 100):
    Log('====================================================')
    Log(f'Vel: {vel}');velocity.append(vel)

    # Matriz pose do objeto a partir do Coppelia
    real_pose = coppelia.Cuboids.GetRealPose('3', 'red') 
    x_pos = real_pose.t[0]

    # Posição x do objeto estimada pelo Kalman
    x_pos_kalman = kalman_filter.update(-vel,x_pos)[0,0]
    Log('Kalman:', x_pos_kalman);kalman.append(x_pos_kalman)
    
    # Posição x do objeto estimada pela integração
    if x_pos_integration == 0:
        x_pos_integration = real_pose.t[0]-Motion.ts*vel
    else:
        x_pos_integration -= Motion.ts*vel
    Log('Integration:', x_pos_integration);integration.append(x_pos_integration)

    # Ruído na velocidade
    if count_vel == 2:
        vel = random.randrange(1,10)*0.01
        coppelia.Conveyor.Move(vel)
        count_vel = 0
    else:
        count_vel += 1

    coppelia.Step()

    # Posição x do objeto a partir do Coppelia
    x_pos_real = coppelia.Cuboids.GetRealPose('3', 'red').t[0]
    Log('Real:', x_pos_real);real.append(x_pos_real)

coppelia.Stop()

SaveRecording()

SaveData('test1', velocity, real, kalman, integration)
Plot('test1')