# Pegando as informações do objeto vindas do coppelia
import numpy as np
import random
deg = np.pi/180; rad = 180/np.pi

from Helpers.paths import Paths
import os
Paths.execution = fr'{Paths.output}\{os.path.splitext(os.path.basename(__file__))[0]}\{os.path.splitext(os.path.basename(Paths.execution))[0]}'
os.makedirs(Paths.execution, exist_ok=True)
from Helpers.log import Log

from Data.targets import GetArucoPickPlace
from Data.transformations import PoseToCart, GetDot
from Models import DH_LBR_iiwa
from Helpers.measures import Real, Ref
from Helpers.input import Motion, Cuboids as Cb, Aruco
from Kinematics.control import JointSpaceController, IsCloseToTarget
from Kinematics.trajectory import TrajectoryPlanning
from Simulators import CoppeliaSim
from Simulators.CoppeliaSim import Drawing, RobotiqGripper, Camera, Conveyor, Cuboids
from VisionProcessing.aruco import ArucoVision

from kalman import kalmanFilter
Q = np.matrix([[1,0],[0,0.01]])
R = np.matrix([1])

kalman = kalmanFilter(Motion.ts, Q, R)
coppelia = CoppeliaSim(scene='test1.ttt')

coppelia.Conveyor = Conveyor()
coppelia.Cuboids = Cuboids()

coppelia.Start(None)

coppelia.Step()
count = 0
vel = 0.01
angle = 0.0

while True:
    pose = coppelia.Cuboids.GetRealPose('3', 'red')
    Log('Estimado:', [kalman.update(-vel)[0][0], 0, 0])
    angle += vel*Motion.ts
    Log('angle:', angle)
    if count==2:
        vel = random.randrange(1,10)*0.01
        Log(f'vel: {vel}')
        coppelia.Conveyor.Move(vel)
        count = 0
    else:
        count += 1
    coppelia.Step()
    pose = coppelia.Cuboids.GetRealPose('3', 'red')
    Log('Real:', pose.t)