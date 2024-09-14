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

Cb.Create.max = 1
Aruco.estimatedRpy = False

coppelia = CoppeliaSim(scene='test2.ttt')
coppelia.Camera = Camera()
coppelia.ArucoVision = ArucoVision(coppelia.Camera)
coppelia.Conveyor = Conveyor()
coppelia.Cuboids = Cuboids()

coppelia.Start(None)

coppelia.Step()
vel = 0.01
count_vel = 0
count_camera = 0
angle = 0

while coppelia.Cuboids.CheckToHandle():
    if len(coppelia.ArucoVision.detected) == 0:
        coppelia.Step()
    else:
        marker = coppelia.ArucoVision.detected[0]
        if count_camera > 20:
            position = 0
            count_camera+=1
        else:
            position = marker.objectWorldT.t[0]
            count_camera+=1
        Log('Estimado:', [kalman.update(-vel, position)[0][0], 0, 0])
        angle += vel*Motion.ts
        Log('angle:', angle)
        if count_vel==2:
            vel = random.randrange(1,10)*0.01
            Log(f'vel: {vel}')
            coppelia.Conveyor.Move(vel)
            count_vel = 0
        else:
            count_vel += 1
        coppelia.Step()
        position = coppelia.Cuboids.GetRealPose('3', 'red')
        Log('Real:', position.t)
        coppelia.Step()

coppelia.Stop()