# Ruído
import numpy as np
import random
deg = np.pi/180; rad = 180/np.pi

from Helpers.paths import Paths
import os
Paths.execution = fr'{Paths.output}\{os.path.splitext(os.path.basename(__file__))[0]}\{os.path.splitext(os.path.basename(Paths.execution))[0]}'
os.makedirs(Paths.execution, exist_ok=True)
from Helpers.log import Log

from Data.pose import Pose
from Data.targets import GetArucoPickPlace
from Data.transformations import PoseToCart, GetDot
from Models import DH_LBR_iiwa
from Helpers.measures import Real, Ref
from Helpers.input import Motion, Cuboids as Cb, Aruco, Gripper, Conveyor as Conv
from Kinematics.control import JointSpaceController, IsCloseToTarget
from Kinematics.trajectory import TrajectoryPlanning
from Simulators import CoppeliaSim
from Simulators.CoppeliaSim import Drawing, RobotiqGripper, Camera, Conveyor, Cuboids
from VisionProcessing.aruco import ArucoVision
# Em tools desativar Scene Hierarchy, ativar Video Recorder, mudar a location da gravação para o root e clicar em "Launch at next simulation start" 
# e ao fim da gravação clicar em OK
from Helpers.record import SaveRecording 
from Helpers.plot import SaveData, Plot
# ===================================================================================================================================================

from Kalman.kalman import kalmanFilter

# Inicializando kalman
Q = np.matrix([[1,0],[0,1e-4]])
R = np.matrix([1])
kalman_filter = kalmanFilter(Motion.ts, Q, R)

# Inicializando Coppelia
Cb.Create.max = 3
robot = DH_LBR_iiwa()
coppelia = CoppeliaSim(scene='test5.ttt')
coppelia.Gripper = RobotiqGripper()
coppelia.Camera = Camera()
coppelia.ArucoVision = ArucoVision(coppelia.Camera)
coppelia.Conveyor = Conveyor()
coppelia.Cuboids = Cuboids()

# Começando a simulação
robot = coppelia.Start(robot)
coppelia.Step()
count_cuboids = 0
moment = None

# Inicializando variáveis
count_vel = 0
vel = Conv.vel
x_pos_integration = 0
count_camera = 0

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
        marker = coppelia.ArucoVision.detected[0]

        if count_camera > 0:
            count_camera = 0
            count_cuboids += 1
            pickPlace = GetArucoPickPlace(robot, marker, count_cuboids)
            align, pick, place, ready, initial = pickPlace
            
            for i, target in enumerate(pickPlace[:4]):
                if i > 0 and not pickPlace[i-1].success:
                    target = ready
                elif i == 1:
                    pick.T = T
                    target = pick

                q0 = qRef0 = qDotRef0 = coppelia.GetJointsPosition(robot)
                traj = TrajectoryPlanning(
                    type = Motion.Trajectory.type, 
                    source = Motion.Trajectory.source, 
                    robot = robot, 
                    q0 = q0, 
                    T1 = target.T, 
                    t = target.t
                )

                target.SaveData(robot, Ref(traj.q, traj.qDot, traj.qDotDot, traj.x, traj.xDot, traj.xDotDot))

                Log("====== Kalman ======", target.T.t[0])
                
                for j, (TRef, qRef, qDotRef, qDotDotRef, xRef, xDotRef) in enumerate(zip(traj.T, traj.q, traj.qDot, traj.qDotDot, traj.x, traj.xDot)):
                    if target == align or target == pick:
                        Log('====================================================')
                        Log(f'Vel: {vel}');velocity.append(vel)

                        if len(coppelia.ArucoVision.detected) == 0:
                            x_pos = 0
                        else:
                            marker_pose = coppelia.ArucoVision.detected[0].objectWorldT # Matriz pose do objeto a partir da câmera
                            x_pos = marker_pose.t[0]

                        # Posição x do objeto estimada pelo Kalman
                        x_pos_kalman = kalman_filter.update(-vel,x_pos)[0,0]
                        Log('Kalman:', x_pos_kalman);kalman.append(x_pos_kalman)

                        # Posição x do objeto estimada pela integração
                        if x_pos_integration == 0:
                            x_pos_integration = marker_pose.t[0]-Motion.ts*vel
                        else:
                            x_pos_integration -= Motion.ts*vel
                        Log('Integration:', x_pos_integration);integration.append(x_pos_integration)

                    # Ruído na velocidade
                    if count_vel == 2:
                        vel = random.randrange(1,3)*0.01
                        coppelia.Conveyor.Move(vel)
                        count_vel = 0
                    else:
                        count_vel += 1
                    
                    # CONTROLE
                    q = coppelia.GetJointsPosition(robot)
                    if qRef is None:
                        qRef = robot.ikine_LMS(TRef).q
                        qDotRef = GetDot([qRef], qRef0)[0]
                        qDotDotRef = GetDot([qDotRef], qDotRef0)[0]
                        qRef0 = qRef; qDotRef0 = qDotRef
                    
                    qDotControl, target.intErr = JointSpaceController(robot, Motion.Kp, Motion.Ki, target.intErr, q, qRef, qDotRef)
                    
                    qControl = q + qDotControl*Motion.ts
                    target.measures.append([
                        Real(qControl, qDotControl, None, PoseToCart(robot.fkine(qControl)), None, None),
                        Ref(qRef, None, None, xRef, None, None)
                    ])
                    target.SaveData(robot)
                    coppelia.SetJointsTargetVelocity(robot, qDotControl); coppelia.Step(xRef[:3])

                    if target == align or target == pick:
                        # Posição x do objeto a partir da câmera
                        x_pos_camera = x_pos
                        Log('Camera:', x_pos_camera);camera.append(x_pos_camera)
                        # Posição x do objeto a partir do Coppelia
                        x_pos_real = coppelia.Cuboids.GetRealPose(marker.id, marker.color).t[0]
                        Log('Real:', x_pos_real);real.append(x_pos_real)

                    if target == pick: 
                        Log("====== Real ======", coppelia.Cuboids.GetRealPose(marker.id, marker.color).t[0])
                        prox, _ = coppelia.Gripper.CheckProximity(target.GripperActuation)
                    else:
                        prox = False
                    if prox:
                        break

                if target == align or target == pick:
                    Log('====================================================')
                    Log(f'Vel: {vel}');velocity.append(vel)

                    # Posição x do objeto estimada pelo Kalman
                    x_pos_kalman = kalman_filter.update(-vel,x_pos)[0,0]
                    Log('Kalman:', x_pos_kalman);kalman.append(x_pos_kalman)

                    # Posição x do objeto estimada pela integração
                    if x_pos_integration == 0:
                        x_pos_integration = marker_pose.t[0]-Motion.ts*vel
                    else:
                        x_pos_integration -= Motion.ts*vel
                    Log('Integration:', x_pos_integration);integration.append(x_pos_integration)

                    T = Pose(
                        x = x_pos_kalman,
                        y = marker.T.t[1], 
                        z = marker.T.t[2], 
                        rpy = marker.T.rpy()
                    )*Gripper.rotation

                coppelia.SetJointsTargetVelocity(robot, [0,0,0,0,0,0,0]); coppelia.Step(xRef[:3])
                target.success = coppelia.Gripper.HandleShape(target.GripperActuation, coppelia)

                if target == align or target == pick:
                    # Posição x do objeto a partir da câmera
                    x_pos_camera = x_pos
                    Log('Camera:', x_pos_camera);camera.append(x_pos_camera)
                    # Posição x do objeto a partir do Coppelia
                    x_pos_real = coppelia.Cuboids.GetRealPose(marker.id, marker.color).t[0]
                    Log('Real:', x_pos_real);real.append(x_pos_real)
        else:
            count_camera += 1
            coppelia.Step()

coppelia.Stop()

SaveRecording()

SaveData('test3', velocity, real, kalman, integration, camera)
Plot('test3', True)