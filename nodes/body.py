#!/usr/bin/env python
#coding: utf-8

import sys
import time
import numpy as np
import random
import rospy
import tf

from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Empty, Header

''' no robot for the moment
from naoMoves import nao_moves as nm
from naoqi import ALProxy
from naoqi import ALBroker
from naoqi import ALModule
'''

############################################# publish obs to brain
pub_signal = rospy.Publisher('signal', String, queue_size = 1)

############################################# global values
''' no robot for the moment
#NAO_IP = "192.168.1.64" #HACK : this should not be hardcoded !
NAO_IP = "146.193.224.102" #HACK : this should not be hardcoded !
port = 9559
speed = 0.1
angles = [0,0] #HACK : will be longer in futur !
'''
Zpitch = 0
Zyaw = 0
stop = False

''' no robot for the moment
############################################# init proxies
myBroker = ALBroker("myBroker","0.0.0.0",0,NAO_IP,port)
hasFallen = False
motionProxy = ALProxy("ALMotion", NAO_IP, port)
memoryProxy = ALProxy("ALMemory", NAO_IP, port)
postureProxy = ALProxy("ALRobotPosture", NAO_IP, port)
faceProxy = ALProxy("ALFaceDetection", NAO_IP, port)
tracker = ALProxy("ALTracker", NAO_IP, port)
'''

############################################# what if new action

#TODO: place origine facing the human (instead of p+y = zero)

''' no robot for the moment
def onReceiveAction(msg):
    action_msg = str(msg.data)
    angles = [float(angle) for angle in action_msg.split("_")]
    angle[0] += Zyaw
    angle[1] += Zpitch
    action = True
'''

def onExit(msg):
    global stop
    stop = True


############################################# main loop
if __name__=="__main__":

    rospy.init_node("body")

    ''' no robot for the moment
    sg.StiffnessOn(motionProxy)
    '''

    listener = tf.TransformListener()
    listener.waitForTransform('/base_footprint','/face_0', rospy.Time(0), rospy.Duration(4.0))

    while not stop:

        test  = listener.getFrameStrings()
        ''' no robot for the moment
        action = False
        '''

        rospy.Subscriber('exit_topic', String, onExit)

        ''' no robot for the moment
        if action:
            time.sleep(1)
            nm.head(motionProxy, speed, angles)
        '''

        if "base_footprint" in test and "robot_head" in test and "face_0" in test:
            rospy.loginfo("frames found! (head condition")

            (pose,rot) = listener.lookupTransform('/robot_head','/face_0', rospy.Time(0))

            x = pose[0]
            y = pose[1]
            z = pose[2]

            Zyaw = np.arctan(y/x) # this should be robot's origine yaw=0
            Zpitch = np.arctan(-z/x) # this should be robot's origine pitch=0

            euler = tf.transformations.euler_from_quaternion(rot)
            r = euler[1]
            p = euler[0]
            y = euler[2]
            '''
            # if perspective taking:
            r = euler[1]
            p = euler[0] - np.pi/2.
            y = -np.sign(euler[2])*(np.abs(euler[2])-np.pi/2.)
            '''
            msg = String()
            msg.data = str(x)+"_"+str(y)+"_"+str(z)+"_"+str(r)+"_"+str(p)+"_"+str(y)
            pub_signal.publish(msg.data)

        rospy.sleep(0.2)

    ''' no robot for the moment
    sg.StiffnessOff(motionProxy)
    '''

    rospy.spin()
