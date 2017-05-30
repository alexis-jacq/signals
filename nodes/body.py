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

import naoMoves.nao_moves as nm
from naoqi import ALProxy
from naoqi import ALBroker
from naoqi import ALModule

############################################# publish obs to brain
pub_signal = rospy.Publisher('signal', String, queue_size = 1)

############################################# global values
#NAO_IP = "192.168.1.64" #HACK : this should not be hardcoded !
NAO_IP = "146.193.224.25" #HACK : this should not be hardcoded !
port = 9559
speed = 0.2
stop = False
phase = 0
t = 0
action = False
angles_py = None

############################################# init proxies
myBroker = ALBroker("myBroker","0.0.0.0",0,NAO_IP,port)
hasFallen = False
motionProxy = ALProxy("ALMotion", NAO_IP, port)
memoryProxy = ALProxy("ALMemory", NAO_IP, port)
postureProxy = ALProxy("ALRobotPosture", NAO_IP, port)
faceProxy = ALProxy("ALFaceDetection", NAO_IP, port)
tracker = ALProxy("ALTracker", NAO_IP, port)

############################################# what if new action

#TODO: place origine facing the human (instead of p+y = zero)

def onTrajectory(msg):
    global action
    global angles_py
    action_msg = str(msg.data)
    angles = [float(angle) for angle in action_msg.split("_")]
    angles_p = angles[:len(angles)/2]
    angles_y = angles[len(angles)/2:]
    rospy.loginfo(action_msg)
    angles_py = zip(angles_p,angles_y)
    action = True

def onExit(msg):
    global stop
    stop = True


############################################# main loop
if __name__=="__main__":

    rospy.init_node("body")

    nm.StiffnessOn(motionProxy)
    tracker.registerTarget("Face", 1)
    tracker.track("Face")

    listener = tf.TransformListener()
    face_found = False
    while not face_found:
        try:
            listener.waitForTransform('/base_footprint','/face_0', rospy.Time(0), rospy.Duration(4.0))
            rospy.loginfo('face detected !')
            face_found = True
        except tf.Exception:
            pass

    while not stop:

        test  = listener.getFrameStrings()
        action = False

        rospy.Subscriber('exit_topic', String, onExit)
        rospy.Subscriber('trajectory', String, onTrajectory)

        if action:
            tracker.stopTracker()
            for (p,y) in angles_py:
                nm.head(motionProxy, speed, [p,y])
                time.sleep(0.1)
            tracker.track("Face")


        if "base_footprint" in test and "robot_head" in test and "face_0" in test:
            #rospy.loginfo("frames found! (head condition")

            (pose,rot) = listener.lookupTransform('/robot_head','/face_0', rospy.Time(0))

            x = pose[0]
            y = pose[1]
            z = pose[2]

            # if the robot wants to look at the head :
            Zyaw = np.arctan(y/x)
            Zpitch = np.arctan(-z/x)

            euler = tf.transformations.euler_from_quaternion(rot)
            '''
            # if no perspective taking
            r = euler[1]
            p = euler[0]
            y = euler[2]
            #'''

            # if perspective taking:
            r = euler[1]
            p = Zpitch + euler[0] - np.pi/2.
            y = Zyaw - np.sign(euler[2])*(np.abs(euler[2])-np.pi/2.)

            msg = String()
            #msg.data = str(x)+"_"+str(y)+"_"+str(z)+"_"+str(r)+"_"+str(p)+"_"+str(y)
            msg.data = str(p)+"_"+str(y)+"_"+str(r)
            pub_signal.publish(msg.data)
            #TODO: slow a lot the frequency of publication

        rospy.sleep(0.01)

    nm.StiffnessOff(motionProxy)

    rospy.spin()
