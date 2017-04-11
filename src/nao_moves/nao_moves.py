#!/usr/bin/env python
#coding: utf-8

import sys
import time
import numpy as np
import random

import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Empty, Header
#from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
import tf

from naoqi import ALProxy
from naoqi import ALBroker
from naoqi import ALModule


#########################################"### moving functions
def StiffnessOn(motionProxy):
	pNames = "Body"
	pStiffnessLists = 1.0
	pTimeLists = 1.0
	motionProxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)

def StiffnessOff(motionProxy):
	speed = 0.1
	motionProxy.setAngles("LShoulderPitch", 1.5, speed)
	motionProxy.setAngles("RShoulderPitch", 1.5, speed)
	time.sleep(2)
	pNames = "Body"
	pStiffnessLists = 0.0
	pTimeLists = 1.0
	motionProxy.stiffnessInterpolation(pNames, pStiffnessLists, pTimeLists)

def trackFace(motionProxy,tracker):
	targetName = "Face"
	faceWidth = 0.1
	tracker.registerTarget(targetName, faceWidth)
	# Then, start tracker.
	motionProxy.setStiffnesses("Head", 1.0)
	tracker.track(targetName)

def head(motionProxy, speed, angles):
	yaw = angles[0] * np.abs(-1.3-1.3)/2. + (-1.3+1.3)/2.
	pitch = angles[1] * np.abs(-0.5-0.4)/2. + (-0.5+0.4)/2.
	motionProxy.setAngles("HeadYaw", yaw, speed)
	motionProxy.setAngles("HeadPitch", pitch, speed)

def arms(motionProxy,tts,speed, angles):
	LShoulderPitch = angles[0] * np.abs(-0.5-0.1)/2. + (-0.5+0.1)/2.
	LShoulderRoll  = angles[1] * np.abs(0.2-0.8)/2. + (0.2+0.8)/2.
	LElbowYaw  = angles[2] * np.abs(-1-1)/2. + (-1+1)/2.
	LElbowRoll  = angles[3] * np.abs(-1+0.2)/2. + (-1-0.2)/2.
	LWristYaw  = -1.6
	LHand = 1

	RShoulderPitch = angles[4] * np.abs(-0.5-0.1)/2. + (-0.5+0.1)/2.
	RShoulderRoll  = angles[5] * np.abs(-0.8-0.2)/2. + (-0.8+0.2)/2.
	RElbowYaw  = angles[6] * np.abs(-1-1)/2. + (-1+1)/2.
	RElbowRoll  = angles[7] * np.abs(0.2-1) + (0.2+1)/2.
	RWristYaw  = 1.6
	RHand = LHand

	motionProxy.setAngles("LShoulderPitch", LShoulderPitch, speed)
	motionProxy.setAngles("RShoulderPitch", RShoulderPitch, speed)
	motionProxy.setAngles("LShoulderRoll", LShoulderRoll, speed)
	motionProxy.setAngles("RShoulderRoll", RShoulderRoll, speed)
	motionProxy.setAngles("LElbowYaw", LElbowYaw, speed)
	motionProxy.setAngles("RElbowYaw", RElbowYaw, speed)
	motionProxy.setAngles("LElbowRoll", LElbowRoll, speed)
	motionProxy.setAngles("RElbowRoll", RElbowRoll, speed)
	motionProxy.setAngles("LWristYaw", LWristYaw, speed)
	motionProxy.setAngles("RWristYaw", RWristYaw, speed)
	motionProxy.setAngles("LHand", LHand, speed)
	motionProxy.setAngles("RHand", RHand, speed)

#def arms_head(motionProxy,tts,speed, angles)
