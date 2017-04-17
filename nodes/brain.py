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
from signalLearner import Drqn

############################################# publish action to main_activity (and to body)
pub_action = rospy.Publisher('action', String, queue_size = 1)

############################################# global values
agent = Drqn(input_size=6, nb_action=2, gamma=0.9)
stop = False
signal = [0,0,0,0,0,0]
reward = 0
received_signal = False
received_reward = False

############################################# functions
def onExit(msg):
    global stop
    stop = True

def onReceiveSignal(msg):
    global signal
    signal_msg = str(msg.data)
    signal = [float(i) for i in signal_msg.split('_')]
    received_signal = True

def onReceiveReward(msg):
    global reward
    reward_msg = str(msg.data)
    reward = float(reward_msg)
    received_reward = True

############################################# main loop
if __name__=="__main__":
    rospy.init_node("brain")

    while not stop:
        rospy.Subscriber('exit_topic', String, onExit)
        rospy.Subscriber('signal', String, onReceiveSignal)
        rospy.Subscriber('reward', String, onReceiveSignal)

        if received_reward and received_signal:
            action = agent.update(reward, signal)
            msg = String()
            msg.data = str(action)
            pub_action.publish(msg)
            received_signal = False
            received_reward = False

        rospy.sleep(0.1)
    rospy.spin()
