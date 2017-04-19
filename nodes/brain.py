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
from learningSignals.agents import Drqn

############################################# publish action to main_activity (and to body)
pub_action = rospy.Publisher('action', String, queue_size = 1)

############################################# global values
agent = Drqn(input_size=3, nb_action=2, gamma=0.9)
stop = False
signal = [0,0,0]
reward = 0
received_signal = False
received_reward = True

############################################# functions
def onExit(msg):
    global stop
    stop = True

def onReceiveSignal(msg):
    global signal
    global received_signal
    signal_msg = str(msg.data)
    signal = [float(i) for i in signal_msg.split('_')]
    received_signal = True
    rospy.loginfo('signal: '+str(signal))

def onReceiveReward(msg):
    global reward
    global received_reward
    reward_msg = str(msg.data)
    reward = float(reward_msg)
    received_reward = True
    rospy.loginfo('reward: '+reward_msg)

############################################# main loop
if __name__=="__main__":
    rospy.init_node("brain")

    while not stop:
        rospy.Subscriber('exit_topic', String, onExit)
        rospy.Subscriber('signal', String, onReceiveSignal)
        rospy.Subscriber('reward', String, onReceiveReward)

        if received_reward and received_signal:
            rospy.loginfo('first update !')
            action = agent.update(reward, signal)
            rospy.loginfo('action: '+str(action))
            msg = String()
            msg.data = str(action)
            pub_action.publish(msg)
            received_signal = False
            received_reward = False

        rospy.sleep(0.1)
    rospy.spin()
