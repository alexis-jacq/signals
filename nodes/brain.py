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

# for training:
pub_scores = rospy.Publisher('scores', String, queue_size = 1)
pub_hiddens = rospy.Publisher('hiddens', String, queue_size = 1)

############################################# global values
agent = Drqn(input_size=3, nb_action=2, gamma=0.9)
stop = False
signal = [0,0,0]
reward = 0
key = 0
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
    new_signal = [float(i) for i in signal_msg.split('_')]
    if new_signal!=signal:
        received_signal = True
        rospy.loginfo('signal: '+str(new_signal))
        signal = new_signal


def onReceiveReward(msg):
    global reward
    global key
    global received_reward
    reward_msg = str(msg.data)
    new_reward, new_key = float(reward_msg.split('_')[0]), float(reward_msg.split('_')[1])
    #rospy.loginfo('new_key: '+str(new_key))
    #rospy.loginfo('key: '+str(key))
    if new_key!=key:
        reward = new_reward
        key = new_key
        received_reward = True
        rospy.loginfo('reward: '+str(reward))

############################################# main loop
if __name__=="__main__":
    rospy.init_node("brain")

    while not stop:
        rospy.Subscriber('exit_topic', String, onExit)
        rospy.Subscriber('signal', String, onReceiveSignal)
        rospy.Subscriber('reward', String, onReceiveReward)

        if received_signal:#received_reward and received_signal:
            #rospy.loginfo('update !')
            action = agent.update(reward, signal)
            #rospy.loginfo('action: '+str(action))
            msg = String()
            msg.data = str(action)+'_'+str(np.random.rand())
            pub_action.publish(msg)
            received_signal = False
            received_reward = False
            #for training:
            msg_scores = String(); msg_hiddens = String()
            msg_scores.data = str(agent.scores*100)
            msg_hiddens.data = str(agent.last_hidden*100)
            pub_scores.publish(msg_scores)
            pub_hiddens.publish(msg_hiddens)

        rospy.sleep(0.3)
    rospy.spin()
