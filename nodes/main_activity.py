#!/usr/bin/env python
#coding: utf-8

import sys
import time
import numpy as np
import random
import rospy
import thread

from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Empty, Header
from Tkinter import *

########################################## ros publishers
# for brain:
pub_reward = rospy.Publisher('reward', String, queue_size=1)

########################################## global values
master = Tk()
var_goal = StringVar()
var_last_action = StringVar()
goal = 0
action = 0

########################################## functions
def onReceiveAction(msg):
    global action
    action_msg = str(msg.data)
    action = float(action_msg)
    var_last_action.set('last_action = '+action_msg)
    reward = 0
    if action>0:
        if goal>0:
            reward = 1
        else:
            reward = -1
    msg = String()
    msg.data = str(reward)
    pub_reward.publish(msg)

def setGoal(new_goal):
    global goal
    goal = new_goal
    var_goal.set('goal = '+str(goal))

########################################## interface objects
Button(master, text='goal 0', height = 10, width = 30, command=lambda:setGoal(0)).grid(row=0, column=0, sticky=W, pady=4)
Button(master, text='goal 1', height = 10, width = 30, command=lambda:setGoal(1)).grid(row=0, column=1, sticky=W, pady=4)
Label(master, height = 10, textvariable = var_goal).grid(row=1, sticky=EW, pady=4)
Label(master, height = 10, textvariable = var_last_action).grid(row=2, sticky=EW, pady=4)

########################################## ros loop
def ros_loop(test):
	while True:
		rospy.Subscriber('action', String, onReceiveAction)
		rospy.sleep(0.1)
	rospy.spin()

######################################### main loop
if __name__=="__main__":
    rospy.init_node("main_activity")
    thread.start_new_thread(ros_loop, ("",))
    mainloop()
