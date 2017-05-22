#!/usr/bin/env python
#coding: utf-8

import sys
import time
import numpy as np
import random
import rospy
import thread
import matplotlib.pyplot as plt

from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Empty, Header
from Tkinter import *

# going to import new version of generative agent..
from learningSignals.agents import Elitist

########################################## ros publishers
# for body:
pub_trajectory_to_generate = rospy.Publisher('trajectory_to_generate', String, queue_size=1)

########################################## global values
master = Tk()
var_mode = StringVar()
signal = [0,0,0]
received_signal = False
window_capacity = 200
window = []
agent = Elitist(window_size = window_capacity, nb_moods = 3)

########################################## functions
def onReceiveSignal(msg):
    global signal
    global received_signal
    signal_msg = str(msg.data)
    new_signal = [float(i) for i in signal_msg.split('_')]
    if new_signal!=signal:
        received_signal = True
        rospy.loginfo('signal: '+str(new_signal))
        signal = new_signal
        window.append(signal)
        if len(window)>window_capacity:
            del window[0]
            agent.learn(zip(*window))

def setMode(new_mode):
    global mode
    mode = new_mode
    var_goal.set('current mode = '+str(mode))

def generate(mode):
    trajectory = agent.generate(mode = mode, length = 1000)
    plt.figure()
    plt.plot(trajectory[0,:],'b')
    plt.plot(trajectory[1,:],'r')
    plt.plot(trajectory[2,:],'g')
    plt.show()
    '''
    msp = String()
    msg.data = '_'.join([])
    pub_trajectory_to_generate.publish(msg)
    '''

########################################## interface objects
Button(master, text='learn 0', height = 10, width = 30, command=lambda:setMode(0)).grid(row=0, column=0, sticky=W, pady=4)
Button(master, text='learn 1', height = 10, width = 30, command=lambda:setMode(1)).grid(row=0, column=1, sticky=W, pady=4)
Button(master, text='learn 2', height = 10, width = 30, command=lambda:setMode(2)).grid(row=0, column=2, sticky=W, pady=4)
Button(master, text='generate 0', height = 10, width = 30, command=lambda:generate(0)).grid(row=1, column=0, sticky=W, pady=4)
Button(master, text='generate 1', height = 10, width = 30, command=lambda:generate(1)).grid(row=1, column=1, sticky=W, pady=4)
Button(master, text='generate 2', height = 10, width = 30, command=lambda:generate(2)).grid(row=1, column=2, sticky=W, pady=4)
Label(master, height = 10, textvariable = var_mode).grid(row=2, sticky=EW, pady=4)

########################################## ros loop
def ros_loop(test):
	while True:
        rospy.Subscriber('signal', String, onReceiveSignal)
		rospy.sleep(0.3)
	rospy.spin()

######################################### main loop
if __name__=="__main__":
    rospy.init_node("main_activity")
    thread.start_new_thread(ros_loop, ("",))
    mainloop()
