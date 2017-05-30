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
from learningSignals.elitist import Elitist

########################################## ros publishers
# for body:
pub_trajectory = rospy.Publisher('trajectory', String, queue_size=1)

########################################## global values
master = Tk()
mode = 0
var_mode = StringVar()
signal = [0,0,0]
received_signal = False
window_capacity = 100
windows = [(1-2*np.random.rand(3,window_capacity))*0.01 for _ in range(3)]
agent = Elitist(window_size=window_capacity/2, nb_moods=3, amorce_size=window_capacity/2)

########################################## functions
def onReceiveSignal(msg):
    global signal
    global received_signal
    global windows
    global agent
    signal_msg = str(msg.data)
    new_signal = [float(i) for i in signal_msg.split('_')]
    if new_signal!=signal:
        received_signal = True
        rospy.loginfo('signal: '+str(new_signal))
        signal = new_signal
        event = np.array(new_signal).reshape(3,1)
        windows[mode] = np.concatenate((windows[mode][:,1:],event),1)
        agent.learn(interval=windows[mode], mood=mode)

def setMode(new_mode):
    global mode
    mode = new_mode
    var_mode.set('current mode = '+str(mode))

def generate(mode):
    trajectory = agent.generate(mood=mode, length=200)
    '''
    fig = plt.figure(1)
    fig.add_subplot(311)
    plt.plot(trajectory[0,:],'b')
    fig.add_subplot(312)
    plt.plot(trajectory[1,:],'r')
    fig.add_subplot(313)
    plt.plot(trajectory[2,:],'g')
    plt.show()
    '''
    #send trajectory p and y (not r) to the robot
    msg = String()
    msg.data = '_'.join([str(i) for i in list(trajectory[0,:50])+list(trajectory[1,:50])])
    #traj = np.sin(np.array(range(100)))
    #msg.data = '_'.join([str(i) for i in traj])
    pub_trajectory.publish(msg)

def plot_window(length):
    last_window = np.array(windows[mode])
    fig = plt.figure(1)
    fig.add_subplot(311)
    plt.plot(last_window[0,:],'b')
    fig.add_subplot(312)
    plt.plot(last_window[1,:],'r')
    fig.add_subplot(313)
    plt.plot(last_window[2,:],'g')
    plt.show()

########################################## interface objects
Button(master, text='learn 0', height = 10, width = 30, command=lambda:setMode(0)).grid(row=0, column=0, sticky=W, pady=4)
Button(master, text='learn 1', height = 10, width = 30, command=lambda:setMode(1)).grid(row=0, column=1, sticky=W, pady=4)
Button(master, text='learn 2', height = 10, width = 30, command=lambda:setMode(2)).grid(row=0, column=2, sticky=W, pady=4)
Button(master, text='generate 0', height = 10, width = 30, command=lambda:generate(0)).grid(row=1, column=0, sticky=W, pady=4)
Button(master, text='generate 1', height = 10, width = 30, command=lambda:generate(1)).grid(row=1, column=1, sticky=W, pady=4)
Button(master, text='generate 2', height = 10, width = 30, command=lambda:generate(2)).grid(row=1, column=2, sticky=W, pady=4)
Button(master, text='plot window', height = 10, width = 30, command=lambda:plot_window(500)).grid(row=2, sticky=EW, pady=4)
Label(master, height = 10, textvariable = var_mode).grid(row=3, sticky=EW, pady=4)

########################################## ros loop
def ros_loop(test):
    while(True):
        rospy.Subscriber('signal', String, onReceiveSignal)
        rospy.sleep(0.01)
	rospy.spin()

######################################### main loop
if __name__=="__main__":
    rospy.init_node("main_activity")
    thread.start_new_thread(ros_loop, ("",))
    mainloop()
