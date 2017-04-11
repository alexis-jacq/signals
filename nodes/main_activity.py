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
from naoStoryTelling import story_gestures as sg
from nextChoice import story_maker as sm
from nextChoice.story_maker import story
from nextChoice.decision2 import *

from naoqi import ALProxy
from naoqi import ALBroker
from naoqi import ALModule

########################################## ros publishers
# for nao_actions:
pub_robot_target = rospy.Publisher('robot_target_topic', String, queue_size=1)
pub_robot_say = rospy.Publisher('robot_say_topic', String, queue_size=1)
pub_robot_say_long = rospy.Publisher('robot_say_long_topic', String, queue_size=1)
pub_robot_point = rospy.Publisher('robot_point_topic', String, queue_size=1)
pub_exit = rospy.Publisher('exit_topic', String, queue_size=1)

# for interface:
pub_human_turn = rospy.Publisher('human_turn_topic', String, queue_size=1)
pub_human_chosen = rospy.Publisher('human_chosen_topic', String, queue_size=1)
pub_human_predict = rospy.Publisher('human_predict_turn_topic', String, queue_size=1)
pub_robot_turn = rospy.Publisher('robot_turn_topic', String, queue_size=1)
pub_robot_chosen = rospy.Publisher('robot_chosen_topic', String, queue_size=1)
pub_new_element = rospy.Publisher('new_element', String, queue_size=1)

# for withmeness
pub_state = rospy.Publisher('state_activity', String, queue_size=1)
########################################## publishing
