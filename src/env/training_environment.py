#!/usr/bin/env python


import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
from training_respawn import Respawn


class Env():
    def __init__(self):
        self.envs_list = {}
        self.record_goals = 0
        self.sequential_goals = 0
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel_l = rospy.Publisher('cmd_vel_l', Twist, queue_size=5)
        self.pub_cmd_vel_r = rospy.Publisher('cmd_vel_r', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.past_distance = 0.

        rospy.on_shutdown(self.shutdown)

    def shutdown(self):

        rospy.loginfo("Stopping TurtleBot")
        self.pub_cmd_vel_l.publish(Twist())
        self.pub_cmd_vel_r.publish(Twist())
        rospy.sleep(1)

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        self.past_distance = goal_distance

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        #print 'yaw', yaw
        #print 'gA', goal_angle

        heading = goal_angle - yaw
        #print 'heading', heading
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 3)

    def getState(self, scan, past_action):
        scan_range = []
        heading = self.heading
        min_range = 0.16
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True

        for pa in past_action:
            scan_range.append(pa)

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        if current_distance < 0.2:
            self.get_goalbox = True

        return scan_range + [heading, current_distance], done

    def setReward(self, state, done):
        current_distance = state[-1]
        heading = state[-2]

        distance_rate = (self.past_distance - current_distance) 
        if distance_rate > 0:
            reward = 200.*distance_rate
        if distance_rate <= 0:
            reward = -8.
        self.past_distance = current_distance
        if done:
            rospy.loginfo("Collision!!")
            rospy.loginfo("record = " + str(self.record_goals))
            if self.record_goals < self.sequential_goals:
                self.record_goals = self.sequential_goals
            self.sequential_goals = 0
            reward = -100.
            self.pub_cmd_vel_l.publish(Twist())
            self.pub_cmd_vel_r.publish(Twist())

        if self.get_goalbox:
            self.sequential_goals += 1
            rospy.loginfo("Goal!!")
            if self.record_goals < self.sequential_goals:
                self.record_goals = self.sequential_goals
            rospy.loginfo("current = " + str(self.sequential_goals))
            rospy.loginfo("record = " + str(self.record_goals))
            reward = 1000.
            self.pub_cmd_vel_l.publish(Twist())
            self.pub_cmd_vel_r.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.moduleRespawns()
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False


        return reward

    def step(self, action, past_action):
        wheel_vel_l = action[0]
        wheel_vel_r = action[1]

        vel_cmd_l = Twist()
        vel_cmd_l.linear.x = wheel_vel_l

        vel_cmd_r = Twist()
        vel_cmd_r.linear.x = wheel_vel_r


        self.pub_cmd_vel_l.publish(vel_cmd_l)
        self.pub_cmd_vel_r.publish(vel_cmd_r)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data, past_action)
        reward = self.setReward(state, done)

        return np.asarray(state), reward, done

    def reset(self):

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
        data = None
 
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                print "scan failed"
                pass
        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.moduleRespawns()
            self.initGoal = False
        else:
            self.goal_x, self.goal_y = self.respawn_goal.moduleRespawns()

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data, [0.,0.])

        return np.asarray(state)
