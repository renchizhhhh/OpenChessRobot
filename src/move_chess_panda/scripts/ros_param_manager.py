#!/home/charles/panda/panda_env310/bin/python3.10

import rospy
from moveit_msgs.msg import ExecuteTrajectoryActionFeedback, MoveGroupFeedback

def update_execute_feedback(msg):
    if msg.status.status == 3:
        rospy.set_param('is_last_command_successful', True)
        rospy.logwarn("execute finished")
    else:
        rospy.set_param('is_last_command_successful', False)
        rospy.logwarn(f"doing, current status is {msg.status.status}")

def update_go_feedback(msg):
    if msg.status.status == 3:
        rospy.set_param('is_last_command_successful', True)
        rospy.logwarn("go finished")
    else:
        rospy.set_param('is_last_command_successful', False)
        rospy.logwarn(f"doing, current status is {msg.status.status}")

if __name__ == '__main__':
    rospy.init_node("param_mamanger", anonymous=True)
    rospy.set_param('is_waiting', False)
    rospy.set_param('is_moving', False)
    rospy.set_param('is_speaking', False)
    rospy.set_param('board_is_localized', False)
    rospy.set_param('is_last_command_successful', False)
    rospy.Subscriber("/execute_trajectory/feedback", ExecuteTrajectoryActionFeedback, update_execute_feedback)
    rospy.Subscriber("/move_group/feedback", ExecuteTrajectoryActionFeedback, update_go_feedback)

    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        rate.sleep()
