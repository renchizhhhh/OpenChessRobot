#!/home/charles/panda/panda_env310/bin/python3.10

from franka_msgs.msg import ErrorRecoveryActionGoal
import rospy

if __name__ == "__main__":
    try:
        rospy.init_node("recovery_sender", anonymous=True, log_level=rospy.INFO)
        pub_recovery_message = rospy.Publisher("/franka_control/error_recovery/goal", ErrorRecoveryActionGoal, latch=False)
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            pub_recovery_message.publish()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
