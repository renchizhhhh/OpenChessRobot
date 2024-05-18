#!/home/charles/panda/panda_env310/bin/python3.10

import rospy

if __name__ == '__main__':
    rospy.init_node("movement_mamanger", anonymous=True)
    rospy.set_param('is_waiting', False)
    rospy.set_param('is_moving', False)
    rospy.set_param('is_speaking', False)
    
    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        if rospy.get_param('is_moving'):
            rospy.logwarn('robot rotating detected')
        rate.sleep()
