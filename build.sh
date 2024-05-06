catkin clean -y
rosdep install --from-paths . --ignore-src --rosdistro noetic -y --skip-keys libfranka
catkin config --extend /opt/ros/${ROS_DISTRO} --cmake-args -DCMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH=${HOME}/panda/libfranka/build
catkin build
