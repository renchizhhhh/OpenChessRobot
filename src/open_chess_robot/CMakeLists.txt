cmake_minimum_required(VERSION 3.0.2)
project(open_chess_robot)

find_package(catkin REQUIRED COMPONENTS
  actionlib_msgs
  message_generation
  rospy
  std_msgs
)

catkin_python_setup()

add_service_files(
  FILES
  RecognizeBoard.srv
)

add_action_files(
  FILES
  ChessMove.action
)

generate_messages(
  DEPENDENCIES
  actionlib_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  actionlib_msgs
  message_runtime
  rospy
  std_msgs
)

catkin_install_python(
  PROGRAMS
  scripts/chess_commander.py
  scripts/chess_robo_player.py
  scripts/chess_robot_recovery.py
  scripts/data_collect_chess_commander.py
  scripts/data_collect_chess_robot.py
  scripts/eva_chess_commander.py
  scripts/eva_chess_robot.py
  scripts/llm_chess_commander.py
  scripts/llm_commentary_manager.py
  scripts/hri_chess_commander.py
  scripts/hri_chess_robot.py
  scripts/ros_param_manager.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY
  config
  docs
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  DIRECTORY scripts/chessrec/config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts/chessrec/config
)

install(
  DIRECTORY scripts/data/games/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts/data/games
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/chessrec/runs/runtime")
  install(
    DIRECTORY scripts/chessrec/runs/runtime/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts/chessrec/runs/runtime
  )
endif()

install(
  PROGRAMS
  scripts/engine/stockfish15
  scripts/engine/stockfish16
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts/engine
)

install(
  DIRECTORY scripts/engine/maia/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts/engine/maia
  USE_SOURCE_PERMISSIONS
  PATTERN "backup" EXCLUDE
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_phase_a_configuration.py)
  catkin_add_nosetests(test/test_phase_b_move_sequences.py)
  catkin_add_nosetests(test/test_phase_c_recognition.py)
  catkin_add_nosetests(test/test_phase_c_geometry.py)
  catkin_add_nosetests(test/test_phase_c_recognition_augment.py)
  catkin_add_nosetests(test/test_phase_c_recognition_projection.py)
  catkin_add_nosetests(test/test_phase_c4_command_result.py)
  catkin_add_nosetests(test/test_llm_commentary.py)
endif()
