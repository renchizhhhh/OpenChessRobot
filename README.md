# OpenChessRobot: An Open-Source Reproducible Chess Robot for Human-Robot Interaction Research

This project introduces OpenChessRobot, an open-source project at the intersection of artificial intelligence, robotics, and human-computer interaction with an aim for reproducibility. This repository hosts the complete source code and documentations for building and utilizing an open-source chess robot designed for human-robot interaction (HRI) research. Our project leverages chess as a standardized testing environment to explore and quantify the dynamics of human interactions with robots.

The OpenChessRobot has been evaluated in an international online survey and expert gameplay. Please checkout our paper for more details: https://doi.org/10.3389/frobt.2025.1436674

## Project Overview

The OpenChessRobot aims to improve the HRI by focusing on both verbal and non-verbal communication modes between humans and robots. A large language model is used to translate the evaluation from the chess engines into human-like speech. Supported by an international survey and expert gameplay, our findings show the robot’s potential in educational and research contexts, while highlighting limitations in human-like interaction

Check out the demo videos on Youtube: 
- OpenChessRobot Demo: https://youtu.be/RenXuiwX4Go
- Demo Video of the Online Study: https://www.youtube.com/shorts/Mj_rZVboH9s 
- Additional Eye Tracking Demo: https://youtu.be/j-b32ILQjtw

### Key Features

- **Chess Piece Recognition**: Utilizes advanced computer vision techniques to identify chess pieces accurately and determine their positions on the board.
- **Robotic Movements**: Executes chess moves with precision using robotic arms, enhancing the interaction realism.
- **Interactive Communication**: Engages with human players through voice commands and robotic gestures, providing a comprehensive HRI experience.
- **Qualitative Evaluations**: Includes ChatGPT to generate human-like game analysis based on the recognized game position.
- **Reproducibility**: Detailed documentation and guidelines are provided to ensure that researchers can replicate the setup and experiments.

## Getting Started

This GitHub repository relies on the materials listed here by default. However, some of these hardware can be replaced easily or we provide suggestions for possible [alternatives](#possible-alternatives-of-the-hardware). 

- a Franka Emika Panda robot arm (Franka Emika, 2020) equipped with a Franka Hand and a customized 3D-printed robot gripper [3D print](). It operates on v5.4.0 firmware.
- a ZED2 StereoLabs camera (StereoLabs, n.d.) and a customized 3D-printed mount for it. 
- a NVIDIA Jetson Nano (NVIDIA, n.d.). 
- a Linux PC with Ubuntu 20.04, with real-time kernel. It is equipped with an Intel I7-8700K processor and an NVIDIA GTX2080 graphics card.
- an external microphone and a speaker connected to the PC. 
- a keyboard connected to the PC. 

### Prerequisites

Before diving into the setup, make sure you have the required hardware components mentioned above and software dependencies installed. 

1. ROS, MoveIt, Franka Control Interface (FCI) \
This project uses ROS and MoveIt to communicate with FCI for the robot control. So it's necessary to install the correct version of each framework. \
Follow this [link](http://wiki.ros.org/noetic/Installation/Ubuntu) to install the `ROS Noetic` for Ubuntu. For people who are not farmiliar with ROS: we recommand to follow the installation guide line by line rather than the convient One-line Installation tutorial. \
Follow this [link](https://moveit.ros.org/install/source/) to install the `MoveIt 1` for Ubuntu. We recommand to download the source code of MoveIt and then move the `ws_moveit` folder that contains `MoveIt` packages to `<your catkin workspace>/src`. \
Follow this [link](https://frankaemika.github.io/docs/installation_linux.html) to install the `libfranka` and `franka_ros` for Ubuntu. If you didn't have the realtime kernel on your Linux installation, this tutorial will also guide you build and install the realtime kernel, which is necessary for FCI.
We recommand to move the `franka_ros` folder to `<your catkin workspace>/src`.

2. Nvidia Driver \
The default `nvidia-driver` is incompatible with the realtime kernel. You will need to follow this [link](https://gist.github.com/pantor/9786c41c03a97bca7a52aa0a72fa9387) to install the nvidia-driver and check if `nvidia-smi` works while using the realtime kernel. Please note that we are using the `nvidia-driver-525.78.01`. 

3. Zed SDK \
If you want to use a Zed2 camera and a Nvidia Jetson for the camera stream, first install the `ZED SDK 3.8.2` from this [link](https://www.stereolabs.com/developers/release/3.8). Then use this [script](https://github.com/stereolabs/zed-sdk/blob/master/camera%20streaming/sender/python/streaming_sender.py) on the Jetson to broadcast the video stream at Jetson's ip address in your local network. 

### Installation

1. Download and build the necessary packages \
First, clone the repository:
```
cd <your catkin workspace>
git clone https://github.com/renchizhhhh/OpenChessRobot.git
```

Idealy, `<your catkin workspace>` now should contain folders: `franka_ros`, `ws_moveit` and the `move_chess_panda`. 

Then, install the missing packages if any:
```
rosdep install --from-paths src --ignore-src --rosdistro noetic -y --skip-keys libfranka
```

Now, build the packages using the Catkin-tools, which should be installed when you set up the moveit before: 
```
# install the Catkin-tools in case you don't have 
# sudo apt-get install python3-catkin-tools
catkin clean -y
catkin config --extend /opt/ros/${ROS_DISTRO} --cmake-args -DCMAKE_BUILD_TYPE=Release -DFranka_DIR:PATH=<Path to your libfranka>/build
catkin build
```
**Note:** After the build, don't forget to source your workspace every time before you want to start the robot. 

2. Python environment and packages

All the packages are tested with Python3.10 so make sure you are using the same version when you have trouble with the package conflicts. 

Setup a virtual env with `virtualenv`: 
```
python -m venv --system-site-packages chess_bot_env  
source chess_bot_env/bin/activate
pip install --upgrade pip
```
Make sure that the ROS installed python packages under `/opt/ros/noetic/lib/python3/dist-packages` are visiable in this venv. Otherwise, you need to source the ROS and try it again. 
```
pip list -v
```
Install all the required python packages:
```
cd <your catkin workspace>/src/move_chess_panda
pip install -r requirements.txt
```
Then build the python package in your workspace:
```
catkin build --this
```

### Configurations of the Project
Important configrations are stored in the `setup_configurations.py`. Please edit the variables in this file before you start the robot. 

Variables for the robot:\
`ROBOT`: the name of the robot. Default is `panda`.  \
`ACC`: the acceleration for the robot movements.\
`VEL` = the velocity for the robot movements.\
`Z_ABOVE_BOARD`: the height above the board for the pre-grasp pose.\
`Z_TO_PIECE`: the depth from the pre-grasp pose to the piece.\
`Z_DROP`: the height above the board for placing the piece.\
`X_OFFSET`: the offset for the gripper on x-axis.\
`Y_OFFSET`: the offset for the gripper on y-axis.

Variables for the camera:\
`CAM_IP`: the local ip address for the Jetson that broadcasting the Zed camera stream. \
`CAMERA`: the left or right camera of the Zed camera to use.\
`SQUARE_SIZE`: the length of the chess squares.\
`MARKER_SIZE`: the length of the ArUco markers.\
`MARKER_TYPE`: the type of the ArUco markers.\
`MODEL_PATH`: the path to the game recognition models.

Variables for the chess engine:\
`MODE`: the chess engine to use (stockfish15 | stockfish16 | maia). \
`DEPTH`: the search depth (only applicable for the stockfish15 | stockfish16).\
`ELO`: the target ELO (only applicable for the stockfish15 | stockfish16).

### Download

#### 3D print
There are two customized 3D-printed parts for the robot. You can find the models under `<your local repository>/src/move_chess_panda/3d_print`. Special thanks to [Thomas de Boer](https://github.com/Thomahawkuru) who helps us create and print the models. 

#### The recognition model
The pretrained models for chess piece recognition can be downloaded from the following link: 
[4TU ResearchData](https://data.4tu.nl/datasets/1cb5bf64-468e-462a-a82e-c847d88a7a86). We provide the pretrained models trained on the synthetic dataset and the finetuned models on our chess set. The chess set we use can be found in the following links: [Chessboard](https://schaakshop.nl/schaakborden/Schaakbord-SB10) (currently out of stock) and [Chess pieces](https://schaakshop.nl/Schaakstukken-Staunton-4b). 

#### The ArUco marker
The markers used to localize the chess board are in the `3d_print_and_markers` folder. Print and place these markers at the edge of the chessboard, as shown in the demo video. Remember to update the `MARKER_SIZE` in the configurations. 

## Use the Robot
### Preparation
Follow the step by step installation guidance to install the project. Prepare the 3D-printed gripper and attach the ArUco markers at the four corners of the chessboard. 

### Data Collection and Finetuning
Launch the `data_collection.launch` and use `data_collect_chess_commander.py`.

### Human Robot Chess Play
Launch the `hri_chess.launch` and use `hri_chess_commander.py`.

## Possible Alternatives of the Hardware and Accessories

| Material               | Suggestions                                                                                          | Potential Actions to Take                                                                                                                                                                                                     |
| ---------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Robot arm and hand     | Not recommanded because it involves quite some work. Although it's possible to use other robot arms. | You may need to import your robot model to make it recognized well by MoveIt and can use MoveGroup for execution. Then you need to adjust all most all the parameters in the setup_configurations.py file to suit your robot. |
| Zed2 Camera            | Can be replaced by any RGB camera                                                                    | Edit the \`get_img\` function and its relevant functions in\`utili/camera_config.py\`                                                                                                                                         |
| NVIDIA Jetson Nano     | Can be removed if you don't need a video streaming via local network to the PC                       | Edit the Zed camera related functions in the \`Camera\` class in\`utili/camera_config.py\`                                                                                                                                    |
| Keyboard               | Can be easily replaced by any keyboard                                                               | Plug and play                                                                                                                                                                                                                 |
| Microphone and speaker | Can be easily replaced by any microphone and speaker                                                 | Plug and play                                                                                                                                                                                                                 |
| PC                     | Can be easily replaced by other PCs with similar Nvidia card and the same Ubuntu system              | Make sure that your nvidia driver is workable with the realtime kernel.                                                                                                                                                       |
| Chess set              | Can be easily replaced by other similar chess sets.                                                  | Retrain the chess recognition model and tweak the configurations for the robot grasping.               

## Datasets and Pretrained Models
The data folder includes real-world chess image data in both raw and processed forms, along with pre-trained and fine-tuned models. These resources can be found in a public data repository [4TU ResearchData](https://data.4tu.nl/datasets/1cb5bf64-468e-462a-a82e-c847d88a7a86). It includes the following key components:

- Raw and Processed Chess Image Data: Essential for training and evaluating the robot's computer vision system.

- Pre-trained and Fine-tuned Models: Facilitate replication of the study and further experimentation.

## How to Contribute

We encourage contributions from the community, whether it's improving the code, expanding the dataset, or refining the documentation. 

## Support

Need help? You can reach out directly via the Issues tab of the Github page for support from the OpenChessRobot team and community.

## License

This project is released under a [MIT License](#license). Feel free to use, modify, and distribute any of the contents of this repository in accordance with the license specifications.

We hope the OpenChessRobot serves as a valuable tool for your research and development in human-robot interaction. Happy experimenting!

## Copyright Notice:
Technische Universiteit Delft hereby disclaims all copyright interest in the program “OpenChessBot: An Open-Source Reproducible Chess Robot for Human-Robot Interaction Research” written by the Author(s). 

© 2024, [Renchi Zhang], [OpenChessBot: An Open-Source Reproducible Chess Robot for Human-Robot Interaction Research]
