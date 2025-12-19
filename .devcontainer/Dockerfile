FROM osrf/ros:jazzy-desktop-full

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000

# Change existing user ID to 1001
RUN usermod -u 1001 ubuntu && \
    groupmod -g 1001 ubuntu
# Create a non-root user
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    mkdir /home/${USERNAME}/.config && \
    chown ${USER_UID}:${USER_GID} /home/${USERNAME}/.config && \
# Install sudo
    apt-get update && \
    apt-get install -y sudo && \
    echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}
# Install packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \ 
    nano \
    htop \
    python3-pip \
    python3-colcon-argcomplete \
    build-essential \
    cmake \ 
    git

WORKDIR /home/ros/ws
ENV SHELL=/bin/bash

# Colored invite in bash
RUN echo 'PS1="\[\e[32m\]\u@\h:\[\e[34m\]\w\[\e[0m\]\$ "' >> /home/${USERNAME}/.bashrc && \
# ROS and colcon autocomplete
    echo "source /opt/ros/jazzy/setup.bash" >> /home/${USERNAME}/.bashrc && \
    echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> /home/${USERNAME}/.bashrc && \
# Useful aliases
    echo "alias ssb='source install/setup.bash'" >> /home/${USERNAME}/.bashrc && \
    echo "alias cb='colcon build --symlink-install'" >> /home/${USERNAME}/.bashrc && \
    echo "alias cbp='colcon build --symlink-install --packages-select'" >> /home/${USERNAME}/.bashrc

#############################
# Any additions place here: #
#############################

# Установка Gazebo Harmonic интеграции и необходимых пакетов
RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-gz-ros2-control-demos \
    ros-jazzy-ur-description \
    ros-jazzy-xacro \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-controller-manager \
    ros-jazzy-ros2-controllers \
    ros-jazzy-ur-controllers
RUN apt-get install -y \
    ros-jazzy-ur-simulation-gz \
    ros-jazzy-ur-moveit-config \
    liburdfdom-tools \
    ros-jazzy-ros2-control

#############################

RUN rm -rf /var/lib/apt/lists/*
