FROM ros:jazzy AS base

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git git-lfs \
    python3-dev \
    python3-tk \
    libgl1 libglfw3 libglfw3-dev \
    libosmesa6-dev \
    libglew-dev \
    mesa-utils \
    x11-apps \
    nano vim screen tmux \
    sudo \
&& rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash sailor && \
    usermod -aG sudo sailor && \
    echo "sailor ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
WORKDIR /home/sailor
USER sailor
ENV HOME=/home/sailor

COPY --chown=sailor:sailor . sailboat_simulator
WORKDIR /home/sailor/sailboat_simulator

# Remove local venv folders (not needed in container, causes colcon issues)
RUN rm -rf venv .venv */venv */.venv

# Do not use a virtual environment because for some reason ROS always uses the system python packages
# Install PyTorch CPU-only first to avoid pulling CUDA dependencies (saves several GBs)
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages
RUN pip3 install -r requirements.txt --break-system-packages

RUN rosdep update -q
USER root
RUN rosdep install --from-paths src --ignore-src -r -y
USER sailor
WORKDIR /home/sailor/sailboat_simulator

# Build ROS workspace (must use bash for ROS setup script)
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/jazzy/setup.bash && colcon build

# Source ROS on every bash session
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \
    echo "source /home/sailor/sailboat_simulator/install/setup.bash" >> ~/.bashrc

# Headless rendering
ENV MUJOCO_GL=osmesa

CMD ["/bin/bash"]
