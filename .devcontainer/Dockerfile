# Use x86_64 architecture (required for CARLA client on M1 Mac)
# CARLA 0.9.16 requires Python 3.10, 3.11, or 3.12
FROM --platform=linux/amd64 python:3.10-bullseye

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# CARLA version
ARG CARLA_VERSION=0.9.16

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    # Graphics and display (for OpenCV and visualization)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # X11 for display forwarding (optional)
    x11-apps \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    # Networking tools
    net-tools \
    iputils-ping \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CARLA dependencies (libjpeg8)
RUN wget http://security.ubuntu.com/ubuntu/pool/main/libj/libjpeg-turbo/libjpeg-turbo8_2.0.3-0ubuntu1_amd64.deb \
    && wget http://security.ubuntu.com/ubuntu/pool/main/libj/libjpeg8-empty/libjpeg8_8c-2ubuntu8_amd64.deb \
    && dpkg -i libjpeg-turbo8_2.0.3-0ubuntu1_amd64.deb \
    && dpkg -i libjpeg8_8c-2ubuntu8_amd64.deb \
    && rm *.deb

# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install Python packages for CARLA client
RUN pip install --no-cache-dir \
    numpy \
    opencv-python \
    opencv-contrib-python \
    pillow \
    matplotlib \
    scipy \
    pyyaml \
    tqdm

# Install PyTorch (CPU version for M1 compatibility through emulation)
# RUN pip install --no-cache-dir \
#     torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install CARLA Python package via pip (works on x86_64 Linux)
RUN pip install --no-cache-dir carla==${CARLA_VERSION}

# Set working directory
WORKDIR /workspace

# Switch back to dialog for any ad-hoc use
ENV DEBIAN_FRONTEND=dialog

# Switch to non-root user
USER $USERNAME

# Verify CARLA installation
RUN python -c "import carla; print(f'✅ CARLA Python client installed successfully')" || \
    echo "⚠️  CARLA import test failed - will retry after dependencies installation"

CMD ["/bin/bash"]
