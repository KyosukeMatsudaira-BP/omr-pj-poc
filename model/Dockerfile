FROM ubuntu:20.04

ENV SHELL /bin/bash
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8
# 環境変数を設定して、インタラクティブプロンプトを回避します。
ARG DEBIAN_FRONTEND=noninteractive

# 参考 https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
# dockerコンテナ立ち上げ時にwslと同じIDのuserを作成する
# macで同様にうまく動作するかは不明
ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -l --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# 必要な依存関係をインストール
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
    wget \
    curl \
    gnupg \
    build-essential \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    git

# debconfディレクトリの権限を修正、ダウンロード時のインタラクティブプロンプトを飛ばすための設定
RUN sudo chmod 777 /var/cache/debconf /var/cache/debconf/*
RUN sudo echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# CUDA11.6のインストール
RUN sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN sudo wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb
RUN sudo dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb
RUN sudo apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
RUN sudo apt-get update
RUN sudo apt-get -y install cuda
RUN sudo rm cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb

# gpuの環境変数
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=utility,compute

# Python 3.10をソースからインストール
RUN sudo wget --no-check-certificate https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz && \
    sudo tar -xf Python-3.10.12.tgz && \
    cd Python-3.10.12 && \
    sudo ./configure --enable-optimizations && \
    sudo make -j $(nproc) && \
    sudo make altinstall && \
    cd .. && \
    sudo rm -rf Python-3.10.12 Python-3.10.12.tgz

# install Poetry
RUN sudo curl -sSL https://install.python-poetry.org | python3.10 -
# Poetryのパスの設定
ENV PATH /home/docker/.local/bin:$PATH
# Poetryが仮想環境を生成しないようにする
RUN sudo /home/docker/.local/bin/poetry config virtualenvs.create false

# install PlantUML (java, graphvis and Japanese fonts)
RUN sudo apt-get update && sudo apt-get install -y default-jdk \
    && sudo apt-get install -y graphviz \
    && sudo apt-get install -y fonts-ipafont fonts-ipaexfont

# install gcloud
RUN sudo curl https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-470.0.0-linux-x86_64.tar.gz > /tmp/google-cloud-sdk.tar.gz \
    && sudo mkdir -p /usr/local/gcloud \
    && sudo tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
    && sudo /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Clean up
RUN sudo apt-get autoremove -y \
    && sudo apt-get clean -y \
    && sudo rm -rf /var/lib/apt/lists/*

# Set PYTHONPATH
ENV PYTHONPATH="/workspace/repo"

WORKDIR /workspace/repo

# poetryで依存関係をインストールするためにpyproject.tomlとpoetry.lockをコピー
COPY pyproject.toml poetry.lock* /workspace/repo/

# Poetryで依存関係をインストール
RUN poetry install --no-root
