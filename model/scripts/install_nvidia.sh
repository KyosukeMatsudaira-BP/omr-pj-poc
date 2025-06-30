sudo apt update
sudo apt install -y linux-headers-$(uname -r)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-driver-535
sudo apt install -y nvidia-container-toolkit
sudo apt install -y nvidia-container-runtime


