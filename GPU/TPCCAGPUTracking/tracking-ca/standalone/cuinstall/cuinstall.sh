#! /bin/bash
modprobe -r nvidia
./NVIDIA-Linux-x86_64-256.35.run -a -q --ui=none
rm -Rf /opt/cuda
mkdir /opt/cuda
./cudatoolkit_3.1_linux_64_ubuntu9.10.run -- --prefix=/opt/cuda
cp mkcudadevnodes.sh /opt/cuda
cp 86-nvidia.rules /etc/udev/rules.d
modprobe nvidia
