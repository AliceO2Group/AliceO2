#! /bin/bash
for i in `seq 0 \`lspci | grep -i nvidia | grep "VGA\|3D" | wc -l\``; do
mknod -m 666 /dev/nvidia$i c 195 $i
chown root:video /dev/nvidia$i
done
/bin/mknod -m 666 /dev/nvidiactl c 195 255
