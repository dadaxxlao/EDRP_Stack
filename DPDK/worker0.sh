#chmod u+x dpdk-boot.sh
#sudo ./dpdk-boot.sh

sudo echo 1 > /sys/module/vfio/parameters/enable_unsafe_noiommu_mode

sudo ifconfig ens192 down

cd /home/ddxxlao/dpdk-stable-22.11.4
sudo ./usertools/dpdk-devbind.py -b vfio-pci 0000:0b:00.0
sudo ./usertools/dpdk-hugepages.py -p 2M --setup 4G

./usertools/dpdk-devbind.py -s
./usertools/dpdk-hugepages.py -s