## **A Cpp Version of HF-NET**

![demo](data/hfnet-1660s-demo.gif)

## **9ms in GeForce GTX 2080 Ti**

## **15-20ms in GeForce GTX 1660s 6G**

Since the origin HF-NET based on Python3, May be it is a little difficult integrate the code with your Cpp projects(some SLAM system). To run the `hfnet_ros` package, you need install the tensorflow from the source. For convience, you can using the docker image here to run this package.

This package allows to:

* Extract feature detectors and descriptors on standard image sequence
* Integrate other slam system which written in cpp.
* Combine the origin [hloc-cpp](https://github.com/ethz-asl/hfnet/tree/master/hloc-cpp) back-end and make a full cpp projects.
* It takes 15-20ms in GeForce GTX 1660s 6G.

### Setup

* python 2.7
* Ubuntu 16.04
* ROS Kinetic
* CUDA = 9.0
* Cudnn = 7.0
* Tensorflow cpp gpu == 1.12 (you need build and install tf1.12 from source, and copy the libs to /usr/local/lib)
* opencv > 3.0

### Docker

Build tensorflow from source is annoying, So if you wanna test the code , you can using docker image.

```bash
sudo docker pull xinliangzhong/1604-tf1.2cc-cuda9.0-cudnn7.0-ros:latest
```

```bash
sudo docker run --gpus all --name="hfnet-ros" --net=host -it -v /YOUR/PATH:/root/data xinliangzhong/1604-tf1.2cc-cuda9.0-cudnn7.0-ros:latest /bin/bash
```



### How to run


```bash
cd /root/data
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone https://github.com/TurtleZhong/hfnet_ros.git

# fetch model
cd hfnet_ros/model/
chmod +x download.sh
./download.sh

catkin_make -DCMAKE_BUILD_TYPE=Release
```

```bash
cd /root/data/catkin_ws/
source devel/setup.bash
roslaunch hfnet_ros hfnet_ros_node.launch
```
Then you will get this in terminal.
```bash
...
detected: 817 pts
IMAGE[480 x 752] cost: 18 ms
...
```

The results image is publish as ros msg. if your start docker container using `--net=host`, you can using rqt to view the results.
