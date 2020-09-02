#include <iostream>
#include <chrono>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include "hfnet_ros/tensorflow_net.h"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace tensorflow;

static sensor_msgs::ImageConstPtr img0_cur;

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg) {
  img0_cur = img_msg;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "hfnet_ros_node");
  ros::NodeHandle n("~");
  image_transport::ImageTransport it(n);
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                 ros::console::levels::Info);
  std::string model_dir = "/root/data/hfnet/hfnet_tf/saved_models/hfnet";
  std::string test_image = "";
  std::string input_img_topic = "/cam0";
  int topk_num = 2000;
  int nms_radius = 4;
  bool flag_show = false;
  
  n.param<std::string>("model_dir", model_dir, model_dir);
  n.param<std::string>("test_image", test_image, test_image);
  n.param<std::string>("input_img_topic", input_img_topic, input_img_topic);
  n.param<int>("topk_kpts_num", topk_num, topk_num);
  n.param<int>("nms_radius", nms_radius, nms_radius);
  n.param<bool>("opencv_show_image", flag_show, flag_show);
  

  ros::Subscriber sub_img0 =
      n.subscribe(input_img_topic, 1, img0_callback);

  image_transport::Publisher pub_image = 
      it.advertise("/debug_pts", 10);

  TensorflowNet network(model_dir, topk_num, nms_radius);
  // read params here
  TensorflowNet::GlobalDescType global_desc;
  TensorflowNet::KeyPointType kpts;
  TensorflowNet::LocalDescType local_desc;
  int kpts_num = -1;
  cv::Mat image = cv::imread(test_image, CV_LOAD_IMAGE_GRAYSCALE);

  auto start = chrono::system_clock::now();
  network.PerformInference(image, kpts_num, &global_desc, &kpts, &local_desc);
  auto end = chrono::system_clock::now();
  std::cout << "cost: " << chrono::duration_cast<chrono::milliseconds>(end-start).count() << " ms" << std::endl;
  ROS_WARN_STREAM("Waiting " << input_img_topic << " image...");



  ros::Rate loop_rate(30);
  while (ros::ok()) {
    ros::spinOnce();  // do callback
    if (img0_cur) {
      // run hfnet-cpp
      cv_bridge::CvImageConstPtr srcImage = cv_bridge::toCvShare(img0_cur, sensor_msgs::image_encodings::MONO8);

      auto start = chrono::system_clock::now();
      network.PerformInference(srcImage->image, kpts_num, &global_desc, &kpts, &local_desc);
      auto end = chrono::system_clock::now();
      std::cout << "IMAGE[" << srcImage->image.size << "] cost: " << chrono::duration_cast<chrono::milliseconds>(end-start).count() << " ms" << std::endl;

      cv::Mat image_copy = srcImage->image.clone();
      cv::cvtColor(image_copy, image_copy, CV_GRAY2BGR);
      for (size_t i = 0; i < kpts_num; i++)
      {
        cv::circle(image_copy, cv::Point(kpts.row(i)(0), kpts.row(i)(1)), 2, cv::Scalar(0,255,0), -1);
      }
      
      std::string tmp = "total kpts: " + std::to_string(kpts_num) + ", cost: " + std::to_string(chrono::duration_cast<chrono::milliseconds>(end-start).count()) + " ms";
      cv::putText(image_copy, tmp, cv::Point(20,20),CV_FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255),2);
      sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image_copy).toImageMsg();
      pub_image.publish(msg);
      img0_cur.reset();
    }
    loop_rate.sleep();
}
return 0;
}
