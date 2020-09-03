#ifndef HFNET_TENSORFLOW_NET_H_
#define HFNET_TENSORFLOW_NET_H_

#include <vector>
#include <memory>
#include <iostream>


#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>

#include <tensorflow/c/c_api.h>

#include <opencv2/opencv.hpp>

#include <Eigen/Core>


using tensorflow::Status;
using tensorflow::Tensor;
using namespace tensorflow;

class TensorflowNet {
  public:
    typedef Eigen::Matrix<float, Eigen::Dynamic, 1> GlobalDescType;
    typedef Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor> KeyPointType;
    typedef Eigen::Matrix<float, Eigen::Dynamic, 256, Eigen::RowMajor> LocalDescType;

    TensorflowNet(const std::string model_path ,
                  const unsigned int topk_num = 2000,
                  const unsigned int nms_radius = 4):
                topk_kpts_(topk_num), nms_radius_(nms_radius) {
        CHECK(tensorflow::MaybeSavedModelDirectory(model_path));
        input_channels_ = 1;
        descriptor_size_ = 4096;
        std::string resample_path = "/usr/local/lib/_resampler_ops.so";
        TF_Status* tf_status = TF_NewStatus();
        

        TF_LoadLibrary(resample_path.c_str(),tf_status);

        std::cout << "TF_GetCode(tf_status): " << TF_GetCode(tf_status) << std::endl;

        LOG(INFO) << "HERE";

        if (TF_GetCode(tf_status) !=TF_OK) {
            std::cerr << "TF_LoadLibrary  _resampler_ops.so ERROR, Load resampler.so failed, Please check.\n";
        }

        // Load model
        tensorflow::SessionOptions session_options;
        // tensorflow::graph::SetDefaultDevice("/gpu:0", &graph_def);
        session_options.config.mutable_gpu_options()->set_allow_growth(true);
        session_options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
        Status status = tensorflow::LoadSavedModel(
                session_options, tensorflow::RunOptions(),
                model_path, {tensorflow::kSavedModelTagServe}, &bundle_);
        if (!status.ok())
            LOG(FATAL) << status.ToString();

        // Check input and output shapes
        tensorflow::GraphDef graph_def = bundle_.meta_graph_def.graph_def();
        bool found_input = false, found_output = false;
        for (auto& node : graph_def.node()) {
                // check graph here
            if(node.name() == "global_descriptor")
                std::cout << "global_desc: " << node.name() << std::endl;
            // std::cout << node.name() << std::endl;
        }

        std::cout << "load suscessfully" << std::endl;
    }

    void PerformInference(const cv::Mat& image, int &kpts_num, GlobalDescType* global_desc, KeyPointType* key_points, LocalDescType* local_desc) {
        CHECK(image.data);
        CHECK(image.isContinuous());

        unsigned height = image.size().height, width = image.size().width;
        // TODO: cleaner way to avoid copying when the image is already float
        // or combine with tensor creation
        cv::Mat *float_image_ptr, tmp;
        if(image.type() != CV_32F) {
            image.convertTo(tmp, CV_32F);
            float_image_ptr = &tmp;
        } else {
            float_image_ptr = &const_cast<cv::Mat&>(image);
        }

        // Prepare input tensor
        Tensor input_tensor(
                tensorflow::DT_FLOAT,
                tensorflow::TensorShape({1, height, width, input_channels_}));
        Tensor keypoints_num(tensorflow::DT_INT32,TensorShape());
        Tensor radius(tensorflow::DT_INT32,TensorShape());
        keypoints_num.scalar<int>()() = 2000;
        radius.scalar<int>()() = 4;
        // TODO: avoid copy if possible
        tensorflow::StringPiece tmp_data = input_tensor.tensor_data();
        std::memcpy(const_cast<char*>(tmp_data.data()), float_image_ptr->data,
                    height * width * input_channels_ * sizeof(float));

 
        // Run inference here average 15-20ms in 1660 
        std::vector<Tensor> outputs;
        Status status = bundle_.session->Run({{"image:0", input_tensor},{"pred/simple_nms/radius", radius},{"pred/top_k_keypoints/k",keypoints_num}},
                                             {"global_descriptor", "keypoints", "local_descriptors"}, {}, &outputs);
        
        if (!status.ok())
            LOG(FATAL) << status.ToString();
        // std::cout << "inference done" << std::endl;
        // Copy result
        float *descriptor_ptr = outputs[0].flat<float>().data();
        Eigen::Map<GlobalDescType> descriptor_map(descriptor_ptr, descriptor_size_);
        Eigen::Matrix<float, Eigen::Dynamic, 1> desc =
                        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>>(descriptor_ptr, 4096, 1);
        int kp_num = outputs[1].shape().dim_size(1);
        kpts_num = kp_num;
        std::cout << "detected: " << kp_num << " pts" << std::endl;
        *global_desc = descriptor_map;  // Copy
        int *kpts_ptr = outputs[1].flat<int>().data();
        Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> kpts_int =
                        Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>>(kpts_ptr, kp_num, 2);
        Eigen::MatrixXf kpts = kpts_int.cast<float>();
        *key_points = kpts;  // Copy


        float *local_desc_ptr = outputs[2].flat<float>().data();
        static const int dim = 256;
        Eigen::Matrix<float, Eigen::Dynamic, dim, Eigen::RowMajor> kpts_desc =
                        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, dim, Eigen::RowMajor>>(local_desc_ptr, kp_num, dim);
        *local_desc = kpts_desc;  // Copy
    }

    unsigned descriptor_size() {
        return descriptor_size_;
    }

  private:
    tensorflow::SavedModelBundle bundle_;
    unsigned descriptor_size_;
    unsigned input_channels_;
    unsigned int topk_kpts_;
    unsigned int nms_radius_;
};

#endif  // HFNET_TENSORFLOW_NET_H_

