#ifndef TENSORRT_INFERENCE_YOLOV5_CU_H
#define TENSORRT_INFERENCE_YOLOV5_CU_H

#include "YOLO.h"
#include <string>
#include <fstream>
#include <iostream>
#include <cassert>
#include <cuda_runtime_api.h>  // 添加这个用于 CUDA 函数
#include "postprocess.h"
#include "preprocess.h"
struct Yolov5Box{
    float x1,y1,x2,y2;
     float landmarks[8];  //关键点4个
     float score;
     int label;
};

struct Yolov5Res {
    std::vector<Yolov5Box> yolov5_results;
};

struct affine_matrix
{
    float i2d[6];
    float d2i[6];
};

class yolov5_cu : public Model {
public:
    explicit yolov5_cu(const YAML::Node &config);
    void LoadEngine();
    std::vector<Yolov5Res> InferenceImages(std::vector<cv::Mat> &vec_img);
    void InferenceFolder(const std::string &folder_name);
    void DrawResults(const std::vector<Yolov5Res> &detections, std::vector<cv::Mat> &vec_img,
                 std::vector<std::string> &image_name);
    ~yolov5_cu();
private:
     float * prob=nullptr;  //trt输出 
     int output_size ;   //trt输出大小 
     int output_candidates; //多少行 640输入是 25200

     const char* input_blob_name = "input"; //onnx 输入  名字
     const char* output_blob_name = "output"; //onnx 输出 名字
    float prob_threshold;
    float nms_threshold;
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    float *affine_matrix_host=nullptr;
    float *affine_matrix_device=nullptr;
    float *decode_ptr_host=nullptr;
    float *decode_ptr_device=nullptr;      
};

#endif // TENSORRT_INFERENCE_YOLOV5_CU_H
