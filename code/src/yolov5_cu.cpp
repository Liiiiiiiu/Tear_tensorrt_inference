#include "yolov5_cu.h"
#include <fstream>
#include <cassert>
#include <iostream>
#include <chrono>

#define MAX_IMAGE_INPUT_SIZE_THRESH 1920 * 1080  // 图片实际输入大小；根据实际修改
#define MAX_OBJECTS 100  // 一次处理的最多目标数；根据实际修改
#define NUM_BOX_ELEMENT 15  // 5 + 8 + NUM_CLASSES
#define NUM_CLASSES 3
#define CHECK(status) \
    do \
    { \
        auto ret = (status); \
        if (ret != 0) \
        { \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort(); \
        } \
    } while (0)

static std::vector<std::string> yolov5_class_labels{"body", "car", "plate"};

// 定义计时器辅助类
class Timer {
public:
    Timer(const std::string &func_name) : func_name(func_name) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << func_name << " executed in " << duration << " ms" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::string func_name;
};

void get_d2i_matrix(affine_matrix &afmt, cv::Size to, cv::Size from)
{
    Timer timer("get_d2i_matrix");  // 计时

    float scale = std::min(to.width / float(from.width), to.height / float(from.height));
    afmt.i2d[0] = scale;
    afmt.i2d[1] = 0;
    afmt.i2d[2] = -scale * from.width * 0.5 + to.width * 0.5;
    afmt.i2d[3] = 0;
    afmt.i2d[4] = scale;
    afmt.i2d[5] = -scale * from.height * 0.5 + to.height * 0.5;

    cv::Mat mat_i2d(2, 3, CV_32F, afmt.i2d);
    cv::Mat mat_d2i(2, 3, CV_32F, afmt.d2i);
    cv::invertAffineTransform(mat_i2d, mat_d2i);
    memcpy(afmt.d2i, mat_d2i.ptr<float>(0), sizeof(afmt.d2i));
}

yolov5_cu::yolov5_cu(const YAML::Node &config) : Model(config)
{
    Timer timer("yolov5_cu::yolov5_cu");  // 计时

    prob_threshold = config["obj_threshold"].as<float>();
    nms_threshold = config["nms_threshold"].as<float>();
}

yolov5_cu::~yolov5_cu()
{
    Timer timer("yolov5_cu::~yolov5_cu");  // 计时

    if (prob)
        delete[] prob;
    if (decode_ptr_host)
        delete[] decode_ptr_host;
    CHECK(cudaFree(img_device));
    CHECK(cudaFreeHost(img_host));
    CHECK(cudaFree(affine_matrix_device));
    CHECK(cudaFreeHost(affine_matrix_host));
    CHECK(cudaFree(decode_ptr_device));
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
}

// 在这里实现 loadTrtModel 函数
void yolov5_cu::LoadEngine()
{
    Timer timer("yolov5_cu::LoadEngine");  // 计时

    char *trtModelStream{nullptr};
    size_t size{0};
    nvinfer1::IRuntime *trtRuntime;
    const std::string engine_file_path = engine_file;
    std::ifstream file(engine_file_path, std::ios::binary);

    // 检查文件是否有效
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];  // 创建存储引擎数据的缓冲区
        assert(trtModelStream);           // 确保内存分配成功
        file.read(trtModelStream, size);  // 读取文件到缓冲区
        file.close();
    }
    else
    {
        std::cerr << "Error: Engine file not found!" << std::endl;
        return;
    }

    // 创建 TensorRT 运行时并反序列化引擎
    trtRuntime = nvinfer1::createInferRuntime(gLogger);
    assert(trtRuntime != nullptr);
    engine = trtRuntime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);

    // 创建执行上下文
    context = engine->createExecutionContext();
    assert(context != nullptr);

    // 获取输出维度
    auto out_dims = engine->getBindingDimensions(1);
    output_size = 1;
    output_candidates = out_dims.d[1];

    for (int j = 0; j < out_dims.nbDims; j++)
    {
        output_size *= out_dims.d[j];  // 计算输出的总大小
    }

    // 获取输入输出绑定索引
    const int inputIndex = engine->getBindingIndex(input_blob_name);
    const int outputIndex = engine->getBindingIndex(output_blob_name);
    assert(inputIndex == 0);
    assert(outputIndex == 1);

    CHECK(cudaMalloc((void **)&buffers[inputIndex], 3 * IMAGE_HEIGHT * IMAGE_WIDTH * sizeof(float)));  // trt输入内存申请
    CHECK(cudaMalloc((void **)&buffers[outputIndex], output_size * sizeof(float)));                  // trt输出内存申请
    CHECK(cudaStreamCreate(&stream));

    decode_ptr_host = new float[1 + MAX_OBJECTS * NUM_BOX_ELEMENT];
    // prepare input data cache in pinned memory
    CHECK(cudaMallocHost((void **)&affine_matrix_host, sizeof(float) * 6));
    CHECK(cudaMallocHost((void **)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CHECK(cudaMalloc((void **)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    CHECK(cudaMalloc((void **)&affine_matrix_device, sizeof(float) * 6));
    CHECK(cudaMalloc((void **)&decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT)));

    std::cout << "Load TRT engine success!" << std::endl;

    delete[] trtModelStream;  // 释放缓冲区
}

std::vector<Yolov5Res> yolov5_cu::InferenceImages(std::vector<cv::Mat> &vec_img)
{
    Timer timer("yolov5_cu::InferenceImages");  // 计时

    std::vector<Yolov5Res> yolov5_results_vec;
    cv::Mat img = vec_img[0];
    Yolov5Res yolov5_res;
    affine_matrix afmt;
    const int inputIndex = engine->getBindingIndex(input_blob_name);
    const int outputIndex = engine->getBindingIndex(output_blob_name);

    get_d2i_matrix(afmt, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), cv::Size(img.cols, img.rows));

    double begin_time = cv::getTickCount();
    float *buffer_idx = (float *)buffers[inputIndex];
    size_t size_image = img.cols * img.rows * 3;
    size_t size_image_dst = IMAGE_HEIGHT * IMAGE_WIDTH * 3;
    memcpy(img_host, img.data, size_image);
    memcpy(affine_matrix_host, afmt.d2i, sizeof(afmt.d2i));
    CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream));

    preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, IMAGE_WIDTH, IMAGE_HEIGHT, affine_matrix_device, stream); // cuda 前处理
    double time_pre = cv::getTickCount();
    double time_pre_ = (time_pre - begin_time) / cv::getTickFrequency() * 1000;
    std::cout << "preprocessing time is " << time_pre_ << " ms" << std::endl;

    double infer_begin_time = cv::getTickCount();
    context->enqueueV2((void **)buffers, stream, nullptr);
    CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(int), stream));
    double infer_end_time = cv::getTickCount();
    double time_infer = (infer_end_time - infer_begin_time) / cv::getTickFrequency() * 1000;
    std::cout << "infer time is " << time_infer << " ms" << std::endl;

    double post_begin_time = cv::getTickCount();
    float *predict = (float *)buffers[outputIndex];
    double post_decode_begin_time = cv::getTickCount();
    decode_kernel_invoker(predict, output_candidates, NUM_CLASSES, 4, prob_threshold, affine_matrix_device, decode_ptr_device, MAX_OBJECTS, stream); // cuda 后处理
    double post_decode_end_time = cv::getTickCount();
    double time_decode = (post_decode_end_time - post_decode_begin_time) / cv::getTickFrequency() * 1000;
    std::cout << "post_decode time is " << time_decode << " ms" << std::endl;

    double post_nms_begin_time = cv::getTickCount();
    nms_kernel_invoker(decode_ptr_device, nms_threshold, MAX_OBJECTS, stream); // cuda nms
    double post_nms_end_time = cv::getTickCount();
    double time_nms = (post_nms_end_time - post_nms_begin_time) / cv::getTickFrequency() * 1000;
    std::cout << "post_nms time is " << time_nms << " ms" << std::endl;

    double post_trans_begin_time = cv::getTickCount();
    CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    double post_trans_end_time = cv::getTickCount();
    double time_trans = (post_trans_end_time - post_trans_begin_time) / cv::getTickFrequency() * 1000;
    std::cout << "trans_nms time is " << time_trans << " ms" << std::endl;


    int boxes_count = 0;
    int count = std::min((int)*decode_ptr_host, MAX_OBJECTS);
    double post_for_begin_time = cv::getTickCount();
    for (int i = 0; i < count; i++)
    {
        int basic_pos = 1 + i * NUM_BOX_ELEMENT;
        int keep_flag = decode_ptr_host[basic_pos + 6];
        if (keep_flag == 1)
        {
            boxes_count += 1;
            Yolov5Box box;
            box.x1 = decode_ptr_host[basic_pos + 0];
            box.y1 = decode_ptr_host[basic_pos + 1];
            box.x2 = decode_ptr_host[basic_pos + 2];
            box.y2 = decode_ptr_host[basic_pos + 3];
            box.score = decode_ptr_host[basic_pos + 4];
            box.label = decode_ptr_host[basic_pos + 5];
            int landmark_pos = basic_pos + 7;
            for (int id = 0; id < 4; id++)
            {
                box.landmarks[2 * id] = decode_ptr_host[landmark_pos + 2 * id];
                box.landmarks[2 * id + 1] = decode_ptr_host[landmark_pos + 2 * id + 1];
            }
            yolov5_res.yolov5_results.push_back(box);
        }
    }
    double post_for_end_time = cv::getTickCount();
    double time_for = (post_for_end_time - post_for_begin_time) / cv::getTickFrequency() * 1000;
    std::cout << "for_nms time is " << time_for << " ms" << std::endl;

    yolov5_results_vec.emplace_back(yolov5_res);
    double post_end_time = cv::getTickCount();
    double post_infer = (post_end_time - post_begin_time) / cv::getTickFrequency() * 1000;
    std::cout << "post time is " << post_infer << " ms" << std::endl;
    return yolov5_results_vec;
}

void yolov5_cu::InferenceFolder(const std::string &folder_name)
{
    std::vector<std::string> image_list = ReadFolder(folder_name);
    int index = 0;
    int batch_id = 0;
    int batch_size = 1;
    std::vector<cv::Mat> vec_Mat(batch_size);
    std::vector<std::string> vec_name(batch_size);
    float total_time = 0;
    for (const std::string &image_name : image_list) {
        index++;
        std::cout << "Processing: " << image_name << std::endl;
        cv::Mat src_img = cv::imread(image_name);
        if (src_img.data) {
            vec_Mat[batch_id] = src_img.clone();
            vec_name[batch_id] = image_name;
            batch_id++;
        }
        if (batch_id == batch_size or index == image_list.size()) {
            auto start_time = std::chrono::high_resolution_clock::now();
            auto det_results = this->InferenceImages(vec_Mat);
            auto end_time = std::chrono::high_resolution_clock::now();
            this->DrawResults(det_results, vec_Mat, vec_name);
            vec_Mat = std::vector<cv::Mat>(batch_size);
            batch_id = 0;
            total_time += std::chrono::duration<float, std::milli>(end_time - start_time).count();
        }
    }
    std::cout << "Average processing time is " << total_time / image_list.size() << "ms" << std::endl;
}  
void yolov5_cu::DrawResults(const std::vector<Yolov5Res> &detections, std::vector<cv::Mat> &vec_img,
                            std::vector<std::string> &image_name) {
    std::vector<cv::Scalar> class_colors(3);
    srand((int) time(nullptr));
    for (cv::Scalar &class_color : class_colors)
        class_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);

    for (int i = 0; i < (int)vec_img.size(); i++) {
        auto org_img = vec_img[i];
        if (!org_img.data)
            continue;

        auto rects = detections[i].yolov5_results;
        for (const auto &rect : rects) {
            char t[256];
            sprintf(t, "%.2f", rect.score);
            std::string name = yolov5_class_labels[rect.label] + "-" + t;
            cv::putText(org_img, name, cv::Point(rect.x1, rect.y1 - 5),
                        cv::FONT_HERSHEY_COMPLEX, 0.7, class_colors[rect.label], 2);
            cv::Rect rst(cv::Point(rect.x1, rect.y1), cv::Point(rect.x2, rect.y2));
            cv::rectangle(org_img, rst, class_colors[rect.label], 2, cv::LINE_8, 0);
        }

        if (!image_name.empty()) {
            int pos = image_name[i].find_last_of('.');
            std::string rst_name = image_name[i].insert(pos, "_");
            std::cout << rst_name << std::endl;
            cv::imwrite(rst_name, org_img);
        }
    }
}
