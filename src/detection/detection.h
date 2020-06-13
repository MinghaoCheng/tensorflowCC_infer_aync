#pragma once

#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <thread>
#include <functional>
#include <queue>

#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/lib/core/errors.h>
#include <tensorflow/core/lib/core/stringpiece.h>
#include <tensorflow/core/lib/core/threadpool.h>
#include <tensorflow/core/lib/io/path.h>
#include <tensorflow/core/lib/strings/str_util.h>
#include <tensorflow/core/lib/strings/stringprintf.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/platform/logging.h>
#include <tensorflow/core/platform/types.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/util/command_line_flags.h>


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "cuda.h"
#include "cuda_runtime_api.h"

enum {
    BUFFER_ID_A = 0,
    BUFFER_ID_B,
    BUFFER_ID_C
    };

// These are all common classes it's handy to reference with no namespace.
using namespace tensorflow;

class Detector
{
private:
    uint8_t batchSize;
    uint32_t input_width;
    uint32_t input_height;
    uint8_t input_depth;

    const string input_layer = "image_arrays:0";
    const string output_layer = "detections:0";
    std::unique_ptr<Session> sess;

    tensorflow::Tensor *BufferA, *BufferB, *BufferC;
    std::vector<tensorflow::Tensor> OutputTensor;
    bool BufferA_busy;
    bool BufferB_busy;
    bool BufferC_busy;

    std::queue<std::vector<cv::Mat>> InFIFO;
    std::queue<tensorflow::Tensor> OutFIFO;
    std::queue<uint16_t> InferQueue;
    void AllocBuffer(void);
    void feed_data(uint8_t BUFFER_ID, std::vector<cv::Mat> imgArray);

    std::function<void(void*)> InferDoneCB;
    void *Instance_ptr;

    std::thread* infer_thread;
    std::thread* feed_thread;
    std::thread* fetch_thread;
    void infer_thread_function(void);
    void feed_thread_function(void);
    void fetch_thread_function(void);
    std::condition_variable cond_infer;
    std::condition_variable cond_feed;
    std::condition_variable cond_fetch;

    std::mutex mutex_infer;
    std::mutex mutex_feed;
    std::mutex mutex_fetch;

    cudaStream_t CUDAstreamA;
    cudaStream_t CUDAstreamB;
    cudaStream_t CUDAstreamC;

    bool ShouldClose;
    bool IsInferReady;
    bool IsFeedReady;
    bool IsFetchReady;

    uint16_t FIFO_capacity = 100;
public:
    Detector(uint8_t BatchSize, uint32_t Input_width, uint32_t Input_height, uint8_t Input_depth, std::string graph_path, std::function<void(void*)> InferDoneCB, void *instance_ptr);
    ~Detector();
    void Terminate(void);
    // static void create_tensor_from_CVimage(std::vector<cv::Mat> imgArray, Tensor &output);
    
    bool Infer(std::vector<cv::Mat> imgArray);
    bool GetOutput(tensorflow::Tensor &TensorOut);
    bool IsOutFIFOEmpty(void);
    bool IsReady(void);
};