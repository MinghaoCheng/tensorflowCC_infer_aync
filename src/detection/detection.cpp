#include "detection.h"

// GPU allocator
#include <tensorflow/core/common_runtime/gpu/gpu_id.h>
#include <tensorflow/core/common_runtime/gpu/gpu_id_utils.h>
#include <tensorflow/core/common_runtime/gpu/gpu_init.h>
#include <tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "cuda.h"
#include "cuda_runtime_api.h"

Status ReadEntireFile(Env *env, const string &filename, tensorflow::Tensor *output)
{
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    string contents;
    contents.resize(file_size);

    std::unique_ptr<RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size)
    {
        return errors::DataLoss("Truncated read of '", filename,"' expected ", file_size, " got ", data.size());
    }
    output->scalar<tstring>()() = tstring(data);
    return Status::OK();
}

Status LoadGraph(const string &graph_file_name, std::unique_ptr<Session> *session)
{
    GraphDef graph_def;
    Status load_graph_status = ReadBinaryProto(Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok())
    {
        return errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
    }
    auto options = SessionOptions();
    options.config.mutable_gpu_options()->set_allow_growth(true);
    tensorflow::ConfigProto &config = options.config;
    config.set_inter_op_parallelism_threads(10);
    config.set_intra_op_parallelism_threads(10);
    config.set_use_per_session_threads(false);  
    session->reset(NewSession(options));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok())
    {
        return session_create_status;
    }
    return Status::OK();
}

Detector::Detector(uint8_t BatchSize, uint32_t Input_width, uint32_t Input_height, uint8_t Input_depth, std::string graph_path, std::function<void(void*)> InferDoneCB, void *instance_ptr)
{
    this->batchSize = BatchSize;
    this->input_width = Input_width;
    this->input_height = Input_height;
    this->input_depth = Input_depth;
    this->BufferA_busy = true;
    this->BufferB_busy = true;
    this->BufferC_busy = true;
    this->InferDoneCB = InferDoneCB;

    this->feed_thread = new std::thread(&Detector::feed_thread_function, this);
    
    this->IsInferReady = false;
    this->IsFeedReady = false;
    this->IsFetchReady = false;

    this->ShouldClose = false;
    Status load_graph_status = LoadGraph(graph_path, &this->sess);
    if (!load_graph_status.ok())
    {
        std::cout << "MESSAGE::dectection.cpp::Loading Graph failed" << "\n";
    }
    this->Instance_ptr = instance_ptr;
    tensorflow::Tensor temp_tensor(DT_UINT8, TensorShape({this->batchSize, this->input_height, this->input_width, this->input_depth}));
    sess->Run({{Detector::input_layer, temp_tensor}}, {this->output_layer}, {}, &this->OutputTensor);
}

Detector::~Detector()
{

}

void Detector::AllocBuffer(void)
{
    TensorShape shape = TensorShape({this->batchSize, this->input_height, this->input_width, this->input_depth});
    PlatformGpuId platform_gpu_id(0);

    GPUMemAllocator *sub_allocatorA =
        new GPUMemAllocator(
            GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
            platform_gpu_id, false /*use_unified_memory*/, {}, {});
    GPUBFCAllocator *allocatorA = new GPUBFCAllocator(sub_allocatorA, shape.num_elements() * sizeof(uint8), "GPU_0_bfc");
    this->BufferA = new Tensor(allocatorA, tensorflow::DT_UINT8, shape);

    GPUMemAllocator *sub_allocatorB =
        new GPUMemAllocator(
            GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
            platform_gpu_id, false /*use_unified_memory*/, {}, {});
    GPUBFCAllocator *allocatorB = new GPUBFCAllocator(sub_allocatorB, shape.num_elements() * sizeof(uint8), "GPU_0_bfc");
    this->BufferB = new Tensor(allocatorB, tensorflow::DT_UINT8, shape);

    GPUMemAllocator *sub_allocatorC =
        new GPUMemAllocator(
            GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie(),
            platform_gpu_id, false /*use_unified_memory*/, {}, {});
    GPUBFCAllocator *allocatorC = new GPUBFCAllocator(sub_allocatorC, shape.num_elements() * sizeof(uint8), "GPU_0_bfc");
    this->BufferC = new Tensor(allocatorC, tensorflow::DT_UINT8, shape);

    cudaStreamCreateWithFlags(&this->CUDAstreamA, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&this->CUDAstreamB, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&this->CUDAstreamC, cudaStreamNonBlocking);
}

void Detector::feed_data(uint8_t BUFFER_ID, std::vector<cv::Mat> imgArray)
{
    Tensor* dst_buffer;
    cudaStream_t *CUDAstream;
    if(BUFFER_ID == BUFFER_ID_A)
    {
        std::cout << "Feed A\n";
        dst_buffer = this->BufferA;
        CUDAstream = &this->CUDAstreamA;
    }
    else if(BUFFER_ID == BUFFER_ID_B)
    {
        std::cout << "Feed B\n";
        dst_buffer = this->BufferB;
        CUDAstream = &this->CUDAstreamB;
    }
    else
    {
        std::cout << "Feed C\n";
        dst_buffer = this->BufferC;
        CUDAstream = &this->CUDAstreamC;
    }

    tensorflow::Tensor temp_tensor(DT_UINT8, TensorShape({this->batchSize, this->input_height, this->input_width, this->input_depth}));
    uint8 *p = temp_tensor.flat<uint8>().data();
    uint8 *dst = dst_buffer->flat<uint8>().data();

    for (uint8_t i = 0; i < this->batchSize; i++)
    {
        cv::Mat fakeMat(this->input_height, this->input_width, CV_8UC3, p + this->input_depth * this->input_height * this->input_width * i);
        cv::cvtColor(imgArray[i], fakeMat, cv::COLOR_BGR2RGB);
    }
    cudaMemcpyAsync(dst, p, this->batchSize * this->input_depth * this->input_height * this->input_width, cudaMemcpyHostToDevice, *CUDAstream);
    std::cout << "Feed done\n";
}

void Detector::infer_thread_function(void)
{
    tensorflow::Tensor *tensor_buffer;
    uint8_t this_infer;

    this->IsInferReady = true;
    while(!this->ShouldClose)
    {
        std::unique_lock<std::mutex> lock(this->mutex_infer);
        this->cond_infer.wait(lock, [this](){return !this->InferQueue.empty();});
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        while(!this->InferQueue.empty())
        {
            this_infer = InferQueue.front();
            this->InferQueue.pop();
            if(this_infer == BUFFER_ID_A)
            {
                std::cout << "Infer A\n";
                tensor_buffer = this->BufferA;
            }
            else if(this_infer == BUFFER_ID_B)
            {
                std::cout << "Infer B\n";
                tensor_buffer = this->BufferB;
            }
            else //if(this_infer == BUFFER_ID_C)
            {
                std::cout << "Infer C\n";
                tensor_buffer = this->BufferC;
            }
            
            Status run_status = sess->Run({{Detector::input_layer, *tensor_buffer}}, {this->output_layer}, {}, &this->OutputTensor);
            if (!run_status.ok())
            {
                std::cout << "MESSAGE::dectection.cpp::Running model failed\n";
            }
            std::cout << "Infer done\n";
            if(this_infer == BUFFER_ID_A)
            {
                this->BufferA_busy = false;
            }
            else if(this_infer == BUFFER_ID_B)
            {
                this->BufferB_busy = false;
            }
            else if(this_infer == BUFFER_ID_C)
            {
                this->BufferC_busy = false;
            }
            this->cond_fetch.notify_one();
            this->cond_feed.notify_one();
        }
    }
    this->sess->Close();
}

void Detector::feed_thread_function(void)
{
    std::vector<cv::Mat> temp;
    this->AllocBuffer();
    this->BufferA_busy = false;
    this->BufferB_busy = false;
    this->BufferC_busy = false;
    this->IsFeedReady = true;
    this->infer_thread = new std::thread(&Detector::infer_thread_function, this);
    this->fetch_thread = new std::thread(&Detector::fetch_thread_function, this);
    while(!this->ShouldClose)
    {
        std::unique_lock<std::mutex> lock(this->mutex_feed);
        this->cond_feed.wait(lock, [this](){return !this->InFIFO.empty();});
        while(!this->InFIFO.empty())
        {
            if(!this->BufferA_busy && !this->InFIFO.empty())
            {
                temp = this->InFIFO.front();
                this->feed_data(BUFFER_ID_A, temp);
                this->InFIFO.pop();
                this->InferQueue.push(BUFFER_ID_A);
                this->BufferA_busy = true;
                this->cond_infer.notify_one();
            }
            if(!this->BufferB_busy && !this->InFIFO.empty())
            {
                temp = this->InFIFO.front();
                this->feed_data(BUFFER_ID_B, temp);
                this->InFIFO.pop();
                this->InferQueue.push(BUFFER_ID_B);
                this->BufferB_busy = true;
                this->cond_infer.notify_one();
            }
            if(!this->BufferC_busy && !this->InFIFO.empty())
            {
                temp = this->InFIFO.front();
                this->feed_data(BUFFER_ID_C, temp);
                this->InFIFO.pop();
                this->InferQueue.push(BUFFER_ID_C);
                this->BufferC_busy = true;
                this->cond_infer.notify_one();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void Detector::fetch_thread_function(void)
{
    tensorflow::Tensor temp_Tensor, tensorA;
    this->IsFetchReady = true;
    while(!this->ShouldClose)
    {
        std::unique_lock<std::mutex> lock(this->mutex_fetch);
        this->cond_fetch.wait(lock);
        if(temp_Tensor.CopyFrom(this->OutputTensor[0], this->OutputTensor[0].shape()))
        {
            this->OutFIFO.push(temp_Tensor);
            this->InferDoneCB(this->Instance_ptr);
        }
    }
}

bool Detector::Infer(std::vector<cv::Mat> imgArray)
{
    if(this->InFIFO.size() >= FIFO_capacity)
    {
        std::cout << "MESSAGE::detection.cpp::Input FIFO reached maximum capacity" << "\n";
        return false;
    }
    this->InFIFO.push(imgArray);
    this->cond_feed.notify_one();
    return true;
}

bool Detector::GetOutput(tensorflow::Tensor &TensorOut)
{
    if(!this->OutFIFO.empty())
    {
        TensorOut = this->OutFIFO.front();
        this->OutFIFO.pop();
        return true;
    }
    return false;
}

void Detector::Terminate(void)
{
    this->ShouldClose = true;
    this->cond_feed.notify_one();
    this->cond_infer.notify_one();
    this->cond_fetch.notify_one();
    this->feed_thread->join();
    this->infer_thread->join();
    this->fetch_thread->join();
}

bool Detector::IsOutFIFOEmpty(void)
{
    return this->OutFIFO.empty();
}

bool Detector::IsReady(void)
{
    return this->IsInferReady || this->IsFeedReady || this->IsFetchReady;
}
