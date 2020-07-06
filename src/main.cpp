#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>

#include <unistd.h>
#include <queue>

#include "detection.h"
#include "videoIO.h"

#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
using namespace cv;

uint8_t batch_size = 10;
uint8_t alignment = 5;
std::queue<std::vector<cv::Mat>> draw_frame_queue;
Detector *detector;

void CalcDoneCB(void *dummy)
{
    static bool first_time = true;

    static std::chrono::steady_clock::time_point t_last;
    static std::chrono::steady_clock::time_point t_now;

    static uint64_t counter = 0;
    static double total;
    double this_time;

    counter += 2;

    if (first_time)
    {
        t_last = std::chrono::steady_clock::now();
        first_time = false;
    }
    else
    {
        t_now = std::chrono::steady_clock::now();
        this_time = (std::chrono::duration_cast<std::chrono::microseconds>(t_now - t_last).count()) / 1000.0;
        total += this_time;
        t_last = t_now;

        std::cout << "Last frame:  " << this_time << "ms,\t"
                  << "FPS:  " << 1000 / (total / (counter - 1)) << ",\t"
                  << "Frames calculated:  " << counter << ",\t"
                  << "Total elapsed time:  " << total << "ms,\t"
                  << "\n";
    }
}

void feed_data(void)
{
    std::string graph_path = "saved_model_bs10/efficientdet-d0_frozen.pb";
    std::vector<std::string> video_paths{"testvideos/camA.mp4", "testvideos/camB.mp4", "testvideos/camH.mp4", "testvideos/camK.mp4", "testvideos/camN.mp4"};
    std::vector<cv::Mat> frame_array_temp;
    std::vector<cv::Mat> batch;
    CV_VideoReader videoReader{video_paths, 5};
    
    while(1)
    {
        if (draw_frame_queue.size() < 10)
        {
            batch.clear();
            frame_array_temp.clear();
            // make batch
            for (uint8_t i = 0; i < batch_size; i += alignment)
            {
                videoReader.ReadNextFrame(frame_array_temp);
                draw_frame_queue.push(frame_array_temp);
                batch.insert(batch.end(), frame_array_temp.begin(), frame_array_temp.end());
            }
            // put imgarray into pipeline
            detector->Infer(batch);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void write_to_videos(void)
{
    std::vector<std::string> video_out_paths = {"testvideos_out/camA.avi",
                                                "testvideos_out/camB.avi",
                                                "testvideos_out/camH.avi",
                                                "testvideos_out/camK.avi",
                                                "testvideos_out/camN.avi"};
    CV_VideoWriter videoWriter{video_out_paths,
                               5,
                               1024,768,
                               30};
    Tensor OutputTensor;
    std::vector<cv::Mat> draw_frame_temp;
    while(1)
    {
        if (detector->GetOutput(OutputTensor))
        {
            for (uint8_t i = 0; i < batch_size; i += alignment)
            {
                draw_frame_temp.clear();
                for (uint j = 0; j < alignment; j++)
                {
                    draw_frame_temp.push_back(draw_frame_queue.front()[j].clone());
                }
                draw_frame_queue.pop();
                auto output_detection = OutputTensor.tensor<float, 3>();
                // get 2D BB
                for (int class_result = 1; class_result <= 5; class_result++)
                {
                    for (int img_index = i; img_index < i + alignment; img_index++)
                    {
                        for (int index = 0; index < 5; index++)
                        {
                            if (output_detection(img_index, index, 6) == class_result)
                            {
                                cv::rectangle(draw_frame_temp[img_index - i],
                                                cv::Point{int(output_detection(img_index, index, 2)), int(output_detection(img_index, index, 1))},
                                                cv::Point{int(output_detection(img_index, index, 4)), int(output_detection(img_index, index, 3))},
                                                cv::Scalar(0, 0, 255));
                            }
                        }
                    }
                }
                videoWriter.Write(draw_frame_temp);
            }
        }
    }
}

int main()
{
    // init AABB pipeline
    std::string graph_path = "saved_model_bs10/efficientdet-d0_frozen.pb";

    detector = new Detector(batch_size, 1024, 768, 3, graph_path, CalcDoneCB, NULL);
    while (!detector->IsReady());

    std::thread feed_thread{feed_data};
    std::thread write_thread{write_to_videos};
    feed_thread.join();
    write_thread.join();
    detector->Terminate();
    delete detector;
}