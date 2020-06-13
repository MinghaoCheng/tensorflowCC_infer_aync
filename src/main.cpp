#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>

#include <unistd.h>
#include <queue>

#include "detection.h"

#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
using namespace cv;

void CalcDoneCB(void *dummy)
{
    static bool first_time = true;

    static std::chrono::steady_clock::time_point t_last;
    static std::chrono::steady_clock::time_point t_now;

    static uint64_t counter = 0;
    static double total;
    double this_time;

    counter++;

    if(first_time)
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
                  << "FPS:  " << 1000/(total/(counter-1)) << ",\t"
                  << "Frames calculated:  " << counter << ",\t"
                  << "Total elapsed time:  " << total << "ms,\t"
                  << "\n";
    }
}

int main()
{
    std::string graph_path = "saved_model_bs5/efficientdet-d0_frozen.pb";
    // std::vector<std::string> imgPath = {"testdata/camA.jpg", "testdata/camB.jpg", "testdata/camH.jpg", "testdata/camK.jpg", "testdata/camN.jpg"};
    uint8_t batch_size = 5;
    uint8_t alignment = 5;

    Detector *detector = new Detector(batch_size, 1024, 768, 3, graph_path, CalcDoneCB, NULL);
    while(!detector->IsReady());

    // Read and prepare images using OpenCV:
    std::cout << "Read and prepare images/videos\n";
    
    // for (uint8_t i = 0; i < batch_size; i++)
    // {
    //     imgArray.push_back(cv::imread(imgPath[i]));
    // }
    // open videos
    const char *video0 = "testvideos/camA.mp4";
    const char *video1 = "testvideos/camB.mp4";
    const char *video2 = "testvideos/camH.mp4";
    const char *video3 = "testvideos/camK.mp4";
    const char *video4 = "testvideos/camN.mp4";
    std::vector<cv::VideoCapture> cap;
    cap.push_back(cv::VideoCapture{video0});
    cap.push_back(cv::VideoCapture{video1});
    cap.push_back(cv::VideoCapture{video2});
    cap.push_back(cv::VideoCapture{video3});
    cap.push_back(cv::VideoCapture{video4});
    if (!cap[0].isOpened() || !cap[1].isOpened() || !cap[2].isOpened() || !cap[3].isOpened() || !cap[4].isOpened())
    {
        std::cout << "Error opening video stream or file\n";
    }
    std::vector<cv::Mat> imgArray;
    std::queue<std::vector<cv::Mat>> frame_queue;
    cv::Mat frame;

    Tensor OutputTensor;
    std::vector<cv::Mat> frame_array_temp;
    uint32_t counter = 0;
    while(1)
    {
        if(frame_queue.size() < 10)
        {
            // make batch
            for (uint8_t i = 0; i < batch_size; i += alignment)
            {
                for (uint8_t j = 0; j < alignment; j ++)
                {
                    cap[j] >> frame;
                    if (frame.empty())
                    {
                        std::cout << "Nomore frame!\n";
                        // When everything done, release the video capture object
                        for(uint8_t i = 0; i < alignment; i++)
                        {
                            cap[i].release();
                        }
                        detector->Terminate();
                        delete detector;
                        std::cout << "done\n";
                        return 0;
                    }
                    imgArray.push_back(frame.clone());
                }
            }

            // put imgarray into pipeline
            frame_queue.push(imgArray);
            detector->Infer(imgArray);
            imgArray.clear();
            std::cout << counter++ <<"\n";
        }
        if(detector->GetOutput(OutputTensor))
        {
            // for (uint j = 0 ; j < batch_size; j++)
            // {
            //     frame_array_temp.push_back(frame_queue.front()[j].clone());
            // }
            frame_queue.pop();
        }
        //     auto output_detection = OutputTensor.tensor<float, 3>();
        //     // get 2D BB
        //     for (uint8_t frame = 0; frame < 5; frame += 5)
        //     {
        //         for (int class_result = 1; class_result <= 5; class_result++)
        //         {
        //             for (int cam = frame; cam < frame + 5; cam++)
        //             {
        //                 for (int index = 0; index < 5; index++)
        //                 {
        //                     if (output_detection(cam, index, 6) == class_result)
        //                     {
        //                         cv::rectangle(frame_array_temp[cam],
        //                                     cv::Point{int(output_detection(cam, index, 2)), int(output_detection(cam, index, 1))}, 
        //                                     cv::Point{int(output_detection(cam, index, 4)), int(output_detection(cam, index, 3))},
        //                                     cv::Scalar(0, 0, 255));
        //                     }
        //                 }
        //             }
        //         }
        //     }
        //     for (uint8_t cam = 0; cam < 5; cam++)
        //     {
        //         char c = cam + 0x30;
        //         cv::imwrite(cv::String("testdata_out/") + c + "_" + std::to_string(counter).c_str() + ".jpg",
        //                     frame_array_temp[cam]);
        //     }
        //     frame_array_temp.clear();
        // }
    }
}