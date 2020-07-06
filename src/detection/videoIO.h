#pragma once

#include <vector>
#include <future>
#include <string>
#include <iostream>
#include <chrono>

#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;

class CV_VideoReader
{
private:
    std::vector<VideoCapture> sources;
public:
    CV_VideoReader(std::vector<std::string> Paths, uint8_t numOfvideos);
    ~CV_VideoReader();
    bool ReadNextFrame(std::vector<Mat> &Frames);
};

class CV_VideoWriter
{
private:
    std::vector<VideoWriter> writers;
public:
    CV_VideoWriter(std::vector<std::string> Paths, uint8_t numOfvideos, uint32_t width, uint32_t height, double fps);
    ~CV_VideoWriter();
    void Write(std::vector<Mat> &Frames);
};