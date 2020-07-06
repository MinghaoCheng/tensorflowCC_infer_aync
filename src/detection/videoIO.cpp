#include "videoIO.h"

CV_VideoReader::CV_VideoReader(std::vector<std::string> Paths, uint8_t numOfvideos)
{
    for (uint8_t i = 0; i < numOfvideos; i++)
    {
        this->sources.push_back(VideoCapture{Paths[i].c_str()});
        if (!this->sources[i].isOpened())
        {
            std::cout << "MESSAGE::videoStreamReader.cpp::Error opening video stream or file " << Paths[i] << "\n";
        }
    }
    cv::setNumThreads(8);
}

CV_VideoReader::~CV_VideoReader()
{
    for (auto &source : this->sources)
    {
        source.release();
    }
}

static Mat Read_frame(VideoCapture &source)
{
    Mat frame;
    source.read(frame);
    return frame.clone();
}

bool CV_VideoReader::ReadNextFrame(std::vector<Mat> &Frames)
{
    // auto t_last = std::chrono::steady_clock::now();
    
    std::vector<std::future<Mat>> frames_temp;
    for (auto &source : this->sources)
    {
        frames_temp.push_back(std::async(std::launch::async, Read_frame, std::ref(source)));
    }
    for (auto &temp : frames_temp)
    {
        Frames.push_back(temp.get().clone());
    }
    // auto t_now = std::chrono::steady_clock::now();
    // std::cout << "frames reading time: " << (std::chrono::duration_cast<std::chrono::microseconds>(t_now - t_last).count()) / 1000.0 << "\n";
    return true;
}


CV_VideoWriter::CV_VideoWriter(std::vector<std::string> Paths, uint8_t numOfvideos, uint32_t width, uint32_t height, double fps)
{
    for (uint8_t i = 0; i < numOfvideos; i++)
    {
        this->writers.push_back(VideoWriter{Paths[i].c_str(), CV_FOURCC('X', 'V', 'I', 'D'), fps, Size(width, height)});
    }
}

CV_VideoWriter::~CV_VideoWriter()
{
    for (auto &writer : this->writers)
    {
        writer.release();
    }
}

static void write_frame(VideoWriter &writer, Mat &frame)
{
    writer.write(frame);
}

void CV_VideoWriter::Write(std::vector<Mat> &Frames)
{
    std::vector<std::future<void>> results;
    for (uint8_t i = 0; i < this->writers.size(); i ++)
    {
        results.push_back(std::async(std::launch::async, write_frame, std::ref(this->writers[i]), std::ref(Frames[i])));
    }
    for (auto &result : results)
    {
        result.get();
    }
}
