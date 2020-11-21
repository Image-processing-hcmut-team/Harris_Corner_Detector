#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

int main(){
    cv::Mat img = cv::imread("./Project/Tho/Linux/data/lena.jpg", cv::IMREAD_GRAYSCALE);
    uchar test = img.at<uchar>(101,150);
    std::cout << double(test);
}

