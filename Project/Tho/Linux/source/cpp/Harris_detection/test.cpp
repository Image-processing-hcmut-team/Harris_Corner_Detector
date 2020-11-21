#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>


int main()
{
    std::string s = "./Project/Tho/Linux/data/screen.png";
    cv::Mat img = cv::imread(s, cv::IMREAD_GRAYSCALE); 
    cv::imshow("Window 1", img);
    cv::waitKey(0);
    for (int i = 0; i<img.rows; i++){
        for (int j = 0; j<img.cols; j++){
            uchar pixel = img.at<uchar>(i,j);
            std::cout << int(pixel) << "\n";
        }
    }
}
