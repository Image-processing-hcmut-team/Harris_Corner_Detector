#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;

int main()
{
	cv::Mat img = cv::imread("./data/lena.jpg");
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	cv::imshow("image", img);
	cv::waitKey(0);
 	return 0;
}