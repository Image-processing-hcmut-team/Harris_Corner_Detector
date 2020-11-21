#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>

#define Linux

#ifdef Linux
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

#ifdef Windows
#include <direct.h>
#define GetCurrentDir _getcwd
#endif

std::string getpath(std::string relative_path);

int main()
{
  std::string s = getpath("/Project/Tho/Code/Testing/Helloworld/smarties.png");
  cv::Mat src = cv::imread(s);
  cv::Mat img_gray = cv::imread(s, cv::IMREAD_GRAYSCALE);
  cv::namedWindow("Detected circle", cv::WINDOW_NORMAL);
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(img_gray, circles, cv::HOUGH_GRADIENT, 1, img_gray.rows / 16, 100, 30, 1, 30);
  for (size_t i = 0; i < circles.size(); i++)
  {
    cv::Vec3i c = circles[i];
    cv::Point center = cv::Point(c[0], c[1]);
    int radius = c[2];
    circle(src, center, radius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);
  }
  imshow("Detected circle", src);
  cv::waitKey(0);
  return 0;
}

std::string getpath(std::string relative_path)
{
  char cCurrentPath[FILENAME_MAX];
  GetCurrentDir(cCurrentPath, sizeof(cCurrentPath));
  cCurrentPath[sizeof(cCurrentPath) - 1] = '\0';
  std::string cppCurrentPath(cCurrentPath);
  cppCurrentPath = cppCurrentPath + relative_path;
  return (cppCurrentPath);
}