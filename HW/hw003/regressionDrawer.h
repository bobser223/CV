//
// Created by Volodymyr Avvakumov on 02.04.2026.
//

#ifndef CODE_REGRESSIONDRAWER_H
#define CODE_REGRESSIONDRAWER_H


#include <opencv2/opencv.hpp>

cv::Point2i worldToImage(const cv::Point2f& p,
                         float minX, float maxX,
                         float minY, float maxY,
                         int width, int height,
                         int margin);
void drawAxes(cv::Mat& image,
              float minX, float maxX,
              float minY, float maxY,
              int margin);

void drawRegressionResult(const std::vector<cv::Point2f>& points,
                          const cv::Point2d& lineParams,   // (a, b)
                          const std::string& title,
                          const std::string& outputFilename,
                          int width = 1000,
                          int height = 800,
                          int margin = 60);

#endif //CODE_REGRESSIONDRAWER_H