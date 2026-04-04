//
// Created by Volodymyr Avvakumov on 02.04.2026.
//


#include "regressionDrawer.h"
cv::Point2i worldToImage(const cv::Point2f& p,
                         float minX, float maxX,
                         float minY, float maxY,
                         int width, int height,
                         int margin) {
    float usableW = static_cast<float>(width - 2 * margin);
    float usableH = static_cast<float>(height - 2 * margin);

    float xNorm = (p.x - minX) / (maxX - minX);
    float yNorm = (p.y - minY) / (maxY - minY);

    int px = static_cast<int>(margin + xNorm * usableW);
    int py = static_cast<int>(height - margin - yNorm * usableH); // переворот по Y

    return {px, py};
}


void drawAxes(cv::Mat& image,
              float minX, float maxX,
              float minY, float maxY,
              int margin) {
    int width = image.cols;
    int height = image.rows;

    if (minX <= 0.0f && 0.0f <= maxX) {
        cv::Point2i p1 = worldToImage({0.0f, minY}, minX, maxX, minY, maxY, width, height, margin);
        cv::Point2i p2 = worldToImage({0.0f, maxY}, minX, maxX, minY, maxY, width, height, margin);
        cv::line(image, p1, p2, cv::Scalar(200, 200, 200), 1);
    }

    if (minY <= 0.0f && 0.0f <= maxY) {
        cv::Point2i p1 = worldToImage({minX, 0.0f}, minX, maxX, minY, maxY, width, height, margin);
        cv::Point2i p2 = worldToImage({maxX, 0.0f}, minX, maxX, minY, maxY, width, height, margin);
        cv::line(image, p1, p2, cv::Scalar(200, 200, 200), 1);
    }
}


void drawRegressionResult(const std::vector<cv::Point2f>& points,
                          const cv::Point2d& lineParams,   // (a, b)
                          const std::string& title,
                          const std::string& outputFilename,
                          int width,
                          int height,
                          int margin) {
    if (points.empty()) {
        throw std::runtime_error("drawRegressionResult: points are empty");
    }

    double a = lineParams.x;
    double b = lineParams.y;

    float minX = points[0].x;
    float maxX = points[0].x;
    float minY = points[0].y;
    float maxY = points[0].y;

    for (const auto& p : points) {
        minX = std::min(minX, p.x);
        maxX = std::max(maxX, p.x);
        minY = std::min(minY, p.y);
        maxY = std::max(maxY, p.y);
    }

    float padX = std::max(1.0f, 0.1f * (maxX - minX));
    float padY = std::max(1.0f, 0.1f * (maxY - minY));

    minX -= padX;
    maxX += padX;
    minY -= padY;
    maxY += padY;

    cv::Mat image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    drawAxes(image, minX, maxX, minY, maxY, margin);

    // Малюємо точки
    for (const auto& p : points) {
        cv::Point2i pi = worldToImage(p, minX, maxX, minY, maxY, width, height, margin);
        cv::circle(image, pi, 4, cv::Scalar(0, 0, 0), cv::FILLED);
    }

    // Малюємо пряму на всьому видимому діапазоні x
    cv::Point2f w1(minX, static_cast<float>(a * minX + b));
    cv::Point2f w2(maxX, static_cast<float>(a * maxX + b));

    cv::Point2i p1 = worldToImage(w1, minX, maxX, minY, maxY, width, height, margin);
    cv::Point2i p2 = worldToImage(w2, minX, maxX, minY, maxY, width, height, margin);

    cv::line(image, p1, p2, cv::Scalar(0, 0, 255), 2);

    // Підпис
    cv::putText(image,
                title,
                cv::Point(30, 40),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(30, 30, 30),
                2);

    std::string eq = "y = " + std::to_string(a) + " * x + " + std::to_string(b);
    cv::putText(image,
                eq,
                cv::Point(30, 80),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(30, 30, 30),
                2);

    if (!cv::imwrite(outputFilename, image)) {
        throw std::runtime_error("Failed to save image: " + outputFilename);
    }
}