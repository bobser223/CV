#ifndef TASK4_H
#define TASK4_H

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

typedef cv::Point2d point;
typedef size_t idx;

struct Model {
    Model(const std::vector<double>& coefficients_input, const std::string& name_input, size_t inliers_cnt) : coefficients(coefficients_input), name(name_input), inliers_count(inliers_cnt) {}
    ~Model() = default;

    std::vector<double> coefficients;
    std::string name;
    size_t inliers_count = 0;
};

std::vector<idx> getNRandomIdx(idx n, size_t size);
std::tuple<idx, idx, idx> get3RandomIdx(size_t size);
std::pair<idx, idx> get2RandomIdx(size_t size);

std::tuple<double, double, double> minimizeQuadratic(
    const std::vector<point>& points,
    idx idx_1,
    idx idx_2,
    idx idx_3
);

double countQuadraticError(const point& p, double a, double b, double c);

std::vector<point> getQuadraticInliers(
    const std::vector<point>& points,
    double a,
    double b,
    double c,
    double inlierThreshold
);

std::tuple<double, double, double> fitQuadraticByInliers(
    const std::vector<point>& inliers
);

Model RANSACforQuadratic(
    std::vector<point> points,
    double inlierThreshold,
    size_t maxIterations
);

std::pair<double, double> minimizeExp(
    const std::vector<point>& points,
    idx idx_1,
    idx idx_2
);

double countExpError(const point& p, double a, double b);

std::vector<point> getExpInliers(
    const std::vector<point>& points,
    double a,
    double b,
    double inlierThreshold
);

std::pair<double, double> fitExpByInliers(
    const std::vector<point>& inliers
);

Model RANSACforExp(
    std::vector<point> points,
    double inlierThreshold,
    size_t maxIterations
);

std::pair<double, double> minimizeLinear(
    const std::vector<point>& points,
    idx idx_1,
    idx idx_2
);

double countLinearError(const point& p, double a, double b);

std::vector<point> getLinearInliers(
    const std::vector<point>& points,
    double a,
    double b,
    double inlierThreshold
);

std::pair<double, double> fitLinearByInliers(
    const std::vector<point>& inliers
);

Model RANSACforLinear(
    std::vector<point> points,
    double inlierThreshold,
    size_t maxIterations
);

double evalModel(const Model& model, double x);

void printModels(
    const std::vector<point>& points,
    const std::vector<Model>& models,
    const Model& winner
);

Model modelSelection(
    std::vector<point> points,
    double inlierThreshold,
    size_t maxIterations
);

#endif