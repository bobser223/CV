//
// Created by Volodymyr Avvakumov on 06.04.2026.
//

#include "task4.h"

#include <random>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <matplot/matplot.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


typedef cv::Point2d point;
typedef size_t idx;


std::vector<idx> getNRandomIdx(idx n, size_t size) {
    static std::mt19937 rng(123456);

    std::vector<idx> indices(size);
    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), rng);

    return {indices.begin(), indices.begin() + n};
}

std::tuple<idx, idx, idx> get3RandomIdx(size_t size) {
    auto indexes = getNRandomIdx(3, size);
    return {indexes[0], indexes[1], indexes[2]};
}

std::tuple<double, double, double> minimizeQuadratic(
    const std::vector<point>& points,
    idx idx_1,
    idx idx_2,
    idx idx_3
) {
    const auto& p1 = points[idx_1];
    const auto& p2 = points[idx_2];
    const auto& p3 = points[idx_3];

    cv::Mat A = (cv::Mat_<double>(3, 3) <<
        p1.x * p1.x, p1.x, 1.0,
        p2.x * p2.x, p2.x, 1.0,
        p3.x * p3.x, p3.x, 1.0
    );

    cv::Mat B = (cv::Mat_<double>(3, 1) <<
        p1.y,
        p2.y,
        p3.y
    );

    cv::Mat X;
    bool ok = cv::solve(A, B, X, cv::DECOMP_SVD);

    if (!ok) return {0,0,0};

    double a = X.at<double>(0, 0);
    double b = X.at<double>(1, 0);
    double c = X.at<double>(2, 0);

    return {a, b, c};
}

double countQuadraticError(const point& p, double a, double b, double c) {

    return p.y - a * p.x * p.x - b * p.x - c;
}

std::vector<point> getQuadraticInliers(const std::vector<point>& points, double a, double b, double c, double inlierThreshold) {
    std::vector<point> inliers;
    inliers.reserve(points.size());
    for (auto& point : points) {
        if (std::abs(countQuadraticError(point, a, b, c)) < inlierThreshold) {
            inliers.push_back(point);
        }
    }
    return inliers;
}

std::tuple<double, double, double> fitQuadraticByInliers(const std::vector<point>& inliers) {
    if (inliers.size() < 3) {
        return {0.0, 0.0, 0.0};
    }

    cv::Mat A((int)inliers.size(), 3, CV_64F);
    cv::Mat B((int)inliers.size(), 1, CV_64F);

    for (int i = 0; i < (int)inliers.size(); ++i) {
        A.at<double>(i, 0) = inliers[i].x * inliers[i].x;
        A.at<double>(i, 1) = inliers[i].x;
        A.at<double>(i, 2) = 1.0;
        B.at<double>(i, 0) = inliers[i].y;
    }

    cv::Mat X;
    bool ok = cv::solve(A, B, X, cv::DECOMP_SVD);

    if (!ok) {
        return {0.0, 0.0, 0.0};
    }

    double a = X.at<double>(0, 0);
    double b = X.at<double>(1, 0);
    double c = X.at<double>(2, 0);

    return {a, b, c};
}

Model RANSACforQuadratic(std::vector<point> points, double inlierThreshold, size_t maxIterations) {

    std::tuple<size_t, idx, idx, idx> winner{0,0,1,2}; // inliers_cnt, idx_1, idx_2, idx_3

    for (size_t i = 0; i < maxIterations; ++i) {
        auto [point_idx_1, point_idx_2, point_idx_3] = get3RandomIdx(points.size());
        auto [a, b, c] = minimizeQuadratic(points, point_idx_1, point_idx_2, point_idx_3);
        auto inliers = getQuadraticInliers(points, a, b, c, inlierThreshold);

        if (inliers.size() > std::get<0>(winner)) {
            winner = {inliers.size(), point_idx_1, point_idx_2, point_idx_3};
        }
    }

    auto [a, b, c] = minimizeQuadratic(points, std::get<1>(winner), std::get<2>(winner), std::get<3>(winner));

    auto inliers = getQuadraticInliers(points, a, b, c, inlierThreshold);

    auto [a_fit, b_fit, c_fit] = fitQuadraticByInliers(inliers);

    return Model({a_fit, b_fit, c_fit},
        "quadratic",
        std::get<0>(winner));
}


// -------------------------------- Exp ---------------------------
std::pair<idx, idx> get2RandomIdx(size_t size) {
    auto indexes = getNRandomIdx(2, size);
    return {indexes[0], indexes[1]};
}

std::pair<double, double> minimizeExp(
    const std::vector<point>& points,
    idx idx_1,
    idx idx_2
) {
    auto [x1, y1] = points[idx_1];
    auto [x2, y2] = points[idx_2];

    if (x1 == x2) return {0,0};
    if (y1 <= 0|| y2 <= 0) return {0,0};

    double b = std::log(y2/y1) / (x2-x1);
    double a = y1 / std::exp(b * x1);
    return {a, b};
}


double countExpError(const point& p, double a, double b) {

    return p.y - a * std::exp(b * p.x);
}

std::vector<point> getExpInliers(const std::vector<point>& points, double a, double b, double inlierThreshold) {
    std::vector<point> inliers;
    inliers.reserve(points.size());
    for (auto& point : points) {
        if (std::abs(countExpError(point, a, b)) < inlierThreshold) {
            inliers.push_back(point);
        }
    }
    return inliers;
}

std::pair<double, double> fitExpByInliers(const std::vector<point>& inliers) {
    std::vector<point> validPoints;
    validPoints.reserve(inliers.size());

    for (const auto& p : inliers) {
        if (p.y > 0.0) {
            validPoints.push_back(p);
        }
    }

    if (validPoints.size() < 2) {
        return {0.0, 0.0};
    }

    cv::Mat A((int)validPoints.size(), 2, CV_64F);
    cv::Mat B((int)validPoints.size(), 1, CV_64F);

    for (int i = 0; i < (int)validPoints.size(); ++i) {
        A.at<double>(i, 0) = validPoints[i].x;
        A.at<double>(i, 1) = 1.0;
        B.at<double>(i, 0) = std::log(validPoints[i].y);
    }

    cv::Mat X;
    bool ok = cv::solve(A, B, X, cv::DECOMP_SVD);

    if (!ok) {
        return {0.0, 0.0};
    }

    double b = X.at<double>(0, 0);
    double ln_a = X.at<double>(1, 0);
    double a = std::exp(ln_a);

    return {a, b};
}


Model RANSACforExp(std::vector<point> points, double inlierThreshold, size_t maxIterations) {

    std::tuple<size_t, idx, idx> winner{0,0,1}; // inliers_cnt, point)idx_1, point_idx_2

    for (size_t i = 0; i < maxIterations; ++i) {
        auto [point_idx_1, point_idx_2] = get2RandomIdx(points.size());
        auto [a, b] = minimizeExp(points, point_idx_1, point_idx_2);
        auto inliers = getExpInliers(points, a, b, inlierThreshold);

        if (inliers.size() > std::get<0>(winner)) {
            winner = {inliers.size(), point_idx_1, point_idx_2};
        }
    }

    auto [a, b] = minimizeExp(points, std::get<1>(winner), std::get<2>(winner));
    auto inliers = getExpInliers(points, a, b, inlierThreshold);
    auto [a_fit, b_fit] = fitExpByInliers(inliers);



    return Model({a_fit, b_fit},
        "exp",
        std::get<0>(winner));
}


// ---------------------------------- linear

std::pair<double, double> minimizeLinear(
    const std::vector<point>& points,
    idx idx_1,
    idx idx_2
) {
    const auto& p1 = points[idx_1];
    const auto& p2 = points[idx_2];

    if (p1.x == p2.x) {
        return {0,0};
    }

    double a = (p2.y - p1.y) / (p2.x - p1.x);
    double b = p1.y - a * p1.x;

    return {a, b};
}

double countLinearError(const point& p, double a, double b) {
    return p.y - (a * p.x + b);
}

std::vector<point> getLinearInliers(
    const std::vector<point>& points,
    double a,
    double b,
    double inlierThreshold
) {
    std::vector<point> inliers;
    inliers.reserve(points.size());

    for (const auto& point : points) {
        if (std::abs(countLinearError(point, a, b)) < inlierThreshold) {
            inliers.push_back(point);
        }
    }

    return inliers;
}

std::pair<double, double> fitLinearByInliers(const std::vector<point>& inliers) {
    if (inliers.size() < 2) {
        return {0.0, 0.0};
    }

    cv::Mat A((int)inliers.size(), 2, CV_64F);
    cv::Mat B((int)inliers.size(), 1, CV_64F);

    for (int i = 0; i < (int)inliers.size(); ++i) {
        A.at<double>(i, 0) = inliers[i].x;
        A.at<double>(i, 1) = 1.0;
        B.at<double>(i, 0) = inliers[i].y;
    }

    cv::Mat X;
    bool ok = cv::solve(A, B, X, cv::DECOMP_SVD);

    if (!ok) {
        return {0.0, 0.0};
    }

    double a = X.at<double>(0, 0);
    double b = X.at<double>(1, 0);

    return {a, b};
}

Model RANSACforLinear(std::vector<point> points, double inlierThreshold, size_t maxIterations) {

    std::tuple<size_t, idx, idx> winner{0, 0, 1}; // inliers_cnt, point_idx_1, point_idx_2

    for (size_t i = 0; i < maxIterations; ++i) {

            auto [point_idx_1, point_idx_2] = get2RandomIdx(points.size());
            auto [a, b] = minimizeLinear(points, point_idx_1, point_idx_2);
            auto inliers = getLinearInliers(points, a, b, inlierThreshold);

            if (inliers.size() > std::get<0>(winner)) {
                winner = {inliers.size(), point_idx_1, point_idx_2};
            }

    }

    auto [a, b] = minimizeLinear(points, std::get<1>(winner), std::get<2>(winner));
    auto inliers = getLinearInliers(points, a, b, inlierThreshold);
    auto [a_fit, b_fit] = fitLinearByInliers(inliers);

    return Model(
        {a_fit, b_fit},
        "linear",
        std::get<0>(winner)
    );
}

// ----------------------------- end ----------------------
double evalModel(const Model& model, double x) {
    if (model.name == "linear") {
        double a = model.coefficients[0];
        double b = model.coefficients[1];
        return a * x + b;
    }

    if (model.name == "quadratic") {
        double a = model.coefficients[0];
        double b = model.coefficients[1];
        double c = model.coefficients[2];
        return a * x * x + b * x + c;
    }

    if (model.name == "exp") {
        double a = model.coefficients[0];
        double b = model.coefficients[1];
        return a * std::exp(b * x);
    }

    return 0.0;
}

void printModels(
    const std::vector<point>& points,
    const std::vector<Model>& models,
    const Model& winner
) {
    if (points.empty()) {
        return;
    }

    const int width = 1200;
    const int height = 800;
    const int margin = 60;

    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    double min_x = points[0].x;
    double max_x = points[0].x;
    double min_y = points[0].y;
    double max_y = points[0].y;

    for (const auto& p : points) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }

    std::vector<double> xs;
    xs.reserve(400);

    double dx = max_x - min_x;
    if (dx == 0.0) dx = 1.0;

    double plot_min_x = min_x - 0.1 * dx;
    double plot_max_x = max_x + 0.1 * dx;

    for (int i = 0; i < 400; ++i) {
        double t = static_cast<double>(i) / 399.0;
        xs.push_back(plot_min_x + t * (plot_max_x - plot_min_x));
    }

    for (const auto& model : models) {
        for (double x : xs) {
            double y = evalModel(model, x);
            if (std::isfinite(y)) {
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);
            }
        }
    }

    double dy = max_y - min_y;
    if (dy == 0.0) dy = 1.0;

    double plot_min_y = min_y - 0.1 * dy;
    double plot_max_y = max_y + 0.1 * dy;

    auto toPixel = [&](double x, double y) -> cv::Point {
        double px = margin + (x - plot_min_x) / (plot_max_x - plot_min_x) * (width - 2 * margin);
        double py = height - margin - (y - plot_min_y) / (plot_max_y - plot_min_y) * (height - 2 * margin);
        return {static_cast<int>(std::round(px)), static_cast<int>(std::round(py))};
    };

    cv::line(canvas, cv::Point(margin, height - margin), cv::Point(width - margin, height - margin), cv::Scalar(0, 0, 0), 1);
    cv::line(canvas, cv::Point(margin, margin), cv::Point(margin, height - margin), cv::Scalar(0, 0, 0), 1);

    for (int i = 0; i <= 10; ++i) {
        double tx = plot_min_x + i * (plot_max_x - plot_min_x) / 10.0;
        cv::Point p = toPixel(tx, plot_min_y);
        cv::line(canvas, cv::Point(p.x, height - margin - 5), cv::Point(p.x, height - margin + 5), cv::Scalar(0, 0, 0), 1);
        cv::putText(canvas, cv::format("%.2f", tx), cv::Point(p.x - 20, height - margin + 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    for (int i = 0; i <= 10; ++i) {
        double ty = plot_min_y + i * (plot_max_y - plot_min_y) / 10.0;
        cv::Point p = toPixel(plot_min_x, ty);
        cv::line(canvas, cv::Point(margin - 5, p.y), cv::Point(margin + 5, p.y), cv::Scalar(0, 0, 0), 1);
        cv::putText(canvas, cv::format("%.2f", ty), cv::Point(5, p.y + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    for (const auto& p : points) {
        cv::circle(canvas, toPixel(p.x, p.y), 4, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA);
    }

    auto getColor = [&](const Model& model) -> cv::Scalar {
        if (model.name == winner.name) {
            return cv::Scalar(0, 0, 255); // red
        }
        if (model.name == "linear") {
            return cv::Scalar(255, 0, 0); // blue
        }
        if (model.name == "quadratic") {
            return cv::Scalar(0, 180, 0); // green
        }
        if (model.name == "exp") {
            return cv::Scalar(255, 0, 255); // magenta
        }
        return cv::Scalar(128, 128, 128);
    };

    for (const auto& model : models) {
        std::vector<cv::Point> polyline;
        polyline.reserve(xs.size());

        for (double x : xs) {
            double y = evalModel(model, x);
            if (std::isfinite(y)) {
                polyline.push_back(toPixel(x, y));
            }
        }

        if (polyline.size() >= 2) {
            int thickness = (model.name == winner.name) ? 4 : 2;
            cv::polylines(canvas, polyline, false, getColor(model), thickness, cv::LINE_AA);
        }
    }

    cv::putText(canvas, "Model selection", cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);

    int legend_y = 60;

    cv::circle(canvas, cv::Point(width - 260, legend_y), 5, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA);
    cv::putText(canvas, "points", cv::Point(width - 240, legend_y + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    legend_y += 30;

    for (const auto& model : models) {
        cv::Scalar color = getColor(model);
        int thickness = (model.name == winner.name) ? 4 : 2;

        cv::line(canvas, cv::Point(width - 270, legend_y), cv::Point(width - 230, legend_y), color, thickness, cv::LINE_AA);

        std::string label = model.name + " (" + std::to_string(model.inliers_count) + " inliers)";
        if (model.name == winner.name) {
            label += " WINNER";
        }

        cv::putText(canvas, label, cv::Point(width - 220, legend_y + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        legend_y += 30;
    }

    cv::imshow("Model selection", canvas);
    cv::waitKey(0);
}

Model modelSelection(std::vector<point> points, double inlierThreshold, size_t maxIterations) {
    if (points.size() < 2) {
        throw std::invalid_argument("Need at least 2 points");
    }

    std::vector<Model> models{
        RANSACforLinear(points, inlierThreshold, maxIterations),
        RANSACforQuadratic(points, inlierThreshold, maxIterations),
        RANSACforExp(points, inlierThreshold, maxIterations)
    };

    std::sort(models.begin(), models.end(), [](const Model& a, const Model& b) {
        return a.inliers_count > b.inliers_count;
    });


    printModels(points, models, models.front());

    return models.front();
}


