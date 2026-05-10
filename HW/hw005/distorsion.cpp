//
// Created by Volodymyr Avvakumov on 09.05.2026.
//




#include <utility>

#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include "data_path.h"

void testDistortion1()
{
	std::string path = "D:\\OpenCV\\Test\\theater.jpeg";
	cv::Mat image = cv::imread(path);
	cv::resize(image, image, image.size() * 3);
	cv::imshow("Picture", image);
	cv::waitKey(0);

	std::vector<std::vector<double>> coefs = { {0., 0.}, {2., 0.}, {-2., 0.}, {0., 0.0003}, {0., -0.000007}, {3., -0.000007} };
	for (auto& k : coefs)
	{
		double k1 = k[0], k2 = k[1];
		cv::Mat distorted_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
		for (size_t i = 0; i < image.rows; ++i)
			for (size_t j = 0; j < image.cols; ++j)
			{
				double x = i, y = j, xc = image.rows / 2, yc = image.cols / 2;
				double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
				double coef_rad = 1. + 1.e-6 * (k1 * r2 + k2 * r2 * r2);
				double xd = xc + (x - xc) * coef_rad;
				double yd = yc + (y - yc) * coef_rad;
				int i1 = int(xd + 0.5), j1 = int(yd + 0.5);
				if (i1 >= 0 && j1 >= 0 && i1 < distorted_image.rows && j1 < distorted_image.cols)
	  			    distorted_image.at<cv::Vec3b>(i1, j1) = image.at<cv::Vec3b>(i, j);
			}
		cv::imshow("Distorted, k1 = " + std::to_string(k1) + ", k2 = " + std::to_string(k2), distorted_image);
		cv::waitKey(0);
	}
}

cv::Mat distort_image_in_pixels(cv::Mat const& image, std::vector<double> const& coef)
{
	double k1 = coef[0], k2 = coef[1];
	cv::Mat distorted_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
	for (size_t i = 0; i < image.rows; ++i)
		for (size_t j = 0; j < image.cols; ++j)
			for (size_t k = 0; k < 5; ++k)
				for (size_t l = 0; l < 5; ++l)
				{
					double x = i + 0.2 * k, y = j + 0.2 * l, xc = image.rows / 2, yc = image.cols / 2;
					double r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
					double coef_rad = 1. + 1.e-6 * (k1 * r2 + k2 * r2 * r2);
					double xd = xc + (x - xc) * coef_rad;
					double yd = yc + (y - yc) * coef_rad;
					int i1 = int(xd + 0.5), j1 = int(yd + 0.5);
					if (i1 >= 0 && j1 >= 0 && i1 < distorted_image.rows && j1 < distorted_image.cols)
						distorted_image.at<cv::Vec3b>(i1, j1) = image.at<cv::Vec3b>(i, j);
				}
	return distorted_image;
}

void testDistortion2()
{
	std::string path = "D:\\OpenCV\\Test\\theater.jpeg";
	cv::Mat image = cv::imread(path);
	cv::resize(image, image, image.size() * 3);
	cv::imshow("Picture", image);
	cv::waitKey(0);

	std::vector<std::vector<double>> coefs = { {0., 0.}, {5., 0.}, {-5., 0.}, {0., 0.03}, {0., -0.007}, {3., -0.000007} };
	for (auto& k : coefs)
	{
		double k1 = k[0], k2 = k[1];
		cv::Mat distorted_image = distort_image_in_pixels(image, k);
		cv::imshow("Distorted, k1 = " + std::to_string(k1) + ", k2 = " + std::to_string(k2), distorted_image);
		cv::waitKey(0);
	}
}

void distort(double x, double y, double& xd, double& yd, double k1, double k2)
{
	double r2 = x * x + y * y;
	double coef_rad = 1. + k1 * r2 + k2 * r2 * r2;
	xd = x * coef_rad;
	yd = y * coef_rad;
}

void undistort(double xd, double yd, double& x, double& y, double k1, double k2)
{
	x = xd, y = yd;
	for (size_t iter = 0; iter < 10; ++iter)
	{
		double x1, y1;
		distort(x, y, x1, y1, k1, k2);
		x -= 0.5 * (x1 - xd);
		y -= 0.5 * (y1 - yd);
	}
}

cv::Mat distort_image_in_absolute(cv::Mat const& image, std::vector<double> const& coef, std::vector<double> const& intrinsics)
{
	double k1 = coef[0], k2 = coef[1];
	double fx = intrinsics[0], fy = intrinsics[1], cx = intrinsics[2], cy = intrinsics[3];
	cv::Mat distorted_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
	for (size_t i = 0; i < image.rows; ++i)
		for (size_t j = 0; j < image.cols; ++j)
			for (size_t k = 0; k < 5; ++k)
				for (size_t l = 0; l < 5; ++l)
				{
					double xp = i + 0.2 * k, yp = j + 0.2 * l;
					double x = (xp - cx) / fx, y = (yp - cy) / fy;
					double xd, yd;
					distort(x, y, xd, yd, k1, k2);
					double xdp = cx + fx * xd, ydp = cy + fy * yd;
					int i1 = int(xdp + 0.5), j1 = int(ydp + 0.5);
					if (i1 >= 0 && j1 >= 0 && i1 < distorted_image.rows && j1 < distorted_image.cols)
						distorted_image.at<cv::Vec3b>(i1, j1) = image.at<cv::Vec3b>(i, j);
				}
	return distorted_image;
}

cv::Mat undistort_image_in_absolute(cv::Mat const& image, std::vector<double> const& coef, std::vector<double> const& intrinsics)
{
	double k1 = coef[0], k2 = coef[1];
	double fx = intrinsics[0], fy = intrinsics[1], cx = intrinsics[2], cy = intrinsics[3];
	cv::Mat undistorted_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
	for (size_t i = 0; i < image.rows; ++i)
		for (size_t j = 0; j < image.cols; ++j)
			for (size_t k = 0; k < 5; ++k)
				for (size_t l = 0; l < 5; ++l)
				{
					double xp = i + 0.2 * k, yp = j + 0.2 * l;
					double x = (xp - cx) / fx, y = (yp - cy) / fy;
					double xd, yd;
					undistort(x, y, xd, yd, k1, k2);
					double xdp = cx + fx * xd, ydp = cy + fy * yd;
					int i1 = int(xdp + 0.5), j1 = int(ydp + 0.5);
					if (i1 >= 0 && j1 >= 0 && i1 < undistorted_image.rows && j1 < undistorted_image.cols)
						undistorted_image.at<cv::Vec3b>(i1, j1) = image.at<cv::Vec3b>(i, j);
				}
	return undistorted_image;
}

void testDistortion3()
{
	std::string path = "D:\\OpenCV\\Test\\theater.jpeg";
	cv::Mat image = cv::imread(path);
	cv::resize(image, image, image.size() * 3);
	cv::imshow("Picture", image);
	cv::waitKey(0);

	std::vector<std::vector<double>> coefs = { {0., 0.}, {2., 0.}, {-0.2, 0.}, {0., 0.1}, {0., -0.1}, {3., -0.1} };
	std::vector<double> intrinsics = { image.rows / 2., image.cols / 2., image.rows / 2., image.cols / 2. };
	for (auto& k : coefs)
	{
		double k1 = k[0], k2 = k[1];
		cv::Mat distorted_image = distort_image_in_absolute(image, k, intrinsics);
		cv::imshow("Distorted, k1 = " + std::to_string(k1) + ", k2 = " + std::to_string(k2), distorted_image);
		cv::waitKey(0);
		cv::Mat undistorted_image = undistort_image_in_absolute(distorted_image, k, intrinsics);
		cv::imshow("Undistorted, k1 = " + std::to_string(k1) + ", k2 = " + std::to_string(k2), undistorted_image);
		cv::waitKey(0);
	}
}

void testDistortion()
{
	//testDistortion1();
	//testDistortion2();
	testDistortion3();
}

std::pair<double, double>radialDistortion(double x_normalized, double y_normalized, double k1, double k2, double k3) {
    const double r2 = x_normalized*x_normalized + y_normalized*y_normalized;
    const double r4 = r2*r2;
    const double coef = 1.0 + k1*r2 + k2*r4 + k3*r4*r2;
    double x_d = x_normalized*coef;
    double y_d = y_normalized*coef;
    return {x_d, y_d};
}

std::pair<double, double>redialUndistorsion(
	double x_distorted,
	double y_distorted,
	double k1,
	double k2,
	double k3) {
	double x = x_distorted;
	double y = y_distorted;
	for (int iter = 0; iter < 10; ++iter)
	{
		const auto [x1, y1] = radialDistortion(x, y, k1, k2, k3);
		x -= 0.5 * (x1 - x_distorted);
		y -= 0.5 * (y1 - y_distorted);
	}
	return {x, y};
}

std::pair<double, double>tangentDistortion(double x_normalized, double y_normalized, double k1, double k2, double k3, double p1, double p2)
{
    //x_d = x*radial + 2p_1*x*y + p_2(r^2+2x^2) <=> radial_x + 2p_1*x*y + p_2(r^2+2x^2)
    //y_d = y*radial + p_1(r^2 +2y^2) + 2p_2*x*y <=> radial_y + p1*(r^2 + 2y^2) + 2*p2*x*y

    const double x2 = x_normalized*x_normalized;
    const double y2 = y_normalized*y_normalized;
    const double r2 = x2 + y2;

    const auto&[radial_x, radial_y] = radialDistortion(x_normalized, y_normalized, k1, k2, k3);
    const double x_d = radial_x + 2*p1*x_normalized*y_normalized + p2*(r2 + 2*x2);
    const double y_d = radial_y + p1*(r2 + 2*y2) + 2*p2*x_normalized*y_normalized;
    return {x_d, y_d};
}

std::pair<double, double> tangentUndistortion(
	double x_distorted,
	double y_distorted,
	double k1,
	double k2,
	double k3,
	double p1,
	double p2
) {
	double x = x_distorted;
	double y = y_distorted;

	for (int iter = 0; iter < 10; ++iter)
	{
		const auto [x1, y1] = tangentDistortion(
			x,
			y,
			k1,
			k2,
			k3,
			p1,
			p2
		);

		x -= 0.5 * (x1 - x_distorted);
		y -= 0.5 * (y1 - y_distorted);
	}

	return {x, y};
}

cv::Mat distortImageTangential(
	const cv::Mat& image,
	double k1,
	double k2,
	double k3,
	double p1,
	double p2,
	double fx,
	double fy,
	double cx,
	double cy
) {

	cv::Mat distorted_image = cv::Mat::zeros(image.rows, image.cols, image.type());

	for (int row = 0; row < image.rows; ++row) {
		for (int col = 0; col < image.cols; ++col) {
			const double y_pixel = row;
			const double x_pixel = col;

			const double x_normalized = (x_pixel - cx) / fx;
			const double y_normalized = (y_pixel - cy) / fy;

			const auto [x_distorted, y_distorted] = tangentDistortion(
				x_normalized,
				y_normalized,
				k1,
				k2,
				k3,
				p1,
				p2
			);

			const double x_distorted_pixel = std::round(cx + fx * x_distorted);
			const double y_distorted_pixel = std::round(cy + fy * y_distorted);


			const int distorted_col = static_cast<int>(x_distorted_pixel);
			const int distorted_row = static_cast<int>(y_distorted_pixel);

			if (
				distorted_row >= 0 && distorted_row < distorted_image.rows &&
				distorted_col >= 0 && distorted_col < distorted_image.cols
			) {
				distorted_image.at<cv::Vec3b>(distorted_row, distorted_col) =
					image.at<cv::Vec3b>(row, col);
			}


			}
	}
	return distorted_image;
}

cv::Mat undistortImageTangential(const cv::Mat& image,
	double k1,
	double k2,
	double k3,
	double p1,
	double p2,
	double fx,
	double fy,
	double cx,
	double cy) {
	cv::Mat undistorted_image = cv::Mat::zeros(image.rows, image.cols, image.type());
	for (int row = 0; row < image.rows; ++row) {
		for (int col = 0; col < image.cols; ++col) {
			const double y_pixel = row;
			const double x_pixel = col;
			const double x_normalized = (x_pixel - cx) / fx;
			const double y_normalized = (y_pixel - cy) / fy;

			const auto [x_distorted, y_distorted] = tangentDistortion(
				x_normalized,
				y_normalized,
				k1,
				k2,
				k3,
				p1,
				p2

			);

			const double x_distorted_pixel = std::ceil(cx + fx * x_distorted);
			const double y_distorted_pixel = std::ceil(cy + fy * y_distorted);


			const int distorted_col = static_cast<int>(x_distorted_pixel);
			const int distorted_row = static_cast<int>(y_distorted_pixel);

			if (
				distorted_row >= 0 && distorted_row < undistorted_image.rows &&
				distorted_col >= 0 && distorted_col < undistorted_image.cols
			) {
				undistorted_image.at<cv::Vec3b>(row, col) = image.at<cv::Vec3b>(distorted_row, distorted_col);
			}

		}
	}
	return undistorted_image;
}

void testPointUndistortion()
{
	double x = 0.2;
	double y = 0.1;

	double k1 = 0.2;
	double k2 = -0.05;
	double k3 = 0.01;
	double p1 = 0.01;
	double p2 = 0.01;

	auto [xd, yd] = tangentDistortion(x, y, k1, k2,k3, p1, p2);

	auto [xu, yu] = tangentUndistortion(xd, yd, k1, k2,k3, p1, p2);

	std::cout << "Original:    " << x  << " " << y  << "\n";
	std::cout << "Distorted:   " << xd << " " << yd << "\n";
	std::cout << "Undistorted: " << xu << " " << yu << "\n";
}

void testDistortImageTangential()
{
	std::string path = PATH_TO_DATA + "hw005/contax139q.jpg";

	cv::Mat image = cv::imread(path);

	if (image.empty())
	{
		std::cout << "Cannot load image\n";
		return;
	}

	cv::imshow("Original", image);
	cv::waitKey(0);

	const double fx = image.cols / 2.0;
	const double fy = image.rows / 2.0;
	const double cx = image.cols / 2.0;
	const double cy = image.rows / 2.0;

	const double k1 = 0.2;
	const double k2 = -0.05;
	const double k3 = 0.01;

	const double p1 = 0.01;
	const double p2 = 0.01;

	cv::Mat distorted = distortImageTangential(
		image,
		k1,
		k2,
		k3,
		p1,
		p2,
		fx,
		fy,
		cx,
		cy
	);

	cv::Mat undistored = undistortImageTangential(
		distorted,
		k1,
		k2,
		k3,
		p1,
		p2,
		fx,
		fy,
		cx,
		cy);

	cv::imshow("Tangential distorted", distorted);
	cv::waitKey(0);

	cv::imshow("Tangential undistorted", undistored);
	cv::waitKey(0);
}