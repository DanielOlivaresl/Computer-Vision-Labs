#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <QColor>
#include <QPoint>
#include <QImage>
#include "computations.h"
#include <queue>

#ifndef A_OBJECTDETECTION
#define A_OBJECTDETECTION


class ObjectDetection{
public:
	static void segmentImage(cv::Mat& image, const cv::Mat& labels, int numSuperpixels, int threshold);
	static void regionGrowing(cv::Mat& image, const std::vector<cv::Point>& seeds, int threshold, cv::Mat& visited, cv::Vec3b regionColor);


	static std::pair<cv::Mat, cv::Mat> generateSuperpixels(const cv::Mat& inputImage, int numberOfSuperpixels, int ruler);
	static void colorSuperpixelBorders(cv::Mat& image, const cv::Mat& labels, const cv::Mat& mask);
	static void createBoundaryMask(cv::Mat& labels, cv::Mat& boundaryMask);
	static void colorBorders(cv::Mat& image, const cv::Mat& boundaryMask, cv::Mat& labels);
	static void adjustBoundaryMask(cv::Mat& mask, bool thinBorders);
	static void fillSuperpixelsWithColors(cv::Mat& image, const cv::Mat& labels);
	static void regionGrowing(cv::Mat& image, cv::Mat& labels, int threshold);
	static void applyRegionGrowing(cv::Mat& image, cv::Mat& labels, int threshold);
	static std::vector<QColor> generateColors();
	static QVector<QVector<QPoint>> connectedN4(QImage& image);
	static  QVector<QPoint> outLine(QImage& image, int i, int j);
	static QPoint nextCell(QPoint& crrPixel, int& currDir, QPoint& savePixel);
	static QVector<QPoint> borderFollow(QImage& img, QPoint& start, QPoint& prev, int direction, int& NBD);
	static QVector<QVector<QPoint>> rasterScan(QImage& img);




	static std::vector<std::string> classifyImage(QImage& image, Eigen::MatrixXd centroids, std::vector < std::function <std::vector<int>(QVector<QPoint>, QImage&)>> functions, std::map<int, std::string> namesMap);
	static void calculateBounds(QList<QPoint> objectBorder, int& minX, int& maxX, int& minY, int& maxY);

};





#endif