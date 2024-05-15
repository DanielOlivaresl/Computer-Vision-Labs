#pragma once

#include<iostream>
#include<vector>
#include<cmath>	
#include<algorithm>
#include<map>

#include<Eigen/Dense>
#include <QString>
#include<QImage>
#include <QPoint>
std::vector<double> euclidean(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point);
std::vector<double> euclidean(Eigen::MatrixXd points, Eigen::VectorXd point); //we pass a matrix of n cols and n rows and a point of n dimensions




Eigen::MatrixXd  calculateCovMatrix(Eigen::MatrixXd data);
std::vector<double> manhalanobis(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point);
int getClosest(std::vector<double> distances);
int getMaxProb(std::vector<double> probabilities);
std::vector<double> max_prob(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point);
int kNearestNeighbours(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point, int k);
std::vector<std::vector<double>> get_matrixConfusion(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> mat, std::vector<std::vector<int>> vectorOfPredictions);
std::vector<std::vector<std::vector<int>>> generatePredictions(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> train, std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> test, int knn);
void normalizeColumn(Eigen::MatrixXd& matrix, int column);


int clockwise(int dir);
int counterclockwise(int dir);



class CrossValidation {
public:
	static std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> leaveOneOut(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, int clas, int el);
	static std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> crossValidation(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes);
	static std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> Restitucion(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes);

};

class ObjectMetrics {
public:
	static std::vector<int> calculateArea(QVector<QPoint> object, QImage& image);
	static std::vector<int> calculatePerimeter(QVector<QPoint> object, QImage& image);
	static std::vector<int> calculateCenterOfGravity(QVector<QPoint> object, QImage& image);
	static Eigen::MatrixXd featureExtraction(std::vector < std::function <std::vector<int>(QVector<QPoint>,QImage&)>> functions, QVector<QPoint> object, QImage& image);

	
};