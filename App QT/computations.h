#pragma once

#include<iostream>
#include<vector>
#include<cmath>	
#include<algorithm>
#include<map>
#include <QDebug>
#include<Eigen/Dense>
#include <QString>
#include<QImage>
#include <QPoint>




std::vector<double> euclidean(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point);
double euclideanDistance(Eigen::VectorXd p1, Eigen::VectorXd p2);
std::vector<double> euclidean(Eigen::MatrixXd points, Eigen::VectorXd point); //we pass a matrix of n cols and n rows and a point of n dimensions
Eigen::VectorXd euclideanGenerelied(Eigen::MatrixXd points, Eigen::VectorXd point);
Eigen::MatrixXd  calculateCovMatrix(Eigen::MatrixXd data);
std::vector<double> manhalanobis(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point);
Eigen::VectorXd manhalanobis(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point);
int getClosest(std::vector<double> distances);
int getClosest(Eigen::VectorXd distances);
int getMaxProb(std::vector<double> probabilities);
Eigen::VectorXd max_prob(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point);
std::vector<double> max_prob(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point);
int kNearestNeighbours(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point, int k);
int kNearestNeighbours(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point, int k);




std::vector<std::vector<double>> get_matrixConfusion(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> mat, std::vector<std::vector<int>> vectorOfPredictions);
std::vector<Eigen::VectorXd> get_matrixConfusion(std::vector<Eigen::MatrixXd> mat, std::vector<std::vector<int>> vectorOfPredictions);
std::vector<std::vector<std::vector<int>>> generatePredictions(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> train, std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> test, int knn);
std::vector<std::vector<std::vector<int>>> generatePredictions(std::vector<Eigen::MatrixXd> train, std::vector<Eigen::MatrixXd> test, int knn);

void normalizeColumn(Eigen::MatrixXd& matrix, int column);



int clockwise(int dir);
int counterclockwise(int dir);


//Fourier transform functions

void fft1D(std::vector<std::complex<double>>& data, bool invert);
std::vector<std::vector<std::complex<double>>> fft2D(const std::vector<std::vector<std::complex<double>>>& data,bool invert);


class CrossValidation {
public:
	static std::vector<std::vector<Eigen::MatrixXd>> leaveOneOut(std::vector<Eigen::MatrixXd> classes, int clas, int el);
	static std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> leaveOneOut(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, int clas, int el);
	static std::vector<std::vector<Eigen::MatrixXd>> crossValidation(std::vector<Eigen::MatrixXd> classes);
	static std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> crossValidation(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes);
	static std::vector<std::vector<Eigen::MatrixXd>> Restitucion(std::vector<Eigen::MatrixXd> classes);
	static std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> Restitucion(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes);

};

class ObjectMetrics {
public:
	
	static std::vector<int> calculateArea(QVector<QPoint> object, QImage& image);
	static std::vector<int> calculatePerimeter(QVector<QPoint> object, QImage& image);
	static std::vector<int> calculateCenterOfGravity(QVector<QPoint> object, QImage& image);
	static Eigen::MatrixXd featureExtraction(std::vector < std::function <std::vector<int>(QVector<QPoint>, QImage&)>> functions, QVector<QPoint> object, QImage& image);
	static double imageMoments(QImage& binarizedImage, int p, int q);
	static double centralMoments(QImage& binarizedImage, int p, int q, double xCent, double yCent);
	static double calculateEccentricity(QImage& image);

};
