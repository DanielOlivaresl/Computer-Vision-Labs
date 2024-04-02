#pragma once

#include<vector>
#include<Eigen/Dense>
#include<iostream>
#include<cmath>	
#include<algorithm>

std::vector<double> euclidean(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point);
Eigen::MatrixXd  calculateCovMatrix(Eigen::MatrixXd data);
std::vector<double> manhalanobis(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point);
int getClosest(std::vector<double> distances);
int getMaxProb(std::vector<double> probabilities);
std::vector<double> max_prob(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point);
int kNearestNeighbours(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point, int k);


class CrossValidation {
	
	std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> CrossValidation::leaveOneOut(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, int clas, int el);
	std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> crossValidation(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes);
	std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> CrossValidation::Restitucion(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes);

};
