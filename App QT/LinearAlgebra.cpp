#include "computations.h"	



/**
 *
 * @brief Helper function that calculates the covariance matrix from a set of data
 * @param Eigen::MatrixXd matrix of n rows and m columns, where the m is the number of features in the set of data, and n is the number of instances of that data
 * @returns Eigen::MatrixXd returns the covariance matrix of dimensions mxm, of the data passed.
 */
Eigen::MatrixXd  Computations::LinearAlgebra::calculateCovMatrix(Eigen::MatrixXd data) {
	/*
	Eigen::MatrixXd transposed = data.transpose();
	// Center the data: subtract the mean of each column from all elements in the column
	Eigen::MatrixXd centered = transposed.rowwise() - transposed.colwise().mean();
	// Calculate the covariance matrix
	Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(centered.rows() - 1);
	*/
	int numSamples = data.rows();
	int numDimensions = data.cols();

	// Calculate the mean vector
	Eigen::VectorXd mean = data.colwise().mean();

	// Subtract the mean from each data point
	Eigen::MatrixXd centered = data.rowwise() - mean.transpose();

	// Calculate the covariance matrix
	Eigen::MatrixXd cov = (centered.transpose() * centered) / double(numSamples - 1);


	return cov;
}

