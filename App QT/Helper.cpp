#include "computations.h"


/**
 * @brief Helper function that calculates the closest distance from a vector of distances
 * @param std::vector<double> distances: vector of distances
 * @return int returns the index of the closest point in the vector
 */
int  Computations::Helper::getClosest(std::vector<double> distances) {
	int min = 0;
	for (int i = 1; i < distances.size(); i++) {
		if (distances.at(i) < distances.at(min)) {
			min = i;
		}
	}
	return min;
}
int Computations::Helper::getClosest(Eigen::VectorXd distances) {
	int min = 0;
	for (int i = 1; i < distances.cols(); i++) {
		if (distances(i) < distances(min)) {
			min = i;
		}
	}
	return min;
}
/**
 * @brief Helper function that calculates the maximum probability from a given vector of probabilities.
 * @param std::vector<double> probabilities: vector of probabilities, should sum up to 1
 * @return int returns the index of the vector which has the maximum probability
 */
int Computations::Helper::getMaxProb(std::vector<double> probabilities) {
	int max = 0;
	for (int i = 1; i < probabilities.size(); i++) {
		if (probabilities.at(i) > probabilities.at(max)) {
			max = i;
		}
	}
	return max;
}

int Computations::Helper::getMaxProb(Eigen::VectorXd probabilities) {
	int max = 0;
	for (int i = 1; i < probabilities.cols(); i++) {
		if (probabilities(i) > probabilities(max)) {
			max = i;
		}
	}
	return max;
}







/**
 * @brief function that calculates the probabilities of a point belonging to a set of classes
 * @param std::vector<Eigen::Matrix<double, Eigen::Dynamic,3>> classes: vector of matrices of size nx3 where each matrix represents a class
 * @param Eigen::Vector3d point point that will be used to calculate the probabilities that it belogns to each of the classes passed
 * @returns std::vector<double> vector of probabilites, that the point belongs to each of the classes
 */
std::vector<double>  Computations::Helper::max_prob(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point) {

	std::vector<double> manhalanobis_distance = Distances::manhalanobis(classes, point);

	std::vector<double> probabilites;

	for (int i = 0; i < classes.size(); i++) {
		Eigen::MatrixXd cov = LinearAlgebra::calculateCovMatrix(classes.at(i));
		double det_cov = cov.determinant();

		double pi_term = pow(2 * EIGEN_PI, (3 / 2));

		double manh_dist = manhalanobis_distance.at(i);

		probabilites.push_back((1 / (
			pi_term * sqrt(det_cov)
			)) *
			exp(-0.5 * manh_dist));
	}

	double sum = 0;
	for (int i = 0; i < probabilites.size(); i++) {
		sum += probabilites.at(i);
	}

	for (int i = 0; i < probabilites.size(); i++) {
		probabilites.at(i) = probabilites.at(i) / sum;
	}
	return probabilites;

}

Eigen::VectorXd Computations::Helper::max_prob(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point) {

	Eigen::VectorXd manhalanobis_distance = Distances::manhalanobis(classes, point);

	Eigen::VectorXd probabilites(classes.size());

	for (int i = 0; i < classes.size(); i++) {
		Eigen::MatrixXd cov = LinearAlgebra::calculateCovMatrix(classes.at(i));
		double det_cov = cov.determinant();

		double pi_term = pow(2 * EIGEN_PI, (3 / 2));

		double manh_dist = manhalanobis_distance(i);

		probabilites(i) = (1 / (
			pi_term * sqrt(det_cov)
			)) *
			exp(-0.5 * manh_dist);
	}

	double sum = 0;
	for (int i = 0; i < probabilites.size(); i++) {
		sum += probabilites(i);
	}

	for (int i = 0; i < probabilites.size(); i++) {
		probabilites(i) = probabilites(i) / sum;
	}



	return probabilites;





}



void Computations::Helper::normalizeColumn(Eigen::MatrixXd & matrix, int column) {
	// getting the col
	Eigen::VectorXd col = matrix.col(column);

	// min val and max val 
	double minVal = col.minCoeff();
	double maxVal = col.maxCoeff();

	// range in (0,1)
	matrix.col(column) = (col.array() - minVal) / (maxVal - minVal);
}

int Computations::Helper::clockwise(int dir)
{
	return (dir % 8) + 1;
}

int Computations::Helper::counterclockwise(int dir)
{
	return (dir + 6) % 8 + 1;
}



