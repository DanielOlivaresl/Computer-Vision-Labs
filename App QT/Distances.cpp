#include "computations.h"




/**
 * @brief Function that calculates the euclidean distance between two points.
 * @param Eigen::VectorXd p1: First Point
 * @param Eigen::VectorXd p2: Second Point
 * @returns double: eucledian distance between two points
 *
 */
double Computations::Distances::euclideanDistance(Eigen::VectorXd p1, Eigen::VectorXd p2) {
	double powSums=0;
	if (p1.size() == p2.size()) {
		for (int i = 0; i < p1.size(); i++) {
			powSums += pow(p1(i) - p2(i), 2);
		}
	}
	return sqrt(powSums);
}

/**
* @brief Function that calculates the euclidean distance between two points
 * @param Eigen::Vector2d vector that represents the first (intiial) point
 * @param Eigen::Vector2d vector that represents the second (destiny) point
 * @return eucleidan distance between the two points
 *
 */
std::vector<double>  Computations::Distances::euclidean(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point) {

	std::vector<Eigen::Vector3d> centroids;


	for (int i = 0; i < classes.size(); i++) {
		//We add the mean of every class
		centroids.push_back(
			Eigen::Vector3d(
				classes.at(i).colwise().mean()

			));



	}
	std::vector<double> distances;

	//We now will calculate the distances for every class

	for (int i = 0; i < classes.size(); i++) {

		distances.push_back(
			sqrt(
				pow(point.x() - centroids.at(i).x(), 2) +
				pow(point.y() - centroids.at(i).y(), 2) +
				pow(point.z() - centroids.at(i).z(), 2)

			));



	}

	return distances;


}
/**
* @brief Function that calculates the euclidean distance between two points in n dimensions
 * @param Eigen::MatrixXd Matrix that contains all the points which we want to calc the distance
 * @param Eigen::VectorXd vector of N dimensions, that we want to know the distance to
 * @return Vector of distances
 *
 */
std::vector<double> Computations::Distances::euclidean(Eigen::MatrixXd points, Eigen::VectorXd point)
{
	std::vector<double> distances(points.rows());

	//we iterate the points
	for (int i = 0; i < points.rows(); i++) {

		//we iterate the dimensions of each point
		double sum = 0;
		for (int j = 0; j < points.cols(); j++) {
			sum += pow(point[j] - points(i, j), 2);
		}
		distances[i] = sqrt(sum);

	}



	return distances;
}

Eigen::VectorXd Computations::Distances::euclidean(std::vector <Eigen::MatrixXd> classes, Eigen::VectorXd point) {

	std::vector<Eigen::VectorXd> centroids;


	for (const auto& cls : classes) {
		centroids.push_back(cls.colwise().mean());
	}
	Eigen::VectorXd distances(classes.size());

	//We now will calculate the distances for every class

	for (int i = 0; i < classes.size(); i++) {

		distances(i) = (point - centroids[i]).norm();
	}

	return distances;


}
Eigen::VectorXd euclideanGenerelied(Eigen::MatrixXd points, Eigen::VectorXd point)
{
	Eigen::VectorXd distances(points.rows());

	//we iterate the points
	for (int i = 0; i < points.rows(); i++) {
		//we iterate the dimensions of each point
		double sum = 0;
		for (int j = 0; j < points.cols(); j++) {
			sum += pow(point[j] - points(i, j), 2);
		}
		distances(i) = sqrt(sum);
	}
	return distances;
}

/**
 * @brief function that calculates the manhalanobis distance from a point and a set of points.
 *
 * @param std::vector<Eigen::Matrix<double,Eigen::Dynamic,3>> classes: vector of matrices of size nx3 where each matrix represents a class which we want to know the distance to
 * @param Eigen::Vector3d point: point from which we will calculate the distance to each class
 * @return std::vector<double> vector that stores the distance calculated from the point to each class
 *
 *
 */
std::vector<double>  Computations::Distances::manhalanobis(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point) {

	std::vector<Eigen::Vector3d> centroids;

	for (int i = 0; i < classes.size(); i++) {
		//We add the mean of every class
		centroids.push_back(
			Eigen::Vector3d(
				classes.at(i).col(0).mean(),
				classes.at(i).col(1).mean(),
				classes.at(i).col(2).mean()
			));

	}

	std::vector<double> distances;


	for (int i = 0; i < classes.size(); i++) {
		//We	first transpose the matrix


		Eigen::MatrixXd cov = LinearAlgebra::calculateCovMatrix(classes.at(i));

		//The point minus the centroid of the current class
		Eigen::Vector3d x_minus_mu = point - centroids.at(i);

		Eigen::MatrixXd inv_cov = cov.inverse();



		Eigen::MatrixXd left_term = x_minus_mu.transpose() * inv_cov;
		Eigen::MatrixXd res = left_term * x_minus_mu;



		distances.push_back(res(0, 0));

	}

	return distances;

}

Eigen::VectorXd Computations::Distances::manhalanobis(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point) {

	std::vector<Eigen::VectorXd> centroids;

	for (int i = 0; i < classes.size(); i++) {
		//We add the mean of every class
		Eigen::VectorXd tmp(classes.at(i).cols());
		for (int j = 0; j < classes.at(i).cols(); j++) {
			tmp(j) = classes.at(i).col(j).mean();
		}
		centroids.push_back(tmp);

	}
	Eigen::VectorXd distances(classes.size());


	for (int i = 0; i < classes.size(); i++) {
		//We	first transpose the matrix


		Eigen::MatrixXd cov = LinearAlgebra::calculateCovMatrix(classes.at(i));

		//The point minus the centroid of the current class
		Eigen::VectorXd x_minus_mu = point - centroids.at(i);

		Eigen::MatrixXd inv_cov = cov.inverse();



		Eigen::MatrixXd left_term = x_minus_mu.transpose() * inv_cov;
		Eigen::MatrixXd res = left_term * x_minus_mu;



		distances(i) = res(0, 0);

	}

	return distances;

}


