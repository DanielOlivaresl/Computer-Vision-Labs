//File for extra computations needed in the labs


#include"computations.h"
//Distance functions

//Euclidean


/**
* @brief Function that calculates the euclidean distance between two points
 * @param Eigen::Vector2d vector that represents the first (intiial) point
 * @param Eigen::Vector2d vector that represents the second (destiny) point
 * @return eucleidan distance between the two points
 *
 */
std::vector<double>  euclidean(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point) {

	std::vector<Eigen::Vector3d> centroids;


	for (int i = 0; i < classes.size(); i++) {
		//We add the mean of every class
		centroids.push_back(
			Eigen::Vector3d(
				classes.at(i).row(0).mean(),
				classes.at(i).row(1).mean(),
				classes.at(i).row(2).mean()
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
 * 
 * @brief Helper function that calculates the covariance matrix from a set of data
 * @param Eigen::MatrixXd matrix of n rows and m columns, where the m is the number of features in the set of data, and n is the number of instances of that data 
 * @returns Eigen::MatrixXd returns the covariance matrix of dimensions mxm, of the data passed. 
 */
Eigen::MatrixXd  calculateCovMatrix(Eigen::MatrixXd data) {

	Eigen::MatrixXd transposed = data.transpose();
	// Center the data: subtract the mean of each column from all elements in the column
	Eigen::MatrixXd centered = transposed.rowwise() - transposed.colwise().mean();
	// Calculate the covariance matrix
	Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(centered.rows() - 1);

	return cov;
}

/**
 * @brief Helper function that calculates the closest distance from a vector of distances
 * @param std::vector<double> distances: vector of distances
 * @return int returns the index of the closest point in the vector
 */
int  getClosest(std::vector<double> distances) {
	int min = 0;
	for (int i = 1; i < distances.size(); i++) {
		if (distances.at(i) < distances.at(min)) {
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
int getMaxProb(std::vector<double> probabilities) {
	int max = 0;
	for (int i = 1; i < probabilities.size(); i++) {
		if (probabilities.at(i) < probabilities.at(max)) {
			max= i;
		}
	}
	return max;
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
std::vector<double>  manhalanobis(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point) {

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


		Eigen::MatrixXd cov = calculateCovMatrix(classes.at(i));

		//The point minus the centroid of the current class
		Eigen::Vector3d x_minus_mu = point - centroids.at(i);

		Eigen::MatrixXd inv_cov = cov.inverse();



		Eigen::MatrixXd left_term = x_minus_mu.transpose() * inv_cov;
		Eigen::MatrixXd res = left_term * x_minus_mu;



		distances.push_back(res(0, 0));

	}

	return distances;

}
/**
 * @brief function that calculates the probabilities of a point belonging to a set of classes
 * @param std::vector<Eigen::Matrix<double, Eigen::Dynamic,3>> classes: vector of matrices of size nx3 where each matrix represents a class
 * @param Eigen::Vector3d point point that will be used to calculate the probabilities that it belogns to each of the classes passed
 * @returns std::vector<double> vector of probabilites, that the point belongs to each of the classes
 */
std::vector<double>  max_prob(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point) {

	std::vector<double> manhalanobis_distance = manhalanobis(classes, point);

	std::vector<double> probabilites;

	for (int i = 0; i < classes.size(); i++) {
		Eigen::MatrixXd cov = calculateCovMatrix(classes.at(i));
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