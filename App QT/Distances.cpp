#include "computations.h"



/**
* @brief Function that calculates the euclidean distance between a point and a set of classes
 * @param std::vector<Eigen::MatrixXd> Vector of matrices that represents each class
 * @param Eigen::VectorXd vector that represents the point
 * @return Eigen::VectorXd of distances between point and classes
 *
 */
Eigen::VectorXd  Computations::Distances::euclidean(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point) {

	//Before making any computations, we will 



	//We will create a matrix that holds the centroids of each class, where each row is a class and the cols are the corresponding dimensions  
	
	//We will create variables for the number of classes, as well as the dims, this will be to initialize the matrix

	int rows = classes.size(); //Number of classes
	int cols = point.size(); //Number of dimensions 


	Eigen::MatrixXd centroids(rows,cols);




	for (int i = 0; i < classes.size(); i++) {
		//We will fill the matrix with the means of every class
	
		centroids.row(i) = classes.at(i).colwise().mean();
	
	}


	//We will now create an Eigen::VectorXd that will store the distances from the point to each centroid stored in the matrix





	Eigen::VectorXd distances(rows);

	//We now will calculate the distances for every class

	for (int i = 0; i < classes.size(); i++) {

		auto classVector = centroids.row(i); //Vector of the centroid of the i'th class
		
		//We now will calculate the difference between the classVector and the point

		Eigen::VectorXd diffVec = classVector - point;

		//Finally to compute the distance we calculate the dot product of the difference vector with itself

		distances(i) = diffVec.transpose() * diffVec;


	}

	return distances;


}








/**
* @brief Function that calculates the euclidean distance between a point and a set of points
 * @param std::vector<Eigen::MatrixXd> Matrix of points
 * @param Eigen::VectorXd vector that represents the point
 * @return Eigen::VectorXd of distances between point and classes
 * */

Eigen::VectorXd Computations::Distances::euclidean(Eigen::MatrixXd points, Eigen::VectorXd point)
{
	
	//We will first create the vector of distances that will be returned from the function 

	Eigen::VectorXd distances(points.rows());


	//Now we will iterate the matrix rows to calculate the distance of each point 

	for (int i = 0; i < points.rows(); i++) {

		//We will first calculate a difference vector between each of the matrix rows and the point

		Eigen::VectorXd diffVec = points.row(i) - point;

		//Now we will calulate the dot product of the difference Vector with itself and fill the distances vector

		distances(i) = diffVec.transpose() * diffVec;


	}


	return distances;



	


}

/**
* @brief Function that calculates the euclidean distance between Two points
 * @param Eigen::VectorXd First Point
 * @param Eigen::VectorXd Second Point
 * @return double distances between two points
 * */

double Computations::Distances::euclidean(Eigen::VectorXd point1, Eigen::VectorXd point2)
{
	//We first compute the difference between the two points

	Eigen::VectorXd diffVec = point1 - point2;

	//We will now calculate the dot product of the difference vector with itself

	return diffVec.transpose() * diffVec;
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


