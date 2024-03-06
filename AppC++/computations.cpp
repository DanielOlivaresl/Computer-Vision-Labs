//File for extra computations needed in the labs

#include<vector>
#include<Eigen/Dense>
#include<iostream>
#include<cmath>	

//Distance functions

//Euclidean


/**
* @brief Function that calculates the euclidean distance between two points
 * @param Eigen::Vector2d vector that represents the first (intiial) point
 * @param Eigen::Vector2d vector that represents the second (destiny) point
 * @return eucleidan distance between the two points
 * 
 * 
 */
double static euclidean(std::vector<Eigen::Matrix<double,Eigen::Dynamic,3>> classes, Eigen::VectorXd point) {
	
	std::vector<Eigen::Vector3d> centroids;


	for (int i = 0; i < classes.size(); i++) {
		centroids.push_back(
			Eigen::Vector3d(
				classes.at(i).row(0).mean(),
				classes.at(i).row(1).mean(),
				classes.at(i).row(2).mean()
			));

		

	}




}


double static manhalanobis(Eigen::Vector2d point) {

	euclidean(Eigen::Vector2d(2, 2), Eigen::Vector2d(2, 2));
}



