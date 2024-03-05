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
double static euclidean(Eigen::Vector2d vec1, Eigen::Vector2d vec2) {
	
	
	
	return sqrt(
		pow(vec1.x() - vec2.x(), 2) +
		pow(vec1.y() - vec2.y(), 2)
	);
	


}


double static manhalanobis(Eigen::Vector2d point) {

	euclidean(Eigen::Vector2d(2, 2), Eigen::Vector2d(2, 2));
}



