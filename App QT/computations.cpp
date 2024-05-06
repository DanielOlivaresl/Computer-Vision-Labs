//File for extra computations needed in the labs


#include"computations.h"
//Distance functions

/**
 * @brief Function that calculates the euclidean distance between two points.
 * @param Eigen::Vector3d p1: First Point
 * @param Eigen::Vector3d p2: Second Point
 * @returns double: eucledian distance between two points
 *
 */
double euclideanDistance(Eigen::Vector3d p1, Eigen::Vector3d p2) {
	return sqrt(
		pow(p1.x() - p2.x(), 2) +
		pow(p1.y() - p2.y(), 2) +
		pow(p1.z() - p2.z(), 2)
	);
}


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
 *
 * @brief Helper function that calculates the covariance matrix from a set of data
 * @param Eigen::MatrixXd matrix of n rows and m columns, where the m is the number of features in the set of data, and n is the number of instances of that data
 * @returns Eigen::MatrixXd returns the covariance matrix of dimensions mxm, of the data passed.
 */
Eigen::MatrixXd  calculateCovMatrix(Eigen::MatrixXd data) {
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
		if (probabilities.at(i) > probabilities.at(max)) {
			max = i;
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


int kNearestNeighbours(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point, int k) {
	//Vector that stores a set of vectors that have the distance to the point, this is because each index in the vector represents a class,
	//This means that each element in the vector is another vector that stores the distances to the point from the ith class
	std::vector<std::vector<double>> distances;

	//We iterate the classes and fill the vectors
	for (int i = 0; i < classes.size(); i++) {
		std::vector<double> currClass;
		//We first iterate the current class and convert each matrix element into a point
		for (int j = 0; j < classes.at(i).rows(); j++) {
			Eigen::Vector3d currPoint = classes.at(i).row(j); //We assign the point

			currClass.push_back(euclideanDistance(point, currPoint));



		}
		//we sort the points before adding them
		std::sort(currClass.begin(), currClass.end());
		//we add the class to the vector
		distances.push_back(currClass);
	}


	//Now that are points are stored and sorted we will begin to check which are the k nearest neighbours, to do this we will first, from each class we will
	//only keep the k first elements because in the worst case the k nearest neighbours are from only one class

	for (int i = 0; i < distances.size(); i++) {
		if (distances.at(i).size() > k) {
			distances.at(i).erase(distances.at(i).begin() + k, distances.at(i).end());
		}
	}
	//Now that each class has their k nearest neighbours we will see which ones are the k closest in general, to do this we will iterate each class for their first
	//element and when we find the one that is the smallest, we will add it to our solution vector and remove that element from the class vector, and repeat this
	//process until the solution vector is of size k

	std::vector<int> solution;

	while (solution.size() < k) {
		//We iterate the classes
		int mindist = 0; //We assume that the first class is the smallest distance and iterate from there
		for (int i = 1; i < distances.size(); i++) {
			if (distances.at(i).at(0) < distances.at(mindist).at(0)) {
				mindist = i;
			}

		}

		//Now we will add that class to the solution and remove that element
		solution.push_back(mindist);
		distances.at(mindist).erase(distances.at(mindist).begin());

	}

	//Now we have the classes of the k nearest neighbours, we will now see what class dominates and make a prediction

	//We will first iterate the classes and check the ocurrence the one with the most occurence will be our final result

	//We will also first assume that the class 0 is the initial prediction and adjust from there
	int res = 0;
	int count = std::count(solution.begin(), solution.end(), res);
	for (int i = 1; i < classes.size(); i++) {
		int currCount = std::count(solution.begin(), solution.end(), i);
		if (currCount > count) {
			res = i;
			count = currCount;
		}
	}
	//The result is the classification prediction
	return res;

}



// 50/50 split
std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> CrossValidation::crossValidation(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes) {

	//We will create a test and train split vector's of matrices 

	std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> testSplit;
	std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> trainingSplit;

	//we will have train and test splits of the same size, and we will fill these randomly and have them be the same size

	//to do this we will first iterate the original data

	for (Eigen::Matrix<double, Eigen::Dynamic, 3> matrix : classes) {
		Eigen::Matrix<double, Eigen::Dynamic, 3> test;
		Eigen::Matrix<double, Eigen::Dynamic, 3> train;


		//we will randomly select indexes to add to the test set, until we have enough
		std::vector<int> indexes;

		while (indexes.size() < (matrix.rows() / 2)) {
			int index = std::rand() % matrix.rows();

			//we check that the index isn't in the list 
			while (std::count(indexes.begin(), indexes.end(), index)) {
				index = std::rand() % matrix.rows();
			}

			//Now that we know that the element isn't in the list we add it
			indexes.push_back(index);
		}

		//We will now fill the test and training sets

		for (int i = 0; i < matrix.rows(); i++) {
			if (std::count(indexes.begin(), indexes.end(), i)) {
				test.conservativeResize(test.rows() + 1, Eigen::NoChange);
				test.row(test.rows() - 1) << matrix.row(i);
			}
			else {
				train.conservativeResize(train.rows() + 1, Eigen::NoChange);
				train.row(train.rows() - 1) << matrix.row(i);
			}
		}

		testSplit.push_back(test);
		trainingSplit.push_back(train);
	}

	//Final result

	std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> result = { testSplit,trainingSplit };


	return result;

}

// n-1/1 split
//In the test set only one class will have data at one time thats the element we left out
std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> CrossValidation::leaveOneOut(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, int clas, int el) {
	std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> testSplit;
	std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> trainingSplit;

	for (int i = 0; i < classes.size(); i++) {
		Eigen::Matrix<double, Eigen::Dynamic, 3> trainMatrix;


		for (int j = 0; j < classes.at(i).rows(); j++) {

			if (i == clas && j == el) {
				testSplit.push_back((classes.at(i).row(el)));

			}
			else {
				trainMatrix.conservativeResize(trainMatrix.rows() + 1, Eigen::NoChange);
				trainMatrix.row(trainMatrix.rows() - 1) = classes.at(i).row(j);

			}

		}

		trainingSplit.push_back(trainMatrix);

	}

	std::vector<std::vector< Eigen::Matrix<double, Eigen::Dynamic, 3> > > result = { testSplit,trainingSplit };

	return result;
}
// both splits of the same size
std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> CrossValidation::Restitucion(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes) {
	std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> testSplit = classes;
	std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> trainingSplit = classes;

	std::vector<std::vector< Eigen::Matrix<double, Eigen::Dynamic, 3> > > result = { testSplit,trainingSplit };

	return result;



}


std::vector<std::vector<std::vector<int>>> generatePredictions(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> train, std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> test, int knn) {

	//we will create a 3D vector where one dimension is the Method (euclidean,manhalanobis, etc.. .) another dimension is the classes and the last dimension is the actual prediction

	std::vector<std::vector<std::vector<int>>> predictions(4, std::vector<std::vector<int>>(test.size(), std::vector<int>(test.at(0).rows())));

	//now we begin to fill our vector

	int classNumber = 0;
	for (Eigen::Matrix<double, Eigen::Dynamic, 3> clas : test) {
		for (int i = 0; i < clas.rows(); i++) {
			//we begin to fill our vector

			//Euclidean
			std::vector<double> distances1 = euclidean(train, clas.row(i));
			int res = getClosest(distances1);
			predictions.at(0).at(classNumber).at(i) = res;

			//Manhalanobis
			std::vector<double> distances2 = manhalanobis(train, clas.row(i));
			res = getClosest(distances2);
			predictions.at(1).at(classNumber).at(i) = res;

			//MaxProb
			std::vector<double> distances3 = max_prob(train, clas.row(i));
			res = getMaxProb(distances3);
			predictions.at(2).at(classNumber).at(i) = res;
			//KNN
			res = kNearestNeighbours(train, clas.row(i), knn);
			predictions.at(3).at(classNumber).at(i) = res;




		}
		classNumber++;
	}

	return predictions;

}


std::vector<std::vector<double>> get_matrixConfusion(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> mat, std::vector<std::vector<int>> vectorOfPredictions) {


	std::vector<std::vector<double>> matConf(mat.size(), std::vector<double>(mat.size(), 0));
	for (int i = 0; i < mat.size(); i++) {
		for (auto v : vectorOfPredictions[i]) {
			matConf[i][v] += 1;
		}
	}
	return matConf;
}
