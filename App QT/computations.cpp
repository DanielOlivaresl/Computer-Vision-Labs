//File for extra computations needed in the labs

#include"computations.h"
//Distance functions












/**
* @brief Function that calculates the euclidean distance between two points in n dimensions
 * @param Eigen::MatrixXd Matrix that contains all the points which we want to calc the distance
 * @param Eigen::VectorXd vector of N dimensions, that we want to know the distance to
 * @return Vector of distances
 *
 
std::vector<double> euclidean(Eigen::MatrixXd points, Eigen::VectorXd point)
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
Eigen::VectorXd euclidean(std::vector <Eigen::MatrixXd> classes, Eigen::VectorXd point) {

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
*/
/**
 *
 * @brief Helper function that calculates the covariance matrix from a set of data
 * @param Eigen::MatrixXd matrix of n rows and m columns, where the m is the number of features in the set of data, and n is the number of instances of that data
 * @returns Eigen::MatrixXd returns the covariance matrix of dimensions mxm, of the data passed.
Eigen::MatrixXd  calculateCovMatrix(Eigen::MatrixXd data) {
	/*
	Eigen::MatrixXd transposed = data.transpose();
	// Center the data: subtract the mean of each column from all elements in the column
	Eigen::MatrixXd centered = transposed.rowwise() - transposed.colwise().mean();
	// Calculate the covariance matrix
	Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(centered.rows() - 1);
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
int  getClosest(std::vector<double> distances) {
	int min = 0;
	for (int i = 1; i < distances.size(); i++) {
		if (distances.at(i) < distances.at(min)) {
			min = i;
		}
	}
	return min;
}
int getClosest(Eigen::VectorXd distances) {
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
int getMaxProb(std::vector<double> probabilities) {
	int max = 0;
	for (int i = 1; i < probabilities.size(); i++) {
		if (probabilities.at(i) > probabilities.at(max)) {
			max = i;
		}
	}
	return max;
}
int getMaxProb(Eigen::VectorXd probabilities) {
	int max = 0;
	for (int i = 1; i < probabilities.cols(); i++) {
		if (probabilities(i) > probabilities(max)) {
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

Eigen::VectorXd manhalanobis(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point) {

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


		Eigen::MatrixXd cov = calculateCovMatrix(classes.at(i));

		//The point minus the centroid of the current class
		Eigen::VectorXd x_minus_mu = point - centroids.at(i);

		Eigen::MatrixXd inv_cov = cov.inverse();



		Eigen::MatrixXd left_term = x_minus_mu.transpose() * inv_cov;
		Eigen::MatrixXd res = left_term * x_minus_mu;



		distances(i) = res(0, 0);

	}

	return distances;

}






/**
 * @brief function that calculates the probabilities of a point belonging to a set of classes
 * @param std::vector<Eigen::Matrix<double, Eigen::Dynamic,3>> classes: vector of matrices of size nx3 where each matrix represents a class
 * @param Eigen::Vector3d point point that will be used to calculate the probabilities that it belogns to each of the classes passed
 * @returns std::vector<double> vector of probabilites, that the point belongs to each of the classes
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

Eigen::VectorXd max_prob(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point) {

	Eigen::VectorXd manhalanobis_distance = manhalanobis(classes, point);

	Eigen::VectorXd probabilites(classes.size());

	for (int i = 0; i < classes.size(); i++) {
		Eigen::MatrixXd cov = calculateCovMatrix(classes.at(i));
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



//int kNearestNeighbours(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point, int k) {
//	//Vector that stores a set of vectors that have the distance to the point, this is because each index in the vector represents a class,
//	//This means that each element in the vector is another vector that stores the distances to the point from the ith class
//	std::vector<std::vector<double>> distances;
//
//	//We iterate the classes and fill the vectors
//	for (int i = 0; i < classes.size(); i++) {
//		std::vector<double> currClass;
//		//We first iterate the current class and convert each matrix element into a point
//		for (int j = 0; j < classes.at(i).rows(); j++) {
//			Eigen::Vector3d currPoint = classes.at(i).row(j); //We assign the point
//
//			currClass.push_back(euclideanDistance(point, currPoint));
//
//
//
//		}
//		//we sort the points before adding them
//		std::sort(currClass.begin(), currClass.end());
//		//we add the class to the vector
//		distances.push_back(currClass);
//	}
//
//
//	//Now that are points are stored and sorted we will begin to check which are the k nearest neighbours, to do this we will first, from each class we will
//	//only keep the k first elements because in the worst case the k nearest neighbours are from only one class
//
//	for (int i = 0; i < distances.size(); i++) {
//		if (distances.at(i).size() > k) {
//			distances.at(i).erase(distances.at(i).begin() + k, distances.at(i).end());
//		}
//	}
//	//Now that each class has their k nearest neighbours we will see which ones are the k closest in general, to do this we will iterate each class for their first
//	//element and when we find the one that is the smallest, we will add it to our solution vector and remove that element from the class vector, and repeat this
//	//process until the solution vector is of size k
//
//	std::vector<int> solution;
//
//	while (solution.size() < k) {
//		//We iterate the classes
//		int mindist = 0; //We assume that the first class is the smallest distance and iterate from there
//		for (int i = 1; i < distances.size(); i++) {
//			if (distances.at(i).at(0) < distances.at(mindist).at(0)) {
//				mindist = i;
//			}
//
//		}
//
//		//Now we will add that class to the solution and remove that element
//		solution.push_back(mindist);
//		distances.at(mindist).erase(distances.at(mindist).begin());
//
//	}
//
//	//Now we have the classes of the k nearest neighbours, we will now see what class dominates and make a prediction
//
//	//We will first iterate the classes and check the ocurrence the one with the most occurence will be our final result
//
//	//We will also first assume that the class 0 is the initial prediction and adjust from there
//	int res = 0;
//	int count = std::count(solution.begin(), solution.end(), res);
//	for (int i = 1; i < classes.size(); i++) {
//		int currCount = std::count(solution.begin(), solution.end(), i);
//		if (currCount > count) {
//			res = i;
//			count = currCount;
//		}
//	}
//	//The result is the classification prediction
//	return res;
//
//}
//
//int kNearestNeighbours(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point, int k) {
//	//Vector that stores a set of vectors that have the distance to the point, this is because each index in the vector represents a class,
//	//This means that each element in the vector is another vector that stores the distances to the point from the ith class
//	std::vector<Eigen::VectorXd> distances;
//
//	//We iterate the classes and fill the vectors
//	for (int i = 0; i < classes.size(); i++) {
//		Eigen::VectorXd currClass(classes.at(i).rows());
//		//We first iterate the current class and convert each matrix element into a point
//		for (int j = 0; j < classes.at(i).rows(); j++) {
//			Eigen::VectorXd currPoint = classes.at(i).row(j); //We assign the point
//
//			currClass(j) = euclideanDistance(point, currPoint);
//
//
//
//		}
//		//we sort the points before adding them
//		std::sort(currClass.begin(), currClass.end());
//		//we add the class to the vector
//		distances.push_back(currClass);
//	}
//
//
//	//Now that are points are stored and sorted we will begin to check which are the k nearest neighbours, to do this we will first, from each class we will
//	//only keep the k first elements because in the worst case the k nearest neighbours are from only one class
//
//	for (int i = 0; i < distances.size(); i++) {
//		if (distances.at(i).size() > k) {
//			distances.at(i).conservativeResize(k);
//		}
//	}
//	//Now that each class has their k nearest neighbours we will see which ones are the k closest in general, to do this we will iterate each class for their first
//	//element and when we find the one that is the smallest, we will add it to our solution vector and remove that element from the class vector, and repeat this
//	//process until the solution vector is of size k
//
//	Eigen::VectorXi solution(k);
//	int KCount = 0;
//
//	while (solution.size() < k) {
//		//We iterate the classes
//		int mindist = 0; //We assume that the first class is the smallest distance and iterate from there
//		for (int i = 1; i < distances.size(); i++) {
//			if (distances.at(i)(0) < distances.at(mindist)(0)) {
//				mindist = i;
//			}
//
//		}
//
//		//Now we will add that class to the solution and remove that element
//		solution(KCount) = mindist;
//		distances[mindist] = distances[mindist].segment(1, distances[mindist].size() - 1);;
//
//	}
//
//	//Now we have the classes of the k nearest neighbours, we will now see what class dominates and make a prediction
//
//	//We will first iterate the classes and check the ocurrence the one with the most occurence will be our final result
//
//	//We will also first assume that the class 0 is the initial prediction and adjust from there
//	int res = 0;
//	int count = std::count(solution.begin(), solution.end(), res);
//	for (int i = 1; i < classes.size(); i++) {
//		int currCount = std::count(solution.begin(), solution.end(), i);
//		if (currCount > count) {
//			res = i;
//			count = currCount;
//		}
//	}
//	//The result is the classification prediction
//	return res;
//
//}
//
//



// 50/50 split

std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> MachineLearning::CrossValidation::crossValidation(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes) {

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


// 50/50 split
std::vector<std::vector<Eigen::MatrixXd>> crossValidation(std::vector<Eigen::MatrixXd> classes) {
	// We will create vectors of matrices for the test and training sets
	std::vector<Eigen::MatrixXd> testSplit;
	std::vector<Eigen::MatrixXd> trainingSplit;

	// Seed the random number generator to ensure different results each run
	std::srand(std::time(nullptr));

	// Iterate over the original dataset
	for (const Eigen::MatrixXd& matrix : classes) {
		Eigen::MatrixXd test;
		Eigen::MatrixXd train;

		// We will randomly select indexes to add to the test set, until we have enough
		std::vector<int> indexes;

		while (indexes.size() < (matrix.rows() / 2)) {
			int index = std::rand() % matrix.rows();

			// Check that the index isn't already in the list
			while (std::count(indexes.begin(), indexes.end(), index)) {
				index = std::rand() % matrix.rows();
			}

			// Now that we know the element isn't in the list we add it
			indexes.push_back(index);
		}

		// We will now fill the test and training sets
		for (int i = 0; i < matrix.rows(); i++) {
			if (std::count(indexes.begin(), indexes.end(), i)) {
				test.conservativeResize(test.rows() + 1, Eigen::NoChange);
				test.row(test.rows() - 1) = matrix.row(i);
			}
			else {
				train.conservativeResize(train.rows() + 1, Eigen::NoChange);
				train.row(train.rows() - 1) = matrix.row(i);
			}
		}

		testSplit.push_back(test);
		trainingSplit.push_back(train);
	}

	// Final result containing test and training splits
	std::vector<std::vector<Eigen::MatrixXd>> result = { testSplit, trainingSplit };
	return result;
}
std::vector<std::vector<Eigen::MatrixXd>> leaveOneOut(std::vector<Eigen::MatrixXd> classes, int clas, int el) {
	std::vector<Eigen::MatrixXd> testSplit;
	std::vector<Eigen::MatrixXd> trainingSplit;

	for (int i = 0; i < classes.size(); i++) {
		Eigen::MatrixXd trainMatrix;

		for (int j = 0; j < classes.at(i).rows(); j++) {
			if (i == clas && j == el) {
				// Correctly handling the creation of a matrix from a single row
				Eigen::MatrixXd singleRowMatrix = classes.at(i).row(j);
				testSplit.push_back(singleRowMatrix);
			}
			else {
				// Ensuring we resize only if needed to add rows
				trainMatrix.conservativeResize(trainMatrix.rows() + 1, Eigen::NoChange);
				trainMatrix.row(trainMatrix.rows() - 1) = classes.at(i).row(j);
			}
		}

		if (trainMatrix.rows() > 0) { // Ensure we only add non-empty matrices
			trainingSplit.push_back(trainMatrix);
		}
	}

	std::vector<std::vector<Eigen::MatrixXd>> result = { testSplit, trainingSplit };
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
std::vector<std::vector<Eigen::MatrixXd>> Restitucion(std::vector<Eigen::MatrixXd> classes) {
	// Directly use the input classes as both test and training splits
	std::vector<Eigen::MatrixXd> testSplit = classes;
	std::vector<Eigen::MatrixXd> trainingSplit = classes;

	// Pack both splits into a single result vector
	std::vector<std::vector<Eigen::MatrixXd>> result = { testSplit, trainingSplit };

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
std::vector<std::vector<std::vector<int>>> generatePredictions(std::vector<Eigen::MatrixXd> train, std::vector<Eigen::MatrixXd> test, int knn) {

	//we will create a 3D vector where one dimension is the Method (euclidean,manhalanobis, etc.. .) another dimension is the classes and the last dimension is the actual prediction

	std::vector<std::vector<std::vector<int>>> predictions(4, std::vector<std::vector<int>>(test.size(), std::vector<int>(test.at(0).rows())));

	//now we begin to fill our vector

	int classNumber = 0;
	for (Eigen::MatrixXd clas : test) {
		for (int i = 0; i < clas.rows(); i++) {
			//we begin to fill our vector
			//Euclidean
			Eigen::VectorXd distances1 = euclidean(train, clas.row(i));
			int res = getClosest(distances1);
			predictions.at(0).at(classNumber).at(i) = res;

			//Manhalanobis
			Eigen::VectorXd  distances2 = manhalanobis(train, clas.row(i));
			res = getClosest(distances2);
			predictions.at(1).at(classNumber).at(i) = res;

			//MaxProb
			Eigen::VectorXd distances3 = max_prob(train, clas.row(i));
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



void normalizeColumn(Eigen::MatrixXd& matrix, int column) {
	// getting the col
	Eigen::VectorXd col = matrix.col(column);

	// min val and max val 
	double minVal = col.minCoeff();
	double maxVal = col.maxCoeff();

	// range in (0,1)
	matrix.col(column) = (col.array() - minVal) / (maxVal - minVal);
}
int clockwise(int dir)
{
	return (dir % 8) + 1;
}
int counterclockwise(int dir)
{
	return (dir + 6) % 8 + 1;
}
void fft1D(std::vector<std::complex<double>>& data,bool invert)
{


	int n = data.size();
	if (n <= 1) {
		return;
	}

	std::vector<std::complex<double>> even(n / 2);
	std::vector<std::complex<double>> odd(n / 2);


	for (int i = 0; i < n/2; i++) {
		even[i] = data[i * 2];
		odd[i] = data[i * 2 + 1];
	}

	fft1D(even,invert);
	fft1D(odd,invert);

	double ang = 2 * M_PI / n * (invert ? -1 : 1);
	std::complex<double> w(1), wn(cos(ang), sin(ang));

	for (int i = 0; i < n / 2; ++i) {
		data[i] = even[i] + w * odd[i];
		data[i + n / 2] = even[i] - w * odd[i];
		if (invert) {
			data[i] /= 2;
			data[i + n / 2] /= 2;
		}
		w *= wn;
	}



}

std::vector<std::vector<std::complex<double>>> fft2D(const std::vector<std::vector<std::complex<double>>>& data,bool inverted)
{

	int height = data.size();
	int width = data[0].size();

	std::vector<std::vector<std::complex<double>>> result = data;
	//Apply fft to rows
	for (int y = 0; y < height; ++y) {
		fft1D(result[y],inverted);
	}

	//Transpose the result
	std::vector<std::vector<std::complex<double>>> transposed(width, std::vector<std::complex<double>>(height));

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			transposed[x][y] = result[y][x];
		}
	}


	// Apply FFT to columns (transposed rows)
	for (size_t x = 0; x < width; ++x) {
		fft1D(transposed[x],inverted);
	}

	// Transpose the result back
	for (size_t y = 0; y < height; ++y) {
		for (size_t x = 0; x < width; ++x) {
			result[y][x] = transposed[x][y];
		}
	}

	return result;




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


std::vector<Eigen::VectorXd> get_matrixConfusion(std::vector<Eigen::MatrixXd> mat, std::vector<std::vector<int>> vectorOfPredictions) {


	std::vector<Eigen::VectorXd> matConf(mat.size(), Eigen::VectorXd::Zero(mat.size()));
	for (int i = 0; i < mat.size(); i++) {
		for (int v : vectorOfPredictions[i]) {
			matConf[i][v] += 1;
		}
	}
	return matConf;
}


std::vector<int> ObjectMetrics::calculateArea(QVector<QPoint> points, QImage& image)
{

	std::sort(points.begin(), points.end(), [](const QPoint& a, const QPoint& b) {
		return a.x() < b.x();
		});

	//we obtain the marginals of each pixel
	std::map<int, QVector<QPoint>> clusters;
	for (const auto& points : points) {
		clusters[points.y()].push_back(points);
	}

	//We get the area
	int area = 0;
	for (const auto& cluster : clusters) {

		for (int j = 0; j < cluster.second.size() - 1; j++) {
			//area += abs(cluster.second[j + 1].x() - cluster.second[j].x());

			if (abs(cluster.second[j + 1].x() - cluster.second[j].x()) == 1) {
				area++;

			}
			else {

				area += abs(cluster.second[j + 1].x() - cluster.second[j].x()) + 1;
			}
		}
	}
	return std::vector<int>(1, area);
}


std::vector<int> ObjectMetrics::calculatePerimeter(QVector<QPoint> object, QImage& image)
{
	int res = object.size();

	std::vector<int> resultingMet;
	resultingMet.push_back(res);
	return resultingMet;
}

std::vector<int> ObjectMetrics::calculateCenterOfGravity(QVector<QPoint> object, QImage& image)
{
	std::vector<int> centerOfGravity(2);

	int moment0 = 0;
	int moment10 = 0;
	int moment01 = 0;

	for (int j = 0; j < object.size(); j++) { //We will iterate the points of the current object
		moment0 += 1;
		moment10 += object[j].x(); //x spatial distribution
		moment01 += object[j].y(); //y spatial distribution

	}

	centerOfGravity[0] = static_cast<int>(moment10 / moment0);
	centerOfGravity[1] = static_cast<int>(moment01 / moment0);



	return centerOfGravity;
}

Eigen::MatrixXd ObjectMetrics::featureExtraction(std::vector<std::function<std::vector<int>(QVector<QPoint>, QImage&)>> functions, QVector<QPoint> object, QImage& image)
{
	int minX = object[0].x();
	int minY = object[0].y();
	int maxX = object[0].x();
	int maxY = object[0].y();
	// Find the bounding box of the object
	for (const QPoint& point : object) {
		if (point.x() < minX) {
			minX = point.x();
		}
		if (point.x() > maxX) {
			maxX = point.x();
		}
		if (point.y() < minY) {
			minY = point.y();
		}
		if (point.y() > maxY) {
			maxY = point.y();
		}
	}

	// Create a new image containing only the object
	QImage objectImage = image.copy(minX - 5, minY - 5, maxX - minX + 10, maxY - minY + 10);
	//QImage objectImageBinary = thereshold(objectImage, 130);
	double e = ObjectMetrics::calculateEccentricity((objectImage));
	Eigen::MatrixXd descriptors(1, functions.size() + 1 + 1);
	int metricCount = 0;
	for (auto func : functions) {



		std::vector<int> currMetric = func(object, image);

		//we iterate the current metrics values

		if (currMetric.size() > 1) {
			descriptors.conservativeResize(descriptors.rows(), descriptors.cols() + currMetric.size() - 1);
		}

		for (auto metric : currMetric) {
			descriptors(0, metricCount) = metric;
			metricCount++;
		}
	}// area, per
	descriptors(0, metricCount) = e;
	double c = imageMoments(objectImage,0,0);
	descriptors(0, metricCount + 1) = c;
	return descriptors;
}

double ObjectMetrics::imageMoments(QImage& binarizedImage, int p, int q)
{
	double res = 0;
	for (int y = 0; y < binarizedImage.height(); y++) {
		for (int x = 0; x < binarizedImage.width(); x++) {
			int pixelVal = qGray(binarizedImage.pixel(x, y));
			if (pixelVal == 255) { // Asume que la imagen está binarizada con fondo negro (0) y objetos blancos (255)
				res += pow(x, q) * pow(y, p);
			}
		}
	}
	return res;
}

double ObjectMetrics::centralMoments(QImage& binarizedImage, int p, int q, double xCent, double yCent)
{
	double res = 0;
	for (int y = 0; y < binarizedImage.height(); y++) {
		for (int x = 0; x < binarizedImage.width(); x++) {
			int pixelVal = qGray(binarizedImage.pixel(x, y));
			if (pixelVal == 255) { // Asume que la imagen está binarizada con fondo negro (0) y objetos blancos (255)
				res += pow(x - xCent, p) * pow(y - yCent, q);
			}
		}
	}
	return res;
}

double ObjectMetrics::calculateEccentricity(QImage& image)
{
	double m00 = imageMoments(image, 0, 0);
	double m10 = imageMoments(image, 1, 0);
	double m01 = imageMoments(image, 0, 1);

	double cx = m10 / m00;
	double cy = m01 / m00;

	double mu20 = centralMoments(image, 2, 0, cx, cy);
	double mu02 = centralMoments(image, 0, 2, cx, cy);
	double mu11 = centralMoments(image, 1, 1, cx, cy);

	double a = sqrt(2 * (mu20 + mu02 + sqrt(pow(mu20 - mu02, 2) + 4 * pow(mu11, 2))));
	double b = sqrt(2 * (mu20 + mu02 - sqrt(pow(mu20 - mu02, 2) + 4 * pow(mu11, 2))));

	double c = sqrt(a * a - b * b);
	double e = c / a;

	return 200.000*e;
}



 */
