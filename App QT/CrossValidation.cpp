#include "machineLearning.h"


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
std::vector<std::vector<Eigen::MatrixXd>> MachineLearning::CrossValidation::crossValidation(std::vector<Eigen::MatrixXd> classes) {
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

std::vector<std::vector<Eigen::MatrixXd>> MachineLearning::CrossValidation::leaveOneOut(std::vector<Eigen::MatrixXd> classes, int clas, int el) {
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
std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> MachineLearning::CrossValidation::leaveOneOut(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, int clas, int el) {
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
std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> MachineLearning::CrossValidation::Restitucion(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes) {
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

