#include "machineLearning.h"



/**
* @brief Function that creates a cross validation split (50/50)
 * @param std::vector<Eigen::MatrixXd> Vector of matrices that represents each class
 * @return std::vector<std::vector<Eigen::MatrixXd>> std::vector of size 2, where the first element is the training split, and the second is the testing split
 *
 */
std::vector<std::vector<Eigen::MatrixXd>> MachineLearning::CrossValidation::crossValidation(std::vector<Eigen::MatrixXd> classes) {


	//We will first create two std::vectors that contain matrices that represent each class, these two vectors represent our training/test splits

	std::vector<Eigen::MatrixXd> testSplit; 
	std::vector<Eigen::MatrixXd> trainSplit;

	
	
	//we will have train and test splits of the same size, and we will fill them randomly 


	//To do this, we will first iterate the classes

	for (Eigen::MatrixXd currentClassMatrix : classes) {
		//We will create variables for the train, and test data

		Eigen::MatrixXd train, test;


		//We will now randomly select the indexes to fill these

		std::vector<int> indexes;

		while (indexes.size() < (currentClassMatrix.rows() / 2)) {
			int index = std::rand() % currentClassMatrix.rows();

			//We check that the element isn't in the index list, if it is we will recalculate the index until we find an index that is unvisited (unadded)

			while (std::count(indexes.begin(), indexes.end(), index)) {
				index = std::rand() % currentClassMatrix.rows();
			}

			//Now that we have a valid index we will add it to the list
			indexes.push_back(index);


		}

		//Now that are indexes are set, we will fill our matrices

		for (int i = 0; i < currentClassMatrix.rows(); i++) {
			if (std::count(indexes.begin(), indexes.end(), i)) {
				test.conservativeResize(test.rows() + 1, Eigen::NoChange);
				test.row(test.rows() - 1) << currentClassMatrix.row(i);
			}
			else {
				train.conservativeResize(train.rows() + 1, Eigen::NoChange);
				train.row(train.rows() - 1) << currentClassMatrix.row(i);
			}
		}

		//we add the current class split to the final data
		testSplit.push_back(test);
		trainSplit.push_back(train);
	}






	return {testSplit,trainSplit};
	
}







/**
* @brief Function that creates a leave one out split (n-1/ 1)
 * @param std::vector<Eigen::MatrixXd> Vector of matrices that represents each class
 * @param int clas: the current class that the element that will be leaved out belongs to
 * @param int el: the index of the element that will be leaved out 
 * @return std::vector<std::vector<Eigen::MatrixXd>> std::vector of size 2, where the first element is the training split, and the second is the testing split
 *
 */
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

/**
* @brief Function that creates a restitution split (n/n) 
 * @param std::vector<Eigen::MatrixXd> Vector of matrices that represents each class
 * @return std::vector<std::vector<Eigen::MatrixXd>> std::vector of size 2, where the first element is the training split, and the second is the testing split
 *
 */
std::vector<std::vector<Eigen::MatrixXd>> MachineLearning::CrossValidation::Restitucion(std::vector<Eigen::MatrixXd> classes) {
	std::vector<Eigen::MatrixXd> testSplit = classes;
	std::vector<Eigen::MatrixXd> trainingSplit = classes;

	std::vector<std::vector< Eigen::MatrixXd>> result = { testSplit,trainingSplit };

	return result;



}
