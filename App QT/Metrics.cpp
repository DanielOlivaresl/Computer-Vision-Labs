#include "machineLearning.h"




std::vector<std::vector<std::vector<int>>> MachineLearning::Metrics::generatePredictions(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> train, std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> test, int knn) {

	//we will create a 3D vector where one dimension is the Method (euclidean,manhalanobis, etc.. .) another dimension is the classes and the last dimension is the actual prediction

	std::vector<std::vector<std::vector<int>>> predictions(4, std::vector<std::vector<int>>(test.size(), std::vector<int>(test.at(0).rows())));

	//now we begin to fill our vector

	int classNumber = 0;
	for (Eigen::Matrix<double, Eigen::Dynamic, 3> clas : test) {
		for (int i = 0; i < clas.rows(); i++) {
			//we begin to fill our vector

			//Euclidean
			std::vector<double> distances1 = Computations::Distances::euclidean(train, clas.row(i));
			int res = Computations::Helper::getClosest(distances1);
			predictions.at(0).at(classNumber).at(i) = res;

			//Manhalanobis
			std::vector<double> distances2 = Computations::Distances::manhalanobis(train, clas.row(i));
			res = Computations::Helper::getClosest(distances2);
			predictions.at(1).at(classNumber).at(i) = res;

			//MaxProb
			std::vector<double> distances3 = Computations::Helper::max_prob(train, clas.row(i));
			res = Computations::Helper::getMaxProb(distances3);
			predictions.at(2).at(classNumber).at(i) = res;
			//KNN
			res = kNearestNeighbours(train, clas.row(i), knn);
			predictions.at(3).at(classNumber).at(i) = res;
		}
		classNumber++;
	}

	return predictions;

}
std::vector<std::vector<std::vector<int>>> MachineLearning::Metrics::generatePredictions(std::vector<Eigen::MatrixXd> train, std::vector<Eigen::MatrixXd> test, int knn) {

	//we will create a 3D vector where one dimension is the Method (euclidean,manhalanobis, etc.. .) another dimension is the classes and the last dimension is the actual prediction

	std::vector<std::vector<std::vector<int>>> predictions(4, std::vector<std::vector<int>>(test.size(), std::vector<int>(test.at(0).rows())));

	//now we begin to fill our vector

	int classNumber = 0;
	for (Eigen::MatrixXd clas : test) {
		for (int i = 0; i < clas.rows(); i++) {
			//we begin to fill our vector
			//Euclidean
			Eigen::VectorXd distances1 = Computations::Distances::euclidean(train, clas.row(i));
			int res = Computations::Helper::getClosest(distances1);
			predictions.at(0).at(classNumber).at(i) = res;

			//Manhalanobis
			Eigen::VectorXd  distances2 = Computations::Distances::manhalanobis(train, clas.row(i));
			res = Computations::Helper::getClosest(distances2);
			predictions.at(1).at(classNumber).at(i) = res;

			//MaxProb
			Eigen::VectorXd distances3 = Computations::Helper::max_prob(train, clas.row(i));
			res = Computations::Helper::getMaxProb(distances3);
			predictions.at(2).at(classNumber).at(i) = res;
			//KNN
			res = kNearestNeighbours(train, clas.row(i), knn);
			predictions.at(3).at(classNumber).at(i) = res;




		}
		classNumber++;
	}

	return predictions;

}





std::vector<std::vector<double>> MachineLearning::Metrics::get_matrixConfusion(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> mat, std::vector<std::vector<int>> vectorOfPredictions) {


	std::vector<std::vector<double>> matConf(mat.size(), std::vector<double>(mat.size(), 0));
	for (int i = 0; i < mat.size(); i++) {
		for (auto v : vectorOfPredictions[i]) {
			matConf[i][v] += 1;
		}
	}
	return matConf;
}


std::vector<Eigen::VectorXd> MachineLearning::Metrics::get_matrixConfusion(std::vector<Eigen::MatrixXd> mat, std::vector<std::vector<int>> vectorOfPredictions) {


	std::vector<Eigen::VectorXd> matConf(mat.size(), Eigen::VectorXd::Zero(mat.size()));
	for (int i = 0; i < mat.size(); i++) {
		for (int v : vectorOfPredictions[i]) {
			matConf[i][v] += 1;
		}
	}
	return matConf;
}


