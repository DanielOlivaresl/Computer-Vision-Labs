#include "machineLearning.h"




std::vector<Eigen::MatrixXd> MachineLearning::Metrics::generatePredictions(std::vector<Eigen::MatrixXd> train, std::vector<Eigen::MatrixXd> test, int knn) {

	//we will create a 3D vector where one dimension is the Method (euclidean,manhalanobis, etc.. .) another dimension is the classes and the last dimension is the actual prediction

	std::vector<Eigen::MatrixXd> predictions(4, Eigen::MatrixXd(test.size(), test[0].size()));

	//now we begin to fill our vector

	int classNumber = 0;
	for (Eigen::MatrixXd clas : test) {
		for (int i = 0; i < clas.rows(); i++) {
			//we begin to fill our vector

			//Euclidean
			Eigen::VectorXd distances1 = Computations::Distances::euclidean(train, clas.row(i));
			int res = Computations::Helper::getClosest(distances1);
			predictions[0](classNumber, i) = res;
			
			//Manhalanobis
			Eigen::VectorXd distances2 = Computations::Distances::manhalanobis(train, clas.row(i));
			res = Computations::Helper::getClosest(distances2);
			predictions[1](classNumber, i) = res;

			//MaxProb
			Eigen::VectorXd distances3 = Computations::Helper::max_prob(train, clas.row(i));
			res = Computations::Helper::getMaxProb(distances3);
			predictions[2](classNumber, i) = res;

			//KNN
			res = kNearestNeighbours(train, clas.row(i), knn);
			predictions[3](classNumber, i) = res;

		}
		classNumber++;
	}

	return predictions;

}



Eigen::MatrixXd MachineLearning::Metrics::get_matrixConfusion(std::vector<Eigen::MatrixXd> mat, Eigen::MatrixXd matrixOfPredictions) {

	Eigen::MatrixXd matConf(mat.size(), mat.size());
	matConf.fill(0);



	//std::vector<std::vector<double>> matConf(mat.size(), std::vector<double>(mat.size(), 0));
	for (int i = 0; i < mat.size(); i++) {
		for (int v : matrixOfPredictions.row(i)) {
			matConf(i,v) += 1;
		}
	}
	return matConf;
}




