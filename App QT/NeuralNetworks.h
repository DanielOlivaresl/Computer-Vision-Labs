#ifndef NEURALNETWORKS_H
#define NEURALNETWORKS_H

#include <Eigen/Dense>

class NeuralNetwork {

public:
	NeuralNetwork();
	~NeuralNetwork();

	void addLayer(); //Method to add any Layer Type
	void compile(); 
	void fit(Eigen::MatrixXd);
		
	void predict(Eigen::MatrixXd);

};




#endif // NEURALNETWORK_H
