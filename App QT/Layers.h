#pragma once
#include <Eigen/Dense>
#include "computations.h"

class Layer {
public:
	
	virtual ~Layer() = default; //Default destructor

	Eigen::VectorXd inputDimension; //We must specify the dimensions of the input, because the dimension of the 
	//input controls the size of the Weights matrix, because the cols of the weights matrix, has to be the same size
	// as the features.
	int numNeurons;
	Eigen::MatrixXd weights; //For most layer types this will be of size n_features x n_neurons
	virtual void initializeWeights()=0; //virtual method that makes the class abstract (it can't be instantiated, only it's subclasses)



};


class Dense : public Layer {
public:

	Dense(Eigen::VectorXd inputDimension,int numNeurons, std::string activationFunction);

	void initializeWeights() override;

	std::function<double(double)> getActivationFunction();


private:
	std::function<double(double)> activationFunction;


};

class SoftMax : Dense {

public:

	std::function<Eigen::VectorXd(Eigen::VectorXd)> activationFunction;
	
	SoftMax(Eigen::VectorXd inputDimension, int numNeurons): 
		Dense(inputDimension,numNeurons, "SoftMax"),
		activationFunction(Computations::ActivationFunctions::softmax) {}


};