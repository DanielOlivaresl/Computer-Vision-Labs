#include "Layers.h"

Dense::Dense(Eigen::VectorXd inputDimension, int numNeurons, std::string activationFunctions)
{

	int cols = inputDimension(1); //Get the rows of the features

	this->inputDimension = inputDimension;
	this->numNeurons = numNeurons;
	this->weights = Eigen::MatrixXd(numNeurons, cols);


	//Instead of passing an std::function we will create a hashmap where we map each string of valid activation 
	// functions to an std::function for ease of use



}

void Dense::initializeWeights()
{
	//we will initialize the weights with random values
	this->weights.setRandom();




}

std::function<double(double)> Dense::getActivationFunction()
{

	return this->activationFunction;

}
