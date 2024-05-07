#pragma once
#include <string>
#include<Eigen/Dense>
#include <QString>
#include "computations.h"

class ML
{
public:
    Eigen::MatrixXd Kmeans(const Eigen::MatrixXd data, int k ); // returns a matrix where each row is a centroid, k centroids so k rows (k = numClasses). 
    int query_centroids(Eigen::MatrixXd inputToClass); // return a an integer representing the most closest centroid

};


class Perceptron
{
public:
    Perceptron(const Eigen::Matrix<double, Eigen::Dynamic, 3>& X_input, const std::string& F_input);
    Eigen::MatrixXd init_data(const Eigen::MatrixXd& X_input);
    Eigen::MatrixXd init_query(const Eigen::VectorXd& input);
    Eigen::VectorXd init_weights(const int& input_shape);
    Eigen::VectorXd get_h(const Eigen::MatrixXd& inputData);
    Eigen::VectorXd sigmoid(const Eigen::VectorXd& h);
    Eigen::VectorXd sigmoid_prime(const Eigen::VectorXd& h);
    Eigen::VectorXd get_activation(const Eigen::VectorXd& h);
    Eigen::VectorXd feed_forward(const Eigen::MatrixXd& X_data);
    Eigen::VectorXd error(const Eigen::VectorXd& Target);
    Eigen::VectorXd error_term(const Eigen::VectorXd& Target);
    Eigen::VectorXd get_increment(const Eigen::VectorXd& target, const float& learning_rate);
    Eigen::VectorXd update_weights(const Eigen::VectorXd& target, const float& learning_rate);
    Eigen::VectorXd query(const Eigen::VectorXd& x);
    void showInfo();
    Eigen::VectorXd weights;
    Eigen::MatrixXd inputData;
private:
    Eigen::VectorXd inputMean;
    Eigen::VectorXd inputStd;
    
    std::string activationFunction;
    

};

