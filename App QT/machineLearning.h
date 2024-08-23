#pragma once
#include <string>
#include<Eigen/Dense>
#include <QString>
#include <algorithm>
#include <cstdlib>
#include <QDebug>

#include "computations.h"
#include "Layers.h"



#ifndef A_MACHINELEARNING
#define A_MACHINELEARNING


class MachineLearning
{
public:

    //Distance Based Models

    
    static int kNearestNeighbours(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point, int k);
    static std::pair<std::vector<Eigen::MatrixXd>, Eigen::MatrixXd> Kmeans(const Eigen::MatrixXd data, int k, float threshold); // returns a matrix where each row is a centroid, k centroids so k rows (k = numClasses). 

    //static void gradientDescent(Eigen::MatrixXd data, Eigen::MatrixXd weights, std::function<double(Eigen::MatrixXd)>& loss);

    static void initializeWeights(std::vector<Eigen::MatrixXd>& weights);
    static Eigen::VectorXd forwardPass(Eigen::MatrixXd data, std::vector<Layer>& layers,std::function<double(Eigen::VectorXd)> l, Eigen::VectorXd target);
    static Eigen::VectorXd backwardPass(Eigen::VectorXd lossGradient,std::vector<Layer>& layers, std::vector<Eigen::MatrixXd> preActivationValues, double alpha = 0.000001);


    class Metrics {
    public:
        static Eigen::MatrixXd get_matrixConfusion(std::vector<Eigen::MatrixXd> mat, Eigen::MatrixXd matrixOfPredictions);

        //Check structure of output before reformatting (Do this next)
        static std::vector<std::vector<std::vector<int>>> generatePredictions(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> train, std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> test, int knn);
        static std::vector<Eigen::MatrixXd> generatePredictions(std::vector<Eigen::MatrixXd> train, std::vector<Eigen::MatrixXd> test, int knn);

    };


    class CrossValidation {
    public:
        static std::vector<std::vector<Eigen::MatrixXd>> leaveOneOut(std::vector<Eigen::MatrixXd> classes, int clas, int el);
        static std::vector<std::vector<Eigen::MatrixXd>> crossValidation(std::vector<Eigen::MatrixXd> classes);
        static std::vector<std::vector<Eigen::MatrixXd>> Restitucion(std::vector<Eigen::MatrixXd> classes);

    };



};






class Perceptron
{
public:

    //Fix

    Perceptron(const Eigen::MatrixXd& X_input);// dado todo el conjunto de datos crea un vector de labels iniciado en 0's
    Perceptron(const std::vector<Eigen::MatrixXd> classes, int class_to_clasiffy); // dado un vector de clases, y el indice de la clase a clasificar, crea un vector de labels con 0's para las demas clases y 1's para la clase corerecta
    Eigen::MatrixXd init_data(const Eigen::MatrixXd inputData);
    Eigen::MatrixXd init_data_classes(const std::vector<Eigen::MatrixXd> classes);
    Eigen::VectorXd init_labels_classes(const std::vector<Eigen::MatrixXd> classes, int class_to_clasiffy);
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
    Eigen::VectorXd labels;
private:
    Eigen::VectorXd inputMean;
    Eigen::VectorXd inputStd;
    Eigen::VectorXd minVal, maxVal;

};












#endif