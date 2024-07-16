#pragma once
#include <string>
#include<Eigen/Dense>
#include <QString>
#include "computations.h"
#include <algorithm>
#include <cstdlib>
#include <QDebug>

#ifndef A_MACHINELEARNING
#define A_MACHINELEARNING


class MachineLearning
{
public:
    //Distance Based Models
    static int kNearestNeighbours(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, Eigen::Vector3d point, int k);
    static int kNearestNeighbours(std::vector<Eigen::MatrixXd> classes, Eigen::VectorXd point, int k);
    static std::pair<std::vector<Eigen::MatrixXd>, Eigen::MatrixXd> Kmeans(const Eigen::MatrixXd data, int k, float threshold); // returns a matrix where each row is a centroid, k centroids so k rows (k = numClasses). 


    class Metrics {
    public:
        static std::vector<std::vector<double>> get_matrixConfusion(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> mat, std::vector<std::vector<int>> vectorOfPredictions);
        static std::vector<Eigen::VectorXd> get_matrixConfusion(std::vector<Eigen::MatrixXd> mat, std::vector<std::vector<int>> vectorOfPredictions);
        static std::vector<std::vector<std::vector<int>>> generatePredictions(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> train, std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> test, int knn);
        static std::vector<std::vector<std::vector<int>>> generatePredictions(std::vector<Eigen::MatrixXd> train, std::vector<Eigen::MatrixXd> test, int knn);

    };


    class CrossValidation {
    public:
        static std::vector<std::vector<Eigen::MatrixXd>> leaveOneOut(std::vector<Eigen::MatrixXd> classes, int clas, int el);
        static std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> leaveOneOut(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes, int clas, int el);
        static std::vector<std::vector<Eigen::MatrixXd>> crossValidation(std::vector<Eigen::MatrixXd> classes);
        static std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> crossValidation(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes);
        static std::vector<std::vector<Eigen::MatrixXd>> Restitucion(std::vector<Eigen::MatrixXd> classes);
        static std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> Restitucion(std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> classes);

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