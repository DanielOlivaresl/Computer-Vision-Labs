// 
#include<QtCharts/qscatterseries.h>
#include <QtCharts/QchartView.h>
#include <QtCharts/qlineseries.h>
#include<QMainWindow>
#include<Eigen/Dense>
#include <QWidget>
#include <QGridLayout>
#include<QtCharts/qchart.h>
#include <QString>
#include <QDebug>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include <QtDataVisualization/Q3DScatter>
#include <QtDataVisualization/QScatterDataProxy>
#include <QtDataVisualization/QScatter3DSeries>
#include <QWidget>
#include <QColor>
#include <QDir>
#include <QImage>
#include "computations.h"
#include "machineLearning.h"


#ifndef A_PLOTS
#define A_PLOTS

class Plots {

    

public:

    static void matrixPlot3D_labels(Eigen::MatrixXd values, std::vector<std::string> labels, std::string col1Name, std::string col2Name, std::string col3Name); // plots a 3d matrix data where each entry in the labels vector is the label of each row
    static void plotMatrix(Eigen::MatrixXd matrixData, std::vector<std::string> colNames); // used to handle the plotMatrix diferent procedures
    static void matrixPlot2D(Eigen::MatrixXd values, std::string col1Name, std::string col2Name); // simple plotting 
    static void matrixPlot3D_subImages(Eigen::MatrixXd values, std::vector<std::string> labels, std::string col1Name, std::string col2Name, std::string col3Name);// labels es el identificador de cada punto, 
    static void scatterPlot(std::vector<std::vector<double>> data); // go watch to do... 
    static void ConfusionMatrix(std::vector<std::vector<std::vector<double>>> matrices, std::vector<std::string> names); // go watch to do...
    static void histogram(std::vector<int> histogram);
    static void plotPerceptrons(std::vector<Perceptron> perceptrons);
    static void plotMatrixClasses(std::vector<Eigen::MatrixXd> mc); // Cada indice del vector es una clase de los datos, plotea con diferentes colores cada matrices de datos
};







#endif