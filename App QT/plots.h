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
#include "computations.h"
#include "machineLearning.h"
class Plots {



public:
    static void matrixPlot2D(Eigen::MatrixXd values,std::string col1Name, std::string col2Name);
    static void scatterPlot(std::vector<std::vector<double>> data);
    static void ConfusionMatrix(std::vector<std::vector<std::vector<double>>> matrices, std::vector<std::string> names);
    static void histogram(std::vector<int> histogram);
    static void plotPerceptrons(std::vector<Perceptron> perceptrons);

};
