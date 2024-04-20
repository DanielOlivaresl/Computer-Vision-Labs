// 
#include <QScatterSeries>
#include <QChartView>
#include <QLineSeries>
#include<QMainWindow>
#include<Eigen/Dense>
#include <QWidget>
#include <QGridLayout>
#include<Qchart>
#include <QchartView>
#include <QString>
#include <QDebug>
#include <QBarSeries>
#include <QBarSet>

#include "computations.h"
class Plots {



public:

    static void scatterPlot(std::vector<std::vector<double>> data);
    static void ConfusionMatrix(std::vector<std::vector<std::vector<double>>> matrices, std::vector<std::string> names);
    static void histogram(std::vector<int> histogram);


};
