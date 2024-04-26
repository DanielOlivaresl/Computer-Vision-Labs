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
class Plots {



public:

    static void scatterPlot(std::vector<std::vector<double>> data);
    static void ConfusionMatrix(std::vector<std::vector<std::vector<double>>> matrices, std::vector<std::string> names);



};
