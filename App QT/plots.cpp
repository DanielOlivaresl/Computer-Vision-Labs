#include "plots.h"	
#include <memory>
#include<QtCharts/QValueAxis>


void Plots::scatterPlot(std::vector<std::vector<double>> data) {
    std::vector<QtCharts::QLineSeries* > series;
    for (const auto& clas : data) {
        QtCharts::QLineSeries* curr = new QtCharts::QLineSeries();
        curr->setName("Series x");
        for (int i = 0; i < clas.size(); i++) {
            curr->append(i, clas.at(i));
        }
        series.push_back(std::move(curr)); // Move the unique_ptr into the vector
    }

    QtCharts::QChart *chart = new QtCharts::QChart();

    for (const auto& el : series) {
        chart->addSeries(el); // Get the raw pointer from the unique_ptr
    }

    // Create axes and attach them to the chart
    QtCharts::QValueAxis* axisX = new QtCharts::QValueAxis;
    QtCharts::QValueAxis* axisY = new QtCharts::QValueAxis;

    axisX->setTitleText("X Axis");
    axisY->setTitleText("Y Axis");

    chart->addAxis(axisY, Qt::AlignLeft);
    chart->addAxis(axisX, Qt::AlignBottom);
    for (const auto& el : series) {
        el->attachAxis(axisX);
        el->attachAxis(axisY);
    }


    // Create a chart view and set the chart
    QtCharts::QChartView* chartView = new QtCharts::QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    // Display the chart view
    chartView->resize(400, 300);
    chartView->show();


   
}
