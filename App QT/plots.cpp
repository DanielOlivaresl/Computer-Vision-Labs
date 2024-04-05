#include "plots.h"	
#include <memory>
#include<QtCharts/QValueAxis>
void Plots::scatterPlot(std::vector<std::vector<double>> data) {
    // Check to avoid out of range errors
    std::vector<std::string> names = { "Restitucion", "Leave one Out", "Cross Validation" };
    if (data.size() > names.size()) {
        // Handle error: The data vector has more elements than there are names
        return;
    }

    // Create a chart
    QtCharts::QChart* chart = new QtCharts::QChart();

    // Create a single set of axes for the chart
    QtCharts::QValueAxis* axisX = new QtCharts::QValueAxis(chart);
    QtCharts::QValueAxis* axisY = new QtCharts::QValueAxis(chart);
    axisX->setTitleText("X Axis");
    axisY->setTitleText("Y Axis");
    axisX->setRange(-1, 5); // Set X axis range, adjust if needed
    axisY->setRange(0, 1);  // Set Y axis range, adjust if needed
    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);

    // Create and add scatter and line series to the chart
    for (size_t i = 0; i < data.size(); ++i) {
        // Create scatter series
        QtCharts::QScatterSeries* scatter = new QtCharts::QScatterSeries(chart);
        scatter->setName(QString::fromStdString(names[i]));
        for (size_t j = 0; j < data[i].size(); ++j) {
            scatter->append(j, data[i][j]);
        }
        chart->addSeries(scatter);
        scatter->attachAxis(axisX);
        scatter->attachAxis(axisY);

        // Create line series
        QtCharts::QLineSeries* line = new QtCharts::QLineSeries(chart);
        line->setName("Line " + QString::fromStdString(names[i])); // Unique name for each line series
        for (size_t j = 0; j < data[i].size(); ++j) {
            line->append(j, data[i][j]);
        }
        chart->addSeries(line);
        line->attachAxis(axisX);
        line->attachAxis(axisY);
    }

    // Create a chart view, set the chart, and display the chart view
    QtCharts::QChartView* chartView = new QtCharts::QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    chartView->resize(400, 300);
    chartView->show();
}
