#include "plots.h"	
#include <memory>
#include<QtCharts/QValueAxis>
#include <QLabel>
//using namespace QtCharts;
void Plots::scatterPlot(std::vector<std::vector<double>> data) {
    // Check to avoid out of range errors
    std::vector<std::string> names = { "Restitucion", "Leave one Out", "Cross Validation" };
    if (data.size() > names.size()) {
        // Handle error: The data vector has more elements than there are names
        return;
    }

    // Create a chart
    QChart* chart = new QChart();

    // Create a single set of axes for the chart
    QValueAxis* axisX = new QValueAxis(chart);
    QValueAxis* axisY = new QValueAxis(chart);
    axisX->setTitleText("X Axis");
    axisY->setTitleText("Y Axis");
    axisX->setRange(-1, 5); // Set X axis range, adjust if needed
    axisY->setRange(0, 1);  // Set Y axis range, adjust if needed
    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);

    // Create and add scatter and line series to the chart
    for (size_t i = 0; i < data.size(); ++i) {
        // Create scatter series
        QScatterSeries* scatter = new QScatterSeries(chart);
        scatter->setName(QString::fromStdString(names[i]));
        for (size_t j = 0; j < data[i].size(); ++j) {
            scatter->append(j, data[i][j]);
        }
        chart->addSeries(scatter);
        scatter->attachAxis(axisX);
        scatter->attachAxis(axisY);

        // Create line series
        QLineSeries* line = new QLineSeries(chart);
        //line->setName("Line " + QString::fromStdString(names[i])); // Unique name for each line series
        for (size_t j = 0; j < data[i].size(); ++j) {
            line->append(j, data[i][j]);
        }
        chart->addSeries(line);
        line->attachAxis(axisX);
        line->attachAxis(axisY);
    }

    // Create a chart view, set the chart, and display the chart view
    QChartView* chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::RenderHint::Antialiasing);
    chartView->resize(400, 300);
    chartView->show();
}
void Plots::ConfusionMatrix(std::vector<std::vector<std::vector<double>>> matrices, std::vector<std::string> names)
{
    int n = matrices.size();

    QWidget* container = new QWidget(); // container of layouts
    QGridLayout* layout = new QGridLayout(container);

    int rows = std::sqrt(n);
    int cols = std::ceil((double)n / rows);

    const int sceneWidth = 500;
    const int sceneHeight = 500;
   
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            if (index < n) {

                QGraphicsScene* scene = new QGraphicsScene(); // new instance of a graphicsScene (to draw the matrix)
                scene->setSceneRect(0, 0, sceneWidth, sceneHeight);
                
                int matrixSize = std::min(matrices[index].size(), matrices[index][0].size()); 
                double squareSize = std::min(sceneWidth / cols, sceneHeight / rows) / matrixSize; // size of each rect


                // computes the sum of each row 
                std::vector<double> rowSums(matrices[index].size(), 0.0);
                for (int x = 0; x < matrices[index].size(); ++x) {
                    for (int y = 0; y < matrices[index][x].size(); ++y) {
                        qDebug() << " matrix" << x << "at value" << matrices[index][x][y];
                        rowSums[x] += matrices[index][x][y];
                    }
                }

                for (int x = 0; x < matrices[index].size(); ++x) {
                    for (int y = 0; y < matrices[index][x].size(); ++y) {
                        double squareX = x * squareSize;
                        double squareY = y * squareSize;

                        // computes the alpha value for each rect in the row
                        float alpha;
                        if (matrices[index][x][y] == 0) {
                            alpha = 0;
                        }
                        else {
                            alpha = matrices[index][x][y] / rowSums[x];
                        }
                        
                        qDebug() << "Alpha value for matrix" << index << "at value" << matrices[index][x][y] << ":" << alpha;
                        
                        // color of the matrix
                        QColor color(Qt::blue);
                        color.setAlphaF(alpha);
                        QBrush brush(color);

                        // Drawing the rect and text
                        QGraphicsRectItem* square = new QGraphicsRectItem(squareX, squareY, squareSize, squareSize);
                        square->setBrush(brush);
                        scene->addItem(square);
                        QString text = QString("P = c - %1\nA = c - %2").arg(x).arg(y);
                        QGraphicsTextItem* textItem = new QGraphicsTextItem(text);

                        // Adjusting the size of the text
                        QFont font = textItem->font();
                        font.setPointSizeF(squareSize * 0.10); // a better way of adjusting the text??
                        textItem->setFont(font);

                        // Adjustiing the position 
                        QFontMetrics fm(font);
                        QRect rect = fm.boundingRect(text);
                        qreal textWidth = rect.width()* .10;
                        qreal textHeight = rect.height() * .10;
                        textItem->setPos((squareX + (squareSize - textWidth) / 2)-10, squareY + (squareSize - textHeight) / 2);

                        scene->addItem(textItem);
                        scene->addItem(textItem);
                    }
                }

                // Setting the view and the alignment
                QGraphicsView* graphicsView = new QGraphicsView(scene);
                graphicsView->setRenderHint(QPainter::Antialiasing);
                graphicsView->setAlignment(Qt::AlignTop | Qt::AlignLeft | Qt::AlignRight); // Alinear la vista arriba y a la izquierda
                
                // settin a box layout
                QVBoxLayout* vbox = new QVBoxLayout();
                vbox->addWidget(graphicsView);

                // setting the title of each confusion matrix
                if (index < names.size()) {
                    QLabel* titleLabel = new QLabel(QString::fromStdString(names[index]));
                    titleLabel->setAlignment(Qt::AlignCenter);
                    vbox->addWidget(titleLabel);
                }

                // Adding the layout to the container
                layout->addLayout(vbox, i, j);
            }
        }
    }

    container->setLayout(layout);
    container->resize(800, 600);
    container->show();
}
