#include "plots.h"	
#include <memory>
#include <QLabel>
#include <QtCharts/QValueAxis>
#include "imageTransformations.h"


void Plots::matrixPlot3D_labels(Eigen::MatrixXd values, std::vector<std::string> labels, std::string col1Name, std::string col2Name, std::string col3Name) {
    // congf of  3D graphic
    Q3DScatter* scatter = new Q3DScatter();
    QWidget* container = QWidget::createWindowContainer(scatter);
    scatter->setSelectionMode(QAbstract3DGraph::SelectionItemAndRow | QAbstract3DGraph::SelectionSlice);
    scatter->scene()->activeCamera()->setCameraPreset(Q3DCamera::CameraPresetFront);
    scatter->axisX()->setTitle(QString::fromStdString(col1Name));
    scatter->axisY()->setTitle(QString::fromStdString(col2Name));
    scatter->axisZ()->setTitle(QString::fromStdString(col3Name));
    scatter->axisX()->setTitleVisible(true);
    scatter->axisY()->setTitleVisible(true);
    scatter->axisZ()->setTitleVisible(true);

    // Normalice (0 to 1)
    normalizeColumn(values, 0);
    normalizeColumn(values, 1);
    normalizeColumn(values, 2);

    // creates a new series for each point
    for (int i = 0; i < values.rows(); ++i) {
        QScatter3DSeries* series = new QScatter3DSeries();
        series->setBaseColor(QColor(Qt::magenta));
        series->setItemSize(0.08);
        series->setItemLabelFormat(QString::fromStdString(labels[i])); // setting for each point an id, name in the dataset
        QScatterDataArray dataPoints;
        dataPoints << QVector3D(values(i, 0), values(i, 1), values(i, 2)); // creating a datapoint with the values of each row
        series->dataProxy()->addItems(dataPoints);
        scatter->addSeries(series);

    }

    // showing
    container->show();
    container->setMinimumSize(800, 600);
}
void Plots::plotMatrix(Eigen::MatrixXd matrixData, std::vector<std::string> colNames)
{
    if (matrixData.cols() > 5)
    {
        return;
    }
    else if (matrixData.cols() == 2)
    {
        matrixPlot2D(matrixData, colNames[0], colNames[1]);
        return;
    }
    else if (matrixData.cols() == 3)
    {
        //matrixPlot3D(matrixData, colNames[0], colNames[1], colNames[2]);
        return;
    }
}
void Plots::matrixPlot2D(Eigen::MatrixXd values, std::string col1Name, std::string col2Name)
{
    QChartView* chartView = new QChartView;
    chartView->setWindowTitle("Graphic of the data");
    QScatterSeries* series = new QScatterSeries();
    series->setName("Data Values");
    series->setMarkerSize(10);
    chartView->setRenderHint(QPainter::Antialiasing);
    chartView->setRubberBand(QChartView::RectangleRubberBand);  // Habilita el zoom de rectángulo
    chartView->setInteractive(true);
    qDebug() << values.rows();
    int t = 0;
    for (int i = 0; i < values.rows(); i++)
    {
        t++;
        // filling with the values of the matrix
        series->append(values(i, 0), values(i, 1));
        qDebug() << values(i, 0) << " " << values(i, 1) << "\n";
    }
    qDebug() << " Final t  " << t;
    QChart* chart = new QChart();
    chart->addSeries(series);
    chart->createDefaultAxes();
    chart->setTitle("DATA");
    chart->axisX()->setTitleText(QString::fromStdString(col1Name));
    chart->axisY()->setTitleText(QString::fromStdString(col2Name));

    // Establecer un color para los puntos
    QColor color(255, 0, 0); // Rojo
    series->setColor(color);
    chart->setAcceptHoverEvents(true);
    chart->setAnimationOptions(QChart::AllAnimations);
    chartView->setChart(chart);
    chartView->resize(800, 600);
    chartView->show();
}
void Plots::matrixPlot3D(Eigen::MatrixXd values,std::vector<std::string> labels, std::string col1Name, std::string col2Name, std::string col3Name)
{
    // configuration of 3Dscatter
    Q3DScatter* scatter = new Q3DScatter();
    scatter->setSelectionMode(QAbstract3DGraph::SelectionItemAndRow | QAbstract3DGraph::SelectionSlice);
    scatter->scene()->activeCamera()->setCameraPreset(Q3DCamera::CameraPresetFront);
    scatter->axisX()->setTitle(QString::fromStdString(col1Name));
    scatter->axisX()->setTitleVisible(true);
    scatter->axisY()->setTitle(QString::fromStdString(col2Name));
    scatter->axisY()->setTitleVisible(true);
    scatter->axisZ()->setTitle(QString::fromStdString(col3Name));
    scatter->axisZ()->setTitleVisible(true);

    // container of scatter
    QWidget* container = QWidget::createWindowContainer(scatter);
    QGridLayout* layout = new QGridLayout();
    container->setLayout(layout);

    layout->addWidget(container, 0, 0, 2, 1); 

    // adding the image label of the photo
    QLabel* imageLabel = new QLabel();
    imageLabel->setMaximumSize(100, 100); 
    layout->addWidget(imageLabel, 0, 1, Qt::AlignTop | Qt::AlignRight); 


    // conf of dataSeries
    QScatterDataProxy* dataProxy = new QScatterDataProxy();
    QScatter3DSeries* dataSeries = new QScatter3DSeries(dataProxy);
    dataSeries->setBaseColor(QColor(Qt::magenta));
    dataSeries->setItemSize(0.08);
    

    QScatterDataArray* dataPoints = new QScatterDataArray();

    normalizeColumn(values, 0);
    normalizeColumn(values, 1);
    normalizeColumn(values, 2);

    // Populate data points
    for (int i = 0; i < values.rows(); ++i) {
        QVector3D point(values(i, 0), values(i, 1), values(i, 2));
        dataPoints->push_back(point);
    }
    QString outputDir = "FilesOut/SubImages";

    std::vector<QLabel*> imLabels;


    QObject::connect(scatter, &Q3DScatter::selectedSeriesChanged, [=](QScatter3DSeries* series) {
        if (series) {
            int item = series->selectedItem();
            dataSeries->setItemLabelFormat(QString::fromStdString(labels[item]));
            QString filename = QString("%1/SubImage_%2.png").arg(outputDir).arg(item);
            QImage image(filename);
            if (!image.isNull()) {
                QLabel* imageLabel = new QLabel();  
                imageLabel->setPixmap(QPixmap::fromImage(image));
                imageLabel->show();
                //imLabels.push_back(imageLabel);
            }
            else {
                qDebug() << "Imagen no encontrada: " << filename;
            }
        }
        else {
            // here is where i need to delete all the imageLabel created
        }
        });


    dataSeries->dataProxy()->addItems(*dataPoints);
    scatter->addSeries(dataSeries);


    



    container->show();
    container->setMinimumSize(800, 600);
}
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
                        qreal textWidth = rect.width() * .10;
                        qreal textHeight = rect.height() * .10;
                        textItem->setPos((squareX + (squareSize - textWidth) / 2) - 10, squareY + (squareSize - textHeight) / 2);

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

void Plots::histogram(std::vector<int> histogram) {

    QChart* chart = new QChart();
    chart->setTitle("Image histogram");

    QBarSeries* barSeries = new QBarSeries();
    barSeries->setName("Bar series");

    QBarSet* barset = new QBarSet("");


    for (int i = 0; i < 256; i++) {
        *barset << histogram[i];
    }

    barSeries->append(barset);
    chart->addSeries(barSeries);





    chart->legend()->setVisible(false);
    chart->createDefaultAxes();
    chart->axisX()->setRange(0, 255);

    // Create chart view
    QChartView* chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);
    chartView->resize(600, 400);
    chartView->show();

}
void Plots::plotMatrixClasses(std::vector<Eigen::MatrixXd> mc)
{
    Q3DScatter* scatter = new Q3DScatter();
    QWidget* container = QWidget::createWindowContainer(scatter);
    scatter->setSelectionMode(QAbstract3DGraph::SelectionItemAndRow | QAbstract3DGraph::SelectionSlice);
    scatter->scene()->activeCamera()->setCameraPreset(Q3DCamera::CameraPresetFront);

    QVector<QColor> baseColors = { QColor(Qt::red), QColor(Qt::green), QColor(Qt::blue), QColor(Qt::yellow), QColor(Qt::cyan), QColor(Qt::magenta), QColor(Qt::gray) };

    for (int idx = 0; idx < mc.size(); ++idx) {
        
        // Data points series
        QScatterDataProxy* dataProxy = new QScatterDataProxy();
        QScatter3DSeries* dataSeries = new QScatter3DSeries(dataProxy);
        dataSeries->setBaseColor(baseColors[idx]);
        dataSeries->setItemSize(0.1);
        dataSeries->setItemLabelFormat(QString("Data - Class %1").arg(idx));

        QScatterDataArray* dataPoints = new QScatterDataArray();

        for (int i = 0; i < mc[idx].rows(); ++i) {
            QVector3D point(mc[idx](i, 0), mc[idx](i, 1), mc[idx](i, 2));
            dataPoints->push_back(point);
        }

        dataSeries->dataProxy()->addItems(*dataPoints);
        scatter->addSeries(dataSeries);
    }

    container->show();
    container->setMinimumSize(800, 600);
}


/*
* void Plots::plotPerceptrons(std::vector<Perceptron> perceptrons) {
    qDebug() << "Numero de perceptrones recibidos " << perceptrons.size();
    for (auto p : perceptrons) {
        p.showInfo();
    }

    Q3DScatter* scatter = new Q3DScatter();
    QWidget* container = QWidget::createWindowContainer(scatter);
    scatter->setSelectionMode(QAbstract3DGraph::SelectionItemAndRow | QAbstract3DGraph::SelectionSlice);
    scatter->scene()->activeCamera()->setCameraPreset(Q3DCamera::CameraPresetFront);

    QVector<QColor> baseColors = { QColor(Qt::red), QColor(Qt::green), QColor(Qt::blue), QColor(Qt::yellow), QColor(Qt::cyan), QColor(Qt::magenta), QColor(Qt::gray) };
    QVector<QColor> transparentColors;
    for (const auto& color : baseColors) {
        QColor transparentColor = color;
        transparentColor.setAlpha(20); // Adjust the transparency
        transparentColors.append(transparentColor);
    }

    for (int idx = 0; idx < perceptrons.size(); ++idx) {
        const auto& perceptron = perceptrons[idx];
        auto weights = perceptron.weights;
        // Data points series
        QScatterDataProxy* dataProxy = new QScatterDataProxy();
        QScatter3DSeries* dataSeries = new QScatter3DSeries(dataProxy);
        dataSeries->setBaseColor(baseColors[idx % baseColors.size()]);
        dataSeries->setItemSize(0.1);
        dataSeries->setItemLabelFormat(QString("Data - Class %1").arg(idx));

        // Plane series
        QScatterDataProxy* planeProxy = new QScatterDataProxy();
        QScatter3DSeries* planeSeries = new QScatter3DSeries(planeProxy);
        planeSeries->setBaseColor(transparentColors[idx % transparentColors.size()]);
        planeSeries->setItemSize(0.03);
        planeSeries->setItemLabelFormat(QString("Plane - Class %1").arg(idx));

        QScatterDataArray* dataPoints = new QScatterDataArray();
        QScatterDataArray* planePoints = new QScatterDataArray();

        // Populate data points
        for (int i = 0; i < perceptron.inputData.rows() / 2; ++i) {
            QVector3D point(perceptron.inputData(i, 0), perceptron.inputData(i, 1), perceptron.inputData(i, 2));
            dataPoints->push_back(point);
        }

        // Populate plane points
        double A = weights[0], B = weights[1], C = weights[2], D = weights[3];
        for (double x = -3; x <= 3; x += 0.3) {
            for (double y = -3; y <= 3; y += 0.3) {
                double z = -(A * x + B * y + D) / C;
                planePoints->push_back(QVector3D(x, y, z));
            }
        }

        dataSeries->dataProxy()->addItems(*dataPoints);
        scatter->addSeries(dataSeries);

        planeSeries->dataProxy()->addItems(*planePoints);
        scatter->addSeries(planeSeries);
    }

    container->show();
    container->setMinimumSize(800, 600);
}
*/
