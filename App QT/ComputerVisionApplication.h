#include <QMainWindow>
#include<QMessageBox>
#include <QMouseEvent>
#include <QDebug>
#include <QLineEdit>
#include <QInputDialog>
#include <QLabel>
#include<QPainter>
#include <QtCharts/QChart>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QChartView>
#include <QtWidgets/QVBoxLayout>
#include <QSettings>
#include <QTimer>
#include <QDockWidget>
#include <QTabBar>
#include<Eigen/Dense>
#include <QMdiArea>
#include <QDir>
#include <QStringList>
#include "ui_ComputerVisionApplication.h"
#include "image.h"
#include <string>
#include "imageTransformations.h"
#include "ObjectDetection.h"
#include<vector>
#include <fstream>
#include <cstdlib>  
#include <ctime>    
#include "reader.h"
#include "machineLearning.h"
#include "plots.h"

#include <iostream>
#include <vector>
#include <sstream>
#include <cstdio>
#include<stdio.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <windows.h>
#undef min
#undef max
#include <algorithm>
#include "reader.h"

QT_BEGIN_NAMESPACE
namespace Ui { class ComputerVisionApplication; }
QT_END_NAMESPACE

QT_USE_NAMESPACE


class ComputerVisionApplication : public QMainWindow
{
    Q_OBJECT

public:
    ComputerVisionApplication(QWidget* parent = nullptr); //Constructor
    ~ComputerVisionApplication(); //Destructor


private slots: // listeners to buttons
    //App handlers
    void on_actionSelect_Image_triggered();
    void InMenuSave();// Implemented
    void InMenuExit();// Implemented
    void InInfo();
    //Image conversion
    void on_actionToGray_triggered();
    void on_actionThreshold_triggered();
    void toRGB();
    void toHSV();
    void toHSL();

    //Distance Procedures

    void on_actionEuclidean_triggered();
    void on_actionManhalanobis_triggered();
    void on_actionMaxProbability_triggered();
    void on_actionKNN_triggered();
    void on_actionConfusion_Matrix_triggered();

    //Cross validation Procedures
    void on_actionVisualize_Plots_triggered();


    //Image processing Procedures

    void on_actionimageProcessingFunction1_triggered();


    // DataSet Procedures

    void on_actionLoadDataSet_triggered(); // used to load images dataset from a folder
    void on_actionReadCSV_triggered(); // used to read an csv onto a 

    //Dock actions

    void handleDockLocationChanged(Qt::DockWidgetArea area);
    void handleTopLevelChanged(bool state);



    //Segmentation
    Eigen::MatrixXd on_actionConected_N4_triggered();
    void on_actionClassify_Image_triggered();

private:
    Ui::ComputerVisionApplication* ui; //UI
    //Dock tabs element
    QDockWidget* sideDock = NULL;
    QTabWidget* dockTabs = NULL;
    QSettings* settings;
    QString lastDirectory;

    std::vector<Image> images;
    std::string currFormat = "RGB";

    bool classify = false;

    int knn = 0;

    QTimer* singleClickTimer;


    //Helper functions
    QLabel* getImageLabel();
    Image* getImage();
    QString getFilePath(std::string extension); // pass in a string the extension, example : .csv , .txt, ... 

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    void mouseClickAction();
    void convertTabToDock(QTabWidget* tabWidget, int tabIndex);
    void doubleClickFunctionality(Qt::MouseButton button);
    void singleClickFunctionality(Qt::MouseButton button, const QPoint& mousePos);

    void updateImage(QImage newImage);

};
