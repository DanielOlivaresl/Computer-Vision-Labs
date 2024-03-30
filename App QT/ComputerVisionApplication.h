#include <QMainWindow>
#include<QMessageBox>
#include <QMouseEvent>
#include <QDebug>
#include <QLineEdit>
#include <QInputDialog>
#include <QLabel>

#include<Eigen/Dense>
#include "ui_ComputerVisionApplication.h"

QT_BEGIN_NAMESPACE
namespace Ui { class ComputerVisionApplication; }
QT_END_NAMESPACE



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
    void toRGB();
    void toHSV();
    void toHSL();

    //Distance Procedures

    void on_actionEuclidean_triggered();
    void on_actionManhalanobis_triggered();
    void on_actionMaxProbability_triggered();
    void on_actionKNN_triggered();


private:
    Ui::ComputerVisionApplication* ui; //UI
    QImage image;
    std::string currFormat = "RGB";
    int numClasses = 0;
    std::vector<QPointF> rectangles;
    std::string currProcess;

    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> matrixClasses;
    bool classify = false;

    int knn = 0;

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    void mouseClickAction();

};


