#include "ComputerVisionApplication.h"
#include "imageTransformations.h"
#include "computations.h"

#include<QPixmap>
#include <QFileDialog>
#include<Eigen/Dense>
#include<QPainter>
//Constructor
ComputerVisionApplication::ComputerVisionApplication(QWidget* parent) :
    QMainWindow(parent), ui(new Ui::ComputerVisionApplication)

{
    ui->setupUi(this);

}
//Destructor
ComputerVisionApplication::~ComputerVisionApplication()
{
    delete ui;
}

void ComputerVisionApplication::on_actionSelect_Image_triggered()
{
    //QMessageBox::information(this, "Greetings", "Hello, World!");
    QString filePath = QFileDialog::getOpenFileName(this, tr("Select Image"), "", tr("Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"));
    if (!filePath.isEmpty()) {
        image.load(filePath);
        image = image.convertToFormat(QImage::Format_RGB888);
        if (!image.isNull()) {
            ui->imageLabel->setPixmap(QPixmap::fromImage(image));
            ui->imageLabel->setFixedSize(QPixmap::fromImage(image).size());

            this->resize(QPixmap::fromImage(image).size());


        }
        else {
            QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image."));

        }

    }
}


void ComputerVisionApplication::on_actionToGray_triggered() {

    if (image.isNull()) {
        QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image"));
        return;
    }
    if (currFormat == "Gray") {
        QMessageBox::information(this, "Message", "Already Gray");
        return;
    }

    currFormat = "Gray";

    ImageTransformations::convertToGray(image);

    //We now update the UI

    if (!image.isNull()) {
        ui->imageLabel->setPixmap(QPixmap::fromImage(image));
        ui->imageLabel->setFixedSize(QPixmap::fromImage(image).size());

        this->resize(QPixmap::fromImage(image).size());
    }
    else {
        QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image."));

    }




}

void ComputerVisionApplication::toRGB() {

}
void ComputerVisionApplication::toHSV() {}

void ComputerVisionApplication::toHSL() {}


void ComputerVisionApplication::on_actionEuclidean_triggered() {

    currProcess = "Euclidean";
    //We first ask for the number of classes if it's not defined
    if (numClasses == 0) {
        //QInputDialog inputDialog;
        //inputDialog.exec(); // Execute the dialog window modally
        bool ok;
        //Variables to make the code more understandable
        int defaultValue = 2;
        int minVal = 0;
        int maxVal = 100;
        int step = 1;
        numClasses = QInputDialog::getInt(this, tr("Number of classes"), tr("Type the number of classes: "), defaultValue, minVal, maxVal, step, &ok);


    }
    else {
        std::string str = std::to_string(numClasses);
        QString qstr = QString::fromStdString(str);
        QMessageBox::information(this, "Number of classes", qstr);


    }

    //We start tracking the position of the mouse so we can see what pixel is being clicked
    ui->imageLabel->setMouseTracking(true);
    //setCentralWidget(ui->imageLabel);



    //Install the event filter

    ui->imageLabel->installEventFilter(this);


}


void ComputerVisionApplication::on_actionManhalanobis_triggered() {
    currProcess = "Manhalanobis";
    //We first ask for the number of classes if it's not defined
    if (numClasses == 0) {
        //QInputDialog inputDialog;
        //inputDialog.exec(); // Execute the dialog window modally
        bool ok;
        //Variables to make the code more understandable
        int defaultValue = 2;
        int minVal = 0;
        int maxVal = 100;
        int step = 1;
        numClasses = QInputDialog::getInt(this, tr("Number of classes"), tr("Type the number of classes: "), defaultValue, minVal, maxVal, step, &ok);


    }
    

    //We start tracking the position of the mouse so we can see what pixel is being clicked
    ui->imageLabel->setMouseTracking(true);
    //setCentralWidget(ui->imageLabel);



    //Install the event filter

    ui->imageLabel->installEventFilter(this);
}
void ComputerVisionApplication::on_actionMaxProbability_triggered() {
    currProcess = "MaxProb";
    //We first ask for the number of classes if it's not defined
    if (numClasses == 0) {
        //QInputDialog inputDialog;
        //inputDialog.exec(); // Execute the dialog window modally
        bool ok;
        //Variables to make the code more understandable
        int defaultValue = 2;
        int minVal = 0;
        int maxVal = 100;
        int step = 1;
        numClasses = QInputDialog::getInt(this, tr("Number of classes"), tr("Type the number of classes: "), defaultValue, minVal, maxVal, step, &ok);


    }
    else {
        

    }

    //We start tracking the position of the mouse so we can see what pixel is being clicked
    ui->imageLabel->setMouseTracking(true);
    //setCentralWidget(ui->imageLabel);



    //Install the event filter

    ui->imageLabel->installEventFilter(this);
}
void ComputerVisionApplication::on_actionKNN_triggered() {
    currProcess = "KNN";
    //We first ask for the number of classes if it's not defined
    if (numClasses == 0) {
        //QInputDialog inputDialog;
        //inputDialog.exec(); // Execute the dialog window modally
        bool ok;
        //Variables to make the code more understandable
        int defaultValue = 2;
        int minVal = 0;
        int maxVal = 100;
        int step = 1;
        numClasses = QInputDialog::getInt(this, tr("Number of classes"), tr("Type the number of classes: "), defaultValue, minVal, maxVal, step, &ok);



        

    }

        bool ok;
        //Variables to make the code more understandable
        int defaultValue = numClasses * 2 + 1;
        int minVal = 1;
        int maxVal = 100000;
        int step = 1;
        knn = QInputDialog::getInt(this, tr("KNN"), tr("Type the number of k: "), defaultValue, minVal, maxVal, step, &ok);

    

    //We start tracking the position of the mouse so we can see what pixel is being clicked
    ui->imageLabel->setMouseTracking(true);
    //setCentralWidget(ui->imageLabel);



    //Install the event filter

    ui->imageLabel->installEventFilter(this);
}

void ComputerVisionApplication::InMenuSave() {}
void ComputerVisionApplication::InMenuExit() {}
void ComputerVisionApplication::InInfo() {}


bool ComputerVisionApplication::eventFilter(QObject* watched, QEvent* event) {

    qDebug() << "Event filter called";

    //We start tracking the position of the mouse so we can see what pixel is being clicked

    if (event->type() == QEvent::MouseButtonPress) {





        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);

        //We check if the classes have been selected
        if (rectangles.size() < numClasses * 2) {
            rectangles.push_back(mouseEvent->pos());
            //We now draw the border
            
            if (rectangles.size() % 2 == 0) { //We have set the dimensions for the rectangle


                int x = rectangles.at(rectangles.size() - 1).x();
                int y = rectangles.at(rectangles.size() - 1).y();

                int width = (rectangles.at(rectangles.size() - 1).x() - rectangles.at(rectangles.size() - 2).x());
                int height = (rectangles.at(rectangles.size() - 1).y() - rectangles.at(rectangles.size() - 2).y());

                

                QPixmap originalPixmap = *(ui->imageLabel)->pixmap();
                QPainter painter(&originalPixmap);
                painter.setPen(Qt::green);

                if (width < 0 || height < 0) {
                    x = rectangles.at(rectangles.size() - 2).x();
                    y = rectangles.at(rectangles.size() - 2).y();
                }

                width = abs(width);
                height = abs(height);

                painter.drawRect(abs(x - width), abs(y - height), width, height);
                painter.drawText(x - width, y +12, QString::fromStdString("Clase: " + std::to_string(rectangles.size()/2 -1)));
                ui->imageLabel->setPixmap(originalPixmap);
            }

        }
        else if(matrixClasses.size()==0){ //If this is false we have already filled the classes  
            //Rectangles have finished drawing we now add the pixel to determine and we also determine the smallest dimensions and fill the matrices

            //First we retrieve the pixel to classify


            int minWidth = INT_MAX;
            int minHeight = INT_MAX;


            for (int i = 0; i < rectangles.size(); i += 2) {
                int width = abs(rectangles.at(i).x() - rectangles.at(i+1).x())+1;
                int height = abs(rectangles.at(i).y() - rectangles.at(i+1).y())+1;

                if (width < minWidth) {
                    minWidth = width;
                }
                if (height < minHeight) {
                    minHeight = height;
                }
            }

            //We now fill the matrices

            for (int i = 0; i < rectangles.size(); i+=2) {
                int startX = std::min(rectangles.at(i).x(), rectangles.at(i + 1).x());
                int startY = std::min(rectangles.at(i).y(), rectangles.at(i + 1).y());

                //we now fill the pixels
                Eigen::Matrix<double, Eigen::Dynamic, 3> matrix;
                for (int x = startX; x < startX + minWidth; ++x) {
                    for (int y = startY; y < startY + minHeight; ++y) {
                        matrix.conservativeResize(matrix.rows() + 1, Eigen::NoChange);
                        QRgb pixelValue = image.pixel(x, y);

                        


                        matrix.row(matrix.rows() - 1) << qRed(pixelValue), qGreen(pixelValue), qBlue(pixelValue);

                    }
                }
                //We add the matrix to the classes
                matrixClasses.push_back(matrix);
            }

            
        }
        else { //We will now only classify pixels
            
            QRgb classifyPixel = image.pixel(mouseEvent->pos());
            Eigen::Vector3d vec(qRed(classifyPixel), qGreen(classifyPixel), qBlue(classifyPixel));


            if (currProcess == "Euclidean") {
                std::vector<double> distances = euclidean(matrixClasses, vec);
                int closestClass = getClosest(distances);

                QString qstr = QString::fromStdString("La clase mas cercana por distance euclidana es: " + std::to_string(closestClass));
                QMessageBox::information(this, "Euclidean Distance Metric", qstr);

            }

            if (currProcess == "Manhalanobis") {
                std::vector<double> distances = manhalanobis(matrixClasses, vec);
                int closestClass = getClosest(distances);

                QString qstr = QString::fromStdString("La clase mas cercana por distance manhalanobis es: " + std::to_string(closestClass));
                QMessageBox::information(this, "Manhalanobis Distance Metric", qstr);

            }

            if (currProcess == "MaxProb") {
                std::vector<double> probabilites = max_prob(matrixClasses, vec);
                int closestClass = getMaxProb(probabilites);

                QString qstr = QString::fromStdString("La clase mas cercana por criterio de maxima probabilidad es: " + std::to_string(closestClass));
                QMessageBox::information(this, "Maximum Probability Metric", qstr);

            }

            if (currProcess == "KNN") {
                int result = kNearestNeighbours(matrixClasses, vec, knn);
                QString qstr = QString::fromStdString("La clase mas cercana por criterio de KNN es: " + std::to_string(result));
                QMessageBox::information(this, "KNN Metric", qstr);

            }
            
        }


    

        



        return true; // Event handled
    }
    return false; // Continue default processing
}


void ComputerVisionApplication::paintEvent(QPaintEvent* event){
    
    if (rectangles.size() >= 2) {
        int width = abs(rectangles.at(rectangles.size() - 1).x() - rectangles.at(rectangles.size() - 2).x());
        int height = abs(rectangles.at(rectangles.size() - 1).y() - rectangles.at(rectangles.size() - 2).y());

        QPainter painter(this);
        painter.setPen(Qt::blue);
        painter.drawRect(rectangles.at(rectangles.size() - 2).x(), rectangles.at(rectangles.size() - 2).y(), width, height);
    }
}


void ComputerVisionApplication::mouseClickAction() {

}
