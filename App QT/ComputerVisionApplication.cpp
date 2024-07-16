#include "ComputerVisionApplication.h"
#include "computations.h"
#include "image.h"
#include<QPixmap>
#include<QPixmap>
#include <QFileDialog>
#include<QMdiSubWindow>
#include<Eigen/Dense>
//Constructor
ComputerVisionApplication::ComputerVisionApplication(QWidget* parent) :
    QMainWindow(parent), ui(new Ui::ComputerVisionApplication)
{
    qDebug() << "Dock moved";
    ui->setupUi(this);
    settings = new QSettings("organization", "application");
    settings->beginGroup("FileDialog");
    lastDirectory = settings->value("lastDirectory", "../Images").toString();
    settings->endGroup();
    //We disable some of the UI buttons
    ui->menuConvert->setDisabled(true);
    ui->menuDistances->setDisabled(true);
    ui->menuCross_Validation->setDisabled(true);
    ui->menuPrueba_Chevy->setDisabled(false);
    this->setFixedSize(1500, 900);
    ui->menuBar->installEventFilter(this);
    singleClickTimer = new QTimer(this);
    singleClickTimer->setSingleShot(true);
    singleClickTimer->setInterval(200);
    ui->Tabs->installEventFilter(this);
    this->setDockOptions(this->AllowNestedDocks);
    QTabBar* tabBar = ui->Tabs->tabBar();
    connect(tabBar, &QTabBar::tabBarDoubleClicked, [tabBar, this](int index) {
        /*QMessageBox::warning(this, tr("Confusion Matrix"), QString::number(index));*/
        convertTabToDock(ui->Tabs, index);
        });


    ui->Tabs->setDocumentMode(true);

}
//Destructor
ComputerVisionApplication::~ComputerVisionApplication()
{

    delete ui;
    settings->beginGroup("FileDialog");
    settings->setValue("lastDirectory", lastDirectory);
    QString dir = settings->value("lastDirectory", "null").toString();
    settings->endGroup();
}
void ComputerVisionApplication::on_actionSelect_Image_triggered()
{
    QString initialDir = lastDirectory.isEmpty() ? QDir::homePath() : lastDirectory;

    QString filePath = QFileDialog::getOpenFileName(this, tr("Select Image"), initialDir + "/peppers.jpg", tr("Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"));


    if (!filePath.isEmpty()) {
        // Update lastDirectory with the new directory
        lastDirectory = QFileInfo(filePath).absolutePath();
        Image* tempImg = new Image();

        if (tempImg->image.load(filePath)) {
            tempImg->image = tempImg->image.convertToFormat(QImage::Format_RGB888);
            if (!tempImg->image.isNull()) {

                QWidget* firsttab = new QWidget();
                QVBoxLayout* layout = new QVBoxLayout();
                QLabel* imageLabel = new QLabel("Image Label");
                imageLabel->setPixmap(QPixmap::fromImage(tempImg->image));
                imageLabel->setFixedSize(QPixmap::fromImage(tempImg->image).size());
                //We start tracking the position of the mouse so we can see what pixel is being clicked
                imageLabel->setMouseTracking(true);
                //Install the event filter
                imageLabel->installEventFilter(this);



                layout->addWidget(imageLabel);
                firsttab->setLayout(layout);
                firsttab->setProperty("Image", QVariant::fromValue(tempImg));
                ui->Tabs->addTab(firsttab, "Tab " + QString::number(ui->Tabs->count() + 1));
                setCentralWidget(ui->Tabs);
                ui->Tabs->setCurrentIndex(ui->Tabs->count() - 1);





            }
            else {
                QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image."));
            }
        }
        else {
            QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image."));
        }
    }
}
void ComputerVisionApplication::on_actionToGray_triggered() {
    Image* image = getImage();
    if (image == NULL) {
        return;
    }


    if ((image->image.isNull())) {
        QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image"));
        return;
    }
    if (currFormat == "Gray") {
        QMessageBox::information(this, "Message", "Already Gray");
        return;
    }

    currFormat = "Gray";

    ImageTransformations::ColorSpaces::convertToGray(image->image);

    //We now update the UI

    if (!(image->image.isNull())) {

        QLabel* imageLabel = getImageLabel();

        imageLabel->setPixmap(QPixmap::fromImage(image->image));
        imageLabel->setFixedSize(QPixmap::fromImage(image->image).size());

        this->resize(QPixmap::fromImage(image->image).size());
    }
    else {
        QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image."));

    }




}


void ComputerVisionApplication::on_actionThreshold_triggered() {
    Image* image = getImage();
    if (image->image.isNull()) {
        QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image"));
        return;
    }



    currFormat = "Thresholded";
    //Variables to make the code more understandable
    //We obtain the thereshold number
    bool ok;
    int defaultValue = 0;
    int minVal = 0;
    int maxVal = 255;
    int step = 1;
    int therseholdNumber = 0;
    therseholdNumber = QInputDialog::getInt(this, tr("Thersehold"), tr("Thersehold number: "), defaultValue, minVal, maxVal, step, &ok);

    image->image = ImageTransformations::ColorSpaces::threshold(image->image, therseholdNumber);

    //We now update the UI
    if (!(image->image.isNull())) {

        QLabel* imageLabel = getImageLabel();

        imageLabel->setPixmap(QPixmap::fromImage(image->image));
        imageLabel->setFixedSize(QPixmap::fromImage(image->image).size());

        this->resize(QPixmap::fromImage(image->image).size());
    }
    else {
        QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image."));

    }


}

void ComputerVisionApplication::toRGB() {}
void ComputerVisionApplication::toHSV() {}

void ComputerVisionApplication::toHSL() {}


void ComputerVisionApplication::on_actionEuclidean_triggered() {

    Image* image = getImage();
    if (image == NULL) {

        QMessageBox::information(this, "Number of classes", "Image is null?");

        return;
    }

    image->currProcess = "Euclidean";
    //We first ask for the number of classes if it's not defined
    if (image->numClasses == 0) {
        bool ok;
        //Variables to make the code more understandable
        int defaultValue = 2;
        int minVal = 0;
        int maxVal = 100;
        int step = 1;
        image->numClasses = QInputDialog::getInt(this, tr("Number of classes"), tr("Type the number of classes: "), defaultValue, minVal, maxVal, step, &ok);


    }
    else {
        std::string str = std::to_string(image->numClasses);
        QString qstr = QString::fromStdString(str);
        QMessageBox::information(this, "Number of classes", qstr);


    }
    QLabel* imageLabel = getImageLabel();

    //We start tracking the position of the mouse so we can see what pixel is being clicked
    imageLabel->setMouseTracking(true);



    //Install the event filter

    imageLabel->installEventFilter(this);


}


void ComputerVisionApplication::on_actionManhalanobis_triggered() {

    Image* image = getImage();
    if (image == NULL) {
        return;
    }
    image->currProcess = "Manhalanobis";
    //We first ask for the number of classes if it's not defined
    if (image->numClasses == 0) {
        //QInputDialog inputDialog;
        //inputDialog.exec(); // Execute the dialog window modally
        bool ok;
        //Variables to make the code more understandable
        int defaultValue = 2;
        int minVal = 0;
        int maxVal = 100;
        int step = 1;
        image->numClasses = QInputDialog::getInt(this, tr("Number of classes"), tr("Type the number of classes: "), defaultValue, minVal, maxVal, step, &ok);


    }

    QLabel* imageLabel = getImageLabel();

    //We start tracking the position of the mouse so we can see what pixel is being clicked
    imageLabel->setMouseTracking(true);
    //setCentralWidget(imageLabel);



    //Install the event filter

    imageLabel->installEventFilter(this);
}
void ComputerVisionApplication::on_actionMaxProbability_triggered() {

    Image* image = getImage();
    if (image == NULL) {
        return;
    }
    image->currProcess = "MaxProb";
    //We first ask for the number of classes if it's not defined
    if (image->numClasses == 0) {
        //QInputDialog inputDialog;
        //inputDialog.exec(); // Execute the dialog window modally
        bool ok;
        //Variables to make the code more understandable
        int defaultValue = 2;
        int minVal = 0;
        int maxVal = 100;
        int step = 1;
        image->numClasses = QInputDialog::getInt(this, tr("Number of classes"), tr("Type the number of classes: "), defaultValue, minVal, maxVal, step, &ok);


    }
    else {


    }
    QLabel* imageLabel = getImageLabel();

    //We start tracking the position of the mouse so we can see what pixel is being clicked
    imageLabel->setMouseTracking(true);
    //setCentralWidget(imageLabel);



    //Install the event filter

    imageLabel->installEventFilter(this);
}
void ComputerVisionApplication::on_actionKNN_triggered() {

    Image* image = getImage();
    if (image == NULL) {
        return;
    }
    image->currProcess = "KNN";
    //We first ask for the number of classes if it's not defined
    if (image->numClasses == 0) {
        //QInputDialog inputDialog;
        //inputDialog.exec(); // Execute the dialog window modally
        bool ok;
        //Variables to make the code more understandable
        int defaultValue = 2;
        int minVal = 0;
        int maxVal = 100;
        int step = 1;
        image->numClasses = QInputDialog::getInt(this, tr("Number of classes"), tr("Type the number of classes: "), defaultValue, minVal, maxVal, step, &ok);





    }

    bool ok;
    //Variables to make the code more understandable
    int defaultValue = image->numClasses * 2 + 1;
    int minVal = 1;
    int maxVal = 100000;
    int step = 1;
    knn = QInputDialog::getInt(this, tr("KNN"), tr("Type the number of k: "), defaultValue, minVal, maxVal, step, &ok);

    QLabel* imageLabel = getImageLabel();


    //We start tracking the position of the mouse so we can see what pixel is being clicked
    imageLabel->setMouseTracking(true);
    //setCentralWidget(imageLabel);



    //Install the event filter

    imageLabel->installEventFilter(this);
}


void ComputerVisionApplication::on_actionConfusion_Matrix_triggered()
{

    Image* image = getImage();
    if (image == NULL) {
        return;
    }
    std::vector<std::vector<std::vector<double>>> matrices; // stores all the matrices 
    if (image->numClasses == 0) {
        QMessageBox::warning(this, tr("Confusion Matrix"), tr("Tienes que ingresar clases para poder calcularla."));
        return;
    }
    if (knn == 0) // ifs there's no knn yet...
    {
        bool ok;
        int defaultValue = image->numClasses * 2 + 1;
        int minVal = 1;
        int maxVal = 100000;
        int step = 1;
        knn = QInputDialog::getInt(this, tr("KNN"), tr("Type the number of k: "), defaultValue, minVal, maxVal, step, &ok);
    }
    std::vector<std::vector<double>> matEuc(image->numClasses, std::vector < double>(image->numClasses, 0)); // instance of a confusion matrix
    std::vector<std::vector<double>> matMan(image->numClasses, std::vector < double>(image->numClasses, 0)); // instance of a confusion matrix
    std::vector<std::vector<double>> matMax(image->numClasses, std::vector < double>(image->numClasses, 0)); // instance of a confusion matrix
    std::vector<std::vector<double>> matKnn(image->numClasses, std::vector < double>(image->numClasses, 0)); // instance of a confusion matrix
    for (size_t i = 0; i < image->matrixClasses.size(); ++i) {
        for (int j = 0; j < image->matrixClasses[i].rows(); ++j) {

            double r = image->matrixClasses[i](j, 0);
            double g = image->matrixClasses[i](j, 1);
            double b = image->matrixClasses[i](j, 2);

            Eigen::Vector3d vec2class(r, g, b);

            std::vector<double> distances1 = Computations::Distances::euclidean(image->matrixClasses, vec2class);
            int closestClass1 = Computations::Helper::getClosest(distances1);
            matEuc[i][closestClass1] += 1;

            std::vector<double> distances2 = Computations::Distances::manhalanobis(image->matrixClasses, vec2class);
            int closestClass2 = Computations::Helper::getClosest(distances2);
            matMan[i][closestClass2] += 1;

            std::vector<double> distances3 = Computations::Helper::max_prob(image->matrixClasses, vec2class);
            int closestClass3 = Computations::Helper::getMaxProb(distances3);
            matMax[i][closestClass3] += 1;

            int result = MachineLearning::kNearestNeighbours(image->matrixClasses, vec2class, knn);
            matKnn[i][result] += 1;

        }
    }
    std::vector<std::string> names = { "Euclidiana", "Manhalanobis", "Maxima Probabilidad", "Knn" };
    matrices.push_back(matEuc);
    matrices.push_back(matMan);
    matrices.push_back(matMax);
    matrices.push_back(matKnn);
    Plots::ConfusionMatrix(matrices, names);
    QString message;


    auto addMatrixToMessage = [&message](const QString& title, const std::vector<std::vector<double>>& matrix) {
        message += title + "\n   ";
        for (int i = 0; i < matrix.size(); ++i) {
            message += "C" + QString::number(i) + " ";
        }

        message += "\n";

        for (int i = 0; i < matrix.size(); i++) {
            for (double val : matrix[i]) {
                message += QString::number(val) + " ";
            }
            message += "\n";
        }
        };

    addMatrixToMessage("Confusion Matrix for Euclidian:", matEuc);
    message += "\n";
    addMatrixToMessage("Confusion Matrix for Manhalanobis:", matMan);
    message += "\n";
    addMatrixToMessage("Confusion Matrix for MaxProb:", matMax);
    message += "\n";
    addMatrixToMessage("Confusion Matrix for KNN:", matKnn);

    QMessageBox::information(this, "Confusion Matrices", message);


}


void ComputerVisionApplication::InMenuSave() {}
void ComputerVisionApplication::InMenuExit() {}
void ComputerVisionApplication::InInfo() {}


void ComputerVisionApplication::on_actionLoadDataSet_triggered()
{
    //We load the image folder
    QMessageBox::information(this, "Dataset loader", "Loading dataset");
    QString directoryPath = QFileDialog::getExistingDirectory(nullptr, "Select Directory", "");


    




    if (!directoryPath.isEmpty()) {

        //we create a directory and a list of images
        QDir directory(directoryPath);
        QStringList images = directory.entryList(QStringList() << "*.jpg", QDir::Files);



        std::vector<QImage> vectorImages; // vector that stores the images 
        std::vector<QString> imageNames;
        std::vector<QImage> subImages; // vector that store the subImages of the data


        

        int count = 0;


        //We will iterate the images in the directory


        bool finish = false;
        std::vector<std::string> classNames;
        count = 0;
        foreach(QString filename, images) {

            /*if (count > 200) {
                break;
            }


            if (filename.split("_").first().toStdString() == "coast") {
                if (count < 50) {
                    count++;

                }
                else {
                    continue;
                }
           }



            if (filename.split("_").first().toStdString() == "forest") {
                if (count < 100) {
                    count++;
                }
                else {
                    continue;
                }
            }


            if (filename.split("_").first().toStdString() == "mountain") {
                if (count < 150) {
                    count++;
                }else{
                    continue;}
            }


            if (filename.split("_").first().toStdString() == "opencountry") {
                    count++;
            }
*/



            classNames.push_back(filename.split("_").first().toStdString());

            
            
            qDebug() << "Archivo encontrado:" << filename;
            QString filePath = directory.absoluteFilePath(filename); // getting the absolute path of each image 
            Image* tempImg = new Image(); // init the image 
            if (tempImg->image.load(filePath)) {
                // if the image loaded succesfull, we keep the process .

                /*
                
                QWidget* firsttab = new QWidget();
                QVBoxLayout* layout = new QVBoxLayout();
                QLabel* imageLabel = new QLabel("Image Dataset ");
                imageLabel->setPixmap(QPixmap::fromImage(tempImg->image));
                imageLabel->setFixedSize(QPixmap::fromImage(tempImg->image).size());
                //We start tracking the position of the mouse so we can see what pixel is being clicked
                imageLabel->setMouseTracking(true);
                //Install the event filter
                imageLabel->installEventFilter(this);

                
                */



                vectorImages.push_back(tempImg->image); // storing the image if its necessary... 
                imageNames.push_back(filename); // storing each imageName in the vector


                // here we call the function that retrieves the objects in the image for each image into a csv 
                //ImageTransformations::imageObjectsToCsv(tempImg->image);


                /*layout->addWidget(imageLabel);
                firsttab->setLayout(layout);
                firsttab->setProperty("Image", QVariant::fromValue(tempImg));
                ui->Tabs->addTab(firsttab, "Tab " + QString::number(ui->Tabs->count() + 1));
                setCentralWidget(ui->Tabs);
                ui->Tabs->setCurrentIndex(ui->Tabs->count() - 1);*/
            }
            else {
                // error hanlder is the image has an error.
                qDebug() << "Error loading image";
            }
        }

        /*qDebug() << "Size of the vector of images " << vectorImages.size();
        qDebug() << "size of one image " << vectorImages[0].size();*/

        //Now that the images are loaded, we will select the .csv to write the subimage metrics

        //QString csvPath = QFileDialog::getSaveFileName(nullptr, "Open CSV File", "FilesOut","CSV Files (*.csv)");





        //we check if the path selected is empty
        //if (csvPath.isEmpty()) {
        //    qDebug() << "Invalid path";
        //    return;
        //}

        //We will now check if the file exists

        //QFile file(csvPath);
        //if (!file.exists()) {
        //    //File doesn't exist, we will create it

        //    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        //        QMessageBox::critical(nullptr, "Error", "Failed to create file.");
        //        return;
        //    }

        //    QTextStream stream(&file);
        //    file.close();
        //    QMessageBox::information(nullptr, "Info", "CSV Created");
        //}
        //else {
        //    qDebug() << "File already exists";
        //}



        /*std::ofstream tempFile(csvPath.toStdString(), std::ios::trunc);
        tempFile.close();*/

        /*for (int i = 1; i < vectorImages.size(); i++)
        {

            qDebug() << "Processing image: " << i;
            ImageTransformations::imageObjectsToCsv(vectorImages[i], imageNames[i], csvPath.toStdString(), subImages);
        }*/

       // qDebug() << "Size of the subImages" << subImages.size() << '\n';
       // QString outputDir = "FilesOut/SubImages";
       // 
       ////We will write the images to the output file        
       //ImageTransformations::storeImages(outputDir.toStdString(),subImages,0);
        
        ImageTransformations::computeGistDescriptor(vectorImages,classNames);


    }
    else {
        qDebug() << "No directory selected.";
    }
}

void ComputerVisionApplication::on_actionReadCSV_triggered()
{
    Eigen::MatrixXd input(4, 2);
    input << 0, 0,
        0, 1,
        1, 0,
        1, 1;
    Eigen::VectorXd labels(4);
    labels << 0, 0, 0, 1;
    Perceptron AND(input);
    Eigen::VectorXd error = AND.error(labels);
    qDebug() << "ERROR inicial";
    for (int j = 0; j < error.size(); j++)
    {
        qDebug() << error(j);
    }
    for (int i = 0; i < 200; i++)
    {
        AND.weights = AND.update_weights(labels, 0.8);
    }
    Eigen::VectorXd error2 = AND.error(labels);
    qDebug() << "Final Error";
    for (int j = 0; j < error2.size(); j++)
    {
        qDebug() << error2(j);
    }
    std::vector<Perceptron> ps;
    ps.push_back(AND);
    Plots::plotPerceptrons(ps);
    qDebug() << "Numero de Iteraciones: [" << 200 << "]";
    qDebug() << "Ecuacion del plano " << AND.weights(0) << "x + " << AND.weights(1) << "y + " << AND.weights(2);
    //QString filePath = getFilePath("csv");
    //DataStore test = read_csvToDataStore(filePath);

    //std::vector <std::string> cols;
    //cols.push_back("Area");
    //cols.push_back("Perimetro");
    //cols.push_back("eXCEN");

    //we normalize the data by iterating the cols

   /* for (int i = 0; i < test.numericData.cols(); i++) {
        normalizeColumn(test.numericData, i);
    }*/

    //Plots::matrixPlot3D_subImages(test.numericData, test.stringData[0], cols[0], cols[1], cols[2]);
    //auto result = ML::Kmeans(test.numericData, 5, 0.001);
    //Plots::plotMatrixClasses(result.first);

}
bool ComputerVisionApplication::eventFilter(QObject* watched, QEvent* event) {
    //we check if the object which triggered the event is the menuBar

    if (watched->objectName() == "menuBar" && event->type() == QEvent::Enter) {

        //We now check if the current tab has an image loaded
        Image* image = getImage();
        if (image != NULL) {


            ui->menuConvert->setDisabled(false);

            //we check if the classes are defined
            if (image->rectangles.size() > 1) {

                ui->menuCross_Validation->setDisabled(false);
                ui->menuPrueba_Chevy->setDisabled(false);
                ui->menuDistances->setDisabled(false);
                return true;
            }
        }
    }

    if (watched->objectName() != "menuBar" && event->type() == QEvent::MouseButtonPress) {

        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        auto eventType = mouseEvent->type();
        auto mousePos = mouseEvent->pos();
        auto button = mouseEvent->button();
        if (!(singleClickTimer->isActive())) {

            disconnect(singleClickTimer, &QTimer::timeout, this, nullptr);

            singleClickTimer->start();
            QObject::connect(singleClickTimer, &QTimer::timeout, this, [this, button, mousePos]() {

                singleClickFunctionality(button, mousePos);
                });
        }
    }
    QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);

    auto button = mouseEvent->button();

    if (event->type() == QEvent::MouseButtonDblClick) {
        singleClickTimer->stop();
        doubleClickFunctionality(button);
    }
    //We start tracking the position of the mouse so we can see what pixel is being clicked
    return false; // Continue default processing
}

void ComputerVisionApplication::on_actionimageProcessingFunction1_triggered() {

    Image* img = getImage();



    /*
    
    //We segment the regions of the image

    cv::Mat image = ImageTransformations::QImageToMat(img->image);
    // Convert the image to grayscale for corner detection
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Detect corners using Shi-Tomasi corner detection
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(grayImage, corners, 50, 0.01, 10);

    // Detect edges using Canny edge detector
    cv::Mat edges;
    cv::Canny(grayImage, edges, 100, 200);

    // Combine corners and edges as seeds
    std::vector<cv::Point> seeds;
    for (const auto& corner : corners) {
        seeds.push_back(cv::Point(corner.x, corner.y));
    }
    for (int y = 0; y < edges.rows; y++) {
        for (int x = 0; x < edges.cols; x++) {
            if (edges.at<uchar>(y, x) > 0) {
                seeds.push_back(cv::Point(x, y));
            }
        }
    }

    int threshold = 10; // Intensity/color difference threshold

    cv::Mat labels;
    cv::Vec3b a;
    ImageTransformations::regionGrowing(image, seeds, threshold, labels, a);



    //Now we will classify the image

    int boundaryExtension = 32; // Number of pixels to pad the image
    int imageSize = 256;
    std::vector<int> orientationsPerScale = { 8,8,8,8 };
    int numberBlock = 4;
    int fc_prefilt = 4;
    auto G = ImageTransformations::createGarborKernels(orientationsPerScale, imageSize + (2 * boundaryExtension));
    int Nfeatures = G[0][0].size() * numberBlock * numberBlock;



    std::vector<double> currentGist;



    
    QImage copyImage = img->image.copy();
    copyImage = ImageTransformations::ColorSpaces::convertToGray(copyImage);


    QSize newImageSize(imageSize, imageSize);
    copyImage = copyImage.scaled(newImageSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);




    auto normalizedImage = ImageTransformations::normalizeImage(copyImage);


    auto output = ImageTransformations::preFilter(normalizedImage);







    currentGist = ImageTransformations::gistGabor(output, numberBlock, G, boundaryExtension);




    std::ofstream outFile("descriptor.txt");


    for (int col = 0; col < currentGist.size(); col++) {
        outFile << currentGist[col] << " ";
    }

    outFile.close();














    // Set up the command to run the Python script
    std::string command = "python predict.py";

    // Additional settings to prevent the command window from showing
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags |= STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;  // Prevents the command window from being shown

    ZeroMemory(&pi, sizeof(pi));

    // Execute the Python script
    if (!CreateProcessA(
        NULL,   // No module name (use command line)
        const_cast<char*>(command.c_str()), // Command line
        NULL,   // Process handle not inheritable
        NULL,   // Thread handle not inheritable
        FALSE,  // Set handle inheritance to FALSE
        0,      // No creation flags
        NULL,   // Use parent's environment block
        NULL,   // Use parent's starting directory 
        &si,    // Pointer to STARTUPINFO structure
        &pi)    // Pointer to PROCESS_INFORMATION structure
        )
    {
        std::cerr << "Failed to run the Python script." << std::endl;
        return;
    }

    // Wait until child process exits.
    WaitForSingleObject(pi.hProcess, INFINITE);

    // Close process and thread handles. 
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
















    // Read the results back from the file
    std::ifstream inFile("result.txt");
    if (!inFile) {
        std::cerr << "Failed to open result.txt for reading." << std::endl;
        return;
    }

    std::vector<double> predictions;
    double prediction;

    double temp;

    // Read all the values from the file into the vector
    while (inFile >> temp) {
        predictions.push_back(temp);
    }

    // Close the file
    inFile.close();

    // Output the predictions
    qDebug() << "Received predictions: ";
    std::string pred = "[";
    for (const auto& prediction : predictions) {
        pred.append(std::to_string(prediction) + ", ");
    }
    pred.append(" ]");
    qDebug() << pred;
    auto it = std::max_element(predictions.begin(), predictions.end());

    // Calculate the index based on the iterator
    int index = std::distance(predictions.begin(), it);
    std::map<int, std::string> classes;
    classes[0] = "coastal";
    classes[1] = "forest";
    classes[2] = "Mountain";
    classes[3] = "Country";



    QPainter painter(&img->image);
    painter.setPen(QPen(Qt::green, 2));  // Color azul y un grosor de 
    QString str = QString::fromStdString(classes[index]);
    painter.drawText(img->image.rect(),Qt::AlignTop | Qt::AlignLeft, str);
    painter.end();




    updateImage(img->image);
    */
}


void ComputerVisionApplication::paintEvent(QPaintEvent* event)
{
    Image* image = getImage();
    if (image == NULL) {
        return;
    }

    if (image->rectangles.size() >= 2) {
        int width = abs(image->rectangles.at(image->rectangles.size() - 1).x() - image->rectangles.at(image->rectangles.size() - 2).x());
        int height = abs(image->rectangles.at(image->rectangles.size() - 1).y() - image->rectangles.at(image->rectangles.size() - 2).y());

        QPainter painter(this);

        painter.setPen(Qt::blue);
        painter.drawRect(image->rectangles.at(image->rectangles.size() - 2).x(), image->rectangles.at(image->rectangles.size() - 2).y(), width, height);
    }
}


void ComputerVisionApplication::on_actionVisualize_Plots_triggered() {

    Image* image = getImage();
    if (image == NULL) {
        return;
    }

    if (image->numClasses == 0) {
        QMessageBox::warning(this, tr("Confusion Matrix"), tr("Tienes que ingresar clases para poder calcularla."));
        return;
    }

    if (knn == 0) // ifs there's no knn yet...
    {
        bool ok;
        int defaultValue = image->numClasses * 2 + 1;
        int minVal = 1;
        int maxVal = 100000;
        int step = 1;
        knn = QInputDialog::getInt(this, tr("KNN"), tr("Type the number of k: "), defaultValue, minVal, maxVal, step, &ok);
    }
    //we now fill a prediction vector

    //we will first separate our data with cross validation

    std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> cvSet = MachineLearning::CrossValidation::crossValidation(image->matrixClasses);
    std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> resSet = MachineLearning::CrossValidation::Restitucion(image->matrixClasses);
    //we will now create the data for each of the predictions

    std::vector<std::vector<std::vector<int>>> resPred = MachineLearning::Metrics::generatePredictions(resSet.at(0), resSet.at(1), knn);
    std::vector<std::vector<std::vector<int>>> cvPred = MachineLearning::Metrics::generatePredictions(cvSet.at(0), cvSet.at(1), knn);
    std::vector<std::vector<std::vector<int>>> looPredictions(4, std::vector<std::vector<int>>(image->matrixClasses.size(), std::vector<int>(image->matrixClasses.at(0).rows())));
    //finally for leave one out it's necesarry to iterate the whole set and apply the method n times



    int currClass = 0;
    for (auto clas : image->matrixClasses) {
        for (int i = 0; i < clas.rows(); i++) {
            std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> currSplit = MachineLearning::CrossValidation::leaveOneOut(image->matrixClasses, currClass, i);

            //Euclidean
            std::vector<double> distances1 = Computations::Distances::euclidean(currSplit.at(1), clas.row(i));
            int res = Computations::Helper::getClosest(distances1);
            looPredictions.at(0).at(currClass).at(i) = res;

            //Manhalanobis
            std::vector<double> distances2 = Computations::Distances::manhalanobis(currSplit.at(1), clas.row(i));
            res = Computations::Helper::getClosest(distances2);
            looPredictions.at(1).at(currClass).at(i) = res;

            //MaxProb
            std::vector<double> distances3 = Computations::Helper::max_prob(currSplit.at(1), clas.row(i));
            res = Computations::Helper::getMaxProb(distances3);
            looPredictions.at(2).at(currClass).at(i) = res;
            ////KNN
            res = MachineLearning::kNearestNeighbours(currSplit.at(1), clas.row(i), knn);
            looPredictions.at(3).at(currClass).at(i) = res;
        }
        currClass++;
    }


    //We create the sets of data and predictions



    std::vector<std::vector<std::vector<std::vector<int>>>> predSets = { resPred,cvPred,looPredictions };
    std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>> testSets = { resSet.at(0),cvSet.at(0),image->matrixClasses };



    //we get the data for the euclidena plot

    //We load the data


    QString message;
    //std:; <std::vector<std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>>>>

    std::vector<std::string> names = { "Restitucion", "Leave One Out", "Cross Validation" };
    std::vector<std::string> methodNames = { "Euclidiana", "Manhalanobis", "Max Prob", "KNN" };
    std::vector<std::vector<std::vector<double>>> allData; // Accumulate data from each iteration

    //we initialize the vector
    for (int i = 0; i < methodNames.size(); i++) {
        std::vector<std::vector<double>> el;

        for (int j = 0; j < names.size(); j++) {
            el.push_back(std::vector<double>());
        }
        allData.push_back(el);
    }



    for (int it = 0; it < 3; it++) {
        std::vector<std::vector<double>> data;

        std::vector<std::vector<double>> eucliMat = MachineLearning::Metrics::get_matrixConfusion(testSets.at(it), predSets.at(it).at(0));
        std::vector<std::vector<double>> manhMat = MachineLearning::Metrics::get_matrixConfusion(testSets.at(it), predSets.at(it).at(1));
        std::vector<std::vector<double>> mxprobMat = MachineLearning::Metrics::get_matrixConfusion(testSets.at(it), predSets.at(it).at(2));
        std::vector<std::vector<double>> knnMat = MachineLearning::Metrics::get_matrixConfusion(testSets.at(it), predSets.at(it).at(3));


        std::vector<std::vector<std::vector<double>>> matrices = {
            eucliMat,
            manhMat,
            mxprobMat,
            knnMat
        };

        auto addMatrixToMessage = [&message](const QString& title, const std::vector<std::vector<double>>& matrix) {
            message += title + "\n   ";
            double prom = 0;
            for (int i = 0; i < matrix.size(); ++i) {
                double sum = 0;
                for (double val : matrix[i]) {
                    sum += val;
                }
                prom += matrix[i][i] / sum;


            }
            message += QString::number(((prom / matrix[0].size()))) + "\n";

        };

        std::string mess = "Confusion Matrix for " + names[it];
        for (int i = 0; i < matrices.size(); i++) {
            addMatrixToMessage(QString::fromStdString(mess), matrices.at(i));
        }
        message += "\n";

        Plots::ConfusionMatrix(matrices, methodNames);



        for (int i = 0; i < matrices.size(); i++) {
            std::vector<double> currentPoints;
            for (int j = 0; j < matrices.at(i).size(); j++) { // Iterate rows
                double rowSum = 0.0;
                for (int k = 0; k < matrices.at(i).at(j).size(); k++) { // Iterate cols

                    rowSum += matrices.at(i).at(j).at(k);
                }
                // Calculate accuracy (normalized diagonal element)
                double diagonalElement = matrices.at(i).at(j).at(j);
                double accuracy = diagonalElement / rowSum;

                //currentPoints.push_back(accuracy);

                allData.at(i).at(it).push_back(accuracy);


            }
            /*data.push_back(currentPoints);*/
        }


        //allData.push_back(data);
        //Plots::scatterPlot(allData.at(0));


//    //We display the data

    }

    QMessageBox::information(this, "Matrices", message);




    int index = 0;
    for (const auto& data : allData) {
        Plots::scatterPlot(data);


    }


}




void ComputerVisionApplication::mouseClickAction() {

}




//Helper functions




QLabel* ComputerVisionApplication::getImageLabel() {

    int tabIndex = ui->Tabs->currentIndex();
    QWidget* currentTab = ui->Tabs->widget(tabIndex); //Current tab
    //We check that the current Tab exists 
    if (currentTab == NULL) {
        QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image"));
        return NULL;
    }
    //We retrievce the layout of the tab
    QLayout* layout = currentTab->layout();
    //we now check that the layout exists and that it contains widgets

    if (layout == NULL || layout->count() <= 0) {
        QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image"));
        return NULL;
    }

    QWidget* widget = layout->itemAt(0)->widget();

    if (widget == NULL) {
        QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image"));
        return NULL;
    }

    QLabel* imageLabel = dynamic_cast<QLabel*>(widget);

    return imageLabel;

}
Image* ComputerVisionApplication::getImage() {

    QWidget* currentTab = ui->Tabs->currentWidget();

    //We check if the current tab is valid
    if (currentTab == NULL) {

        return NULL;
    }

    QVariant property = currentTab->property("Image"); //We store the image property

    if (!property.isValid()) {
        QMessageBox::warning(this, "Error", "Not an image");
        return NULL;
    }


    //Now that we know that the property is valid we may retrieve it
    Image* image = (Image*)property.value<Image*>();
    return image;
}


void ComputerVisionApplication::doubleClickFunctionality(Qt::MouseButton button) {

    Image* image = getImage();
    if (image == NULL) {

        return;
    }



    if (button == Qt::LeftButton) {

        image->numClasses += 1;
        return;
    }

    if (button == Qt::RightButton) {

        image->numClasses -= 1;
        return;
    }


}

void ComputerVisionApplication::singleClickFunctionality(Qt::MouseButton button, const QPoint& mousePos) {
    qDebug() << "Exeecuting";

    Image* image = getImage();
    if (image == NULL) {
        return;
    }



    //we check what click it was (left click add points, right click reset classes)


    if (button == Qt::RightButton) {

        image->rectangles.clear();
        //image->numClasses = 2;
        image->matrixClasses.clear();
        QLabel* imageLabel = getImageLabel();
        imageLabel->setPixmap(QPixmap::fromImage(image->image));

        return;
    }


    if (button == Qt::LeftButton) {
        //We check if the classes have been selected
        if (image->rectangles.size() < image->numClasses * 2) {
            image->rectangles.push_back(mousePos);
            //We now draw the border

            if (image->rectangles.size() % 2 == 0) { //We have set the dimensions for the rectangle


                int x = image->rectangles.at(image->rectangles.size() - 1).x();
                int y = image->rectangles.at(image->rectangles.size() - 1).y();

                int width = (image->rectangles.at(image->rectangles.size() - 1).x() - image->rectangles.at(image->rectangles.size() - 2).x());
                int height = (image->rectangles.at(image->rectangles.size() - 1).y() - image->rectangles.at(image->rectangles.size() - 2).y());


                QLabel* imageLabel = getImageLabel();

                QPixmap originalPixmap = (imageLabel)->pixmap();
                QPainter painter(&originalPixmap);
                painter.setPen(Qt::green);

                if (width < 0 || height < 0) {
                    x = image->rectangles.at(image->rectangles.size() - 2).x();
                    y = image->rectangles.at(image->rectangles.size() - 2).y();
                }

                width = abs(width);
                height = abs(height);

                painter.drawRect(abs(x - width), abs(y - height), width, height);
                painter.drawText(x - width, y + 12, QString::fromStdString("Clase: " + std::to_string(image->rectangles.size() / 2 - 1)));
                imageLabel->setPixmap(originalPixmap);
            }


            if (image->rectangles.size() == image->numClasses * 2) {
                int minWidth = INT_MAX;
                int minHeight = INT_MAX;


                for (int i = 0; i < image->rectangles.size(); i += 2) {
                    int width = abs(image->rectangles.at(i).x() - image->rectangles.at(i + 1).x()) + 1;
                    int height = abs(image->rectangles.at(i).y() - image->rectangles.at(i + 1).y()) + 1;

                    if (width < minWidth) {
                        minWidth = width;
                    }
                    if (height < minHeight) {
                        minHeight = height;
                    }
                }

                //We now fill the matrices

                for (int i = 0; i < image->rectangles.size(); i += 2) {
                    int startX = std::min(image->rectangles.at(i).x(), image->rectangles.at(i + 1).x());
                    int startY = std::min(image->rectangles.at(i).y(), image->rectangles.at(i + 1).y());

                    //we now fill the pixels
                    Eigen::Matrix<double, Eigen::Dynamic, 3> matrix;
                    for (int x = startX; x < startX + minWidth; ++x) {
                        for (int y = startY; y < startY + minHeight; ++y) {
                            matrix.conservativeResize(matrix.rows() + 1, Eigen::NoChange);
                            QRgb pixelValue = image->image.pixel(x, y);




                            matrix.row(matrix.rows() - 1) << qRed(pixelValue), qGreen(pixelValue), qBlue(pixelValue);

                        }
                    }
                    //We add the matrix to the classes
                    image->matrixClasses.push_back(matrix);
                }
            }

        }
        else if (image->matrixClasses.size() == 0) { //If this is false we have already filled the classes  
            //Rectangles have finished drawing we now add the pixel to determine and we also determine the smallest dimensions and fill the matrices

            //First we retrieve the pixel to classify


            int minWidth = INT_MAX;
            int minHeight = INT_MAX;


            for (int i = 0; i < image->rectangles.size(); i += 2) {
                int width = abs(image->rectangles.at(i).x() - image->rectangles.at(i + 1).x()) + 1;
                int height = abs(image->rectangles.at(i).y() - image->rectangles.at(i + 1).y()) + 1;

                if (width < minWidth) {
                    minWidth = width;
                }
                if (height < minHeight) {
                    minHeight = height;
                }
            }

            //We now fill the matrices

            for (int i = 0; i < image->rectangles.size(); i += 2) {
                int startX = std::min(image->rectangles.at(i).x(), image->rectangles.at(i + 1).x());
                int startY = std::min(image->rectangles.at(i).y(), image->rectangles.at(i + 1).y());

                //we now fill the pixels
                Eigen::Matrix<double, Eigen::Dynamic, 3> matrix;
                for (int x = startX; x < startX + minWidth; ++x) {
                    for (int y = startY; y < startY + minHeight; ++y) {
                        matrix.conservativeResize(matrix.rows() + 1, Eigen::NoChange);
                        QRgb pixelValue = image->image.pixel(x, y);




                        matrix.row(matrix.rows() - 1) << qRed(pixelValue), qGreen(pixelValue), qBlue(pixelValue);

                    }
                }
                //We add the matrix to the classes
                image->matrixClasses.push_back(matrix);
            }


        }
        else { //We will now only classify pixels

            QRgb classifyPixel = image->image.pixel(mousePos);
            Eigen::Vector3d vec(qRed(classifyPixel), qGreen(classifyPixel), qBlue(classifyPixel));


            if (image->currProcess == "Euclidean") {
                std::vector<double> distances = Computations::Distances::euclidean(image->matrixClasses, vec);
                int closestClass = Computations::Helper::getClosest(distances);

                QString qstr = QString::fromStdString("La clase mas cercana por distance euclidana es: " + std::to_string(closestClass));
                QMessageBox::information(this, "Euclidean Distance Metric", qstr);

            }

            if (image->currProcess == "Manhalanobis") {
                std::vector<double> distances = Computations::Distances::manhalanobis(image->matrixClasses, vec);
                int closestClass = Computations::Helper::getClosest(distances);

                QString qstr = QString::fromStdString("La clase mas cercana por distance manhalanobis es: " + std::to_string(closestClass));
                QMessageBox::information(this, "Manhalanobis Distance Metric", qstr);

            }

            if (image->currProcess == "MaxProb") {
                std::vector<double> probabilites = Computations::Helper::max_prob(image->matrixClasses, vec);
                int closestClass = Computations::Helper::getMaxProb(probabilites);

                QString qstr = QString::fromStdString("La clase mas cercana por criterio de maxima probabilidad es: " + std::to_string(closestClass));
                QMessageBox::information(this, "Maximum Probability Metric", qstr);

            }

            if (image->currProcess == "KNN") {
                int result = MachineLearning::kNearestNeighbours(image->matrixClasses, vec, knn);
                QString qstr = QString::fromStdString("La clase mas cercana por criterio de KNN es: " + std::to_string(result));
                QMessageBox::information(this, "KNN Metric", qstr);

            }

        }








        return; // Event handled
    }
}


void ComputerVisionApplication::convertTabToDock(QTabWidget* tabWidget, int tabIndex) {
    QWidget* widget = tabWidget->widget(tabIndex);
    QString title = tabWidget->tabText(tabIndex);

    //We remove the widget from the tab

    tabWidget->removeTab(tabIndex);

    //we now create a tabbed element that will store the docks

    if (this->sideDock == NULL) {
        sideDock = new QDockWidget("Docked Tabs", this);
        sideDock->setAllowedAreas(Qt::RightDockWidgetArea);
        sideDock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetClosable);
        this->addDockWidget(Qt::RightDockWidgetArea, sideDock);
    }

    if (this->dockTabs == NULL) {
        dockTabs = new QTabWidget(this);
    }






    // Create a new dock widget and set it up

    sideDock->setWidget(dockTabs);
    addDockWidget(Qt::RightDockWidgetArea, sideDock);


    QDockWidget* dockWidget = new QDockWidget(title, this); //Dock that will be the content of each tab

    //connect(dockWidget,&QDockWidget::dockLocationChanged,this,&QMainWindow::)

    dockWidget->setWidget(widget); //Content of tab that was deleted
    dockWidget->setAllowedAreas(Qt::RightDockWidgetArea | Qt::LeftDockWidgetArea);

    //dockWidget->setFeatures(QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetClosable);

    dockTabs->addTab(dockWidget, "Dock 1"); //we add the dock to the tabs


}

void ComputerVisionApplication::handleDockLocationChanged(Qt::DockWidgetArea area) {
    QMessageBox::information(this, "Maximum Probability Metric", "message");

    qDebug() << "Dock moved";

}

void ComputerVisionApplication::handleTopLevelChanged(bool topLevel) {
    if (topLevel) {
        qDebug() << "Dock is now a top-level window";
    }

}


void ComputerVisionApplication::updateImage(QImage newImage) {
    QLabel* imagelabel = getImageLabel();
    imagelabel->setPixmap(QPixmap::fromImage(newImage));
    imagelabel->setFixedSize(QPixmap::fromImage(newImage).size());

}
QString ComputerVisionApplication::getFilePath(std::string extension)
{
    // filter of files
    QString filter = QString("Archivos de texto (*.%1)").arg(QString::fromStdString(extension));

    QString filePath = QFileDialog::getOpenFileName(nullptr, "Seleccionar archivo", "", filter);

    if (!filePath.isEmpty()) {
        qDebug() << "Archivo seleccionado:" << filePath;
        return filePath;
    }
    else {
        qDebug() << "Ningún archivo seleccionado.";
    }
    return QString("");
}

Eigen::MatrixXd ComputerVisionApplication::on_actionConected_N4_triggered() {
    Image* image = getImage();
    Image tmp = *image;
    if (image == NULL) {
        QMessageBox::warning(this, tr("Load Image"), tr("Failed, Image is NULL"));

        return Eigen::MatrixXd(0, 0);
    }
    if (image->image.isNull()) {
        QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image"));
        return Eigen::MatrixXd(0, 0);
    }


    QVector<QVector<QPoint>> objects = ObjectDetection::connectedN4(image->image); //Each element of this list represents a object in the image







    for (int i = 0; i < objects.size(); i++) {

        qDebug() << objects[i].size();


    }


    qDebug() << "Object Size : " << objects.size();


















    for (int i = 0; i < objects.size(); i++) {
        int minX, minY, maxX, maxY;
        ObjectDetection::calculateBounds(objects[i], minX, minY, maxX, maxY);

        QPainter painter(&image->image);
        painter.setPen(QPen(Qt::blue, 2));  // Color azul y un grosor de 5

        //we draw the object border
        QVector<QPoint> points = objects[i];
        for (const QPoint& point : points) {
            painter.drawPoint(point);
            
        }

    }



    Eigen::MatrixXd descritorsReturn(objects.size(), 3);
    std::ofstream outFile("FilesOut/objects.csv", std::ios::app);
    if (!outFile.is_open()) {
        std::cerr << "No se pudo abrir el archivo para escritura." << std::endl;
        return Eigen::MatrixXd(0, 0);
    }
    //We iterate over the objects in an image

    std::vector<QImage> subImages = ImageTransformations::calculatezSubImage(image->image);

    for (int i = 0; i < objects.size(); i++) {

        //we get the perimeter
        descritorsReturn(i, 0) = ImageTransformations::ObjectMetrics::calculatePerimeter(objects[i], image->image)[0];
        
        //we get the area
        descritorsReturn(i, 1) = ImageTransformations::ObjectMetrics::calculatePerimeter(objects[i], image->image)[0];

        //We get the excentricity
        descritorsReturn(i, 2) = ImageTransformations::ObjectMetrics::calculateEccentricity(subImages[i]);


        outFile << descritorsReturn(i, 1) << "," << descritorsReturn(i, 0) << "," << descritorsReturn(i, 2) << "," << descritorsReturn(i, 3) << " object " + std::to_string(i + 1) << std::endl;
    }
    outFile.close();

    if (!(image->image.isNull()))
    {

        QLabel* imageLabel = getImageLabel();

        imageLabel->setPixmap(QPixmap::fromImage(image->image));
        imageLabel->setFixedSize(QPixmap::fromImage(image->image).size());

        this->resize(QPixmap::fromImage(image->image).size());
    }
    else
    {
        QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image."));

    }
    return descritorsReturn;
}


void ComputerVisionApplication::on_actionClassify_Image_triggered() {

    QStringList opciones;
    opciones << "K-nn" << "MaxProb" << "Euclidian" << "Mahalanobis";

    bool ok;
    QInputDialog::getItem(nullptr, "Select Option", "Select any option:", opciones, 0, false, &ok);
    qDebug() << "Entering";
    Image* image = getImage();


    QVector<QVector<QPoint>> objects = ObjectDetection::connectedN4(image->image); //Each element of this list represents a object in the image


    std::vector < std::function<std::vector<int>(QVector<QPoint>, QImage&)>> func = {
        //ImageTransformations::ObjectMetrics::calculateArea,
        //ImageTransformations::ObjectMetrics::calculateCenterOfGravity,
        ImageTransformations::ObjectMetrics::calculatePerimeter
    };


    //we first will get the centroids from the kmeans algorithm, but in this case, we don't have them so we will randomly generate 5 centroids


    QString filePath = "filesout/objects.csv";
    DataStore test = Reader::read_csvToDataStore(filePath);


    
    auto result = MachineLearning::Kmeans(test.numericData, 5, 0.001);

    // Fill the matrix with random values
    //Eigen::MatrixXd centroids= result.second;





    ////We will pass our labeled images through the kmeans algorithm and see which classification we get



    std::map<int, std::string> namesMap;

    namesMap[0] = "Cola de Pato";
    namesMap[1] = "Tornillo";
    namesMap[2] = "Rondana";
    namesMap[3] = "Allen";
    namesMap[4] = "Argolla";

    //Allen y Gancho no claramente identificados

    QImage imageTest = ImageTransformations::ColorSpaces::threshold(image->image, 130);
    qDebug() << "antes";
    std::vector<std::string> classif = ObjectDetection::classifyImage(imageTest, result.second, func, namesMap);

    //we now label the image that we just processed

    std::map<std::string, int> resultsMap;
    resultsMap["Cola de Pato"] = 0;
    resultsMap["Tornillo"] = 0;
    resultsMap["Rondana"] = 0;
    resultsMap["Allen"] = 0;
    resultsMap["Argolla"] = 0;

    //QVector<QVector<QPoint>> objects = ImageTransformations::connectedN4(image->image); //Each element of this list represents a object in the image
    QPainter painter(&image->image);
    int minX, minY, maxX, maxY;
    for (int i = 0; i < objects.size(); i++) {
        ObjectDetection::calculateBounds(objects[i], minX, maxX, minY, maxY);

        painter.setPen(QPen(Qt::green, 2));  // Color azul y un grosor de 
        painter.drawRect(minX, minY, maxX - minX + 10, maxY - minY + 10);  // Rectángulo en la posición (50,50) con ancho 300 y alto 200
        QString str = QString::fromStdString(classif[i]);
        resultsMap[classif[i]]++;
        painter.setPen(QPen(Qt::red, 2));  // Color azul y un grosor de 5

        painter.drawText(minX - 5, maxY - 5, 100, 20, Qt::AlignLeft, str);

    }
    std::vector<std::string> itemNames = { "Cola de Pato" ,"Tornillo" , "Rondana", "Allen" , "Argolla" };
    for (auto item : itemNames) {
        qDebug() << "Hay " << resultsMap[item] <<" "<< item;

    }


    if (!(image->image.isNull()))
    {
        
        QLabel* imageLabel = getImageLabel();

        imageLabel->setPixmap(QPixmap::fromImage(image->image));
        imageLabel->setFixedSize(QPixmap::fromImage(image->image).size());

        this->resize(QPixmap::fromImage(image->image).size());
    }
    else
    {
        QMessageBox::warning(this, tr("Load Image"), tr("Failed to load the image."));

    }

}
