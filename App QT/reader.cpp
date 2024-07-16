#include "reader.h"
#include "ObjectDetection.h"
#include "imageTransformations.h"






Eigen::MatrixXd Reader::read_csv(QString filePath) {
    qDebug() << filePath << "Archivo recibido " << '\n';
    std::ifstream file(filePath.toStdString());
    if (!file.is_open()) {
        qDebug() << "Error al abrir el archivo: " << filePath << '\n';
        return Eigen::MatrixXd();
    }

    std::vector<std::vector<double>> data; // stores the data values 

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row; // Vector to store the data of each row

        std::istringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) {
            // check if the field is numeric 
            bool isNumeric = true;
            for (char c : field) {
                if (!std::isdigit(c) && c != '.' && c != '-') {
                    isNumeric = false;
                    break;
                }
            }
            if (isNumeric) { // is the field is numeric then we can store that in our matrix
                double value = std::stod(field);
                row.push_back(value);
            }
            else {
                // if the field is not numeric the we store that in 
            }
        }

        // pushing into the vector store
        data.push_back(row);
    }

    // vector to  Eigen::MatrixXd
    qDebug() << "size rows " << data.size();
    qDebug() << "size cols " << data[0].size();
    Eigen::MatrixXd matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[i].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }

    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[i].size(); ++j) {
            qDebug() << matrix(i, j) << " ";
        }
        qDebug() << '\n';
    }


    qDebug() << "size rows " << matrix.rows();
    qDebug() << "size cols " << matrix.cols();
    // Cerrar el archivo
    file.close();
    return matrix;
}
DataStore Reader::read_csvToDataStore(QString filePath) {
    qDebug() << filePath << "Archivo recibido" << '\n';
    std::ifstream file(filePath.toStdString());
    if (!file.is_open()) {
        qDebug() << "Error al abrir el archivo: " << filePath << '\n';
        return DataStore();
    }

    std::vector<std::vector<double>> numericData;
    std::vector<std::vector<std::string>> stringData;

    std::string line;
    if (std::getline(file, line)) { // Leer la primera línea para inicializar stringData
        std::istringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) {
            if (field.empty() || std::any_of(field.begin(), field.end(), [](char c) { return !std::isdigit(c) && c != '.' && c != '-'; })) {
                stringData.push_back(std::vector<std::string>()); // Añade un vector para cada columna de texto
            }
        }
    }

    file.clear(); // Limpia los flags de eof()
    file.seekg(0); // Vuelve al inicio del archivo para leer de nuevo

    while (std::getline(file, line)) {
        std::vector<double> numericRow; // Vector to store numeric data of each row
        std::istringstream ss(line);
        std::string field;
        size_t textIndex = 0; // Índice para el vector de vectores de texto
        while (std::getline(ss, field, ',')) {
            if (!field.empty() && std::all_of(field.begin(), field.end(), [](char c) { return std::isdigit(c) || c == '.' || c == '-'; })) {
                numericRow.push_back(std::stod(field));
            }
            else {
                if (textIndex < stringData.size()) {
                    stringData[textIndex++].push_back(field);
                }
            }
        }
        numericData.push_back(numericRow);
    }

    Eigen::MatrixXd matrix(numericData.size(), numericData.empty() ? 0 : numericData[0].size());
    for (size_t i = 0; i < numericData.size(); ++i) {
        for (size_t j = 0; j < numericData[i].size(); ++j) {
            matrix(i, j) = numericData[i][j];
        }
    }

    DataStore result;
    result.numericData = matrix;
    result.stringData = stringData;

    file.close();
    return result;
}




void Reader::storeImages(std::string path, std::vector<QImage> images, int counter)
{
    QString qPath = QString::fromStdString(path);
    QDir().mkpath(qPath);
    for (auto image : images) {
        QString imagePath = QDir(qPath).filePath(QString("SubImage_%1.png").arg(++counter));
        image.save(imagePath);
    }



}




void Reader::imageObjectsToCsv(QImage& image, QString fileName, std::string csvFileName, std::vector<QImage>& subimages)
{

    // calling the function that retrieves the information 
    QVector<QVector<QPoint>> objects = ObjectDetection::connectedN4(image);

    //we first reset the current csv, and then we will start appending to it


    std::ofstream outFile(csvFileName, std::ios::app);

    if (!outFile.is_open()) {
        qDebug()<< "No se pudo abrir el archivo para escritura.";
        return;
    }

    for (int i = 0; i < objects.size(); i++) {
        int minX, maxX, minY, maxY;
        ObjectDetection::calculateBounds(objects[i], minX, maxX, minY, maxY);
        // getting the subimage of the object
        QVector<QPoint> pointsS = objects[i];

        // Find the bounding box of the object


        // Create a new image containing only the object
        QImage objectImage = image.copy(minX - 5, minY - 5, maxX - minX + 10, maxY - minY + 10);

        QImage objectImageBinary = ImageTransformations::ColorSpaces::threshold(objectImage, 130);
        subimages.push_back(objectImageBinary);
        double e = ImageTransformations::ObjectMetrics::calculateEccentricity((objectImageBinary));
        QImage ImageBinary = ImageTransformations::ColorSpaces::threshold(image, 130);



        std::vector < std::function<std::vector<int>(QVector<QPoint>, QImage&)>> func = {
            //ImageTransformations::ObjectMetrics::calculateArea,
            //ImageTransformations::ObjectMetrics::calculateCenterOfGravity,
            ImageTransformations::ObjectMetrics::calculatePerimeter
        };


        for (int i = 0; i < objects.size(); i++) {//we iterate the objects
            Eigen::MatrixXd descritorsReturn = ImageTransformations::ObjectMetrics::featureExtraction(func, objects[i], ImageBinary);

            for (int j = 0; j < descritorsReturn.cols(); j++) { //we iterate the features of the objects 
                outFile << descritorsReturn(0, j) << ",";
            }
            //we end the object in the csv
            outFile << "Object " << i << " " << fileName.toStdString() << "\n";

        }
        outFile.close();
    }
}




