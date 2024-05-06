#include "reader.h"

Eigen::MatrixXd read_csv(QString filePath) {
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
            qDebug() << matrix(i,j) <<" ";
        }
        qDebug() << '\n';
    }


    qDebug() << "size rows " << matrix.rows();
    qDebug() << "size cols " << matrix.cols();
    // closing
    file.close();
    return matrix;
}
