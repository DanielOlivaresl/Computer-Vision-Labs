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
            qDebug() << matrix(i,j) <<" ";
        }
        qDebug() << '\n';
    }


    qDebug() << "size rows " << matrix.rows();
    qDebug() << "size cols " << matrix.cols();
    // Cerrar el archivo
    file.close();
    return matrix;
}
DataStore read_csvToDataStore(QString filePath) {
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
