#include <fstream>
#include <cstdlib> 
#include<Eigen/Dense>
#include <string>
#include<qstring.h>
#include <QDebug>
#include <cctype>
struct DataStore
{
	Eigen::MatrixXd numericData;
	std::vector<std::vector<std::string>> stringData;
};
Eigen::MatrixXd read_csv(QString filePath); // this reader does not store string variables
DataStore read_csvToDataStore(QString filePath);

