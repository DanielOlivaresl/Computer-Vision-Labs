#include <fstream>
#include <cstdlib> 
#include<Eigen/Dense>
#include <string>
#include<qstring.h>
#include <QDebug>
#include <cctype>
#include <QDir>
#include <QImage>

#ifndef A_READER
#define A_READER

struct DataStore
{
	Eigen::MatrixXd numericData;
	std::vector<std::vector<std::string>> stringData;
};


class Reader {
public:
	static Eigen::MatrixXd read_csv(QString filePath); // this reader does not store string variables
	static DataStore read_csvToDataStore(QString filePath);
	static void imageObjectsToCsv(QImage& image, QString fileName, std::string csvFileName, std::vector<QImage>& subimages);
	static void storeImages(std::string path, std::vector<QImage> images, int counter);

};


#endif
