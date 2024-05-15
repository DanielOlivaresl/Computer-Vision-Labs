#include <fstream>
#include <cstdlib>  
#include <ctime>    
#include <cmath>

#include <QPoint>
#include <QImage>
#include <vector>
#include <QPainter>
#include <QRect>
#include <QColor>

#include "computations.h"
#include <QMessageBox>

class ImageTransformations {

public:
	static QImage convertToGray(QImage& image);
	static std::vector<std::vector<QRgb>> getPixels(QImage image);
	static QImage createImage(std::vector<std::vector<QRgb>> matrix);
	static QImage negativeImage(QImage image);
	static QImage logTransform(QImage image, double c);
	static QImage gammaTransform(QImage, double c, double alpha);

	//Histogram Processing
	static std::vector<double> normalizeHistogram(std::vector<int> histogram);
	static std::vector<int> linearHistogramEqualization(QImage image, int maxVal);
	static std::vector<int> computeHistogram(QImage image);
	static QImage histogramToImage(std::vector<int> histogram, QImage image);
	//Histogram Specification
	static std::vector<int> equalizationHistogram(std::vector<int> histogram, int maxVal);
	std::vector<int> histogramSpecification(std::vector<int> histogram, std::vector<double> specifiedHistogram, int maxVal);


	static void imageObjectsToCsv(QImage& image,QString filaname , int i, std::vector<QImage>& subimages); // used to transform an image into a csv with the information of all the objects in the image 

	static QImage thereshold(QImage& image, int thresholded);
	static QVector<QVector<QPoint>> connectedN4(QImage& image);
	static  QVector<QPoint> outLine(QImage& image, int i, int j);




	static QPoint nextCell( QPoint& crrPixel, int &currDir, QPoint& savePixel);

	static QVector<QPoint> borderFollow(QImage& img, QPoint& start,  QPoint& prev, int direction, int& NBD);
	static QVector<QVector<QPoint>> rasterScan(QImage & img);







	//unclassified methods

	static std::vector<std::string> classifyImage(QImage& image, Eigen::MatrixXd centroids, std::vector < std::function <std::vector<int>(QVector<QPoint>, QImage&)>> functions, std::map<int, std::string> namesMap);



};





