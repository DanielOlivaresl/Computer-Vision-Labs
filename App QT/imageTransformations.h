#include <fstream>
#include <cstdlib>  
#include <ctime>    
#include <cmath>
<<<<<<< HEAD
#include <complex>
#include<string>
=======
#include <tuple>

>>>>>>> 7aec56e2c974d9237c6bde378b1d486b4bfd8c68
#include <QPoint>
#include <QImage>
#include <vector>
#include <QPainter>
#include <QRect>
#include <QColor>
#include <QDir>
#include <QVector>
#include <QMessageBox>



#include "computations.h"

class ImageTransformations {

public:
	static QImage convertToGray(QImage& image);
	static std::vector<std::vector<QRgb>> getPixels(QImage image);
	static QImage createImage(std::vector<std::vector<QRgb>> matrix);
	static QImage negativeImage(QImage image);
	static QImage logTransform(QImage image, double c);
	static QImage gammaTransform(QImage, double c, double alpha);

	static QImage normalizeImage(const QImage& image);

	//Domain Tranforms
	static QImage DiscreteFFT(QImage image);
	static QImage InverseFFT(QImage image);


	//Kernels
	QImage createGaborKernel(int ksize,double sigma, double theta, double lambda, double gamma, double psi );
	static std::vector<QImage> createGarborKernels(const std::vector<int>& orientationsPerScale, int imageSize);


	//Filters

	QImage gaborFilter(const QImage &image,const QImage &kernel);
	static QImage preFilter(const QImage& image, double fc = 4.0); //Prefilters the image byt preforming high-pass & local contrast normalization 
	//DownSampling
	static QImage downN(QImage& img, int N);

	static QImage padArray(QImage image, int padSize);



	//Histogram Processing
	static std::vector<double> normalizeHistogram(std::vector<int> histogram);
	static std::vector<int> linearHistogramEqualization(QImage image, int maxVal);
	static std::vector<int> computeHistogram(QImage image);
	static QImage histogramToImage(std::vector<int> histogram, QImage image);
	//Histogram Specification
	static std::vector<int> equalizationHistogram(std::vector<int> histogram, int maxVal);
	std::vector<int> histogramSpecification(std::vector<int> histogram, std::vector<double> specifiedHistogram, int maxVal);




	static void imageObjectsToCsv(QImage& image, QString fileName, std::string csvFileName, std::vector<QImage>& subimages);
	static QImage threshold(QImage& image, int threshold);
	static QVector<QVector<QPoint>> connectedN4(QImage& image);
	static  QVector<QPoint> outLine(QImage& image, int i, int j);




	static QPoint nextCell(QPoint& crrPixel, int& currDir, QPoint& savePixel);

	static QVector<QPoint> borderFollow(QImage& img, QPoint& start, QPoint& prev, int direction, int& NBD);
	static QVector<QVector<QPoint>> rasterScan(QImage& img);







	//unclassified methods

	static std::vector<std::string> classifyImage(QImage& image, Eigen::MatrixXd centroids, std::vector < std::function <std::vector<int>(QVector<QPoint>, QImage&)>> functions, std::map<int, std::string> namesMap);
	static void calculateBounds(QList<QPoint> objectBorder, int &minX, int &maxX,int &minY, int &maxY );
	static std::vector<QImage> calculatezSubImage(QImage& image);
	static void storeImages(std::string path, std::vector<QImage> images,int counter);

	static std::vector<std::vector<double>> computeGistDescriptor(std::vector<QImage> images);
	static std::vector<double> gistGabor(const QImage& image, int w,std::vector<QImage> g,int boundaryExtension);

};
