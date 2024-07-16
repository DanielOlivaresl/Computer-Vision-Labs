#ifndef IMAGETRANSFORMATIONS_H
#define IMAGETRANSFORMATIONS_H



#include <fstream>
#include <cstdlib>  
#include <ctime>    
#include <cmath>
#include <tuple>

#include <QPoint>
#include <QImage>
#include <vector>
#include <QPainter>
#include <QRect>
#include <QColor>
#include <QDir>
#include <QVector>
#include <QMessageBox>
#include <QStack>
#include <stack>
#include "computations.h"
#include <queue>
#include "rapidcsv.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>




class ImageTransformations {

public:

	static QImage createImage(std::vector<std::vector<QRgb>> matrix);
	static std::vector<std::vector<QRgb>> getPixels(QImage image);
	static std::vector<std::vector<double>> normalizeImage(const QImage& image);
	static cv::Mat QImageToMat(const QImage& inputImage, bool inCloneImageData = true);
	static std::vector<QImage> calculatezSubImage(QImage& image);


	class ColorSpaces {
	public:
		static QImage convertToGray(QImage& image);
		static QImage threshold(QImage& image, int threshold);

	};

	class ObjectMetrics {
	public:
		static std::vector<int> calculateArea(QVector<QPoint> object, QImage& image);
		static std::vector<int> calculatePerimeter(QVector<QPoint> object, QImage& image);
		static std::vector<int> calculateCenterOfGravity(QVector<QPoint> object, QImage& image);
		static Eigen::MatrixXd featureExtraction(std::vector < std::function <std::vector<int>(QVector<QPoint>, QImage&)>> functions, QVector<QPoint> object, QImage& image);
		static double imageMoments(QImage& binarizedImage, int p, int q);
		static double centralMoments(QImage& binarizedImage, int p, int q, double xCent, double yCent);
		static double calculateEccentricity(QImage& image);

	};


	class PixelOperations {
	public:
		static QImage negativeImage(QImage image);
		static QImage logTransform(QImage image, double c);
		static QImage gammaTransform(QImage, double c, double alpha);
	};

	class SpatialOperations {
	public:
		static std::vector<std::vector<double>> downN(std::vector<std::vector<double>>& img, int N);
		static std::vector<std::vector<double>> padArray(std::vector<std::vector<double>> image, int padSize);
		QImage gaborFilter(const QImage& image, const QImage& kernel);
		static std::vector<std::vector<double>> preFilter(const std::vector<std::vector<double>>& image, double fc = 4.0); //Prefilters the image byt preforming high-pass & local contrast normalization 

	};

	
	class DomainTransforms {
	public:
		static QImage DiscreteFFT(QImage image);
		static QImage InverseFFT(QImage image);
	};


	class Kernels {
	public:
		QImage createGaborKernel(int ksize, double sigma, double theta, double lambda, double gamma, double psi);
		static std::vector < std::vector<std::vector<double>>> createGarborKernels(const std::vector<int>& orientationsPerScale, int imageSize);

	};
	
	
	
	class Histogram {
	public:
		static std::vector<double> normalizeHistogram(std::vector<int> histogram);
		static std::vector<int> linearHistogramEqualization(QImage image, int maxVal);
		static std::vector<int> computeHistogram(QImage image);
		static QImage histogramToImage(std::vector<int> histogram, QImage image);
		//Histogram Specification
		static std::vector<int> equalizationHistogram(std::vector<int> histogram, int maxVal);
		std::vector<int> histogramSpecification(std::vector<int> histogram, std::vector<double> specifiedHistogram, int maxVal);

	};

	


	

	















	
	//unclassified methods
	static std::vector<std::vector<double>> computeGistDescriptor(std::vector<QImage> images, std::vector<std::string> imageClass);
	static std::vector<double> gistGabor(std::vector<std::vector<double>>& image, int w, std::vector<std::vector<std::vector<double>>> g, int boundaryExtension);

};




#endif