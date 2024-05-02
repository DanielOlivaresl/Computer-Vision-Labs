#include <QImage>
#include <vector>
#include <QPainter>
#include <QRect>
#include <QColor>


#include <cmath>
#include "computations.h"
#include <QMessageBox>
class ImageTransformations {

public:
	static QImage convertToGray(QImage &image);
	static std::vector<std::vector<QRgb>> getPixels(QImage image);
	static QImage createImage(std::vector<std::vector<QRgb>> matrix);
	static QImage negativeImage(QImage image);
	static QImage logTransform(QImage image, double c);
	static QImage gammaTransform(QImage, double c, double alpha);

	//Histogram Processing
	static std::vector<double> normalizeHistogram(std::vector<int> histogram); 
	static std::vector<int> linearHistogramEqualization(QImage image, int maxVal);
	static std::vector<int> computeHistogram(QImage image);
	static QImage histogramToImage(std::vector<int> histogram,QImage image);
		//Histogram Specification
	static std::vector<int> equalizationHistogram(std::vector<int> histogram, int maxVal);
	std::vector<int> histogramSpecification(std::vector<int> histogram,std::vector<double> specifiedHistogram, int maxVal);

	static void thereshold(QImage& image, int thresholded);
	static QVector<QVector<QPoint>> connectedN4(QImage& image);
	static  QVector<QPoint> outLine(QImage& image, int i, int j);
};

 
	