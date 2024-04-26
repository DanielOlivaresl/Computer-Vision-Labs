#include <QImage>
#include <vector>
#include <QPainter>
#include <QRect>
#include <QColor>


class ImageTransformations {

public:
	
	static std::vector<std::vector<QRgb>> getPixels(QImage image);
	static QImage createImage(std::vector<std::vector<QRgb>> matrix);

	static void convertToGray(QImage &image);
	static void thereshold(QImage& image, int thresholded);

	static QVector<QVector<QPoint>> connectedN4(QImage& image);
	static  QVector<QPoint> outLine(QImage& image, int i, int j);
	



	
};

 
	