#include <QImage>
#include <vector>

class ImageTransformations {

public:
	static void convertToGray(QImage &image);
	static std::vector<std::vector<QRgb>> getPixels(QImage image);
	static QImage createImage(std::vector<std::vector<QRgb>> matrix);

	
};


	