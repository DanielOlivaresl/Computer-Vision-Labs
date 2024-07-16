#include "imageTransformations.h"

QImage ImageTransformations::ColorSpaces::convertToGray(QImage& image) {


	QImage newImage(image.width(), image.height(), QImage::Format_Grayscale8);


	for (int x = 0; x < image.width(); x++) {
		for (int y = 0; y < image.height(); y++) {
			double avg = 0;
			avg += qGreen(image.pixel(x, y));
			avg += qRed(image.pixel(x, y));
			avg += qBlue(image.pixel(x, y));
			avg = avg / 3;
			newImage.setPixelColor(x, y, qRgb(avg, avg, avg));


		}
	}



	return newImage;

}


QImage ImageTransformations::ColorSpaces::threshold(QImage& image, int threshold) {
	if (image.isNull()) {
		return image;
	}
	//We pass to gray color
	QImage gray = convertToGray(image);
	QImage thresholdedImage(gray.size(), QImage::Format_Grayscale8);





	for (int y = 0; y < gray.height(); ++y) {
		const uchar* srcRow = gray.constScanLine(y);
		uchar* dstRow = thresholdedImage.scanLine(y);
		for (int x = 0; x < gray.width(); ++x) {
			dstRow[x] = srcRow[x] > threshold ? 255 : 0;
		}
	}

	return thresholdedImage;


}