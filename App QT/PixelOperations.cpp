#include "imageTransformations.h"



QImage ImageTransformations::PixelOperations::negativeImage(QImage image) {

	//We will first convert the image to grayscale
	image = ColorSpaces::convertToGray(image);

	//We then obtain the matrix of pixels
	std::vector<std::vector<QRgb>> matrix = getPixels(image);

	//Now we will compute the negative

	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			matrix[i][j] = qRgb(255 - matrix[i][j], 255 - matrix[i][j], 255 - matrix[i][j]);
		}
	}

	//We create the image
	QImage result = createImage(matrix);

	//We attempt to save the image

	if (result.isNull()) {
		return result;
	}

	if (!result.save(image.text() + "negative.png")) {
		return result;
	}

	return result;



}

QImage ImageTransformations::PixelOperations::logTransform(QImage image, double c)
{
	//We will first convert the image to grayscale
	image = ColorSpaces::convertToGray(image);

	//We then obtain the matrix of pixels
	std::vector<std::vector<QRgb>> matrix = getPixels(image);

	//we will now iterate the image and apply a log transform to each point

	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			int intensity = qGray(matrix[i][j]);
			int tranformedVal = c * log(1 + intensity);
			tranformedVal = qBound(0, tranformedVal, 255);
			matrix[i][j] = qRgb(tranformedVal, tranformedVal, tranformedVal);
		}
	}
	//We create the image
	QImage result = createImage(matrix);

	//We attempt to save the image

	if (result.isNull()) {
		return result;
	}

	if (!result.save(image.text() + "log.png")) {
		return result;
	}

	return result;


}

QImage ImageTransformations::PixelOperations::gammaTransform(QImage image, double c, double alpha) {

	image = ColorSpaces::convertToGray(image);

	std::vector<std::vector<QRgb>> matrix = getPixels(image);

	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			int intensity = qGray(matrix[i][j]);
			int transform = c * (pow(intensity, alpha));
			transform = qBound(0, transform, 255);
			matrix[i][j] = qRgb(transform, transform, transform);
		}
	}
	QImage res = createImage(matrix);
	return res;

}
