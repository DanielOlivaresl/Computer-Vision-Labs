#include "imageTransformations.h"

 void ImageTransformations::convertToGray(QImage &image) {
	if (image.isNull()) {
		return ;
	}
	//We obtain the image matrix
	std::vector<std::vector<QRgb>> matrix = getPixels(image);
	int height = matrix.size();
	int width = matrix[0].size();


	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
	
			QRgb currentPixel = matrix[y][x];
			int value = qRed(currentPixel) * 0.299+ qGreen(currentPixel) * 0.587+ qBlue(currentPixel) * 0.114;
			matrix[y][x] = qRgb(value,value,value);
			
		}
	}
	

	image = createImage(matrix);
	

}

 std::vector<std::vector<QRgb>> ImageTransformations::getPixels(QImage image) {
	
	if (image.isNull()) {
		return std::vector<std::vector<QRgb>>();
	}

	std::vector<std::vector<QRgb>> rgbMatrix(image.height(), std::vector<QRgb>(image.width()));

	for (int y = 0; y < image.height(); y++) {
		for (int x = 0; x < image.width(); x++) {
			//We access the pixel at (x,y)
			QRgb pixelValue = image.pixel(x, y);
		
			//We store the pixel value in the matrix 
			rgbMatrix[y][x] = pixelValue;
		}
	}

	// Now rgbMatrix contains the RGB values of each pixel in the image
	// You can access the RGB values like this:
	// int red = qRed(rgbMatrix[y][x]);
	// int green = qGreen(rgbMatrix[y][x]);
	// int blue = qBlue(rgbMatrix[y][x]);

	return rgbMatrix;
}

 QImage ImageTransformations::createImage(std::vector<std::vector<QRgb>> matrix){
	
	if (matrix.empty() || matrix[0].empty()) {
		return QImage(); //We return an empty image
	}

	int height = matrix.size();
	int width = matrix[0].size();

	QImage image(width, height, QImage::Format_ARGB32);

	for (int y = 0; y < height; y++) {
		QRgb* rowData = reinterpret_cast<QRgb*>(image.scanLine(y));
		for (int x = 0; x < width; x++) {
			rowData[x] = matrix[y][x];
		}
	}
	return image;
}
