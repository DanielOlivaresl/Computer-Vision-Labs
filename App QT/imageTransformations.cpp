#include "imageTransformations.h"

void ImageTransformations::convertToGray(QImage& image) {
	if (image.isNull()) {
		return;
	}
	//We obtain the image matrix
	std::vector<std::vector<QRgb>> matrix = getPixels(image);
	int height = matrix.size();
	int width = matrix[0].size();


	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			QRgb currentPixel = matrix[y][x];
			int value = qRed(currentPixel) * 0.299 + qGreen(currentPixel) * 0.587 + qBlue(currentPixel) * 0.114;
			matrix[y][x] = qRgb(value, value, value);

		}
	}


	image = createImage(matrix);
}


void ImageTransformations::thereshold(QImage& image, int threshold) {
	if (image.isNull()) {
		return;
	}
	

	//We pass to gray color
	convertToGray(image);

	//We obtain the image matrix
	std::vector<std::vector<QRgb>> matrix = getPixels(image);
	int height = matrix.size();
	int width = matrix[0].size();
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			QRgb currentPixel = matrix[y][x];
			if (qRed(currentPixel) >= threshold) {
				matrix[y][x] = qRgb(255, 255, 255);
			}
			else {
				matrix[y][x] = qRgb(0, 0, 0);
			}

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

 bool pointExistsInVector(const QVector<QVector<QPoint>>& vector, int i, int j) {
	 for (const QVector<QPoint>& pintsVector : vector) {
		 for (const QPoint& point : pintsVector) {
			 if (point.x() == i && point.y() == j) {
				 return true; // El punto se encontró en uno de los contornos
			 }
		 }
	 }
	 return false; // El punto no se encontró
 }


 QVector<QVector<QPoint>> ImageTransformations::connectedN4(QImage& image) {
	 
	 thereshold(image,130);
	 std::vector<std::vector<QRgb>> matrix = getPixels(image);
	 
	 //x-axis count
	 int i = 0;
	 //y-axis count
	 int j = 0;
	 //aux count
	 int count = 0;

	 int height = matrix.size();
	 int width = matrix[0].size();


	 QVector<QVector<QPoint>> objects;
	 

	 while (i < matrix.size()) {
		 int jPrevious = 0;
		 while (j< matrix[i].size()) {
			 if (qRed(matrix[i][j]) > 250 && !pointExistsInVector(objects, i, j) && count ==0) {
				 objects.append(outLine(image, i, j));
				 i = 0;
				 j = 0;

			 }
			 else if (qRed(matrix[i][j]) > 250 && !pointExistsInVector(objects, i, j) && count != 0) {
				 i += 1;
				 continue;
			 }else if (qRed(matrix[i][j]) > 250 && pointExistsInVector(objects, i, j) && count == 0) {
				 count += 1;
			 }else if (qRed(matrix[i][j]) > 250 && pointExistsInVector(objects, i, j) && count != 0 && !pointExistsInVector(objects, i, jPrevious)) {
				 count = 0;
			 }else if (qRed(matrix[i][j]) ==0  && pointExistsInVector(objects, i, jPrevious) && count != 0) {
				 count = 0;
			 }
			 jPrevious = j;
			 j += 1;
			
					

		 }
		 i += 1;
		 j = 0;
	 }
	 
	 
	 return objects;
 }



 /**
 * @brief Function that find object online
 * @param QImage: Image to segment
 * @param int i: x-axis coordinate
 * @param int j: y-axis coordinate
 *
 */
 QVector<QPoint> ImageTransformations::outLine(QImage& image, int i, int j) {
	 //we define the objects vector
	 QVector<QPoint> object;
	 //we define variables that we'll use later
	 int currI = i;
	 int currJ = j;
	 int d = 0;
	 std::vector<std::vector<QRgb>> matrix = getPixels(image);
	 while (true) {
		 int q = (d + 3) % 4;
		 bool breakFlag = false; // for brak in principal loop

		 for (int p = 0; p < 4; ++p) {
			 int dir = (q + p) % 4;
			 int nextI = currI;
			 int nextJ = currJ;

			 switch (dir) {
			 case 0: // up
				 nextI--;
				 break;
			 case 1: // right
				 nextJ++;
				 break;
			 case 2: // down
				 nextI++;
				 break;
			 case 3: // left
				 nextJ--;
				 break;
			 }

			 // We check the pixel value
			 if (nextI >= 0 && nextI < image.height() && nextJ >= 0 && nextJ < image.width() &&
				 qRed(matrix[nextI][nextJ]) == 255) {
				 currI = nextI;
				 currJ = nextJ;
				 d = dir;

				 object.append(QPoint(currI, currJ));
				 breakFlag = true;
				 break;
			 }
		 }

		 // stop condition
		 if (!breakFlag || (currI == i && currJ == j)) {
			 break;
		 }
	 }
	 return object;
 }



 