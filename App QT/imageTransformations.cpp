#include "imageTransformations.h"

QImage ImageTransformations::convertToGray(QImage& image) {
	if (image.isNull()) {
		return image;
	}
	//We obtain the image matrix
	std::vector<std::vector<QRgb>> matrix = getPixels(image);
	int height = matrix.size();
	int width = matrix[0].size();
	//We then convert the image to gray



	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			QRgb currentPixel = matrix[y][x];
			int value = qRed(currentPixel) * 0.299 + qGreen(currentPixel) * 0.587 + qBlue(currentPixel) * 0.114;
			matrix[y][x] = qRgb(value, value, value);

		}
	}


	image = createImage(matrix);
	return image;

}

void ImageTransformations::imageObjectsToCsv(QImage& image, QString filaname, int i, std::vector<QImage>& subimages)
{
	qDebug() << "ENTRANDO en " << i << '\n';
	// calling the function that retrieves the information 
	QVector<QVector<QPoint>> objects = ImageTransformations::connectedN4(image);
	std::ofstream outFile("FilesOut/objects.csv", std::ios::app);

	if (!outFile.is_open()) {
		std::cerr << "No se pudo abrir el archivo para escritura." << std::endl;
		return;
	}

	for (int i = 0; i < objects.size(); i++) {
		// getting the subimage of the object
		QVector<QPoint> pointsS = objects[i];
		int minX = pointsS[0].x();
		int minY = pointsS[0].y();
		int maxX = pointsS[0].x();
		int maxY = pointsS[0].y();

		// Find the bounding box of the object
		for (const QPoint& point : pointsS) {
			if (point.x() < minX) {
				minX = point.x();
			}
			if (point.x() > maxX) {
				maxX = point.x();
			}
			if (point.y() < minY) {
				minY = point.y();
			}
			if (point.y() > maxY) {
				maxY = point.y();
			}
		}

		// Create a new image containing only the object
		QImage objectImage = image.copy(minY - 5, minX - 5, maxY - minY + 10, maxX - minX + 10);
		subimages.push_back(objectImage);
		descritorsReturn(i, 0) = objects[i].size();

		//we get the area
		 //let's  suppose we have an image whit an objects outline 
		 /*
			 -------------------------
			 |00000000000000000000000|
			 |00000011111111110000000|
			 |00000010000000010000000|
			 |00000010000000010000000|
			 |00000010000000010000000|
			 |00000010000000010000000|
			 |00000010011110010000000|
			 |00000010010010010000000|
			 |00000010010010010000000|
			 |00000011110011110000000|
			 |00000000000000000000000|
			 -------------------------

		 */
		 //The perimeter is the size of objects outline
		 //First, we sort the tuples of poinst
		QVector<QPoint> points = objects[i];
		std::sort(points.begin(), points.end(), [](const QPoint& a, const QPoint& b) {
			return a.x() < b.x();
			});

		//we obtain the marginals of each pixel
		std::map<int, QVector<QPoint>> clusters;
		for (const auto& points : points) {
			clusters[points.y()].push_back(points);
		}

		//We get the area
		int area = 0;

		for (const auto& cluster : clusters) {
	//We first store the functions to be applied to a list
	std::vector < std::function<std::vector<double>(QVector<QPoint>, QImage&)>> func = {
		ObjectMetrics::calculateArea,
		ObjectMetrics::calculateCenterOfGravity,
		ObjectMetrics::calculatePerimeter
	};
	
	


	




	for (int i = 0; i < objects.size(); i++) {//we iterate the objects

		Eigen::MatrixXd descritorsReturn = ObjectMetrics::featureExtraction(func, objects[i], image);



		for (int j = 0; j < descritorsReturn.cols(); j++) { //we iterate the features of the objects 
			outFile << descritorsReturn(i, j) << ",";
		}
		outFile << "object" << std::to_string(i + 1) << "At " << filaname.toStdString() << std::endl;



	}
	outFile.close();
	qDebug() << "EXITO" << '\n';
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

QImage ImageTransformations::createImage(std::vector<std::vector<QRgb>> matrix) {

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


QImage ImageTransformations::negativeImage(QImage image) {

	//We will first convert the image to grayscale
	image = convertToGray(image);

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

QImage ImageTransformations::logTransform(QImage image, double c)
{
	//We will first convert the image to grayscale
	image = convertToGray(image);

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

QImage ImageTransformations::gammaTransform(QImage image, double c, double alpha) {

	image = convertToGray(image);

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

std::vector<double> ImageTransformations::normalizeHistogram(std::vector<int> histogram) {

	int count = 0; // Total Number of pixels

	for (int i = 0; i < histogram.size(); i++) {
		count += histogram[i];
	}

	std::vector<double> normalizedHist(256, 0);

	for (int i = 0; i < 256; i++) {
		normalizedHist[i] = static_cast<double>(histogram[i]) / count;
	}


	return normalizedHist;
}

std::vector<int> ImageTransformations::linearHistogramEqualization(QImage image, int maxVal)
{
	//Max val parameter will specify the range of the equalization it will be from [0-maxVal]
	std::vector<int> hist = computeHistogram(image);

	std::vector<double> normalizedHistogram = normalizeHistogram(hist);

	//we will now apply transformations to the normalized histogram


	std::vector<int> res(256, 0);

	for (int i = 0; i < 256; i++) {
		double sum = 0;
		for (int j = 0; j < i; j++) {
			sum += normalizedHistogram[j];
		}
		res[i] = maxVal * sum;


	}
	return res;

}



std::vector<int> ImageTransformations::computeHistogram(QImage image)
{
	//We fill the histogram
	image = ImageTransformations::convertToGray(image);
	std::vector<std::vector<QRgb>> matrix = ImageTransformations::getPixels(image);

	std::vector<int> histogram(256, 0);

	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			histogram[qGray(matrix[i][j])]++;
		}
	}

	return histogram;

}

QImage ImageTransformations::histogramToImage(std::vector<int> transformedHist, QImage originalImage)
{


	//we Remap the values

	std::vector<std::vector<QRgb>> matrix = getPixels(originalImage);

	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			int pixelVal = transformedHist[qGray(matrix[i][j])];

			matrix[i][j] = qRgb(pixelVal, pixelVal, pixelVal);
		}
	}
	QImage res = createImage(matrix);
	return res;

}

//Histogram Specification functions
//Histogram Specification: Process of modyfing a histogram to achieve a certain histogram shape

//We compute the discrete pdf (Probabiltiy Distribution Function)
std::vector<int> ImageTransformations::equalizationHistogram(std::vector<int> histogram, int maxVal)
{
	std::vector < double > normalized = normalizeHistogram(histogram);
	std::vector <int> res(256, 0);
	double sum = 0;
	for (int i = 0; i < 256; i++) {
		sum += normalized[i];
		res[i] = maxVal * sum;



	}



	return res;
}

std::vector<int> ImageTransformations::histogramSpecification(std::vector<int> histogram, std::vector<double> specifiedHistogram, int maxVal)
{

	std::vector<int> g(256, 0);
	double sum = 0;
	for (int i = 0; i < 256; i++) {
		sum += specifiedHistogram[i];
		g[i] = sum * maxVal;


	}

	std::vector<int> eqHist = equalizationHistogram(histogram, maxVal);

	std::vector<int> res(256, 0);

	for (int i = 0; i < 256; i++) {

	}
	return eqHist;


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

	thereshold(image, 130);
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
		while (j < matrix[i].size()) {
			if (qRed(matrix[i][j]) > 250 && !pointExistsInVector(objects, i, j) && count == 0) {
				objects.append(outLine(image, i, j));
				i = 0;
				j = 0;

			}
			else if (qRed(matrix[i][j]) > 250 && !pointExistsInVector(objects, i, j) && count != 0) {
				i += 1;
				continue;
			}
			else if (qRed(matrix[i][j]) > 250 && pointExistsInVector(objects, i, j) && count == 0) {
				count += 1;
			}
			else if (qRed(matrix[i][j]) > 250 && pointExistsInVector(objects, i, j) && count != 0 && !pointExistsInVector(objects, i, jPrevious)) {
				count = 0;
			}
			else if (qRed(matrix[i][j]) == 0 && pointExistsInVector(objects, i, jPrevious) && count != 0) {
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

void ImageTransformations::classifyImage(QImage& image, Eigen::MatrixXd centroids, std::vector < std::function <std::vector<double>(QVector<QPoint>, QImage&)>> functions, std::map<int,std::string> namesMap){
	
	//first we will apply all the transformations that were applied to images in the dataset, in order to get the metrics


	QVector<QVector<QPoint>> objects;
	objects = connectedN4(image);
	
	

	//we iterate each of the objects and get they're features

	std::vector<Eigen::MatrixXd> objectMetrics;

	std::vector<int> classification; //Vector that stores the classification of each object


	for (auto object : objects) {
		Eigen::VectorXd pointToClassify = ObjectMetrics::featureExtraction(functions, object, image);
		objectMetrics.push_back(pointToClassify);
		//Once we got they're features we will calculate the distance of each object to the centroid given by Kmeans

		std::vector<double> distances =euclidean(objectMetrics[objectMetrics.size() - 1], pointToClassify);
		classification.push_back(getClosest(distances));

	}


	//now that we have classified the objects we print the results


	for (int i = 0; i < classification.size(); i++) {
		std::string className;
		if (namesMap.find(classification[i]) == namesMap.end()) {
			className = "None";
		}else {
		
			className = namesMap[i];
		}
		qDebug() << "Object[" << i << "] belongs to class: " << className <<"\n";
	}



















}


